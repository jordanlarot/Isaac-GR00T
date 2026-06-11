# GR00T EEF — Post-download robot deploy guide

What to do after downloading the finetuned **end-effector (EEF)** checkpoint to the robot.

**Model:** `jordanlarot/gr00t-pick-place-bottle-eef-10k` (local: `/home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k`)  
**Task:** pick-place bottle (bimanual demos; deploy is right-arm only)  
**Deploy client:** `gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py`  
**Modality config:** baked into checkpoint `processor_config.json` (`new_embodiment`)

Related docs:

- [commands.md](commands.md) — operator runbook (EEF + joint terminal commands)
- [eef-deploy-jerk-report.md](eef-deploy-jerk-report.md) — hardware jerk/sway analysis from `runs/`
- [eef-deploy-umi-takeaways.md](eef-deploy-umi-takeaways.md) — UMI paper applied: horizon, latency, relative vs absolute
- [gr00t-eef-conversion-results.md](gr00t-eef-conversion-results.md) — dataset conversion and training results
- [gr00t-eef-training-plan.md](gr00t-eef-training-plan.md) — full EE training plan
- [environments.md](environments.md) — Python 3.12 (repo root) vs Python 3.10 (`Isaac-GR00T/.venv`)
- [inference-and-deployment.md](inference-and-deployment.md) — ACT/Pi0 deploy modes (joint-space policies)

---

## Summary

Downloading the checkpoint is only the first step. The EEF model speaks a **different observation/action language** than the current robot stack:

| | EEF checkpoint | Current robot stack |
|---|---|---|
| **State in** | `left_eef_9d`, `right_eef_9d` (9D xyz+rot6d) + grippers + 3 cameras | 14D joint angles via `robot_api_server` |
| **Action out** | Absolute EEF targets (9D per arm) + gripper scalars | 14D joint targets via `/action` |

You need a small deploy bridge: **read EE pose → feed GR00T → convert EEF targets to joint commands → execute**.

GR00T handles **relative → absolute** EEF conversion internally during inference. You do **not** manually integrate deltas in the deploy client.

---

## Architecture

Three processes, same pattern as the joint-space GR00T deploy:

```
┌─────────────────────────┐     HTTP :5000      ┌──────────────────────────────┐
│  robot_api_server.py    │ ◄────────────────── │  deploy_groot_realman_eef.py │
│  (on robot)             │                     │  ee_pose → EEF obs → IK      │
│  state, ee_pose, images │                     │  → 14D joint /action         │
└─────────────────────────┘                     └────────────┬─────────────────┘
                                                             │ ZMQ :5555
                                                             ▼
                                                ┌──────────────────────────┐
                                                │  run_gr00t_server.py     │
                                                │  relative EEF → absolute │
                                                └──────────────────────────┘
```

**What changes for EEF:** observation builder (`ee_pose` → `left/right_eef_9d`), IK executor, and 20D action chunk. Server decodes **relative trajectory** (UMI-style) to absolute EEF; robot receives **absolute** joint commands.

---

## What already exists vs what is missing

| Component | Status | Notes |
|-----------|--------|-------|
| EEF checkpoint | Done | `/home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k` |
| `run_gr00t_server.py` | Done | Relative EEF → absolute in `decode_action` |
| `deploy_groot_realman_eef.py` | Done | EEF obs, IK, prefetch, run logging |
| `eef_utils.py`, `realman_ik.py` | Done | quat/rot6d + QPIK IK (calibrated to dataset frame) |
| `robot_api_server.py` | Done | `/observation` returns `state`, `ee_pose` (14D), images |
| TRT engines for EEF | Optional | Build per `commands.md`; ~4–5 Hz on Orin |
| Stale-action skip (UMI PD1.2) | **Not yet** | See `eef-deploy-umi-takeaways.md` Tier 2 |
| `deploy_groot_realman.py` | Joint only | Wrong client for EEF checkpoint |

The **joint-space** checkpoint (`gr00t-pick-bottle-realman`) + `deploy_groot_realman.py` remain valid as a fallback without IK.

---

## Step-by-step after download

### Step 0 — Prerequisites

**On the robot:**

```bash
# Terminal 1 — arm drivers
cd ~/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch rm_driver dual_rm_65_driver.launch.py
```

**On a GPU machine** (or robot if it has a GPU): Isaac-GR00T env with the downloaded checkpoint.

```bash
# Download checkpoint (if not already local)
# From repo root — huggingface-cli or Python:
#   snapshot_download("jordanlarot/gr00t-pick-place-bottle-eef-10k", local_dir=...)
```

Set `HF_TOKEN` if the model repo is private.

---

### Step 1 — Verify the model (no robot)

Run **open-loop eval** on the GPU machine before touching hardware. This confirms the checkpoint loads, the embodiment config matches, and action shapes are correct.

```bash
cd Isaac-GR00T

uv run python gr00t/eval/open_loop_eval.py \
  --model-path jordanlarot/gr00t-pick-place-bottle-eef-10k \
  --dataset-path ../data/pick_place_bottle_v2_gr00t_eef \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/RealMan/realman_dual_arm_eef_config.py \
  --traj-ids 0 50 101 150 201 \
  --action-horizon 16 \
  --modality-keys left_eef_9d right_eef_9d left_gripper right_gripper
```

Use a local path instead of the HF repo id if you downloaded to disk.

EEF error metrics are in metres / rotation units — not directly comparable to joint-space MAE.

---

### Step 2 — Start the robot API server

```bash
cd /home/r2d3/pickup-objects
export PYTHONPATH=~/pickup-objects/src:$PYTHONPATH
python scripts/robot_api_server.py
```

Smoke-test (must include `ee_pose` with 14 floats for EEF deploy):

```bash
curl -s http://localhost:5000/observation | python3 -c "import sys,json; d=json.load(sys.stdin); print('ee_pose' in d, len(d.get('ee_pose',[])))"
```

**14D `ee_pose` layout:** `[l_xyz(3), l_quat_xyzw(4), r_xyz(3), r_quat_xyzw(4)]` metres + quaternion per arm (each arm in its own Realman base frame). Conversion to `left/right_eef_9d` is in `gr00t/eval/real_robot/realman/eef_utils.py`.

---

### Step 3 — Start the GR00T policy server

On the GPU machine:

```bash
cd Isaac-GR00T
source .venv/bin/activate
source scripts/activate_orin.sh

python gr00t/eval/run_gr00t_server.py \
  --model-path /home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
```

Optional TRT: add `--trt-engine-path .../gr00t-pick-place-bottle-eef-10k-trt/engines --trt-mode dit_only` (see `commands.md`).

The server normalizes actions and converts **relative EEF trajectory → absolute EEF targets** (UMI PD2.1; not step-to-step deltas).

---

### Step 4 — EEF deploy client (IK path implemented)

`deploy_groot_realman_eef.py` handles observation building, IK (`realman_ik.py`), prefetch inference, and run logging. Right arm only; left pinned to observed state.

**Dry run** (inference + IK, no `/action` POST):

```bash
source .venv/bin/activate && source scripts/activate_orin.sh

python gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py \
  --task "pick up bottle" \
  --robot-url http://localhost:5000 \
  --policy-host localhost \
  --policy-port 5555 \
  --dry-run --debug
```

**Closed loop (recommended starting flags):**

```bash
python gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py \
  --task "pick up bottle" \
  --robot-url http://localhost:5000 \
  --policy-host localhost \
  --policy-port 5555 \
  --open-loop-horizon 6 \
  --hz 8 \
  --auto-close-grip \
  --grip-close-threshold 0.95
```

**Tuning jerk/sway:** see [eef-deploy-jerk-report.md](eef-deploy-jerk-report.md) and [eef-deploy-umi-takeaways.md](eef-deploy-umi-takeaways.md). Try `--open-loop-horizon 6–8` before 16; enable TRT on the server.

Confirm startup log shows `GR00T EEF CLOSED-LOOP DEPLOYMENT` and `Initialising RealMan IK`.

---

## End-to-end data flow

```
1. Robot sensors
   udp_arm_position (xyz + quat per arm)
   joint_states (14D)
   3× cameras

2. robot_api_server /observation
   → ee_pose or eef_9d + grippers + images

3. deploy client builds GR00T obs
   video:  top_camera, left_wrist, right_wrist
   state:  left_eef_9d, left_gripper, right_eef_9d, right_gripper
   language: "pick up bottle"

4. GR00T server (run_gr00t_server.py)
   → denormalize
   → relative EEF trajectory + current eef state → absolute EEF targets
   → gripper absolutes

5. deploy_groot_realman_eef.py
   → QPIK IK: absolute EEF → joint targets

6. robot_api_server /action
   → Hardware_Bridge → movej_cmd + gripper
```

---

## Implementation checklist

- [x] **Bridge:** `get_ee_poses_at_time()` in `ros2_bridge.py`
- [x] **API:** `/observation` returns `ee_pose` (14D)
- [x] **IK:** `realman_ik.py` — EEF 9D → 6 joints (right arm deploy)
- [x] **Deploy client:** `deploy_groot_realman_eef.py`
- [x] **Right-arm-only:** left arm pinned to observed state
- [x] **Dry run:** `--dry-run` runs inference + IK without motion
- [ ] **TRT:** build engines for EEF checkpoint (optional speedup)
- [ ] **UMI PD1.2:** stale-action skip after inference latency
- [ ] **Open-loop eval** on held-out EEF trajectories (offline smoothness check)

---

## Known limitations

- **Separate arm frames:** left and right EEF are in each arm's Realman base frame, not a shared world frame. No cross-arm calibration needed.
- **No joint ↔ EEF checkpoint transfer:** action head dimensions differ; always use the matching checkpoint and embodiment config.
- **Reset variability:** EEF start poses vary ~35 mm across episodes. Relative action representation mitigates drift vs absolute-only targets.
- **Data collection gap:** current `collect_data.py` logs joints only, not EE pose. Existing training data already has `observation.ee_pose`; re-enable EE logging for future demos (see [gr00t-eef-training-plan.md](gr00t-eef-training-plan.md) Step 8).

---

## Quick reference — key files

| File | Role |
|------|------|
| `gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py` | EEF closed-loop client |
| `gr00t/eval/real_robot/realman/eef_utils.py` | `ee_pose` ↔ `eef_9d` conversion |
| `gr00t/eval/real_robot/realman/realman_ik.py` | QPIK IK (dataset-frame calibrated) |
| `gr00t/eval/run_gr00t_server.py` | Policy server (relative → absolute EEF) |
| `pickup-objects/scripts/robot_api_server.py` | Robot HTTP API (`ee_pose`, `state`, images) |
| `pickup-objects/src/r2d3/hardware/ros2_bridge.py` | ROS2 bridge, `get_ee_poses_at_time()` |
| `commands.md` | Operator commands (EEF + joint) |
| `eef-deploy-jerk-report.md` / `eef-deploy-umi-takeaways.md` | Hardware tuning guides |

---

## Alternative: joint-space deploy (no IK)

Fallback if IK or EEF checkpoint is unavailable — see **Joint deploy** in [commands.md](commands.md):

- Checkpoint: `/home/r2d3/checkpoints/gr00t-pick-bottle-realman`
- Client: `deploy_groot_realman.py` (14D joint actions directly; no IK)

The EEF checkpoint is the recommended path for bottle pick; joint deploy avoids the IK layer.
