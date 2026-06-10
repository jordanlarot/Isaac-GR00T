# GR00T EEF — Post-download robot deploy guide

What to do after downloading the finetuned **end-effector (EEF)** checkpoint to the robot.

**Model:** `jordanlarot/gr00t-pick-place-bottle-eef-10k`  
**Task:** pick-place bottle (bimanual demos; first deploy can be right-arm only)  
**Training config:** `Isaac-GR00T/examples/RealMan/realman_dual_arm_eef_config.py`

Related docs:

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
┌─────────────────────────┐     HTTP :5000      ┌──────────────────────────┐
│  robot_api_server.py    │ ◄────────────────── │  deploy_groot_realman.py │
│  (on robot)             │                     │  (robot or GPU machine)  │
│  cameras, state, action │                     │  closed-loop client      │
└─────────────────────────┘                     └────────────┬─────────────┘
                                                             │ ZMQ :5555
                                                             ▼
                                                ┌──────────────────────────┐
                                                │  run_gr00t_server.py     │
                                                │  (GPU machine)           │
                                                │  GR00T inference         │
                                                └──────────────────────────┘
```

**What changes for EEF:** the observation builder and action executor in the deploy client, plus `robot_api_server` must expose EE pose. The GR00T server stays the same idea — load the EEF checkpoint and serve over ZMQ.

---

## What already exists vs what is missing

| Component | Status | Notes |
|-----------|--------|-------|
| EEF checkpoint on HF | Done | `jordanlarot/gr00t-pick-place-bottle-eef-10k` |
| `realman_dual_arm_eef_config.py` | Done | Defines EEF modality keys and `RELATIVE` action rep |
| `run_gr00t_server.py` | Done | Loads checkpoint, serves policy over ZMQ |
| `deploy_groot_realman.py` | **Joint only** | Expects `left_arm` / `right_arm` joint keys; sends 14D joints |
| `robot_api_server.py` | **Joint only** | `/observation` returns 14D `state`; `/action` expects 14D joints |
| `Hardware_Bridge_ROS2.py` | **Partial** | Subscribes to `udp_arm_position` but does not expose EE in API |
| IK / Cartesian executor | **Missing** | No EEF → joint conversion in deploy path yet |

The **joint-space** checkpoint and its deploy path remain valid if you want a quicker first hardware test without IK. This doc is specifically for the **EEF** checkpoint.

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
# On robot, from r2d3-training root
python scripts/robot_api_server.py
```

Today this only returns joint state. Steps 3–5 extend it for EEF deploy.

Smoke-test the current API:

```bash
curl http://localhost:5000/observation | python3 -m json.tool | head
```

---

### Step 3 — Expose EE pose in observations

The policy needs live EEF state in the same format as training.

**Source (already subscribed in bridge):**

- `/left_arm_controller/rm_driver/udp_arm_position`
- `/right_arm_controller/rm_driver/udp_arm_position`

**Layout (14D `ee_pose`, same as dataset):**

| Index | Meaning |
|-------|---------|
| 0–2 | Left TCP position (x, y, z) in **metres** |
| 3–6 | Left quaternion **(x, y, z, w)** |
| 7–9 | Right TCP position (x, y, z) in **metres** |
| 10–13 | Right quaternion **(x, y, z, w)** |

**Coordinate frame:** each arm is in **its own Realman base frame**. Left and right are not in a shared world frame — treat them as independent modality keys (same as training).

**Convert to GR00T `left_eef_9d` / `right_eef_9d` (9D each):**

```
[x, y, z, R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2]]
```

where the 6D rotation is the **first two rows** of the rotation matrix from the quaternion. Reuse the logic in `scripts/convert_to_eef_gr00t.py` (`_quat_xyzw_to_rot6d`).

**Grippers:** slice from joint state — left index 6, right index 13 (same as dataset).

**Work items:**

1. Add `get_ee_pose()` (or similar) to `Hardware_Bridge_ROS2.py` — pack `l_ee_pose` / `r_ee_pose` into the 14D layout above.
2. Extend `robot_api_server.py` `/observation` to return `ee_pose` (and optionally precomputed `left_eef_9d` / `right_eef_9d`).

---

### Step 4 — Start the GR00T policy server

On the GPU machine:

```bash
cd Isaac-GR00T

uv run python gr00t/eval/run_gr00t_server.py \
  --model-path jordanlarot/gr00t-pick-place-bottle-eef-10k \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/RealMan/realman_dual_arm_eef_config.py \
  --device cuda \
  --host 0.0.0.0 \
  --port 5555
```

Use the local checkpoint path if you downloaded to the robot or a specific directory.

The server:

- Normalizes / denormalizes actions
- Converts **relative EEF deltas → absolute EEF targets** using the current `left_eef_9d` / `right_eef_9d` in the observation

---

### Step 5 — Convert EEF actions to robot commands (IK or Cartesian)

After inference, the policy returns **absolute EEF targets** per arm plus gripper scalars. The robot stack today only accepts **joint** commands via `movej_cmd`.

Pick one execution path:

**Option A — IK → joint commands (fits current `robot_api_server`)**

1. Take absolute EEF target (9D xyz+rot6d) for the arm being controlled.
2. Convert rot6d back to a rotation matrix / quaternion.
3. Run inverse kinematics for the RM65 → 6 joint angles.
4. Pack into 14D: `[l_arm(6), l_gripper(1), r_arm(6), r_gripper(1)]`.
5. POST to `http://<ROBOT_IP>:5000/action`.

**Option B — Realman Cartesian motion API**

Send the absolute EEF target directly to the Realman driver's Cartesian motion interface (if available in your ROS2 stack). This bypasses explicit IK in your deploy client but still requires correct frame and safety limits.

**Grippers:** absolute scalars — pass through directly (not relative).

**Scope for first test (recommended):** right arm only. Pin left arm EEF/joints to the current observed state (same pattern as `pin_left_arm_to_state` in `deploy_groot_realman.py`).

---

### Step 6 — Update the deploy client

`Isaac-GR00T/gr00t/eval/real_robot/realman/deploy_groot_realman.py` must be extended (or a new `deploy_groot_realman_eef.py` added).

**Observation builder** — replace joint key mapping:

```python
# Today (joint checkpoint):
state_by_key = {
    "left_arm": state[0:6],
    "left_gripper": state[6:7],
    ...
}

# EEF checkpoint:
state_by_key = {
    "left_eef_9d": left_eef_9d,    # from ee_pose → rot6d conversion
    "left_gripper": state[6:7],
    "right_eef_9d": right_eef_9d,
    "right_gripper": state[13:14],
}
```

**Action parser** — policy returns EEF keys, not joint slices. Concatenate in modality order: `left_eef_9d`, `left_gripper`, `right_eef_9d`, `right_gripper` → then run Step 5 conversion before `/action`.

**Closed-loop pattern** — keep the existing receding-horizon loop (re-infer every `open_loop_horizon` steps, default 8). Only the observation/action formats change.

---

### Step 7 — Dry run, then closed loop

**Dry run** (no motion):

```bash
cd Isaac-GR00T

uv run python gr00t/eval/real_robot/realman/deploy_groot_realman.py \
  --task "pick up bottle" \
  --robot-url http://<ROBOT_IP>:5000 \
  --policy-host <GPU_IP> \
  --policy-port 5555 \
  --hz 15 \
  --dry-run \
  --debug
```

Confirm logged EEF targets look reasonable vs live EE pose.

**Closed loop:**

```bash
uv run python gr00t/eval/real_robot/realman/deploy_groot_realman.py \
  --task "pick up bottle" \
  --robot-url http://<ROBOT_IP>:5000 \
  --policy-host <GPU_IP> \
  --policy-port 5555 \
  --hz 15 \
  --open-loop-horizon 8 \
  --max-steps 500
```

Start with a low `max-steps`, clear workspace, and e-stop ready. Match `--hz` to the 10–15 Hz collection rate.

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
   → relative EEF deltas + current eef state → absolute EEF targets
   → gripper absolutes

5. deploy client (new step)
   → IK or Cartesian: absolute EEF → joint targets

6. robot_api_server /action
   → Hardware_Bridge → movej_cmd + gripper
```

---

## Implementation checklist

Use this as a task list for the EEF deploy PR:

- [ ] **Bridge:** `get_ee_pose()` in `Hardware_Bridge_ROS2.py`
- [ ] **API:** `/observation` returns `ee_pose` (14D) or `left_eef_9d` / `right_eef_9d`
- [ ] **IK:** EEF target → 6 joint angles (per arm); validate against RM65 limits
- [ ] **Deploy client:** EEF observation builder + EEF action → joint conversion
- [ ] **Right-arm-only mode:** pin left arm to observed state
- [ ] **Safety:** reuse EE position limits in `check_safety_limits`
- [ ] **Dry run:** verify shapes and magnitudes before motion
- [ ] **Open-loop on robot:** compare one-step EEF prediction vs live EE (optional sanity check)

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
| `Isaac-GR00T/examples/RealMan/realman_dual_arm_eef_config.py` | EEF modality keys, `RELATIVE` EEF actions |
| `scripts/convert_to_eef_gr00t.py` | quat → rot6d conversion (training reference) |
| `Isaac-GR00T/gr00t/eval/run_gr00t_server.py` | Policy server |
| `Isaac-GR00T/gr00t/eval/real_robot/realman/deploy_groot_realman.py` | Closed-loop client (needs EEF update) |
| `scripts/robot_api_server.py` | Robot HTTP API |
| `src/teleop/Hardware_Bridge_ROS2.py` | ROS2 bridge, EE pose subscribers |

---

## Alternative: joint-space deploy (no IK)

If you need hardware running **before** the EEF bridge is built, use the joint-space checkpoint instead:

- Model: joint finetune (e.g. `gr00t_pick_bottle_realman/checkpoint-10000`)
- Config: `realman_dual_arm_config.py`
- Deploy: existing `deploy_groot_realman.py` without changes

That path outputs 14D joint actions directly. The EEF checkpoint is better for manipulation generalization but requires the extra conversion layer described above.
