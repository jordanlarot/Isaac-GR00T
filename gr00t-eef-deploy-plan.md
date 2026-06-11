# GR00T EEF Deploy — Implementation Plan

**Date:** 2026-06-09  
**Checkpoint:** `jordanlarot/gr00t-pick-place-bottle-eef-10k`  
**Local path:** `/home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k`  
**Task:** pick up bottle (first hardware test: **right arm only**)

Related docs:

- [gr00t-eef-conversion-results.md](gr00t-eef-conversion-results.md) — dataset conversion and training
- [gr00t-eef-robot-deploy.md](gr00t-eef-robot-deploy.md) — post-download deploy guide
- [commands.md](commands.md) — operator runbook (EEF + joint)
- [eef-deploy-jerk-report.md](eef-deploy-jerk-report.md) — hardware jerk/sway analysis
- [eef-deploy-umi-takeaways.md](eef-deploy-umi-takeaways.md) — UMI tuning (`--open-loop-horizon`, latency)

---

## Goal

Run the **EEF finetuned checkpoint** on the RealMan R2D3 hardware using the same three-process architecture as joint-space deploy:

```
robot_api_server (:5000)  ←→  deploy client  ←→  run_gr00t_server (:5555)
```

The EEF model expects **9D end-effector pose + gripper** per arm (not 6D joint angles). The deploy bridge must:

1. Read live EE pose from the robot stack
2. Feed GR00T in the same format as training (`left_eef_9d`, `right_eef_9d`, grippers, 3 cameras)
3. Receive **absolute EEF targets** from the policy server (relative→absolute is handled inside GR00T)
4. Convert EEF targets → joint commands (IK) before POSTing to `/action`

---

## Current State Audit

### Done

| Item | Location | Notes |
|------|----------|-------|
| EEF checkpoint downloaded | `/home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k` | Embodiment tag `new_embodiment`; modality keys baked into `processor_config.json` |
| EEF modality config in checkpoint | `processor_config.json` → `new_embodiment` | State/action keys: `left_eef_9d`, `left_gripper`, `right_eef_9d`, `right_gripper`; EEF actions `RELATIVE` |
| Policy server (PyTorch + TRT) | `gr00t/eval/run_gr00t_server.py` | TRT DiT-only via `--trt-engine-path --trt-mode dit_only` |
| Joint-space deploy client | `gr00t/eval/real_robot/realman/deploy_groot_realman.py` | Closed-loop loop, logging, video, grip ratchet |
| Robot HTTP API | `pickup-objects/scripts/robot_api_server.py` | `/observation`, `/action`, right-arm-only execution |
| EE pose in ROS bridge | `pickup-objects/src/r2d3/hardware/ros2_bridge.py` | `get_ee_poses_at_time()` already exists; used by data collection |
| EE pose in dataset | training parquets | `observation.ee_pose` 14D layout documented in pickup-objects |

### Still open for EEF

| Item | Status | Notes |
|------|--------|-------|
| TRT engines for EEF checkpoint | Optional | Build per `commands.md`; ~4–5 Hz on Orin |
| Open-loop eval on EEF checkpoint | Not verified | Offline sanity check before hardware |
| Stale-action skip (UMI PD1.2) | Not implemented | See `eef-deploy-umi-takeaways.md` Tier 2 |
| `examples/RealMan/realman_dual_arm_eef_config.py` | Not in repo | Config is in checkpoint `processor_config.json` |

### Done (2026-06-11)

| Item | Location |
|------|----------|
| `/observation` `ee_pose` (14D) | `pickup-objects/scripts/robot_api_server.py` |
| EEF deploy client + IK | `deploy_groot_realman_eef.py`, `eef_utils.py`, `realman_ik.py` |
| Operator runbook | `commands.md` (EEF section first) |
| Jerk / UMI tuning docs | `eef-deploy-jerk-report.md`, `eef-deploy-umi-takeaways.md` |

### Important repo-specific corrections

The deploy guide references `Hardware_Bridge_ROS2.py`; on this robot the active bridge is:

- `pickup-objects/src/r2d3/hardware/ros2_bridge.py`

EE pose is already subscribed via `Armstate` (`get_current_arm_state_result`), not only `udp_arm_position`. `get_ee_poses_at_time()` returns 14D `[l_xyz, l_quat_xyzw, r_xyz, r_quat_xyzw]`.

---

## Data Formats (must match training)

### EE pose layout (14D, metres + xyzw quaternion)

| Index | Field |
|-------|-------|
| 0–2 | Left TCP position (m) |
| 3–6 | Left quaternion (x, y, z, w) |
| 7–9 | Right TCP position (m) |
| 10–13 | Right quaternion (x, y, z, w) |

Each arm is in **its own Realman base frame** (not a shared world frame).

### GR00T EEF 9D (`left_eef_9d` / `right_eef_9d`)

```
[x, y, z, R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2]]
```

6D rotation = first two rows of the rotation matrix from the quaternion.

### Grippers

- Left: joint state index 6 (normalised 0–1)
- Right: joint state index 13

### Policy I/O (per step, after server decode)

| Modality | Dims | Rep |
|----------|------|-----|
| `left_eef_9d` | 9 | Absolute target (server converts relative→absolute) |
| `left_gripper` | 1 | Absolute |
| `right_eef_9d` | 9 | Absolute |
| `right_gripper` | 1 | Absolute |

Deploy client must IK **right arm EEF** → 6 joints, pin left arm to observed joints (same as today).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  pickup-objects (robot)                                                 │
│  ros2_bridge.get_ee_poses_at_time() + get_state() + cameras             │
│       ↓                                                                 │
│  robot_api_server  GET /observation                                     │
│    { state: [14], ee_pose: [14], images: {...} }                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ HTTP
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Isaac-GR00T  deploy_groot_realman_eef.py  (new or extended client)     │
│    ee_pose → left_eef_9d / right_eef_9d                                 │
│    build GR00T obs → PolicyClient.get_action()                          │
│    EEF action chunk → IK → 14D joint command → POST /action             │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ ZMQ :5555
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  run_gr00t_server.py  (Orin docker, optional TRT dit_only)              │
│    Gr00tPolicy: normalize → infer → relative EEF → absolute EEF         │
└─────────────────────────────────────────────────────────────────────────┘
```

GR00T server does **not** need `--modality-config-path` if the checkpoint's `processor_config.json` contains the `new_embodiment` entry (it does). The deploy client pulls modality keys from the server via `PolicyClient.get_modality_config()`.

---

## Implementation Phases

### Phase 0 — Verify checkpoint (no robot)

**Goal:** Confirm the EEF checkpoint loads and produces sensible actions on held-out dataset trajectories.

**Tasks:**

1. Obtain or symlink EEF dataset locally (`pick_place_bottle_v2_gr00t_eef`) if not already present.
2. Run open-loop eval against the downloaded checkpoint:

```bash
cd Isaac-GR00T
source .venv/bin/activate && source scripts/activate_orin.sh

python gr00t/eval/open_loop_eval.py \
  --model-path /home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k \
  --dataset-path /home/r2d3/datasets/pick-bottle-eef \
  --embodiment-tag NEW_EMBODIMENT \
  --traj-ids 0 50 101 \
  --action-horizon 16 \
  --modality-keys left_eef_9d right_eef_9d left_gripper right_gripper
```

3. Inspect trajectory plots / MSE. EEF errors are in metres / rotation units — not comparable to joint MAE.

**Acceptance:** Checkpoint loads without shape errors; open-loop EEF trajectories look reasonable vs dataset.

**Effort:** ~1 hour (assuming dataset available).

---

### Phase 1 — Expose EE pose in robot API

**Goal:** `/observation` returns live EE pose alongside joint state.

**Repo:** `pickup-objects`

**Tasks:**

1. In `robot_api_server.py` `get_observation()`:
   - Call `hardware.get_ee_poses_at_time(time.time())` (or synchronous `get_state()` + latest EE if simpler for deploy latency).
   - Add `"ee_pose": [...]` (14 floats) to JSON response.
   - Optionally add precomputed `"left_eef_9d"` / `"right_eef_9d"` (9 floats each) to keep conversion logic in one place.

2. Smoke test:

```bash
curl http://localhost:5000/observation | python3 -m json.tool | head -40
```

3. Verify EE values update when arm moves and units are metres (not mm).

**Acceptance:** `/observation` includes valid `ee_pose`; positions change when arm moves; quaternions normalised.

**Effort:** ~2–4 hours.

---

### Phase 2 — Shared EEF pose utilities

**Goal:** Single source of truth for quat ↔ rot6d conversion (matches training).

**Repo:** `Isaac-GR00T` (preferred, used by deploy client) or `pickup-objects` (if API precomputes eef_9d)

**Tasks:**

1. Add `gr00t/eval/real_robot/realman/eef_utils.py` with:
   - `quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray`  # (4,) → (6,)
   - `rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray`     # (6,) → (3,3) with Gram-Schmidt if needed
   - `ee_pose_14d_to_eef_9d(ee_pose: np.ndarray, arm: str) -> np.ndarray`  # arm in {"left","right"}
   - `eef_9d_to_quat_xyzw(eef_9d: np.ndarray) -> tuple[np.ndarray, np.ndarray]`  # pos, quat for IK input

2. Unit tests against known values from one dataset parquet row (golden test).

3. Port logic from training `convert_to_eef_gr00t.py` if/when that script is added to the repo; until then, implement from doc spec in [gr00t-eef-conversion-results.md](gr00t-eef-conversion-results.md).

**Acceptance:** Round-trip quat → rot6d → rotmat → quat within tolerance on dataset sample.

**Effort:** ~3–4 hours.

---

### Phase 3 — Inverse kinematics (EEF → joints)

**Goal:** Convert absolute right-arm EEF target (9D) to 6 joint angles for `/action`.

**This is the highest-risk phase.** Pick one approach and validate offline before closed-loop.

#### Option A — Numerical IK (recommended first)

- Use current joint state as seed.
- Optimise 6 joint angles to minimise pose error vs target EEF.
- Libraries: `pinocchio` + RM65 URDF, or `scipy.optimize` with forward kinematics from Realman SDK / existing FK in repo.
- Enforce joint limits from Realman RM65 spec.

#### Option B — Realman Cartesian motion API

- Bypass explicit IK in deploy client; send Cartesian target to driver (`movep` / Cartesian CANFD if available in ROS stack).
- Requires verifying frame convention matches training and safety limits still apply.

#### Option C — Analytic IK

- Only if RM65 analytic IK is available in your driver stack; often fragile near singularities.

**Tasks:**

1. Spike: given one dataset `(ee_pose, state)` pair, verify IK reproduces joint angles within ~0.05 rad.
2. Add `gr00t/eval/real_robot/realman/realman_ik.py`:
   - `eef_9d_to_joint_angles(target_eef_9d, seed_joints, arm="right") -> np.ndarray`  # (6,)
3. Log IK residual (position mm, orientation deg) in deploy dry-run.

**Acceptance:** Offline IK error < 5 mm position on 10 random dataset frames; no joint limit violations.

**Effort:** ~1–3 days (depends on FK/URDF availability).

---

### Phase 4 — EEF deploy client

**Goal:** Closed-loop client that speaks EEF to GR00T and joints to the robot.

**Recommended:** New file `deploy_groot_realman_eef.py` (keep joint client unchanged for fallback).

**Tasks:**

1. **Observation builder** — replace hardcoded joint mapping:

```python
# From raw["ee_pose"] or precomputed eef_9d fields:
state_by_key = {
    "left_eef_9d": left_eef_9d,
    "left_gripper": state[6:7],
    "right_eef_9d": right_eef_9d,
    "right_gripper": state[13:14],
}
```

2. **Action parser** — EEF chunk is 20D per step (not 14D):

```python
# Modality order from server: left_eef_9d(9), left_gripper(1), right_eef_9d(9), right_gripper(1)
```

3. **Execute path:**
   - Pin left arm joints + gripper to observed state (extend `pin_left_arm_to_state` or equivalent for EEF chunk).
   - IK right `right_eef_9d` → 6 joints.
   - Pack 14D: `[obs_left_joints(6), obs_left_grip(1), ik_right_joints(6), model_right_grip(1)]`.
   - POST to `/action`.

4. Reuse from joint client:
   - Receding-horizon loop (`open_loop_horizon`)
   - Logging (`RunLogger`, `steps.jsonl`)
   - Video recording thread
   - `--auto-close-grip`, `--dry-run`, `--debug`

5. CLI flags mirror joint client; add `--require-ee-pose` to fail fast if API missing EE data.

**Acceptance:** Dry-run logs show EEF obs shapes matching server modality config; executed 14D joints are finite and within limits.

**Effort:** ~1 day after Phase 2–3 complete.

---

### Phase 5 — Policy server + optional TRT

**Goal:** Serve EEF checkpoint from Orin with best available latency.

**Tasks:**

1. Start PyTorch server (baseline):

```bash
python gr00t/eval/run_gr00t_server.py \
  --model-path /home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 --host 0.0.0.0 --port 5555
```

2. Build TRT DiT-only engines (one-time):

```bash
python scripts/deployment/build_trt_pipeline.py \
  --model-path /home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k \
  --dataset-path /home/r2d3/datasets/pick-bottle-eef \
  --embodiment-tag NEW_EMBODIMENT \
  --export-mode dit_only \
  --output-dir /home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k-trt
```

3. Restart server with `--trt-engine-path .../engines --trt-mode dit_only`.

4. Use `--policy-host localhost` in deploy client (server on same Orin).

**Acceptance:** Server prints EEF modality keys on startup; inference_ms in deploy logs drops from ~650 ms → ~200–250 ms (PyTorch local) or ~220 ms (TRT DiT-only per Orin benchmarks).

**Effort:** TRT build ~30 min; server config ~15 min.

---

### Phase 6 — Hardware validation

**Goal:** Safe closed-loop pick attempt with EEF checkpoint.

**Pre-flight:**

- Head camera angle set (grippers visible at rest)
- Reset to episode start pose
- rogent + robot_api + policy server running
- Clear workspace, e-stop ready

**Progression:**

| Step | Command | Pass criteria |
|------|---------|---------------|
| 6a Dry run | `--dry-run --debug` | EEF obs/action shapes OK; IK residuals logged |
| 6b Short live | `--max-steps 30 --open-loop-horizon 6 --hz 8` | Arm moves toward bottle; no safety stops |
| 6c Full run | `--max-steps 500 --auto-close-grip` | Task attempt; review `runs/` logs + video |

**Recommended first settings:**

```bash
python gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py \
  --task "pick up bottle" \
  --policy-host localhost --policy-port 5555 \
  --robot-url http://localhost:5000 \
  --open-loop-horizon 6 --hz 8 \
  --auto-close-grip --grip-close-threshold 0.95
```

If still jerky: try `--open-loop-horizon 4`. After TRT, try `--open-loop-horizon 1` for true closed-loop. See [eef-deploy-jerk-report.md](eef-deploy-jerk-report.md).

**Effort:** ~2–4 hours on hardware (iterative tuning).

---

### Phase 7 — Documentation & operator runbook — **Done**

- [x] `commands.md` — EEF section (checkpoint, TRT, deploy client, tuning flags)
- [x] `CLAUDE.md` / `AGENTS.md` — EEF vs joint deploy, recommended `--open-loop-horizon 6 --hz 8`
- [x] [gr00t-eef-robot-deploy.md](gr00t-eef-robot-deploy.md) — status table + deploy commands
- [x] [eef-deploy-jerk-report.md](eef-deploy-jerk-report.md), [eef-deploy-umi-takeaways.md](eef-deploy-umi-takeaways.md)

---

## Task Checklist (copy for PR tracking)

### pickup-objects

- [x] `/observation` returns `ee_pose` (14D)
- [ ] (Optional) `/observation` returns precomputed `left_eef_9d` / `right_eef_9d`
- [x] Document EE field in robot API docstring

### Isaac-GR00T

- [x] `eef_utils.py` — quat ↔ rot6d conversion + tests
- [x] `realman_ik.py` — EEF → joint IK with seed joints
- [x] `deploy_groot_realman_eef.py` — EEF obs builder + IK execute path
- [x] `commands.md` EEF operator section
- [x] `eef-deploy-jerk-report.md`, `eef-deploy-umi-takeaways.md` — hardware tuning docs
- [ ] Open-loop eval passes on EEF checkpoint
- [ ] TRT engines built for EEF checkpoint
- [ ] UMI stale-action skip in deploy client
- [ ] (Optional) Port `examples/RealMan/realman_dual_arm_eef_config.py` for reproducibility

---

## Testing Matrix

| Test | Where | Blocks |
|------|-------|--------|
| quat→rot6d golden test | unit test | Phase 4 |
| open_loop_eval EEF | GPU/Orin | Phase 5–6 |
| curl `/observation` has ee_pose | robot | Phase 4 |
| deploy dry-run shape check | robot + server | Phase 6a |
| IK offline on dataset frames | dev machine | Phase 3 |
| deploy live 30 steps | robot | Phase 6b |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| IK inaccuracy / singularities | Seed from current joints; log residual; limit max joint delta per step |
| EE frame mismatch vs training | Compare live `ee_pose` to dataset parquet at same reset pose |
| Inference still too slow for H=1 | TRT DiT-only + localhost server; start with H=16 |
| Gripper 10 Hz vs cameras 15 Hz | Run deploy at `--hz 10` |
| EEF vs joint checkpoint confusion | Separate deploy scripts; distinct `commands.md` sections |
| No FK/URDF for RM65 | Priority spike in Phase 3 before writing deploy client |

---

## Fallback: Joint-space deploy (no IK)

If EEF bridge is blocked on IK, continue hardware testing with the joint checkpoint:

- Model: `/home/r2d3/checkpoints/gr00t-pick-bottle-realman`
- Client: existing `deploy_groot_realman.py`
- See current [commands.md](commands.md)

---

## Suggested Execution Order

```
Phase 0  open-loop eval          (verify checkpoint)
   ↓
Phase 1  robot API ee_pose      (pickup-objects)
   ↓
Phase 2  eef_utils              (Isaac-GR00T)
   ↓
Phase 3  IK spike + module       (Isaac-GR00T)  ← critical path
   ↓
Phase 4  deploy_groot_realman_eef.py
   ↓
Phase 5  server + TRT
   ↓
Phase 6  hardware dry-run → live
   ↓
Phase 7  docs
```

**Estimated total:** 4–7 days depending on IK/FK availability.

---

## Quick Reference — Key Files

| File | Repo | Role |
|------|------|------|
| `/home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k/` | local | EEF finetuned checkpoint |
| `pickup-objects/scripts/robot_api_server.py` | pickup-objects | HTTP API — needs `ee_pose` |
| `pickup-objects/src/r2d3/hardware/ros2_bridge.py` | pickup-objects | `get_ee_poses_at_time()` |
| `gr00t/eval/run_gr00t_server.py` | Isaac-GR00T | Policy server (+ TRT flags) |
| `gr00t/eval/real_robot/realman/deploy_groot_realman.py` | Isaac-GR00T | Joint client (unchanged) |
| `gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py` | Isaac-GR00T | **To create** — EEF client |
| `gr00t/eval/real_robot/realman/eef_utils.py` | Isaac-GR00T | **To create** — pose conversion |
| `gr00t/eval/real_robot/realman/realman_ik.py` | Isaac-GR00T | **To create** — IK |
| `commands.md` | Isaac-GR00T | Operator runbook |
