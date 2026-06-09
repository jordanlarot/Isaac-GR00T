# Phase 3 Implementation Plan — `realman_ik.py`

**Date:** 2026-06-09  
**Phase:** 3 of the GR00T EEF deploy plan (see `gr00t-eef-deploy-plan.md`)

## Goal

A pure-math IK module: takes a 9D EEF target (from the GR00T policy) and current joint state, returns 6 joint angles. No hardware connection. Validated offline against dataset `(ee_pose, state)` pairs.

---

## Step 0 — Verify QPIK imports offline

Before writing any module code, confirm that `ik_qp.py` can be imported without a live robot connection. `ik_qp.py` has `from Robotic_Arm.rm_robot_interface import *` at the top, which is the only risk.

**Action:** run a one-liner in the Isaac-GR00T venv:

```bash
cd /home/r2d3/Isaac-GR00T
env -u PYTHONPATH .venv/bin/python -c "
import sys
sys.path.insert(0, '/home/r2d3/X-VLA_copy/teleoperation_IK_v1.0.0/python3')
sys.path.insert(0, '/home/r2d3/X-VLA_copy/teleoperation_IK_v1.0.0/RM_API2-1.0.6/Python')
from ik_qp import QPIK, deg2rad
solver = QPIK('RM65B', 0.01)
print('QPIK init OK')
"
```

**Two outcomes:**
- **OK** — proceed to Step 1 as planned.
- **ImportError on `rm_robot_interface`** — the RM SDK is a C extension that may not be importable inside the GR00T venv. In that case, the plan adapts: `realman_ik.py` would live in the XVLA Python directory (where the SDK is installed), and the GR00T deploy client imports it via `sys.path`. Both outcomes are handled in the design below.

---

## Step 1 — Design `realman_ik.py`

**File:** `gr00t/eval/real_robot/realman/realman_ik.py` (preferred) or `X-VLA_copy/teleoperation_IK_v1.0.0/python3/realman_ik.py` if SDK can't import in GR00T venv.

**Public API (one class, one method):**

```python
class RealmanIK:
    def __init__(self, arm: str = "right"):
        ...

    def eef_9d_to_joint_angles(
        self,
        target_eef_9d: np.ndarray,   # (9,) [x,y,z, rot6d(6)]
        seed_joints: np.ndarray,      # (6,) current joint angles, radians
        n_iterations: int = 15,
    ) -> np.ndarray:                  # (6,) joint angles, radians
        ...
```

**What `__init__` does:**
- Loads `robot_config.yaml` (hardcode path relative to XVLA dir, or accept path kwarg).
- Instantiates `QPIK("RM65B", 0.01)`.
- Calls `set_install_angle`, `set_work_cs_params`, `set_tool_cs_params` per arm.
- Sets joint limits (same constants as `VLAArmController`).
- No hardware connection, no `rm_create_robot_arm`.

**What `eef_9d_to_joint_angles` does:**

```
Step 1 — Extract xyz and rot6d from the 9D input
  xyz    = target_eef_9d[:3]           # (3,) metres
  rot6d  = target_eef_9d[3:]           # (6,)

Step 2 — Recover rotation matrix (Gram-Schmidt)
  R = eef_utils.rot6d_to_rotmat(rot6d) # (3,3)
  # Gram-Schmidt ensures valid rotation even from noisy policy output

Step 3 — Rotation matrix → Euler XYZ (extrinsic) = Rz @ Ry @ Rx convention
  from scipy.spatial.transform import Rotation
  rx, ry, rz = Rotation.from_matrix(R).as_euler('XYZ')
  # 'XYZ' (uppercase) = extrinsic = matches pose_to_matrix() in ik_rbtutils.py

Step 4 — Build QPIK target matrix
  from ik_rbtutils import pose_to_matrix
  Td = pose_to_matrix([xyz[0], xyz[1], xyz[2], rx, ry, rz])

Step 5 — Iterative solve
  q_sol = seed_joints.copy()
  for _ in range(n_iterations):
      q_sol = self._solver.sovler(q_sol, Td)

Step 6 — Safety: NaN check + joint limit check
  if np.isnan(q_sol).any(): raise ValueError(...)
  clip or raise on joint limit violations (configurable)

Step 7 — Return q_sol (6,) radians
```

**What it does NOT do:**
- No `rm_movej_canfd` — execution stays in the robot API server.
- No `RoboticArm` object — no hardware sockets opened.
- No `VLAArmController` — we use QPIK directly.

---

## Step 2 — Test file: `test_realman_ik.py`

**File:** `tests/gr00t/eval/real_robot/realman/test_realman_ik.py`

**TDD sequence (RED → GREEN → REFACTOR per function):**

### Test class 1 — `TestRealmanIKInit`

| Test | What it checks |
|------|----------------|
| `test_right_arm_initialises` | `RealmanIK("right")` succeeds without hardware |
| `test_left_arm_initialises` | Same for left |
| `test_bad_arm_raises` | `RealmanIK("middle")` → `ValueError` |

### Test class 2 — `TestEef9dToJointAngles` (offline dataset spike)

Source data: load one `(ee_pose, state)` pair from the EEF dataset parquet. The `state` gives ground-truth joint angles (6D per arm). The `ee_pose` gives the 14D raw pose we convert to `eef_9d`.

| Test | What it checks |
|------|----------------|
| `test_output_shape` | Returns `(6,)` array |
| `test_no_nans` | No NaN in output |
| `test_within_joint_limits` | All joints within RM65 limits |
| `test_fk_position_residual` | FK of solution ≤ 5 mm from target position |
| `test_fk_rotation_residual` | FK rotation error ≤ 5 deg |
| `test_10_random_dataset_frames` | Loop over 10 frames; all pass position residual ≤ 5 mm |
| `test_wrong_input_shape_raises` | `eef_9d` of shape `(8,)` → `ValueError` |

**Why FK-based residual tests instead of joint-angle comparison:**
The IK is underdetermined near many configurations — multiple joint solutions reach the same EEF pose. Comparing FK output position/orientation against the target is the right acceptance criterion, not comparing joints to dataset ground truth.

### Test class 3 — `TestEulerConvention`

One targeted unit test that the `as_euler('XYZ')` → `pose_to_matrix` roundtrip is self-consistent: given a known rotation matrix, convert it to Euler angles, build the QPIK target matrix, check that the top-left 3×3 of that matrix matches the original rotation matrix. This pins the convention so a future scipy upgrade or a copy-paste mistake doesn't silently break IK.

---

## Step 3 — Offline spike: validate on 10 dataset frames

This is both the acceptance test above and an interactive check to see actual residuals:

```python
# spike script or pytest --capture=no output:
for frame in frames:
    ee_pose_14d = frame["observation.ee_pose"]
    eef_9d = ee_pose_14d_to_eef_9d(ee_pose_14d, "right")
    seed   = np.zeros(6)  # cold start seed — worst case
    q_sol  = ik.eef_9d_to_joint_angles(eef_9d, seed)
    T_fk   = ik._solver.robot.fkine(q_sol)
    pos_err = np.linalg.norm(T_fk[:3, 3] - eef_9d[:3]) * 1000
    print(f"frame {i}: pos_err={pos_err:.2f} mm")
```

Acceptance: all 10 frames ≤ 5 mm. If any exceed 5 mm, increase `n_iterations` or investigate whether there is a `work_cs`/`tool_cs` mismatch.

---

## Step 4 — Ruff + commit

```bash
env -u PYTHONPATH ruff check --fix gr00t/eval/real_robot/realman/realman_ik.py
env -u PYTHONPATH ruff format gr00t/eval/real_robot/realman/realman_ik.py
# run full test suite to check no regressions
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --with pytest \
  python -m pytest tests/gr00t/eval/real_robot/realman/ -q
```

---

## Key risks and mitigations

| Risk | Mitigation |
|------|------------|
| SDK can't import in GR00T venv | Move `realman_ik.py` to XVLA python3 dir; deploy client adds that dir to `sys.path` |
| `install_angle`/`tool_cs` mismatch with training frame | Run FK on known dataset `(joints, ee_pose)` pair and compare — if residual ≠ 0 at training pose, there is a config mismatch |
| IK diverges from cold start seed | Use actual observed joint state as seed (not zeros) — deploy client always has live state |
| Convention drift (scipy version change) | `TestEulerConvention` pins this |

---

## Acceptance criteria (Phase 3 done)

- [x] `RealmanIK("right")` initialises without hardware
- [x] 12 dataset frames: IK position residual ≤ 0.14 mm, orientation ≤ 0.03 deg (bar was 5 mm / 5 deg)
- [x] No joint limit violations on any test frame
- [x] All tests green (17 IK + 13 eef_utils), ruff clean
- [x] Euler convention pinned (`TestEulerConvention`) — NOTE: scipy `'xyz'` lowercase (extrinsic), not `'XYZ'` as drafted in Step 1

## Implementation notes (what differed from the draft)

1. **qpSWIFT had to be installed into the GR00T venv** — built from local source:
   `uv pip install --python .venv/bin/python --no-build-isolation /home/r2d3/open_droids_controllers/r2d3/qp-tools/python`
   (plus `setuptools`). After that, QPIK imports fully offline; `realman_ik.py` lives in
   `gr00t/eval/real_robot/realman/` as preferred.

2. **Frame calibration — the XVLA yaml values are wrong for our dataset.** FK with the yaml
   right-arm config missed the dataset ee_pose by ~570 mm / 90°. Calibrated against the EEF
   dataset: `work_cs rz = -pi/2` (both arms, not 3.14159/-1.571) and a **135 mm Z tool offset on
   BOTH arms** (yaml had 0 for left). With these, FK reproduces dataset ee_pose to <0.5 mm /
   <0.15° on every frame checked across 3 episodes. `TestFkMatchesDataset` pins this.

3. **Adaptive iterations instead of fixed 15.** QPIK is velocity-clamped (~0.9 ms/iter on Orin):
   15 iterations only converges for nearby targets; a 0.2 m jump needs ~75. The solver loop
   early-exits when FK position residual < 0.5 mm (`pos_tol_m`), capped at `max_iterations=100`.
   Measured: small step ~2 ms, 0.2 m jump ~83 ms (one-off after resets — fine at 10 Hz).

4. **Unreachable-target guard added after adversarial verification.** QPIK clamps joints to
   limits, so an out-of-workspace target returned a within-limits solution whose FK was 1.2 m
   from the target — silently wrong. `eef_9d_to_joint_angles` now raises ValueError when the
   final FK residual exceeds `max_residual_m` (default 5 mm), mirroring VLA_move.py's
   `pred_err > 0.3` check.

## Verification results (adversarial, beyond the unit tests)

- **A. Conversion vs dataset builder:** `eef_utils.ee_pose_14d_to_eef_9d` matches the dataset's
  precomputed `observation.{left,right}_eef_9d` columns to 3e-8 (float32 precision) across
  21 episodes.
- **B. FK calibration at scale:** 206 frames × 2 arms across 21 episodes — worst 0.72 mm / 0.16°.
- **C. End-to-end deploy simulation:** walked 3 full episodes (487 steps) solving
  `IK(action.right_eef_9d)` seeded from the previous solution; reproduced the teleoperator's
  recorded joint commands to max 0.38°, mean 0.1°, no drift, no exceptions.
- **D. Noise robustness:** rot6d perturbed with sigma=0.05 (non-orthonormal, policy-like) +
  5 mm position noise — all solved, worst residual 0.48 mm (Gram-Schmidt path works).
- **E. Unreachable target:** found the silent-wrong-pose gap, fixed with the residual guard
  (note 4), pinned by `test_unreachable_target_raises`.
