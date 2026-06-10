# EEF-Pose Model Client — Agent Prompt

Use this prompt in Agent mode (or another AI session) to build a client that connects a model outputting **absolute end-effector (EEF) poses** to the existing IK + robot execution stack in this repo.

---

## Context

I'm working on the R2D3 dual-arm robot (RealMan RM_65) in `/home/r2d3/X-VLA_copy`. XVLA already has a working IK + robot execution stack that I want to reuse for a **different model** that outputs **absolute end-effector (EEF) poses** directly — not XVLA-style 20-D delta actions.

---

## Existing Infrastructure (Do NOT Reimplement IK)

The IK solver and robot control live here:

```
/home/r2d3/X-VLA_copy/teleoperation_IK_v1.0.0/python3/
├── ik_qp.py          # QPIK solver (core IK)
├── ik_rbtdef.py      # RM65B robot model
├── ik_rbtutils.py    # pose_to_matrix(), etc.
├── VLA_move.py       # Production wrapper: connect, IK, execute, gripper
├── ik_test.py        # Standalone IK reference
└── robot_config.yaml # Arm IPs, install_angle, work_cs, tool_cs
```

RealMan API:

```
/home/r2d3/X-VLA_copy/teleoperation_IK_v1.0.0/RM_API2-1.0.6/Python/Robotic_Arm/
```

---

## What XVLA Does Today (Reference Only)

`run_xvla.sh` starts:

1. `~/X-VLA/inference_server.py` — XVLA model HTTP server on port 8000
2. `xvla_client_v2.py` — cameras, proprio, HTTP inference, world↔robot frame transforms, then calls `VLA_move.py`

I do **not** need the XVLA inference client. I need a **new, simpler client** that takes EEF poses from my model and sends them through the existing IK stack.

### XVLA pipeline (for comparison)

```
Cameras + robot state
        ↓
   xvla_client_v2.py  ──HTTP POST──►  inference_server.py (X-VLA model)
        ↓                                      ↓
   20-D action deltas                   predicted actions
        ↓
   frame transforms (world ↔ robot)
        ↓
   VLA_move.py (IK solver)
        ↓
   robot hardware (joint commands + gripper)
```

---

## My Model's Output

Fill in before running:

| Field | Your value |
|-------|------------|
| Pose frame | robot frame or world frame? |
| Orientation format | Euler XYZ radians, quaternion, 6D rotation, other? |
| Which arm(s) | right only, left only, or dual-arm? |
| Model interface | local Python call, HTTP server, other? |
| Model server path | if applicable |

Example assumptions (edit as needed):

- Outputs **absolute EEF pose** per arm (not deltas)
- Format: `[x, y, z, rx, ry, rz]` + gripper
- Frame: robot frame
- Orientation: Euler XYZ radians
- Arms: right arm only

---

## Required Pose Format for `VLA_move.step()`

```python
target_pose = [x, y, z, rx, ry, rz]  # meters, Euler XYZ radians, robot frame
gripper = 0.0  # open  to  1.0  # closed
result = controller.step(target_pose, gripper)
```

### Key config from `robot_config.yaml`

**Right arm:**

- IP: `169.254.128.19`
- `install_angle: [0, -45, 0]` (deg)
- `work_cs: [0, 0, 0, 0, 0, 3.14159]`
- `tool_cs: [0, 0, 0.135, 0, 0, 0]` — 135 mm Z tool offset

**Left arm:**

- IP: `169.254.128.18`
- `install_angle: [0, 45, 0]` (deg)
- `work_cs: [0, 0, 0, 0, 0, -1.571]`
- `tool_cs: [0, 0, 0, 0, 0, 0]`

---

## What to Build

Create a new client script, e.g. `eef_pose_client.py`, in:

```
/home/r2d3/X-VLA_copy/teleoperation_IK_v1.0.0/python3/
```

### Requirements

1. **Initialize robot** via `VLA_move.init_vla(["right_arm"])` (or both arms if needed).

2. **Accept model output** — one of:
   - Direct function call with EEF pose
   - HTTP client to my model server (similar pattern to `xvla_client_v2.py` but simpler)
   - CLI with hardcoded/test poses for validation

3. **Convert pose format** if my model uses quaternions/6D/etc. → `[x, y, z, rx, ry, rz]` Euler radians.

4. **Apply frame transform** if my model outputs world-frame poses (reuse mount/transform logic from `xvla_client_v2.py` only if needed).

5. **Call `controller.step(target_pose, gripper)`** — let `VLA_move.py` handle IK, `rm_movej_canfd`, and gripper.

6. **Run a control loop** at configurable Hz (e.g. 15 Hz), with:
   - Convergence detection (optional)
   - Safe-stop on error via `VLA_move.emergency_stop()`
   - Status logging from `step()` return dict (`status`, `pred_err`, `cmd_joints`, etc.)

7. **Support `SIM_MODE`** flag — compute IK but don't move robot (mirror pattern from `xvla_client_v2.py`).

### `VLA_move.py` API to use

```python
from VLA_move import init_vla, RIGHT_CONTROLLER, LEFT_CONTROLLER
from VLA_move import right_arm_move, left_arm_move, get_arm_state, emergency_stop

init_vla(["right_arm"])
controller = RIGHT_CONTROLLER

result = controller.step(target_pose, gripper)
# result keys: status, pred_err, cmd_joints, act_joints, cap_reason, ...
```

---

## Do NOT

- Reimplement IK — use `VLA_move.py` / `QPIK` as-is
- Copy the full XVLA client (cameras, 20-D proprio, circuit breaker, etc.) unless explicitly requested
- Modify `ik_qp.py` or `VLA_move.py` unless there is a clear bug

---

## Reference Files to Read First

| File | Why |
|------|-----|
| `VLA_move.py` | `VLAArmController.step()`, `init_vla()`, `right_arm_move()`, `get_arm_state()` |
| `ik_test.py` | Minimal IK usage example |
| `xvla_client_v2.py` | Frame transform logic only, if poses are in world frame |
| `robot_config.yaml` | Arm network config and coordinate frames |
| `ik_rbtutils.py` | `pose_to_matrix()` — expected Euler convention |

---

## Deliverables

1. `eef_pose_client.py` — main client
2. Optional `run_eef_model.sh` — launcher script (model server + client, similar to `run_xvla.sh`)
3. Brief usage instructions in comments at top of the script

---

## Test Plan

1. **`SIM_MODE=True`** — verify IK solves without moving robot
2. **Single test pose via CLI** — verify arm moves correctly
3. **Loop with my model** — verify sustained execution at target Hz

---

## Full Agent Prompt (Copy-Paste)

```
I'm working on the R2D3 dual-arm robot (RealMan RM_65) in /home/r2d3/X-VLA_copy.
I have a model that outputs absolute EEF poses (not XVLA 20-D deltas). Reuse the
existing IK stack in teleoperation_IK_v1.0.0/python3/ — do NOT reimplement IK.

Read these files first:
- teleoperation_IK_v1.0.0/python3/VLA_move.py
- teleoperation_IK_v1.0.0/python3/ik_test.py
- teleoperation_IK_v1.0.0/python3/robot_config.yaml
- teleoperation_IK_v1.0.0/python3/xvla_client_v2.py (frame transforms only, if needed)

Build teleoperation_IK_v1.0.0/python3/eef_pose_client.py that:
1. Initializes robot via VLA_move.init_vla()
2. Accepts EEF poses [x,y,z,rx,ry,rz] + gripper from my model (CLI/HTTP/direct)
3. Converts orientation/frame if needed to robot-frame Euler radians
4. Calls controller.step(target_pose, gripper) for IK + execution
5. Runs a control loop at configurable Hz with logging and emergency_stop()
6. Supports SIM_MODE (IK only, no physical motion)

Pose format for VLA_move.step(): [x,y,z,rx,ry,rz] meters/radians, robot frame.
Right arm tool_cs has 135mm Z offset per robot_config.yaml.

Do not copy the full XVLA client. Do not modify ik_qp.py or VLA_move.py unless
there is a clear bug. Add optional run_eef_model.sh launcher.

Test: SIM_MODE first, then single CLI pose, then model loop.
Confirm pose frame and orientation format with me if ambiguous before implementing.
```

---

## Notes

- XVLA outputs **deltas** in world frame; this client expects **absolute** poses (or you add a delta→absolute step).
- `VLA_move.step()` runs iterative IK (15 iterations), safety checks (joint limits, unreachable pose), then `rm_movej_canfd`.
- Gripper threshold in XVLA client is 0.5; `VLA_move` uses the same threshold internally.
