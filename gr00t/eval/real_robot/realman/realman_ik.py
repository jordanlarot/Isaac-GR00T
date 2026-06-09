# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RealMan RM65 inverse kinematics for the GR00T EEF deploy bridge (Phase 3).

Converts an absolute 9D EEF target (``[x, y, z, rot6d]``, GR00T's XYZ_ROT6D —
the format the EEF checkpoint outputs) into 6 joint angles, by wrapping the
QPIK solver from the XVLA teleoperation stack. Pure math: no robot connection
is opened; execution stays in robot_api_server's ``/action``.

Frame calibration
-----------------
The frame parameters below were calibrated against the EEF training dataset
(pick-place-bottle-v2-gr00t-eef), NOT taken from the XVLA robot_config.yaml:
with ``work_cs`` rz = -pi/2 and a 135 mm Z tool offset on BOTH arms, QPIK's
FK reproduces the dataset ``observation.ee_pose`` from the dataset joint
state to <0.5 mm / <0.15 deg on every episode checked. The XVLA yaml uses
different work_cs/tool_cs values because XVLA operated in its own world
frame; the GR00T checkpoint was trained in the frame the ROS bridge reports
(``get_ee_poses_at_time``), so that is the frame this IK must solve in.
``TestFkMatchesDataset`` pins this calibration against golden dataset rows.

The XVLA IK stack location can be overridden with the ``REALMAN_IK_STACK_DIR``
environment variable.
"""

import os
import sys

from gr00t.eval.real_robot.realman.eef_utils import rot6d_to_rotmat
import numpy as np
from scipy.spatial.transform import Rotation


IK_STACK_DIR = os.environ.get(
    "REALMAN_IK_STACK_DIR", "/home/r2d3/X-VLA_copy/teleoperation_IK_v1.0.0"
)

# Calibrated against the EEF training dataset — see module docstring.
_ARM_CONFIGS = {
    "left": {
        "install_angle_deg": [0, 45, 0],
        "work_cs": [0, 0, 0, 0, 0, -np.pi / 2],
        "tool_cs": [0, 0, 0.135, 0, 0, 0],
    },
    "right": {
        "install_angle_deg": [0, -45, 0],
        "work_cs": [0, 0, 0, 0, 0, -np.pi / 2],
        "tool_cs": [0, 0, 0.135, 0, 0, 0],
    },
}

# RM65 joint limits, degrees (same constants as VLA_move.py / ik_test.py).
_Q_MIN_DEG = np.array([-178.0, -130.0, -135.0, -178.0, -128.0, -360.0])
_Q_MAX_DEG = np.array([178.0, 130.0, 135.0, 178.0, 128.0, 360.0])


def _ensure_ik_stack_on_path() -> None:
    for sub in ("python3", "RM_API2-1.0.6/Python"):
        path = os.path.join(IK_STACK_DIR, sub)
        if path not in sys.path:
            sys.path.insert(0, path)


def rotmat_to_euler_xyz(rotmat: np.ndarray) -> np.ndarray:
    """Rotation matrix -> ``[rx, ry, rz]`` in QPIK's pose convention.

    ``pose_to_matrix`` (ik_rbtutils.py) builds R = Rz(rz) @ Ry(ry) @ Rx(rx),
    i.e. extrinsic x-y-z — scipy's lowercase ``'xyz'``.
    """
    return Rotation.from_matrix(np.asarray(rotmat, dtype=float)).as_euler("xyz")


class RealmanIK:
    """Offline QPIK wrapper for one RM65 arm in the training-data frame."""

    def __init__(self, arm: str = "right"):
        if arm not in _ARM_CONFIGS:
            raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
        self.arm = arm
        self.q_min_deg = _Q_MIN_DEG.copy()
        self.q_max_deg = _Q_MAX_DEG.copy()

        _ensure_ik_stack_on_path()
        from ik_qp import QPIK

        cfg = _ARM_CONFIGS[arm]
        self._solver = QPIK("RM65B", 0.01)
        self._solver.set_install_angle(cfg["install_angle_deg"], "deg")
        self._solver.set_work_cs_params(cfg["work_cs"])
        self._solver.set_tool_cs_params(cfg["tool_cs"])
        self._solver.set_joint_limit_min(self.q_min_deg, "deg")
        self._solver.set_joint_limit_max(self.q_max_deg, "deg")

    def fk(self, joints: np.ndarray) -> np.ndarray:
        """Forward kinematics: 6 joint angles (rad) -> 4x4 TCP pose, dataset frame."""
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,):
            raise ValueError(f"joints must have shape (6,), got {joints.shape}")
        return np.asarray(self._solver.robot.fkine(joints))

    def eef_9d_to_joint_angles(
        self,
        target_eef_9d: np.ndarray,
        seed_joints: np.ndarray,
        max_iterations: int = 100,
        pos_tol_m: float = 5e-4,
        max_residual_m: float = 0.005,
    ) -> np.ndarray:
        """Solve IK for an absolute 9D EEF target.

        QPIK is a velocity-clamped differential solver (~0.9 ms/iteration on
        Orin), so iterations needed scale with distance to the target. The
        loop exits as soon as the FK position residual drops below
        ``pos_tol_m``: small deploy steps converge in a few iterations, while
        a large jump (e.g. 0.2 m after an episode reset) needs ~75.

        Args:
            target_eef_9d: ``[x, y, z, rot6d(6)]`` in metres, dataset frame.
            seed_joints: Current joint angles in radians, shape (6,) — the
                solver iterates from here, so live observed state gives the
                solution branch closest to the arm's actual configuration.
            max_iterations: Iteration cap (safety against unreachable targets).
            pos_tol_m: Early-exit position tolerance in metres.
            max_residual_m: Maximum acceptable FK position residual of the
                final solution. QPIK clamps joints to their limits, so for an
                out-of-workspace target it returns a within-limits solution
                whose FK lands far from the target — this check turns that
                into an error instead of a silent wrong pose. (VLA_move.py
                guards the same failure with its ``pred_err > 0.3`` check.)

        Returns:
            Joint angles in radians, shape (6,).

        Raises:
            ValueError: on bad input shapes, NaNs in the solution, a solution
                outside the RM65 joint limits, or a final FK position residual
                above ``max_residual_m`` (unreachable / non-converged target).
        """
        target_eef_9d = np.asarray(target_eef_9d, dtype=float)
        if target_eef_9d.shape != (9,):
            raise ValueError(f"target_eef_9d must have shape (9,), got {target_eef_9d.shape}")
        seed_joints = np.asarray(seed_joints, dtype=float)
        if seed_joints.shape != (6,):
            raise ValueError(f"seed_joints must have shape (6,), got {seed_joints.shape}")

        from ik_rbtutils import pose_to_matrix

        rotmat = rot6d_to_rotmat(target_eef_9d[3:])
        rx, ry, rz = rotmat_to_euler_xyz(rotmat)
        Td = pose_to_matrix([*target_eef_9d[:3], rx, ry, rz])

        q_sol = seed_joints.copy()
        residual_m = np.inf
        for _ in range(max_iterations):
            q_sol = self._solver.sovler(q_sol, Td)
            T = self._solver.robot.fkine(q_sol)
            residual_m = np.linalg.norm(np.asarray(T)[:3, 3] - target_eef_9d[:3])
            if residual_m < pos_tol_m:
                break
        q_sol = np.asarray(q_sol, dtype=float)

        if np.isnan(q_sol).any():
            raise ValueError(f"IK produced NaN joints for target {target_eef_9d[:3]}")
        if residual_m > max_residual_m:
            raise ValueError(
                f"IK did not converge: FK position residual {residual_m * 1000:.1f} mm "
                f"> {max_residual_m * 1000:.1f} mm for target {np.round(target_eef_9d[:3], 3)} "
                f"(unreachable target or max_iterations={max_iterations} too low)"
            )
        q_deg = np.degrees(q_sol)
        if (q_deg < self.q_min_deg - 1e-6).any() or (q_deg > self.q_max_deg + 1e-6).any():
            raise ValueError(
                f"IK solution violates joint limits: {np.round(q_deg, 1)} deg "
                f"(limits {self.q_min_deg} .. {self.q_max_deg})"
            )
        return q_sol
