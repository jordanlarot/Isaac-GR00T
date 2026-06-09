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

"""RealMan EEF pose conversion utilities for the GR00T deploy bridge.

The EEF checkpoint speaks 9D end-effector pose per arm
(``[x, y, z, R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2]]`` — translation plus
the first two rows of the rotation matrix, GR00T's native ``XYZ_ROT6D``), while
the robot stack reports a 14D ``ee_pose`` of position + xyzw quaternion per arm.

These helpers convert between the two. They delegate to GR00T's
``EndEffectorPose`` (``gr00t/data/state_action/pose.py``) so the deploy path uses
the *exact* same convention that built the training dataset
(``scripts/convert_to_eef_gr00t.py``) — there is a single source of truth for the
quat <-> rot6d math, eliminating any train/deploy drift.

14D ee_pose layout (metres + xyzw quaternion, each arm in its own RealMan base
frame)::

    [0:3]   left  TCP position (x, y, z)
    [3:7]   left  quaternion (x, y, z, w)
    [7:10]  right TCP position (x, y, z)
    [10:14] right quaternion (x, y, z, w)
"""

from gr00t.data.state_action.pose import EndEffectorPose
from gr00t.data.types import ActionFormat
import numpy as np


# ee_pose slices per arm: (position slice, quaternion slice)
_ARM_SLICES = {
    "left": (slice(0, 3), slice(3, 7)),
    "right": (slice(7, 10), slice(10, 14)),
}


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a 6D rotation (first two rows of the matrix).

    Args:
        quat: Quaternion in xyzw (scalar-last, scipy/ROS) order, shape (4,).

    Returns:
        6D rotation ``[R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2]]``, shape (6,).
    """
    pose = EndEffectorPose(
        rotation=np.asarray(quat, dtype=float), rotation_type="quat", rotation_order="xyzw"
    )
    return pose.rot6d


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Convert a 6D rotation back to a 3x3 rotation matrix.

    Applies Gram-Schmidt orthonormalisation, so a non-orthonormal 6D input
    (e.g. a noisy policy output) still yields a valid right-handed rotation.

    Args:
        rot6d: 6D rotation (first two rows flattened), shape (6,).

    Returns:
        Orthonormal rotation matrix, shape (3, 3).
    """
    return EndEffectorPose._rot6d_to_matrix(np.asarray(rot6d, dtype=float))


def ee_pose_14d_to_eef_9d(ee_pose: np.ndarray, arm: str) -> np.ndarray:
    """Slice one arm out of the 14D ee_pose and convert it to GR00T's 9D EEF.

    Args:
        ee_pose: 14D ``[l_xyz, l_quat_xyzw, r_xyz, r_quat_xyzw]``.
        arm: ``"left"`` or ``"right"``.

    Returns:
        9D ``[x, y, z, rot6d(6)]`` for the requested arm, shape (9,).
    """
    if arm not in _ARM_SLICES:
        raise ValueError(f"arm must be 'left' or 'right', got {arm!r}")
    ee_pose = np.asarray(ee_pose, dtype=float)
    if ee_pose.shape != (14,):
        raise ValueError(f"ee_pose must have shape (14,), got {ee_pose.shape}")

    pos_slice, quat_slice = _ARM_SLICES[arm]
    pose = EndEffectorPose(
        translation=ee_pose[pos_slice],
        rotation=ee_pose[quat_slice],
        rotation_type="quat",
        rotation_order="xyzw",
    )
    return pose.xyz_rot6d


def eef_9d_to_quat_xyzw(eef_9d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 9D EEF pose to (position, xyzw quaternion) for IK input.

    Args:
        eef_9d: 9D ``[x, y, z, rot6d(6)]``.

    Returns:
        ``(position, quat_xyzw)`` — position shape (3,), unit quaternion shape (4,).
    """
    eef_9d = np.asarray(eef_9d, dtype=float)
    if eef_9d.shape != (9,):
        raise ValueError(f"eef_9d must have shape (9,), got {eef_9d.shape}")
    pose = EndEffectorPose.from_action_format(eef_9d, ActionFormat.XYZ_ROT6D)
    return pose.translation, pose.quat_xyzw
