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

"""Tests for RealMan EEF pose conversion utilities (deploy bridge).

These utilities must match the convention used to build the EEF training dataset
(scripts/convert_to_eef_gr00t.py): an xyzw quaternion is turned into a rotation
matrix and the 6D rotation is the *first two rows* of that matrix
(``[R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2]]``). This is GR00T's native
``XYZ_ROT6D`` convention as implemented by ``EndEffectorPose``.

The golden sample below is a real ``observation.ee_pose`` row from the joint
dataset (pick-bottle-v2-gr00t, episode 0, frame 10) — the same source column the
EEF dataset was derived from.
"""

from gr00t.eval.real_robot.realman import eef_utils
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


# Real 14D ee_pose row: [l_xyz, l_quat_xyzw, r_xyz, r_quat_xyzw]
GOLDEN_EE_POSE_14D = np.array(
    [
        0.43790000677108765,
        0.11265800148248672,
        -0.2248930037021637,
        0.6647500395774841,
        -0.11177468299865723,
        0.7272984385490417,
        0.12903808057308197,
        0.39914798736572266,
        -0.06168900057673454,
        -0.24007900059223175,
        -0.16885802149772644,
        0.67503821849823,
        0.09500017762184143,
        0.7118885517120361,
    ],
    dtype=np.float64,
)


def _expected_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """Independent reference: xyzw quat -> matrix -> first two rows flattened."""
    mat = Rotation.from_quat(quat_xyzw).as_matrix()
    return mat[:2, :].flatten()


class TestQuatXyzwToRot6d:
    def test_identity_quat(self):
        # Identity rotation -> first two rows of the identity matrix.
        rot6d = eef_utils.quat_xyzw_to_rot6d(np.array([0.0, 0.0, 0.0, 1.0]))
        np.testing.assert_allclose(rot6d, [1, 0, 0, 0, 1, 0], atol=1e-8)

    def test_matches_scipy_reference(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            quat = Rotation.random(random_state=rng).as_quat()  # xyzw
            np.testing.assert_allclose(
                eef_utils.quat_xyzw_to_rot6d(quat), _expected_rot6d(quat), atol=1e-10
            )

    def test_output_shape(self):
        assert eef_utils.quat_xyzw_to_rot6d(np.array([0.0, 0.0, 0.0, 1.0])).shape == (6,)


class TestRot6dToRotmat:
    def test_identity(self):
        rotmat = eef_utils.rot6d_to_rotmat(np.array([1, 0, 0, 0, 1, 0], dtype=float))
        np.testing.assert_allclose(rotmat, np.eye(3), atol=1e-8)

    def test_returns_valid_rotation(self):
        # rot6d_to_rotmat must produce an orthonormal, right-handed matrix even
        # from a non-orthonormal 6D input (Gram-Schmidt).
        rng = np.random.default_rng(1)
        for _ in range(20):
            quat = Rotation.random(random_state=rng).as_quat()
            rot6d = _expected_rot6d(quat)
            rotmat = eef_utils.rot6d_to_rotmat(rot6d)
            np.testing.assert_allclose(rotmat @ rotmat.T, np.eye(3), atol=1e-8)
            np.testing.assert_allclose(np.linalg.det(rotmat), 1.0, atol=1e-8)

    def test_inverts_quat_to_rot6d(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            quat = Rotation.random(random_state=rng).as_quat()
            expected_mat = Rotation.from_quat(quat).as_matrix()
            rot6d = eef_utils.quat_xyzw_to_rot6d(quat)
            np.testing.assert_allclose(eef_utils.rot6d_to_rotmat(rot6d), expected_mat, atol=1e-8)


class TestEePose14dToEef9d:
    def test_left_arm_matches_reference(self):
        eef9d = eef_utils.ee_pose_14d_to_eef_9d(GOLDEN_EE_POSE_14D, "left")
        assert eef9d.shape == (9,)
        np.testing.assert_allclose(eef9d[:3], GOLDEN_EE_POSE_14D[0:3], atol=1e-8)
        np.testing.assert_allclose(eef9d[3:], _expected_rot6d(GOLDEN_EE_POSE_14D[3:7]), atol=1e-8)

    def test_right_arm_matches_reference(self):
        eef9d = eef_utils.ee_pose_14d_to_eef_9d(GOLDEN_EE_POSE_14D, "right")
        assert eef9d.shape == (9,)
        np.testing.assert_allclose(eef9d[:3], GOLDEN_EE_POSE_14D[7:10], atol=1e-8)
        np.testing.assert_allclose(eef9d[3:], _expected_rot6d(GOLDEN_EE_POSE_14D[10:14]), atol=1e-8)

    def test_invalid_arm_raises(self):
        with pytest.raises(ValueError):
            eef_utils.ee_pose_14d_to_eef_9d(GOLDEN_EE_POSE_14D, "middle")

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            eef_utils.ee_pose_14d_to_eef_9d(np.zeros(13), "left")


class TestEef9dToQuatXyzw:
    def test_returns_pos_and_unit_quat(self):
        eef9d = eef_utils.ee_pose_14d_to_eef_9d(GOLDEN_EE_POSE_14D, "right")
        pos, quat = eef_utils.eef_9d_to_quat_xyzw(eef9d)
        np.testing.assert_allclose(pos, GOLDEN_EE_POSE_14D[7:10], atol=1e-8)
        assert quat.shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-8)

    def test_roundtrip_recovers_rotation(self):
        # ee_pose -> eef_9d -> (pos, quat) must recover the original rotation
        # (compare as matrices to avoid quaternion double-cover sign ambiguity).
        for arm, qslice in (("left", slice(3, 7)), ("right", slice(10, 14))):
            eef9d = eef_utils.ee_pose_14d_to_eef_9d(GOLDEN_EE_POSE_14D, arm)
            _, quat = eef_utils.eef_9d_to_quat_xyzw(eef9d)
            recovered = Rotation.from_quat(quat).as_matrix()
            original = Rotation.from_quat(GOLDEN_EE_POSE_14D[qslice]).as_matrix()
            np.testing.assert_allclose(recovered, original, atol=1e-6)


def test_full_roundtrip_quat_rot6d_rotmat_quat():
    """Phase 2 acceptance: quat -> rot6d -> rotmat -> quat within tolerance."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        quat = Rotation.random(random_state=rng).as_quat()
        rot6d = eef_utils.quat_xyzw_to_rot6d(quat)
        rotmat = eef_utils.rot6d_to_rotmat(rot6d)
        recovered_quat = Rotation.from_matrix(rotmat).as_quat()
        # Compare rotations, not raw quats (double cover): R(q) == R(q').
        np.testing.assert_allclose(
            Rotation.from_quat(recovered_quat).as_matrix(),
            Rotation.from_quat(quat).as_matrix(),
            atol=1e-8,
        )
