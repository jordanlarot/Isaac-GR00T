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

"""Tests for RealMan EEF -> joint IK (deploy bridge, Phase 3).

The IK wraps the QPIK solver from the XVLA teleoperation stack
(/home/r2d3/X-VLA_copy/teleoperation_IK_v1.0.0). The frame parameters
(install_angle, work_cs, tool_cs) were calibrated against the EEF training
dataset: with work_cs rz = -pi/2 and a 135 mm Z tool offset on BOTH arms,
QPIK's FK reproduces the dataset ``observation.ee_pose`` from the dataset
joint state to <0.5 mm / <0.15 deg. (The XVLA robot_config.yaml uses different
work_cs/tool_cs values because XVLA operated in its own world frame.)

Golden frames below are real (state, ee_pose) rows from the EEF dataset
(pick-place-bottle-v2-gr00t-eef, episode 0, frames 10 and 120).

State layout: [l_arm(6), l_gripper(1), r_arm(6), r_gripper(1)] (radians).
ee_pose layout: [l_xyz(3), l_quat_xyzw(4), r_xyz(3), r_quat_xyzw(4)].
"""

import os

from gr00t.eval.real_robot.realman import eef_utils
from gr00t.eval.real_robot.realman.realman_ik import IK_STACK_DIR, RealmanIK, rotmat_to_euler_xyz
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


pytestmark = pytest.mark.skipif(
    not os.path.isdir(IK_STACK_DIR),
    reason=f"XVLA IK stack not found at {IK_STACK_DIR}",
)

DATASET_DIR = "/home/r2d3/datasets/pick-place-bottle-v2-gr00t-eef"

# Real rows from episode 0 of the EEF dataset.
GOLDEN_FRAMES = {
    10: {
        "state": np.array(
            [
                -0.4999774098396301,
                -2.030604124069214,
                -0.39412569999694824,
                1.5412886142730713,
                1.3157823085784912,
                0.004571899771690369,
                0.911,
                0.446318656206131,
                2.024444341659546,
                0.6518098711967468,
                -1.5462794303894043,
                -1.288176417350769,
                -0.3757159411907196,
                0.958,
            ]
        ),
        "ee_pose": np.array(
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
            ]
        ),
    },
    120: {
        "state": np.array(
            [
                -0.5021063089370728,
                -2.0227341651916504,
                -0.3864826261997223,
                1.5643402338027954,
                1.3096922636032104,
                0.004571899771690369,
                0.912,
                0.870074450969696,
                1.5091458559036255,
                0.9680212736129761,
                -1.2578134536743164,
                -0.9573767781257629,
                -0.16502465307712555,
                0.483,
            ]
        ),
        "ee_pose": np.array(
            [
                0.43838098645210266,
                0.11201199889183044,
                -0.22509700059890747,
                0.6640141010284424,
                -0.1256842315196991,
                0.7260620594024658,
                0.12697479128837585,
                0.5521569848060608,
                -0.1336749941110611,
                -0.12302500009536743,
                -0.1651848554611206,
                0.7028411030769348,
                0.10929503291845322,
                0.6832151412963867,
            ]
        ),
    },
}

ARM_SLICES = {
    "left": {"joints": slice(0, 6), "pos": slice(0, 3), "quat": slice(3, 7)},
    "right": {"joints": slice(7, 13), "pos": slice(7, 10), "quat": slice(10, 14)},
}


def _fk_errors(ik: RealmanIK, joints, ee_pose, arm):
    """(position error mm, rotation error deg) of FK(joints) vs dataset ee_pose."""
    sl = ARM_SLICES[arm]
    T = ik.fk(joints)
    pos_err_mm = np.linalg.norm(T[:3, 3] - ee_pose[sl["pos"]]) * 1000
    R_ds = Rotation.from_quat(ee_pose[sl["quat"]]).as_matrix()
    cos = np.clip((np.trace(R_ds.T @ T[:3, :3]) - 1) / 2, -1, 1)
    rot_err_deg = np.degrees(np.arccos(cos))
    return pos_err_mm, rot_err_deg


class TestRealmanIKInit:
    def test_right_arm_initialises(self):
        assert RealmanIK("right") is not None

    def test_left_arm_initialises(self):
        assert RealmanIK("left") is not None

    def test_bad_arm_raises(self):
        with pytest.raises(ValueError):
            RealmanIK("middle")


class TestFkMatchesDataset:
    """Pin the frame calibration: FK(dataset joints) must equal dataset ee_pose.

    If these fail, the install_angle/work_cs/tool_cs in realman_ik.py no longer
    match the frame the training data was collected in — IK output would be
    silently wrong on hardware.
    """

    @pytest.mark.parametrize("frame", [10, 120])
    @pytest.mark.parametrize("arm", ["right", "left"])
    def test_fk_matches_golden(self, arm, frame):
        ik = RealmanIK(arm)
        golden = GOLDEN_FRAMES[frame]
        joints = golden["state"][ARM_SLICES[arm]["joints"]]
        pos_err_mm, rot_err_deg = _fk_errors(ik, joints, golden["ee_pose"], arm)
        assert pos_err_mm < 2.0, f"FK position error {pos_err_mm:.2f} mm"
        assert rot_err_deg < 0.5, f"FK rotation error {rot_err_deg:.2f} deg"


class TestEulerConvention:
    """Pin rotmat -> euler against pose_to_matrix (QPIK's target convention)."""

    def test_roundtrip_through_pose_to_matrix(self):
        import sys

        sys.path.insert(0, os.path.join(IK_STACK_DIR, "python3"))
        from ik_rbtutils import pose_to_matrix

        rng = np.random.default_rng(7)
        for _ in range(20):
            R = Rotation.random(random_state=rng).as_matrix()
            rx, ry, rz = rotmat_to_euler_xyz(R)
            T = np.asarray(pose_to_matrix([0.0, 0.0, 0.0, rx, ry, rz]))
            np.testing.assert_allclose(T[:3, :3], R, atol=1e-10)


class TestEef9dToJointAngles:
    @pytest.fixture(scope="class")
    def right_ik(self):
        return RealmanIK("right")

    def _target_and_seed(self, frame_target, frame_seed):
        """Target eef_9d from one golden frame, seed joints from another (warm start)."""
        target = eef_utils.ee_pose_14d_to_eef_9d(GOLDEN_FRAMES[frame_target]["ee_pose"], "right")
        seed = GOLDEN_FRAMES[frame_seed]["state"][ARM_SLICES["right"]["joints"]]
        return target, seed

    def test_output_shape_and_dtype(self, right_ik):
        target, seed = self._target_and_seed(10, 10)
        q = right_ik.eef_9d_to_joint_angles(target, seed)
        assert q.shape == (6,)
        assert q.dtype == np.float64

    def test_no_nans(self, right_ik):
        target, seed = self._target_and_seed(10, 120)
        assert not np.isnan(right_ik.eef_9d_to_joint_angles(target, seed)).any()

    def test_within_joint_limits(self, right_ik):
        target, seed = self._target_and_seed(120, 10)
        q_deg = np.degrees(right_ik.eef_9d_to_joint_angles(target, seed))
        assert (q_deg >= right_ik.q_min_deg - 1e-6).all()
        assert (q_deg <= right_ik.q_max_deg + 1e-6).all()

    @pytest.mark.parametrize("frame_target,frame_seed", [(10, 120), (120, 10)])
    def test_fk_residual_warm_start(self, right_ik, frame_target, frame_seed):
        """Solve toward one golden pose seeded from the other (realistic deploy step)."""
        target, seed = self._target_and_seed(frame_target, frame_seed)
        q = right_ik.eef_9d_to_joint_angles(target, seed)
        T = right_ik.fk(q)
        pos_err_mm = np.linalg.norm(T[:3, 3] - target[:3]) * 1000
        R_target = eef_utils.rot6d_to_rotmat(target[3:])
        cos = np.clip((np.trace(R_target.T @ T[:3, :3]) - 1) / 2, -1, 1)
        rot_err_deg = np.degrees(np.arccos(cos))
        assert pos_err_mm < 5.0, f"IK position residual {pos_err_mm:.2f} mm"
        assert rot_err_deg < 5.0, f"IK rotation residual {rot_err_deg:.2f} deg"

    def test_recovers_dataset_joints_when_seeded_nearby(self, right_ik):
        """Seeded at the ground-truth joints, the solution must stay there."""
        golden = GOLDEN_FRAMES[10]
        target = eef_utils.ee_pose_14d_to_eef_9d(golden["ee_pose"], "right")
        seed = golden["state"][ARM_SLICES["right"]["joints"]]
        q = right_ik.eef_9d_to_joint_angles(target, seed)
        np.testing.assert_allclose(q, seed, atol=0.05)  # rad, plan acceptance

    def test_unreachable_target_raises(self, right_ik):
        """An out-of-workspace target must raise, not silently return a wrong pose.

        QPIK clamps joints to limits, so the solution for an unreachable target
        is within limits but its FK lands far from the target — without a final
        residual check the deploy client would command a wrong-but-valid pose.
        """
        target, seed = self._target_and_seed(10, 10)
        target = target.copy()
        target[0] += 1.5  # 1.5 m beyond the RM65 workspace
        with pytest.raises(ValueError, match="residual"):
            right_ik.eef_9d_to_joint_angles(target, seed)

    def test_wrong_target_shape_raises(self, right_ik):
        with pytest.raises(ValueError):
            right_ik.eef_9d_to_joint_angles(np.zeros(8), np.zeros(6))

    def test_wrong_seed_shape_raises(self, right_ik):
        with pytest.raises(ValueError):
            right_ik.eef_9d_to_joint_angles(np.zeros(9), np.zeros(7))


@pytest.mark.skipif(not os.path.isdir(DATASET_DIR), reason="EEF dataset not on disk")
class TestDatasetSpike:
    """Phase 3 acceptance: IK residual <= 5 mm on 10 random dataset frames."""

    def test_ten_random_frames(self):
        import pandas as pd

        rng = np.random.default_rng(3)
        ik = RealmanIK("right")
        episodes = ["episode_000000.parquet", "episode_000050.parquet", "episode_000101.parquet"]
        residuals = []
        for ep in episodes:
            df = pd.read_parquet(f"{DATASET_DIR}/data/chunk-000/{ep}")
            idx = rng.choice(len(df) - 1, size=4, replace=False)
            for i in idx:
                # Seed from the previous frame's joints: deploy-realistic warm start.
                seed = np.asarray(df.iloc[max(i - 1, 0)]["observation.state"][7:13], dtype=float)
                ee = np.asarray(df.iloc[i]["observation.ee_pose"], dtype=float)
                target = eef_utils.ee_pose_14d_to_eef_9d(ee, "right")
                q = ik.eef_9d_to_joint_angles(target, seed)
                T = ik.fk(q)
                residuals.append(np.linalg.norm(T[:3, 3] - target[:3]) * 1000)
        worst = max(residuals)
        assert len(residuals) >= 10
        assert worst < 5.0, f"worst IK residual {worst:.2f} mm over {len(residuals)} frames"
