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

"""Tests for the EEF deploy client's pure conversion functions (Phase 4).

The closed-loop controller itself is network-bound (robot API + ZMQ policy
server) and is exercised on hardware; these tests cover the format-critical
pieces: observation building, action key slicing, and the EEF action ->
14D joint command conversion (including the IK-failure hold path).

Golden data is a real (state, ee_pose) row from the EEF dataset
(pick-place-bottle-v2-gr00t-eef, episode 0, frame 10).
"""

import base64
import os
import types

from gr00t.eval.real_robot.realman import eef_utils
from gr00t.eval.real_robot.realman.deploy_groot_realman_eef import (
    action_key_slices,
    eef_action_to_joint_command,
    parse_observation_eef,
)
from gr00t.eval.real_robot.realman.realman_ik import IK_STACK_DIR, RealmanIK
import numpy as np
import pytest


CANONICAL_ACTION_KEYS = ["left_eef_9d", "left_gripper", "right_eef_9d", "right_gripper"]

# Real row from episode 0, frame 10 of the EEF dataset.
GOLDEN_STATE_14D = np.array(
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
)
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
    ]
)


def _b64_jpeg(h=8, w=8):
    import cv2

    img = np.zeros((h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _modality_configs(state_keys, video_keys, language_keys=("annotation.human.task_description",)):
    mk = lambda keys: types.SimpleNamespace(modality_keys=list(keys))  # noqa: E731
    return {"state": mk(state_keys), "video": mk(video_keys), "language": mk(language_keys)}


def _raw_obs():
    return {
        "state": GOLDEN_STATE_14D.tolist(),
        "ee_pose": GOLDEN_EE_POSE_14D.tolist(),
        "gripper_force": [1.0, 2.0],
        "images": {
            "top_camera": _b64_jpeg(),
            "left_wrist": _b64_jpeg(),
            "right_wrist": _b64_jpeg(),
        },
    }


class TestActionKeySlices:
    def test_canonical_order(self):
        slices = action_key_slices(CANONICAL_ACTION_KEYS)
        assert slices["left_eef_9d"] == slice(0, 9)
        assert slices["left_gripper"] == slice(9, 10)
        assert slices["right_eef_9d"] == slice(10, 19)
        assert slices["right_gripper"] == slice(19, 20)

    def test_total_dims_is_20(self):
        slices = action_key_slices(CANONICAL_ACTION_KEYS)
        assert sum(s.stop - s.start for s in slices.values()) == 20

    def test_respects_server_key_order(self):
        keys = ["right_eef_9d", "right_gripper", "left_eef_9d", "left_gripper"]
        slices = action_key_slices(keys)
        assert slices["right_eef_9d"] == slice(0, 9)
        assert slices["left_gripper"] == slice(19, 20)

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="right_arm"):
            action_key_slices(["left_eef_9d", "right_arm"])


class TestParseObservationEEF:
    def test_state_keys_built_from_ee_pose(self):
        configs = _modality_configs(
            state_keys=["left_eef_9d", "left_gripper", "right_eef_9d", "right_gripper"],
            video_keys=["top_camera"],
        )
        obs = parse_observation_eef(_raw_obs(), configs, "pick up bottle")

        right = obs["state"]["right_eef_9d"]
        assert right.shape == (1, 1, 9)
        np.testing.assert_allclose(
            right[0, 0],
            eef_utils.ee_pose_14d_to_eef_9d(GOLDEN_EE_POSE_14D, "right"),
            atol=1e-6,
        )
        left = obs["state"]["left_eef_9d"]
        np.testing.assert_allclose(
            left[0, 0],
            eef_utils.ee_pose_14d_to_eef_9d(GOLDEN_EE_POSE_14D, "left"),
            atol=1e-6,
        )
        assert obs["state"]["left_gripper"].shape == (1, 1, 1)
        np.testing.assert_allclose(obs["state"]["left_gripper"][0, 0, 0], 0.911, atol=1e-6)
        np.testing.assert_allclose(obs["state"]["right_gripper"][0, 0, 0], 0.958, atol=1e-6)

    def test_video_and_language(self):
        configs = _modality_configs(
            state_keys=["right_eef_9d"], video_keys=["top_camera", "right_wrist"]
        )
        obs = parse_observation_eef(_raw_obs(), configs, "pick up bottle")
        assert obs["video"]["top_camera"].shape == (1, 1, 8, 8, 3)
        assert obs["video"]["top_camera"].dtype == np.uint8
        assert obs["language"]["annotation.human.task_description"] == [["pick up bottle"]]

    def test_missing_ee_pose_raises_clear_error(self):
        raw = _raw_obs()
        del raw["ee_pose"]
        configs = _modality_configs(state_keys=["right_eef_9d"], video_keys=["top_camera"])
        with pytest.raises(ValueError, match="ee_pose"):
            parse_observation_eef(raw, configs, "pick up bottle")


@pytest.mark.skipif(
    not os.path.isdir(IK_STACK_DIR), reason=f"XVLA IK stack not found at {IK_STACK_DIR}"
)
class TestEefActionToJointCommand:
    @pytest.fixture(scope="class")
    def ik(self):
        return RealmanIK("right")

    @pytest.fixture(scope="class")
    def slices(self):
        return action_key_slices(CANONICAL_ACTION_KEYS)

    def _golden_action_20d(self):
        """20D action whose right EEF target is the golden frame's own pose."""
        action = np.zeros(20)
        action[0:9] = eef_utils.ee_pose_14d_to_eef_9d(GOLDEN_EE_POSE_14D, "left")
        action[9] = 0.9
        action[10:19] = eef_utils.ee_pose_14d_to_eef_9d(GOLDEN_EE_POSE_14D, "right")
        action[19] = 0.42
        return action

    def test_left_side_pinned_to_state(self, ik, slices):
        execute, info = eef_action_to_joint_command(
            self._golden_action_20d(), slices, GOLDEN_STATE_14D, ik
        )
        assert execute.shape == (14,)
        np.testing.assert_allclose(execute[0:7], GOLDEN_STATE_14D[0:7], atol=1e-9)

    def test_right_joints_reach_target(self, ik, slices):
        action = self._golden_action_20d()
        execute, info = eef_action_to_joint_command(action, slices, GOLDEN_STATE_14D, ik)
        assert info["ik_ok"]
        T = ik.fk(execute[7:13])
        pos_err_mm = np.linalg.norm(T[:3, 3] - action[10:13]) * 1000
        assert pos_err_mm < 5.0
        assert info["ik_residual_mm"] < 5.0

    def test_right_gripper_from_model(self, ik, slices):
        execute, _ = eef_action_to_joint_command(
            self._golden_action_20d(), slices, GOLDEN_STATE_14D, ik
        )
        np.testing.assert_allclose(execute[13], 0.42, atol=1e-9)

    def test_ik_failure_holds_position(self, ik, slices):
        """Unreachable target: command must hold the observed state, not move."""
        action = self._golden_action_20d()
        action[10] += 1.5  # right target 1.5 m out of workspace
        execute, info = eef_action_to_joint_command(action, slices, GOLDEN_STATE_14D, ik)
        assert not info["ik_ok"]
        np.testing.assert_allclose(execute, GOLDEN_STATE_14D, atol=1e-9)
        assert "error" in info
