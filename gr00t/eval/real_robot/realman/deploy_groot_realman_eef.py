#!/usr/bin/env python3
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

"""Closed-loop GR00T EEF policy deployment for the RealMan dual-arm robot (R2D3).

Architecture (same three processes as the joint-space client):
    robot_api_server.py  (:5000)  — hardware: cameras, joint state, ee_pose, actions
    run_gr00t_server.py  (:5555)  — GR00T EEF model inference (ZMQ)
    THIS SCRIPT                   — obs(ee_pose) → policy → IK → joint commands

Differences from deploy_groot_realman.py (which stays unchanged as the
joint-space fallback):

  * Observations: the policy consumes ``left_eef_9d`` / ``right_eef_9d``
    (built from the robot API's 14D ``ee_pose``) + grippers, not joint angles.
  * Actions: the policy returns a 20D chunk per step
    (left_eef_9d(9), left_gripper(1), right_eef_9d(9), right_gripper(1)),
    already absolute (the server converts relative→absolute).
  * Execution: the right-arm 9D EEF target is converted to 6 joint angles via
    RealmanIK (QPIK, seeded from the observed joints), then packed into the
    same 14D ``/action`` payload as the joint client. Left arm stays pinned
    to the observed state. If IK fails (unreachable / non-converged), the
    command holds the observed state for that step and logs a warning.

Usage:
    python gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py \\
        --task "pick up bottle" \\
        --policy-host localhost --policy-port 5555 \\
        --robot-url http://localhost:5000 \\
        --open-loop-horizon 16 --hz 10 \\
        --auto-close-grip

Requires: robot_api_server exposing ``ee_pose`` in /observation, qpSWIFT in
the venv, and the XVLA IK stack on disk (see realman_ik.py).
Recommended first runs: --dry-run --debug, then --max-steps 30 --hz 10.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import signal
import time
from typing import Any

from gr00t.eval.real_robot.realman.deploy_groot_realman import (
    GR00TRealmanController,
    RunLogger,
    decode_image,
)
from gr00t.eval.real_robot.realman.realman_ik import RealmanIK
from loguru import logger
import numpy as np


# Per-key action dims for this embodiment; 20D total per chunk step.
EEF_ACTION_DIMS = {
    "left_eef_9d": 9,
    "left_gripper": 1,
    "right_eef_9d": 9,
    "right_gripper": 1,
}


def action_key_slices(action_keys: list[str]) -> dict[str, slice]:
    """Map each action key to its slice of the concatenated action vector.

    The concatenation order follows ``action_keys`` (the server's modality
    config order) — the same order parse_action_chunk uses to build the
    flat vector.
    """
    slices: dict[str, slice] = {}
    offset = 0
    for key in action_keys:
        if key not in EEF_ACTION_DIMS:
            raise ValueError(
                f"Unexpected action key {key!r} for the EEF embodiment "
                f"(expected keys: {sorted(EEF_ACTION_DIMS)}). "
                "Is the policy server running the EEF checkpoint?"
            )
        dim = EEF_ACTION_DIMS[key]
        slices[key] = slice(offset, offset + dim)
        offset += dim
    return slices


def parse_observation_eef(
    raw: dict[str, Any],
    modality_configs: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """Build the Policy API observation dict for the EEF embodiment.

    Same nesting/batching as the joint client's parse_observation_gr00t
    ((1, T, ...) with T=1), but state keys are built from the robot API's
    14D ``ee_pose`` instead of joint angles.
    """
    if "ee_pose" not in raw:
        raise ValueError(
            "/observation response has no 'ee_pose' field — the EEF client needs it. "
            "Update pickup-objects robot_api_server.py (Phase 1) or check the robot API."
        )

    from gr00t.eval.real_robot.realman import eef_utils

    state = np.array(raw["state"], dtype=np.float32)  # (14,) joints+grippers
    ee_pose = np.array(raw["ee_pose"], dtype=np.float64)  # (14,) poses
    state_by_key = {
        "left_eef_9d": eef_utils.ee_pose_14d_to_eef_9d(ee_pose, "left"),
        "left_gripper": state[6:7],
        "right_eef_9d": eef_utils.ee_pose_14d_to_eef_9d(ee_pose, "right"),
        "right_gripper": state[13:14],
    }

    obs: dict[str, Any] = {"video": {}, "state": {}, "language": {}}

    for key in modality_configs["video"].modality_keys:
        img = decode_image(raw["images"][key])  # (H, W, 3) uint8
        obs["video"][key] = img[np.newaxis, ...][np.newaxis, ...]  # (1, 1, H, W, C)

    for key in modality_configs["state"].modality_keys:
        arr = np.asarray(state_by_key[key], dtype=np.float32)
        obs["state"][key] = arr[np.newaxis, ...][np.newaxis, ...]  # (1, 1, D)

    for key in modality_configs["language"].modality_keys:
        obs["language"][key] = [[task]]

    return obs


def eef_action_to_joint_command(
    action_step: np.ndarray,
    slices: dict[str, slice],
    state_14d: np.ndarray,
    ik: RealmanIK,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert one 20D EEF action step into the 14D joint command for /action.

    Left arm + left gripper are pinned to the observed state (right-arm-only
    deployment, same policy as the joint client). The right 9D EEF target is
    solved to 6 joints with QPIK seeded from the observed right joints.

    Returns:
        (execute_action_14d, info) where info contains ``ik_ok``,
        ``ik_residual_mm``, ``ik_ms`` and, on failure, ``error``.
        On IK failure the command is the observed state (hold position).
    """
    action_step = np.asarray(action_step, dtype=float)
    state_14d = np.asarray(state_14d, dtype=float)
    target_right = action_step[slices["right_eef_9d"]]
    right_gripper = float(action_step[slices["right_gripper"]][0])
    seed = state_14d[7:13]

    info: dict[str, Any] = {"ik_ok": False, "ik_residual_mm": None, "ik_ms": None}
    t0 = time.time()
    try:
        q_right = ik.eef_9d_to_joint_angles(target_right, seed)
    except ValueError as e:
        info["ik_ms"] = (time.time() - t0) * 1000.0
        info["error"] = str(e)
        return state_14d.copy(), info
    info["ik_ms"] = (time.time() - t0) * 1000.0

    T = ik.fk(q_right)
    info["ik_residual_mm"] = float(np.linalg.norm(T[:3, 3] - target_right[:3]) * 1000.0)
    info["ik_ok"] = True

    execute = np.empty(14, dtype=float)
    execute[0:7] = state_14d[0:7]  # left arm + left gripper pinned to observed
    execute[7:13] = q_right
    execute[13] = right_gripper
    return execute, info


def _fmt_eef(arr: np.ndarray, slices: dict[str, slice]) -> str:
    """Compact right-side view of a 20D EEF action step."""
    a = np.asarray(arr, dtype=np.float32)
    r = a[slices["right_eef_9d"]]
    g = a[slices["right_gripper"]][0]
    return f"R_xyz=[{', '.join(f'{x:+.4f}' for x in r[:3])}] R_grip={g:+.4f}"


class EEFRunLogger(RunLogger):
    """RunLogger with per-step IK telemetry appended to each steps.jsonl record."""

    def log_step(self, *, ik_info: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._pending_extra = {
            "ik_ok": ik_info.get("ik_ok") if ik_info else None,
            "ik_residual_mm": ik_info.get("ik_residual_mm") if ik_info else None,
            "ik_ms": round(ik_info["ik_ms"], 1) if ik_info and ik_info.get("ik_ms") else None,
        }
        # Reimplement the parent record write with the extra fields (the parent
        # writes the line immediately, so we cannot append afterwards).
        record: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "step": kwargs["step"],
            "chunk_idx": kwargs["chunk_idx"],
            "is_infer_step": kwargs["is_infer_step"],
            "state": kwargs["state_14d"].tolist(),
            "model_action": kwargs["model_action_14d"].tolist(),
            "execute_action": kwargs["execute_action_14d"].tolist(),
            "grip_locked": kwargs["grip_locked"],
            **self._pending_extra,
        }
        raw_obs = kwargs.get("raw_obs", {})
        if raw_obs.get("gripper_force") is not None:
            record["gripper_force"] = raw_obs["gripper_force"]
        if kwargs.get("inference_ms") is not None:
            record["inference_ms"] = round(kwargs["inference_ms"], 1)
        if kwargs.get("loop_ms") is not None:
            record["loop_ms"] = round(kwargs["loop_ms"], 1)

        self._steps_file.write(json.dumps(record) + "\n")
        self._steps_file.flush()


class GR00TRealmanEEFController(GR00TRealmanController):
    """Closed-loop EEF client: ee_pose obs → GR00T EEF policy → IK → /action."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        expected = set(EEF_ACTION_DIMS)
        if set(self.action_keys) != expected:
            raise RuntimeError(
                f"Policy server action keys {self.action_keys} do not match the EEF "
                f"embodiment {sorted(expected)} — is the server running the EEF checkpoint?"
            )
        if set(self.state_keys) != expected:
            raise RuntimeError(
                f"Policy server state keys {self.state_keys} do not match the EEF "
                f"embodiment {sorted(expected)}."
            )
        self.action_slices = action_key_slices(self.action_keys)

        logger.info("Initialising RealMan IK (right arm, QPIK)...")
        self.ik = RealmanIK("right")
        logger.info("IK ready (frame calibration pinned by test_realman_ik.py).")

        # Swap the run logger for the IK-aware variant, preserving the run dir
        # the parent already created.
        if self.logger is not None:
            self.logger.__class__ = EEFRunLogger

    def _build_policy_obs(self, raw: dict[str, Any]) -> dict[str, Any]:
        return parse_observation_eef(raw, self.modality_configs, self.task)

    def run(self, max_steps: int = 500):
        dt = 1.0 / self.hz

        logger.info("=" * 60)
        logger.info("GR00T EEF CLOSED-LOOP DEPLOYMENT — RIGHT ARM ONLY")
        logger.info("=" * 60)
        logger.info(f"Task              : {self.task}")
        logger.info(f"Rate              : {self.hz} Hz")
        logger.info(f"Model horizon     : {self.action_horizon}")
        logger.info(f"Execute per infer : {self.open_loop_horizon}")
        logger.info(f"Max steps         : {max_steps}")
        logger.info(f"Robot API         : {self.robot_url}")
        logger.info(f"Dry run           : {self.dry_run}")
        if self.auto_close_grip:
            logger.info(
                f"Grip ratchet      : ON  "
                f"(threshold={self.grip_close_threshold}, lock={self.grip_lock_value})"
            )
        logger.info("Press Ctrl+C to stop")

        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

        step = 0
        pred_chunk: list[np.ndarray] | None = None
        actions_from_chunk_completed = 0
        grip_locked = False
        ik_failures = 0

        try:
            while step < max_steps:
                loop_start = time.time()

                raw = self._fetch_obs()
                state_14d = np.array(raw["state"], dtype=np.float32)

                is_infer_step = (
                    pred_chunk is None or actions_from_chunk_completed >= self.open_loop_horizon
                )
                inference_ms: float | None = None
                if is_infer_step:
                    infer_start = time.time()
                    pred_chunk = self._infer_action_chunk(raw)  # 20D steps
                    inference_ms = (time.time() - infer_start) * 1000.0
                    actions_from_chunk_completed = 0
                    if self.debug:
                        lines = [f"INFER step={step} | model returned {len(pred_chunk)} actions"]
                        for idx, a in enumerate(pred_chunk):
                            marker = ">" if idx < self.open_loop_horizon else " "
                            lines.append(
                                f"  {marker} chunk[{idx}]: {_fmt_eef(a, self.action_slices)}"
                            )
                        logger.debug("\n".join(lines))

                assert pred_chunk is not None
                model_action_20d = pred_chunk[actions_from_chunk_completed]
                execute_action_14d, ik_info = eef_action_to_joint_command(
                    model_action_20d, self.action_slices, state_14d, self.ik
                )

                if not ik_info["ik_ok"]:
                    ik_failures += 1
                    logger.warning(
                        f"IK failed at step={step} — holding position. {ik_info.get('error')}"
                    )

                if self.auto_close_grip:
                    right_grip = execute_action_14d[13]
                    if not grip_locked and right_grip < self.grip_close_threshold:
                        grip_locked = True
                        logger.info(f"GRIP LOCK engaged at step={step} (grip={right_grip:.4f})")
                    if grip_locked:
                        execute_action_14d = execute_action_14d.copy()
                        execute_action_14d[13] = min(right_grip, self.grip_lock_value)

                if self.debug:
                    residual = ik_info.get("ik_residual_mm")
                    logger.debug(
                        f"STEP {step} chunk_idx={actions_from_chunk_completed}\n"
                        f"  model EEF      : {_fmt_eef(model_action_20d, self.action_slices)}\n"
                        f"  IK             : ok={ik_info['ik_ok']} "
                        f"residual={residual if residual is None else f'{residual:.2f}mm'} "
                        f"({ik_info['ik_ms']:.1f} ms)\n"
                        f"  execute joints : R=[{', '.join(f'{x:+.4f}' for x in execute_action_14d[7:13])}] "
                        f"grip={execute_action_14d[13]:+.4f}"
                    )

                robot_response = self._execute_action(execute_action_14d)
                if self.debug and not self.dry_run and robot_response is not None:
                    logger.debug(f"EXECUTED step={step} | status={robot_response.get('status')}")

                if self.logger is not None:
                    self.logger.log_step(
                        step=step,
                        chunk_idx=actions_from_chunk_completed,
                        is_infer_step=is_infer_step,
                        state_14d=state_14d,
                        model_action_14d=model_action_20d,  # 20D — field name kept for tooling
                        execute_action_14d=execute_action_14d,
                        grip_locked=grip_locked,
                        raw_obs=raw,
                        inference_ms=inference_ms,
                        loop_ms=(time.time() - loop_start) * 1000.0,
                        ik_info=ik_info,
                    )

                actions_from_chunk_completed += 1
                step += 1

                if step % 30 == 0 and not self.debug:
                    logger.info(
                        f"Step {step:4d} | chunk_idx={actions_from_chunk_completed - 1} | "
                        f"{_fmt_eef(model_action_20d, self.action_slices)} | "
                        f"ik_failures={ik_failures}"
                    )

                elapsed = time.time() - loop_start
                sleep = dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)
                elif elapsed > dt * 1.5:
                    logger.warning(
                        f"Step {step} took {elapsed * 1000:.0f}ms (target {dt * 1000:.0f}ms)"
                    )

        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            self._stop()
            if self.logger is not None:
                self.logger.finalize(step)
            logger.info(f"Completed {step} steps ({ik_failures} IK failures).")


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop GR00T EEF deployment for RealMan robot (right arm only)"
    )
    parser.add_argument(
        "--task",
        default="pick up bottle",
        help='Language instruction (must match training, default: "pick up bottle")',
    )
    parser.add_argument(
        "--robot-url",
        default="http://localhost:5000",
        help="Base URL of robot_api_server (default: http://localhost:5000)",
    )
    parser.add_argument(
        "--policy-host",
        default="localhost",
        help="Host running run_gr00t_server.py with the EEF checkpoint (default: localhost)",
    )
    parser.add_argument(
        "--policy-port",
        type=int,
        default=5555,
        help="ZMQ port of the GR00T policy server (default: 5555)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Action execution frequency in Hz (default: 10 — gripper state publishes at 10 Hz)",
    )
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=None,
        help=(
            "Steps to execute from each action chunk before re-inferring "
            "(default: 1 = true closed-loop; use 16 at current Orin inference latency)"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max total steps before auto-stop (default: 500)",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=15000,
        help="ZMQ request timeout in ms (default: 15000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Full loop incl. inference and IK (residuals logged) but no robot motion",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Timestamped debug logs: model EEF chunk, IK result, executed joints per step",
    )
    parser.add_argument(
        "--auto-close-grip",
        action="store_true",
        help=(
            "Gripper ratchet: once the commanded right-gripper value drops below "
            "--grip-close-threshold, lock it to --grip-lock-value and never reopen."
        ),
    )
    parser.add_argument(
        "--grip-close-threshold",
        type=float,
        default=0.80,
        help="Right-gripper value below which the ratchet engages (default: 0.80)",
    )
    parser.add_argument(
        "--grip-lock-value",
        type=float,
        default=0.35,
        help="Right-gripper value to lock to once the ratchet engages (default: 0.35)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./runs",
        help="Directory for per-run logs (steps.jsonl incl. IK telemetry, meta.json, videos/)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable all logging (steps.jsonl, meta.json, video recording).",
    )
    parser.add_argument(
        "--no-record-video",
        action="store_true",
        help="Log steps.jsonl/meta.json but skip per-camera video recording.",
    )

    args = parser.parse_args()

    controller = GR00TRealmanEEFController(
        robot_url=args.robot_url,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        task=args.task,
        open_loop_horizon=args.open_loop_horizon,
        hz=args.hz,
        timeout_ms=args.timeout_ms,
        dry_run=args.dry_run,
        debug=args.debug,
        auto_close_grip=args.auto_close_grip,
        grip_close_threshold=args.grip_close_threshold,
        grip_lock_value=args.grip_lock_value,
        log_dir=None if args.no_log else args.log_dir,
        record_video=not args.no_record_video,
    )
    controller.run(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
