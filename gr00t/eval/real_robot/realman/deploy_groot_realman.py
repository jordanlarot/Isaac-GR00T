#!/usr/bin/env python3
"""
Closed-loop GR00T policy deployment for the RealMan dual-arm robot (R2D3).

Architecture:
    robot_api_server.py  (:5000)  — hardware: cameras, joint state, execute actions
    run_gr00t_server.py  (:5555)  — GR00T model inference (ZMQ, typically in Docker)
    THIS SCRIPT               — fetches obs → calls GR00T → sends right-arm actions

The GR00T policy server handles normalization/denormalization and relative→absolute
action conversion via its processor (same path as open_loop_eval.py). This client
must feed observations in the exact Policy API format expected by the checkpoint.

Usage:
    python gr00t/eval/real_robot/realman/deploy_groot_realman.py \\
        --task "pick up bottle" \\
        --policy-host 192.168.104.105 \\
        --policy-port 5555 \\
        --robot-url http://localhost:5000

Only the RIGHT arm and right gripper execute actions.
Left arm slots in the 14D payload are pinned to the current observed state.
"""

from __future__ import annotations

import argparse
import base64
import signal
import time
from typing import Any

import cv2
import numpy as np
import requests

# Camera names must match robot_api_server / config_loader / dataset modality.json
CAMERA_KEYS = ["top_camera", "left_wrist", "right_wrist"]


def decode_image(b64_str: str) -> np.ndarray:
    """Decode base64 JPEG (RGB) from robot_api_server → uint8 HWC numpy array."""
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def parse_observation_gr00t(
    raw: dict[str, Any],
    modality_configs: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """Build Policy API observation dict (matches open_loop_eval.parse_observation_gr00t).

    Converts flat robot state + camera images into nested video/state/language dicts
    with batch dimension B=1. Video/state arrays have shape (1, T, ...) where T comes
    from the modality config delta_indices (typically T=1 for this embodiment).
    """
    state = np.array(raw["state"], dtype=np.float32)  # (14,)
    state_by_key = {
        "left_arm": state[0:6],
        "left_gripper": state[6:7],
        "right_arm": state[7:13],
        "right_gripper": state[13:14],
    }

    obs: dict[str, Any] = {"video": {}, "state": {}, "language": {}}

    for key in modality_configs["video"].modality_keys:
        img = decode_image(raw["images"][key])  # (H, W, 3) uint8
        # (T, H, W, C) for T=1, then add batch dim → (1, T, H, W, C)
        obs["video"][key] = img[np.newaxis, ...][np.newaxis, ...]

    for key in modality_configs["state"].modality_keys:
        arr = state_by_key[key].astype(np.float32)  # (D,) or (1,)
        # (T, D) for T=1, then add batch dim → (1, T, D)
        obs["state"][key] = arr[np.newaxis, ...][np.newaxis, ...]

    for key in modality_configs["language"].modality_keys:
        obs["language"][key] = [[task]]

    return obs


def parse_action_chunk(action: dict[str, np.ndarray], action_keys: list[str]) -> list[np.ndarray]:
    """Convert policy action dict to list of concatenated 14D vectors (matches open_loop_eval).

    Each returned vector has shape (14,) in modality key order:
        left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)

    Actions are already unnormalized and converted to absolute physical units by
    Gr00tPolicy.decode_action on the server side.
    """
    horizon = action[action_keys[0]].shape[1]
    parsed: list[np.ndarray] = []
    for t in range(horizon):
        step = np.concatenate(
            [np.atleast_1d(action[key][0, t]) for key in action_keys],
            axis=0,
        )
        parsed.append(step.astype(np.float32))
    return parsed


def pin_left_arm_to_state(action_14d: np.ndarray, state_14d: np.ndarray) -> np.ndarray:
    """Keep left arm/gripper at observed state; only right side uses model output."""
    out = action_14d.copy()
    out[0:7] = state_14d[0:7]
    return out


class GR00TRealmanController:
    """Closed-loop client bridging robot_api_server and the GR00T ZMQ policy server."""

    def __init__(
        self,
        robot_url: str,
        policy_host: str,
        policy_port: int,
        task: str,
        open_loop_horizon: int | None = None,
        hz: float = 15.0,
        timeout_ms: int = 15000,
        dry_run: bool = False,
        debug: bool = False,
    ):
        from gr00t.policy.server_client import PolicyClient

        self.robot_url = robot_url.rstrip("/")
        self.task = task
        self.hz = hz
        self.dry_run = dry_run
        self.debug = debug

        print(f"Connecting to GR00T policy server at {policy_host}:{policy_port} ...")
        self.policy = PolicyClient(
            host=policy_host,
            port=policy_port,
            timeout_ms=timeout_ms,
            strict=False,
        )
        if not self.policy.ping():
            raise RuntimeError(
                f"Cannot reach GR00T server at {policy_host}:{policy_port}. "
                "Make sure run_gr00t_server.py is running."
            )
        print("GR00T policy server connected.")

        self.modality_configs = self.policy.get_modality_config()
        self.video_keys = self.modality_configs["video"].modality_keys
        self.state_keys = self.modality_configs["state"].modality_keys
        self.action_keys = self.modality_configs["action"].modality_keys
        self.language_keys = self.modality_configs["language"].modality_keys
        self.action_horizon = len(self.modality_configs["action"].delta_indices)

        if open_loop_horizon is None:
            open_loop_horizon = min(8, self.action_horizon)
        if open_loop_horizon <= 0 or open_loop_horizon > self.action_horizon:
            raise ValueError(
                f"--open-loop-horizon must be in [1, {self.action_horizon}]; "
                f"got {open_loop_horizon}"
            )
        self.open_loop_horizon = open_loop_horizon

        print(f"  video keys      : {self.video_keys}")
        print(f"  state keys      : {self.state_keys}")
        print(f"  action keys     : {self.action_keys}")
        print(f"  language keys   : {self.language_keys}")
        print(f"  action horizon  : {self.action_horizon}")
        print(f"  execute steps   : {self.open_loop_horizon} per inference call")

        missing_cams = set(self.video_keys) - set(CAMERA_KEYS)
        if missing_cams:
            print(f"  WARNING: checkpoint expects cameras {missing_cams} not in CAMERA_KEYS")

    def _fetch_obs(self) -> dict[str, Any]:
        resp = requests.get(f"{self.robot_url}/observation", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

    def _build_policy_obs(self, raw: dict[str, Any]) -> dict[str, Any]:
        return parse_observation_gr00t(raw, self.modality_configs, self.task)

    def _execute_action(self, action_14d: np.ndarray):
        if self.dry_run:
            return
        resp = requests.post(
            f"{self.robot_url}/action",
            json={"action": action_14d.tolist()},
            timeout=2.0,
        )
        resp.raise_for_status()

    def _stop(self):
        if self.dry_run:
            return
        try:
            requests.post(f"{self.robot_url}/stop", timeout=2.0)
            print("Emergency stop sent.")
        except Exception as e:
            print(f"Warning: could not send stop: {e}")

    def _infer_action_chunk(self, raw: dict[str, Any]) -> tuple[list[np.ndarray], np.ndarray]:
        """Run one policy inference and return (chunk steps, reference state)."""
        obs = self._build_policy_obs(raw)
        action_dict, _ = self.policy.get_action(obs)
        chunk = parse_action_chunk(action_dict, self.action_keys)
        state_14d = np.array(raw["state"], dtype=np.float32)
        return chunk, state_14d

    def run(self, max_steps: int = 500):
        dt = 1.0 / self.hz

        print(f"\n{'='*60}")
        print("GR00T CLOSED-LOOP DEPLOYMENT — RIGHT ARM ONLY")
        print(f"{'='*60}")
        print(f"Task              : {self.task}")
        print(f"Rate              : {self.hz} Hz")
        print(f"Model horizon     : {self.action_horizon}")
        print(f"Execute per infer : {self.open_loop_horizon}")
        print(f"Max steps         : {max_steps}")
        print(f"Robot API         : {self.robot_url}")
        print(f"Dry run           : {self.dry_run}")
        print(f"\nPress Ctrl+C to stop\n{'='*60}\n")

        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

        step = 0
        pred_chunk: list[np.ndarray] | None = None
        ref_state_14d: np.ndarray | None = None
        actions_from_chunk_completed = 0

        try:
            while step < max_steps:
                loop_start = time.time()

                # Re-infer when starting or after executing open_loop_horizon steps
                # (same receding-horizon pattern as examples/DROID/main_gr00t.py)
                if (
                    pred_chunk is None
                    or actions_from_chunk_completed >= self.open_loop_horizon
                ):
                    raw = self._fetch_obs()
                    pred_chunk, ref_state_14d = self._infer_action_chunk(raw)
                    actions_from_chunk_completed = 0

                    if self.debug:
                        s = np.array(raw["state"], dtype=np.float32)
                        a0 = pin_left_arm_to_state(pred_chunk[0], s)
                        print(
                            f"[infer step {step}] state R={s[7:10].round(3)} "
                            f"grip={s[13]:.3f} | action0 R={a0[7:10].round(3)} "
                            f"grip={a0[13]:.3f}"
                        )

                assert pred_chunk is not None and ref_state_14d is not None
                action_14d = pin_left_arm_to_state(
                    pred_chunk[actions_from_chunk_completed],
                    ref_state_14d,
                )
                self._execute_action(action_14d)
                actions_from_chunk_completed += 1
                step += 1

                if step % 30 == 0 and not self.debug:
                    print(
                        f"Step {step:4d} | chunk_idx={actions_from_chunk_completed - 1} | "
                        f"R_cmd={action_14d[7:10].round(3)} grip={action_14d[13]:.3f}"
                    )

                elapsed = time.time() - loop_start
                sleep = dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)
                elif elapsed > dt * 1.5:
                    print(
                        f"Warning: step {step} took {elapsed*1000:.0f}ms "
                        f"(target {dt*1000:.0f}ms)"
                    )

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self._stop()
            print(f"Completed {step} steps.")


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop GR00T deployment for RealMan robot (right arm only)"
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
        help="Host running run_gr00t_server.py (default: localhost)",
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
        default=15.0,
        help="Action execution frequency in Hz — match training collection rate (default: 15)",
    )
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=None,
        help=(
            "Steps to execute from each action chunk before re-inferring "
            f"(default: min(8, model horizon))"
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
        help="Run inference loop but do not send actions to the robot",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print state vs predicted action on every inference call",
    )

    args = parser.parse_args()

    controller = GR00TRealmanController(
        robot_url=args.robot_url,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        task=args.task,
        open_loop_horizon=args.open_loop_horizon,
        hz=args.hz,
        timeout_ms=args.timeout_ms,
        dry_run=args.dry_run,
        debug=args.debug,
    )
    controller.run(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
