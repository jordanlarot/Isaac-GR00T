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
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import requests
from loguru import logger

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


def _debug_timestamp() -> str:
    """Human-readable timestamp with millisecond precision."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _fmt_joints(arr: np.ndarray) -> str:
    """Format 14D vector: L_arm(6), L_grip, R_arm(6), R_grip."""
    a = np.asarray(arr, dtype=np.float32)
    return (
        f"L_arm=[{', '.join(f'{x:+.4f}' for x in a[0:6])}] "
        f"L_grip={a[6]:+.4f} "
        f"R_arm=[{', '.join(f'{x:+.4f}' for x in a[7:13])}] "
        f"R_grip={a[13]:+.4f}"
    )


def _fmt_right_arm(arr: np.ndarray) -> str:
    """Compact right-arm view (the only side we execute)."""
    a = np.asarray(arr, dtype=np.float32)
    return f"R_arm=[{', '.join(f'{x:+.4f}' for x in a[7:13])}] R_grip={a[13]:+.4f}"


def _print_debug_inference(
    *,
    step: int,
    chunk: list[np.ndarray],
    open_loop_horizon: int,
) -> None:
    lines = [f"INFER step={step} | model returned {len(chunk)} actions"]
    for idx, model_action in enumerate(chunk):
        marker = ">" if idx < open_loop_horizon else " "
        lines.append(f"  {marker} chunk[{idx}]: {_fmt_joints(model_action)}")
    lines.append(f"  executing chunk[0:{open_loop_horizon}] before next inference")
    logger.debug("\n" + "\n".join(lines))


def _print_debug_step(
    *,
    step: int,
    chunk_idx: int,
    state_14d: np.ndarray,
    model_action_14d: np.ndarray,
    execute_action_14d: np.ndarray,
) -> None:
    delta_14d = execute_action_14d - state_14d
    logger.debug(
        f"STEP {step} chunk_idx={chunk_idx}\n"
        f"  observed state : {_fmt_joints(state_14d)}\n"
        f"  model output   : {_fmt_joints(model_action_14d)}\n"
        f"  execute (sent) : {_fmt_joints(execute_action_14d)}\n"
        f"  right delta    : {_fmt_right_arm(delta_14d)}"
    )


def _print_debug_execute_result(
    *,
    step: int,
    execute_action_14d: np.ndarray,
    dry_run: bool,
    elapsed_ms: float,
    robot_response: dict[str, Any] | None = None,
) -> None:
    if dry_run:
        logger.debug(
            f"DRY-RUN step={step} | skipped POST "
            f"(would send {_fmt_right_arm(execute_action_14d)})"
        )
        return
    status = robot_response.get("status", "ok") if robot_response else "ok"
    logger.debug(
        f"EXECUTED step={step} | robot_api POST ok ({elapsed_ms:.1f} ms) "
        f"| status={status} | sent {_fmt_right_arm(execute_action_14d)}"
    )


class RunLogger:
    """Per-run structured logger: JSONL step log + per-camera MP4 videos.

    Video frames are captured by a background thread that polls the robot API
    at a fixed rate, completely independent of the control loop. This ensures
    smooth video even when inference steps stall the loop for ~700 ms.

    Directory layout::

        <log_dir>/run_YYYYMMDD_HHMMSS/
            meta.json        — run config parameters
            steps.jsonl      — one JSON line per control step
            summary.json     — written on finalize (total steps, duration)
            videos/
                top_camera.mp4
                left_wrist.mp4
                right_wrist.mp4
    """

    def __init__(
        self,
        log_dir: str,
        config: dict[str, Any],
        camera_keys: list[str],
        hz: float,
        record_video: bool = True,
        robot_url: str | None = None,
        debug: bool = False,
    ):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_dir, f"run_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        with open(os.path.join(self.run_dir, "meta.json"), "w") as f:
            json.dump({"run_id": ts, **config}, f, indent=2)

        # Configure loguru: plain stderr (same look as print) + timestamped file.
        log_level = "DEBUG" if debug else "INFO"
        logger.remove()
        logger.add(sys.stderr, format="<level>{message}</level>", level=log_level, colorize=True)
        log_path = os.path.join(self.run_dir, "run.log")
        self._log_handler_id = logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
            level="DEBUG",
            encoding="utf-8",
        )

        self._steps_file = open(os.path.join(self.run_dir, "steps.jsonl"), "w")
        self._camera_keys = camera_keys
        self._hz = hz
        self._record_video = record_video
        self._start_time = time.time()

        # Frames captured by background thread; lock guards concurrent appends.
        self._video_frames: dict[str, list[np.ndarray]] = {k: [] for k in camera_keys}
        self._frame_lock = threading.Lock()
        self._capture_thread: threading.Thread | None = None
        self._capturing = False

        logger.info(f"Logging run to: {self.run_dir}")
        if record_video and robot_url:
            self._start_capture_thread(robot_url)
        elif record_video:
            logger.warning("record_video=True but no robot_url — video disabled")

    def _start_capture_thread(self, robot_url: str) -> None:
        """Poll robot API at self._hz in a daemon thread for continuous camera frames."""
        robot_url = robot_url.rstrip("/")
        dt = 1.0 / self._hz
        logger.info(f"Video recording: ON  (cameras: {self._camera_keys}, {self._hz:.0f} Hz)")

        def _capture() -> None:
            while self._capturing:
                t0 = time.time()
                try:
                    resp = requests.get(f"{robot_url}/observation", timeout=3.0)
                    if resp.ok:
                        raw = resp.json()
                        frames = {}
                        for key in self._camera_keys:
                            if key in raw.get("images", {}):
                                frames[key] = decode_image(raw["images"][key])
                        if frames:
                            with self._frame_lock:
                                for key, frame in frames.items():
                                    self._video_frames[key].append(frame)
                except Exception:
                    pass
                elapsed = time.time() - t0
                remaining = dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        self._capturing = True
        self._capture_thread = threading.Thread(target=_capture, daemon=True, name="video-capture")
        self._capture_thread.start()

    def log_step(
        self,
        *,
        step: int,
        chunk_idx: int,
        is_infer_step: bool,
        state_14d: np.ndarray,
        model_action_14d: np.ndarray,
        execute_action_14d: np.ndarray,
        grip_locked: bool,
        raw_obs: dict[str, Any],
        inference_ms: float | None = None,
        loop_ms: float | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "chunk_idx": chunk_idx,
            "is_infer_step": is_infer_step,
            "state": state_14d.tolist(),
            "model_action": model_action_14d.tolist(),
            "execute_action": execute_action_14d.tolist(),
            "grip_locked": grip_locked,
        }
        if raw_obs.get("gripper_force") is not None:
            record["gripper_force"] = raw_obs["gripper_force"]
        if inference_ms is not None:
            record["inference_ms"] = round(inference_ms, 1)
        if loop_ms is not None:
            record["loop_ms"] = round(loop_ms, 1)

        self._steps_file.write(json.dumps(record) + "\n")
        self._steps_file.flush()

    def finalize(self, total_steps: int) -> None:
        # Stop capture thread before touching frame buffers.
        if self._capture_thread is not None:
            self._capturing = False
            self._capture_thread.join(timeout=5.0)

        elapsed = time.time() - self._start_time
        self._steps_file.close()

        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump({"total_steps": total_steps, "duration_s": round(elapsed, 2)}, f, indent=2)

        if self._record_video:
            video_dir = os.path.join(self.run_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            with self._frame_lock:
                frames_snapshot = {k: list(v) for k, v in self._video_frames.items()}
            for key, frames in frames_snapshot.items():
                if not frames:
                    continue
                h, w = frames[0].shape[:2]
                out_path = os.path.join(video_dir, f"{key}.mp4")
                writer = cv2.VideoWriter(
                    out_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self._hz,
                    (w, h),
                )
                for frame in frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
                logger.info(f"Saved {key}: {out_path}  ({len(frames)} frames)")

        logger.info(f"Run log saved: {self.run_dir}  ({total_steps} steps, {elapsed:.1f}s)")
        logger.remove(self._log_handler_id)


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
        auto_close_grip: bool = False,
        grip_close_threshold: float = 0.80,
        grip_lock_value: float = 0.35,
        log_dir: str | None = None,
        record_video: bool = True,
    ):
        from gr00t.policy.server_client import PolicyClient

        self.robot_url = robot_url.rstrip("/")
        self.task = task
        self.hz = hz
        self.dry_run = dry_run
        self.debug = debug
        self.auto_close_grip = auto_close_grip
        self.grip_close_threshold = grip_close_threshold
        self.grip_lock_value = grip_lock_value
        self._log_dir = log_dir
        self._record_video = record_video

        logger.info(f"Connecting to GR00T policy server at {policy_host}:{policy_port} ...")
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
        logger.info("GR00T policy server connected.")

        self.modality_configs = self.policy.get_modality_config()
        self.video_keys = self.modality_configs["video"].modality_keys
        self.state_keys = self.modality_configs["state"].modality_keys
        self.action_keys = self.modality_configs["action"].modality_keys
        self.language_keys = self.modality_configs["language"].modality_keys
        self.action_horizon = len(self.modality_configs["action"].delta_indices)

        # Default 1: re-infer every step so each command uses fresh obs/state.
        # Longer horizons replay a fixed absolute trajectory and drift if the loop
        # is slower than --hz or the arm does not track every setpoint in time.
        if open_loop_horizon is None:
            open_loop_horizon = 1
        if open_loop_horizon <= 0 or open_loop_horizon > self.action_horizon:
            raise ValueError(
                f"--open-loop-horizon must be in [1, {self.action_horizon}]; "
                f"got {open_loop_horizon}"
            )
        self.open_loop_horizon = open_loop_horizon

        logger.info(f"  video keys      : {self.video_keys}")
        logger.info(f"  state keys      : {self.state_keys}")
        logger.info(f"  action keys     : {self.action_keys}")
        logger.info(f"  language keys   : {self.language_keys}")
        logger.info(f"  action horizon  : {self.action_horizon}")
        logger.info(f"  execute steps   : {self.open_loop_horizon} per inference call")

        missing_cams = set(self.video_keys) - set(CAMERA_KEYS)
        if missing_cams:
            logger.warning(f"checkpoint expects cameras {missing_cams} not in CAMERA_KEYS")

        self.logger: RunLogger | None = None
        if self._log_dir is not None:
            self.logger = RunLogger(
                log_dir=self._log_dir,
                config={
                    "task": task,
                    "robot_url": robot_url,
                    "policy_host": policy_host,
                    "policy_port": policy_port,
                    "hz": hz,
                    "open_loop_horizon": self.open_loop_horizon,
                    "action_horizon": self.action_horizon,
                    "dry_run": dry_run,
                    "auto_close_grip": auto_close_grip,
                    "grip_close_threshold": grip_close_threshold,
                    "grip_lock_value": grip_lock_value,
                },
                camera_keys=self.video_keys,
                hz=hz,
                record_video=self._record_video,
                robot_url=robot_url,
                debug=debug,
            )

    def _fetch_obs(self) -> dict[str, Any]:
        resp = requests.get(f"{self.robot_url}/observation", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

    def _build_policy_obs(self, raw: dict[str, Any]) -> dict[str, Any]:
        return parse_observation_gr00t(raw, self.modality_configs, self.task)

    def _execute_action(self, action_14d: np.ndarray) -> dict[str, Any] | None:
        if self.dry_run:
            return None
        resp = requests.post(
            f"{self.robot_url}/action",
            json={"action": action_14d.tolist()},
            timeout=2.0,
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {"status": "ok", "body": resp.text}

    def _stop(self):
        if self.dry_run:
            return
        try:
            requests.post(f"{self.robot_url}/stop", timeout=2.0)
            logger.info("Emergency stop sent.")
        except Exception as e:
            logger.warning(f"Could not send stop: {e}")

    def _infer_action_chunk(self, raw: dict[str, Any]) -> list[np.ndarray]:
        """Run one policy inference and return the absolute action chunk (length H)."""
        obs = self._build_policy_obs(raw)
        action_dict, _ = self.policy.get_action(obs)
        return parse_action_chunk(action_dict, self.action_keys)

    def run(self, max_steps: int = 500):
        dt = 1.0 / self.hz

        logger.info("=" * 60)
        logger.info("GR00T CLOSED-LOOP DEPLOYMENT — RIGHT ARM ONLY")
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

        try:
            while step < max_steps:
                loop_start = time.time()

                # Fresh observation every control step (matches pickup-objects DeployLoop).
                raw = self._fetch_obs()
                state_14d = np.array(raw["state"], dtype=np.float32)

                # Re-infer when starting or after executing open_loop_horizon steps.
                # Use open_loop_horizon=1 for true closed-loop on hardware.
                is_infer_step = pred_chunk is None or actions_from_chunk_completed >= self.open_loop_horizon
                inference_ms: float | None = None
                if is_infer_step:
                    infer_start = time.time()
                    pred_chunk = self._infer_action_chunk(raw)
                    inference_ms = (time.time() - infer_start) * 1000.0
                    actions_from_chunk_completed = 0
                    if self.debug:
                        _print_debug_inference(
                            step=step,
                            chunk=pred_chunk,
                            open_loop_horizon=self.open_loop_horizon,
                        )

                assert pred_chunk is not None
                model_action_14d = pred_chunk[actions_from_chunk_completed]
                execute_action_14d = pin_left_arm_to_state(model_action_14d, state_14d)

                if self.auto_close_grip:
                    right_grip = execute_action_14d[13]
                    if not grip_locked and right_grip < self.grip_close_threshold:
                        grip_locked = True
                        logger.info(f"GRIP LOCK engaged at step={step} (grip={right_grip:.4f})")
                    if grip_locked:
                        execute_action_14d = execute_action_14d.copy()
                        execute_action_14d[13] = min(right_grip, self.grip_lock_value)

                if self.debug:
                    _print_debug_step(
                        step=step,
                        chunk_idx=actions_from_chunk_completed,
                        state_14d=state_14d,
                        model_action_14d=model_action_14d,
                        execute_action_14d=execute_action_14d,
                    )

                exec_start = time.time()
                robot_response = self._execute_action(execute_action_14d)
                if self.debug:
                    _print_debug_execute_result(
                        step=step,
                        execute_action_14d=execute_action_14d,
                        dry_run=self.dry_run,
                        elapsed_ms=(time.time() - exec_start) * 1000.0,
                        robot_response=robot_response,
                    )

                if self.logger is not None:
                    self.logger.log_step(
                        step=step,
                        chunk_idx=actions_from_chunk_completed,
                        is_infer_step=is_infer_step,
                        state_14d=state_14d,
                        model_action_14d=model_action_14d,
                        execute_action_14d=execute_action_14d,
                        grip_locked=grip_locked,
                        raw_obs=raw,
                        inference_ms=inference_ms,
                        loop_ms=(time.time() - loop_start) * 1000.0,
                    )

                actions_from_chunk_completed += 1
                step += 1

                if step % 30 == 0 and not self.debug:
                    logger.info(
                        f"Step {step:4d} | chunk_idx={actions_from_chunk_completed - 1} | "
                        f"R_cmd={execute_action_14d[7:10].round(3)} grip={execute_action_14d[13]:.3f}"
                    )

                elapsed = time.time() - loop_start
                sleep = dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)
                elif elapsed > dt * 1.5:
                    logger.warning(
                        f"Step {step} took {elapsed*1000:.0f}ms "
                        f"(target {dt*1000:.0f}ms)"
                    )

        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            self._stop()
            if self.logger is not None:
                self.logger.finalize(step)
            logger.info(f"Completed {step} steps.")


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
            "Steps to execute from each action chunk before re-inferring. "
            "Use 1 on the real robot (default) so every command uses a fresh observation; "
            "larger values (e.g. 8) only if the loop reliably meets --hz and the arm tracks."
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
        help=(
            "Timestamped debug logs: model chunk on inference, model output vs "
            "pinned execute command, and robot_api POST result each step"
        ),
    )
    parser.add_argument(
        "--auto-close-grip",
        action="store_true",
        help=(
            "Gripper ratchet: once the commanded right-gripper value drops below "
            "--grip-close-threshold, lock it to --grip-lock-value and never reopen. "
            "Eliminates open/close jitter during bottle approach."
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
        help=(
            "Directory for per-run logs (steps.jsonl, meta.json, videos/). "
            "Each run gets its own timestamped subdirectory. Pass --no-log to disable."
        ),
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
        auto_close_grip=args.auto_close_grip,
        grip_close_threshold=args.grip_close_threshold,
        grip_lock_value=args.grip_lock_value,
        log_dir=None if args.no_log else args.log_dir,
        record_video=not args.no_record_video,
    )
    controller.run(max_steps=args.max_steps)


if __name__ == "__main__":
    main()
