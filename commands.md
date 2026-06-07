in docker t1:
```
docker run -it --rm --runtime nvidia --gpus all   --ipc=host   --ulimit memlock=-1   --ulimit stack=67108864   --network host   -v "$(pwd)":/workspace/repo   -v /home/r2d3/checkpoints:/home/r2d3/checkpoints:ro   -v /home/r2d3/datasets:/home/r2d3/datasets:ro   -v "${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface   -e HF_TOKEN="${HF_TOKEN}"   -w /workspace/repo   gr00t-orin   bash


```
python gr00t/eval/run_gr00t_server.py   --model-path /home/r2d3/checkpoints/gr00t-pick-bottle-realman   --embodiment-tag NEW_EMBODIMENT   --device cuda:0   --host 0.0.0.0   --port 5555
```

t2 rogent
```
ros2 launch rogent rogent.launch.py 
```

t3 robot api (observations) 
```
cd /home/r2d3/pickup-objects
export PYTHONPATH=~/pickup-objects/src:$PYTHONPATH
python scripts/robot_api_server.py
```

t4 — dry run + debug (slow; inference only)
```
source .venv/bin/activate
source scripts/activate_orin.sh

python gr00t/eval/real_robot/realman/deploy_groot_realman.py   --task "pick up bottle"   --policy-host 192.168.104.105   --policy-port 5555   --robot-url http://localhost:5000 --dry-run --debug
```

t4 — live robot (recommended settings)
```
source .venv/bin/activate && source scripts/activate_orin.sh

python gr00t/eval/real_robot/realman/deploy_groot_realman.py \
  --task "pick up bottle" \
  --policy-host 192.168.104.105 \
  --policy-port 5555 \
  --robot-url http://localhost:5000 \
  --open-loop-horizon 16 \
  --hz 15 \
  --auto-close-grip \
  --grip-close-threshold 0.95
```

Logs are written automatically to `./runs/run_YYYYMMDD_HHMMSS/` (override with `--log-dir`).

**Flags:**
- `--open-loop-horizon 16` — use the full trained horizon (model was trained at 16). Produces the most arm movement; expect a brief jerk every ~16 steps due to inference latency (~700 ms).
- `--auto-close-grip` — gripper ratchet: once the right gripper command drops below 0.80, it locks to 0.35 and never reopens. Eliminates the open/close jitter during bottle approach. Safe to omit if you want the raw model output.
- `--grip-close-threshold 0.80` — (default) tune higher (e.g. 0.90) to engage the ratchet earlier, lower (e.g. 0.70) to engage later.
- `--grip-lock-value 0.35` — (default) how closed the gripper locks to once engaged.
- `--debug` — timestamped per-step logs: model chunk on inference, model vs execute command, robot_api POST result, and a `GRIP LOCK engaged` line when the ratchet fires.
- `--hz 10` — use if gripper state is only published at 10 Hz.
- `--dry-run` — runs inference loop without sending any commands to the robot.
- `--log-dir ./runs` — (default) directory for per-run logs.
- `--no-log` — disable all logging (steps.jsonl, meta.json, video).
- `--no-record-video` — log steps.jsonl/meta.json but skip video recording.

**Run log layout** (`./runs/run_YYYYMMDD_HHMMSS/`):
- `meta.json` — all run parameters (task, hz, horizon, grip settings)
- `steps.jsonl` — one JSON line per step: `timestamp`, `step`, `chunk_idx`, `is_infer_step`, `state` (14D), `model_action` (14D), `execute_action` (14D), `grip_locked`, `gripper_force`, `inference_ms`, `loop_ms`
- `summary.json` — total steps and wall-clock duration
- `videos/top_camera.mp4`, `left_wrist.mp4`, `right_wrist.mp4` — full-run camera recordings at `--hz`

**note**: put the head camera to at least 300 
```
ros2 topic pub --once /servo_control/move servo_interfaces/msg/ServoMove "{servo_id: 5, angle: 350}"
```

sometimes its 300 or 350

the head camera should be able to see the grippers in view when in resting position

Set starting pose 
```
cd ~/pickup-objects
uv run python scripts/reset_to_episode_start.py \
  --file datasets/pick-bottle/data/chunk_000/episode_000000.parquet \
  --both-arms
```