# CLAUDE.md — Isaac GR00T N1.7

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills.
The repo contains the model, training pipeline, evaluation harness, and deployment tooling.

- **Language:** Python 3.10 (dGPU, Orin); Python 3.12 (Thor, DGX Spark — see deployment dir)
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** setuptools (see `pyproject.toml`)
- **CI:** internal GitLab CI (`.gitlab-ci.yml` + includes under `ci/`, not shipped to the public GitHub EA repo); public GitHub Actions (`.github/workflows/`)

## Quick-start commands

```bash
# Install (dev mode with all extras)
uv sync --all-extras

# Lint and format (uses ruff via pre-commit)
pre-commit run --all-files

# Run CPU tests
python -m pytest tests/ -m "not gpu" -v --timeout=300

# Run GPU tests
python -m pytest tests/ -m gpu -v --timeout=300

# Build package
uv build

# Validate lockfile
uv lock --locked
```

## Code style

- Formatter: `ruff format` (double quotes, spaces, line-length 100)
- Linter: `ruff check` with rules E, F, I (ignores E501)
- Config lives in `pyproject.toml` under `[tool.ruff]`
- Run `pre-commit run --all-files` before committing

## Directory layout

```
gr00t/                    # Main package
  configs/                #   Training, data, and model configs
  data/                   #   Data loading, embodiment tags, dataset processing
  eval/                   #   Evaluation and deployment clients
    run_gr00t_server.py   #     ZMQ policy inference server
    open_loop_eval.py     #     Offline replay against a dataset
    rollout_policy.py     #     Sim rollouts (PyTorch or TensorRT)
    sim/                  #     LIBERO, SimplerEnv, RoboCasa harnesses
    real_robot/           #     Hardware deployment clients
      realman/            #       RealMan dual-arm (R2D3) closed-loop client
      SO100/              #       SO-100 eval example
  experiment/             #   Training pipeline (launch_finetune.py, trainer.py)
  model/                  #   Model architecture (N1.7, base, modules)
  policy/                 #   Policy inference (Gr00tPolicy, server/client)
examples/                 # Per-embodiment example configs and READMEs
scripts/                  # Deployment, conversion, and utility scripts
  deployment/             #   Platform install scripts (dgpu, orin, thor, spark)
  activate_orin.sh        #   Orin runtime library paths (also spark, thor)
tests/                    # pytest suite (markers: gpu, not gpu)
getting_started/          # User-facing guides and notebooks
docker/                   # Container build (dgpu + edge profiles via build.sh)
commands.md               # Local R2D3 operator runbook (robot + policy terminals)
```

## Key entry points

- **Fine-tune:** `bash examples/finetune.sh --base-model-path <path> --dataset-path <path> --embodiment-tag <tag> --output-dir <dir>`
- **Inference server:** `python gr00t/eval/run_gr00t_server.py --model-path <path> --embodiment-tag <tag>`
- **Open-loop eval:** `python gr00t/eval/open_loop_eval.py` (dataset replay; see script `--help`)
- **Sim rollout:** `python gr00t/eval/rollout_policy.py` (supports `--trt-engine-path` on supported platforms)
- **RealMan live deploy:** `python gr00t/eval/real_robot/realman/deploy_groot_realman.py` (see below)
- **ONNX export:** `python scripts/deployment/export_onnx_n1d7.py`
- **TensorRT build:** `python scripts/deployment/build_trt_pipeline.py`
- **Benchmark:** `python scripts/deployment/benchmark_inference.py`

## Real robot deployment (RealMan R2D3)

Three-process architecture (see `commands.md` for local terminal commands):

1. **`run_gr00t_server.py` (:5555)** — GR00T ZMQ policy server (normalization, relative→absolute actions). Often runs in the Orin Docker image on a GPU host.
2. **`robot_api_server.py` (:5000)** — lives in the separate `pickup-objects` repo; cameras, joint state, action execution.
3. **`deploy_groot_realman.py`** — closed-loop client: fetch obs → policy `get_action` → POST right-arm command only (left arm pinned to observed state).

**Recommended live settings:**

- `--open-loop-horizon 16` — use the full trained horizon for maximum arm reach. Produces a brief jerk every ~16 steps due to inference latency (~700 ms); acceptable in practice.
- `--auto-close-grip` — gripper ratchet: once the right-gripper command drops below `--grip-close-threshold` (default 0.80), locks to `--grip-lock-value` (default 0.35) and never reopens. Eliminates open/close jitter during bottle approach. Tune threshold if the lock engages too early or late.
- `--hz 15` — match training collection rate; use `--hz 10` if gripper state is only published at 10 Hz.
- Omit `--debug` for production runs (adds verbose per-step logging; also prints `GRIP LOCK engaged` when the ratchet fires).

**Avoid on hardware:** `--open-loop-horizon 1` at current Orin inference speeds (~700 ms) — the arm will not move because `chunk[0]` is always a hold-position command at 1.5 Hz effective rate.

**`--debug`** prints timestamped logs: model chunk on inference, model output vs pinned execute command, and `robot_api` POST result each step.

**Task string** must match training (e.g. `"pick up bottle"`).

**Run logging** is on by default. Each run creates `./runs/run_YYYYMMDD_HHMMSS/` containing:
- `meta.json` — run parameters
- `steps.jsonl` — per-step record: observed state (14D), model prediction (14D), executed action (14D), grip lock status, gripper force, inference latency, loop timing
- `summary.json` — total steps and duration
- `videos/` — one MP4 per camera at `--hz`

Use `--log-dir <path>` to change the output root, `--no-record-video` to skip video (saves memory), or `--no-log` to disable entirely.

## Testing

- Test markers: `gpu` (requires GPU), default is CPU-safe
- Fixtures live in `tests/fixtures/` and `demo_data/`
- CI runs CPU and GPU tests in separate jobs with 300s timeout

## Deployment platforms

- **dGPU (H100, A100, RTX):** CUDA 12.8 — `scripts/deployment/dgpu/install_deps.sh`; container via `docker/Dockerfile` (`bash docker/build.sh`, supports x86_64 and aarch64)
- **Jetson Orin:** CUDA 12.6, Python 3.10 — `scripts/deployment/orin/install_deps.sh`; container `scripts/deployment/orin/Dockerfile` (`bash docker/build.sh --profile=orin`)
- **Jetson Thor:** CUDA 13.0, Python 3.12 — `scripts/deployment/thor/install_deps.sh`; container via `bash docker/build.sh --profile=thor`
- **DGX Spark:** CUDA 13.0, Python 3.12 — `scripts/deployment/spark/install_deps.sh`; container via `bash docker/build.sh --profile=spark`

Each Jetson/Spark platform ships an `activate_*.sh` helper (`scripts/activate_orin.sh`, `scripts/activate_spark.sh`, `scripts/activate_thor.sh`) that exports platform-specific library paths. For dGPU, `source .venv/bin/activate` is sufficient.

**Orin inference notes:** PyTorch eager is ~2.9 Hz on Orin (see `scripts/deployment/README.md` benchmarks). TensorRT on Orin is **DiT-only** (`--export-mode dit_only`); full backbone TRT is not supported on TRT 10.3. `run_gr00t_server.py` uses PyTorch `Gr00tPolicy` today — wiring TRT into the ZMQ server requires extra integration.

For full TRT build/verify steps and per-platform benchmarks, see `scripts/deployment/README.md`.
