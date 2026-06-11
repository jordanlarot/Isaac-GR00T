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
- **RealMan EEF deploy:** `python gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py` (see below; checkpoint `gr00t-pick-place-bottle-eef-10k`)
- **RealMan joint deploy:** `python gr00t/eval/real_robot/realman/deploy_groot_realman.py` (joint checkpoint fallback)
- **ONNX export:** `python scripts/deployment/export_onnx_n1d7.py`
- **TensorRT build:** `python scripts/deployment/build_trt_pipeline.py`
- **Benchmark:** `python scripts/deployment/benchmark_inference.py`

## Real robot deployment (RealMan R2D3)

Three-process architecture (see `commands.md` for local terminal commands):

1. **`run_gr00t_server.py` (:5555)** — GR00T ZMQ policy server (normalization, relative→absolute actions). Often runs in the Orin Docker image on a GPU host.
2. **`robot_api_server.py` (:5000)** — lives in the separate `pickup-objects` repo; cameras, joint state, `ee_pose`, action execution.
3. **Deploy client** — closed-loop: fetch obs → policy `get_action` → POST right-arm command only (left arm pinned to observed state).
   - **EEF:** `deploy_groot_realman_eef.py` — `ee_pose` → EEF obs → IK → joints (`gr00t-pick-place-bottle-eef-10k`)
   - **Joint:** `deploy_groot_realman.py` — joint obs/actions directly (`gr00t-pick-bottle-realman`)

**EEF recommended live settings** (see `eef-deploy-umi-takeaways.md`, `eef-deploy-jerk-report.md`):

- `--open-loop-horizon 6` — UMI-style execute horizon (try 4–8; **16** causes boundary jerk on Orin)
- `--hz 8` or `--hz 10` — slower commanding reduces tracking lag; gripper state at 10 Hz
- `--auto-close-grip` — gripper ratchet (threshold default 0.80, lock 0.35)
- TRT DiT-only on server (`--trt-mode dit_only`) — ~4–5 Hz vs ~1.5–2.9 Hz PyTorch eager

**Joint recommended live settings:**

- `--open-loop-horizon 16` with TRT; `--hz 10`–`15`
- **Avoid** `--open-loop-horizon 1` without TRT (~700 ms infer) — arm barely moves

**Action representation (EEF):** model predicts **relative trajectory** (UMI-style); server decodes to **absolute** EEF; client IK → **absolute** joint commands. Not delta execution on the robot.

**`--debug`** surfaces DEBUG-level messages on the terminal (model chunk on inference, model output vs pinned execute command, `robot_api` POST result, `GRIP LOCK engaged`). These are always written to `run.log` even without `--debug`.

**Task string** must match training (e.g. `"pick up bottle"`).

**Run logging** is on by default. Each run creates `./runs/run_YYYYMMDD_HHMMSS/` containing:
- `meta.json` — run parameters
- `run.log` — full timestamped log at DEBUG level (loguru); captures everything even without `--debug` on the terminal
- `steps.jsonl` — per-step record: state (14D), model action (14D joint / 20D EEF), execute action (14D), grip lock, gripper force, inference/loop timing; EEF adds `ik_ok`, `ik_residual_mm`, `ik_ms`
- `summary.json` — total steps and duration
- `videos/` — one smooth MP4 per camera at `--hz`, recorded by a background thread independent of inference timing

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

**Orin inference notes:** PyTorch eager is ~2.9 Hz on Orin; TRT DiT-only (~4.6 Hz) is the best achievable on Orin because TRT 10.3 cannot compile the LLM backbone. `run_gr00t_server.py` supports TRT via `--trt-engine-path <engines-dir> --trt-mode dit_only` — build engines once with `build_trt_pipeline.py --export-mode dit_only` (the `llm_bf16.engine` failure is expected and harmless). See `commands.md` for the full build + run sequence.

For full TRT build/verify steps and per-platform benchmarks, see `scripts/deployment/README.md`.
