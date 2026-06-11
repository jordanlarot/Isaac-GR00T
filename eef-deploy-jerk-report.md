# EEF Deploy Jerk Analysis — RealMan R2D3

**Date:** 2026-06-11  
**Scope:** End-effector (EEF) pose deployment only — `deploy_groot_realman_eef.py` + `gr00t-pick-place-bottle-eef-10k`  
**Data source:** `./runs/` logs (19 EEF runs on 2026-06-11; primary deep-dive on `run_20260611_143136`)  
**Companion:** [eef-deploy-umi-takeaways.md](eef-deploy-umi-takeaways.md) — UMI paper applied to tuning (`--open-loop-horizon`, latency)

---

## Executive summary

The robot feels **jerky** because of a **prediction ↔ execution mismatch at chunk boundaries**, not because IK is broken or the wrong deploy script is running (recent runs are correctly on the EEF path).

Three mechanisms stack on top of each other:

1. **Chunk-boundary replan discontinuity** — every 16 steps the model emits a new `chunk[0]` EEF target that can be **50–125 mm** away from where the previous chunk was heading (`chunk[15]`). That snaps the commanded joints by **0.4–0.9 rad** (~23–52°) in a single step.
2. **Open-loop execution drift** — with `--open-loop-horizon 16`, the arm chases a fixed 16-step EEF plan while the physical state lags (tracking error up to **~0.3 rad** at chunk end). The next replan "corrects" from a different state, amplifying the snap.
3. **Slow inference without TRT** — PyTorch eager inference is **~700 ms** on Orin. Prefetch hides this after step 0, but the control loop still runs at **~8–15 Hz commanded** vs **~125 ms target period**, so steps bunch up and occasional boundary waits still appear.

IK and frame calibration are **healthy in recent runs** (0 IK failures, sub-mm residuals). An earlier run (`run_20260609_181001`) had a separate problem: the model predicted **unreachable EEF targets** (75 IK failures), causing hold-position stutter — that is a model/workspace issue, not the primary jerk in today's runs.

---

## Correct EEF stack (reference)

| Component | Correct setting |
|-----------|-----------------|
| Checkpoint | `/home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k` |
| Server | `run_gr00t_server.py --embodiment-tag NEW_EMBODIMENT` |
| Client | `deploy_groot_realman_eef.py` (not `deploy_groot_realman.py`) |
| Robot API | `robot_api_server.py` — must expose `ee_pose` (14D) in `/observation` |
| Action flow | Model 20D EEF → IK → 14D joints → `/action` |

Modality keys in the EEF checkpoint:

- **State:** `left_eef_9d`, `left_gripper`, `right_eef_9d`, `right_gripper`
- **Action:** same keys, `RELATIVE` EEF with `XYZ_ROT6D` (server converts to absolute)

---

## Pipeline: where prediction meets execution

```
robot_api_server          deploy_groot_realman_eef.py           run_gr00t_server.py
─────────────────         ─────────────────────────────         ───────────────────
ee_pose (14D)      →      left/right_eef_9d (9D each)    →      GR00T inference
joint state (14D)  →      + grippers + 3 cameras                returns 16-step
                          │                                     20D EEF chunk
                          ▼
                    RealmanIK (QPIK)
                    EEF target → 6 joint angles
                          │
                          ▼
                    POST /action (14D joints, right arm only)
```

**What the logs record per step** (`steps.jsonl`):

| Field | Meaning |
|-------|---------|
| `model_action` | 20D absolute EEF targets from policy (after server denorm + relative→absolute) |
| `execute_action` | 14D joint command actually sent (after IK, left arm pinned) |
| `state` | Observed joint state from robot |
| `is_infer_step` | `true` at chunk boundaries (steps 0, 16, 32, …) |
| `inference_ms` | Prefetch wait at boundary (0 ms = prefetch succeeded) |
| `ik_ok` / `ik_residual_mm` | IK convergence quality |

The jerk you feel is mostly visible as a large jump in `execute_action[7:13]` (right-arm joints) on `is_infer_step: true` rows.

---

## Evidence from `run_20260611_143136` (most recent EEF run)

**Config:** `--hz 8`, `--open-loop-horizon 16`, live (not dry-run), grip ratchet ON  
**Duration:** 123 steps / 17.5 s  
**IK:** 0 failures, max residual 0.49 mm

### Timing

| Metric | Value | Target (@ 8 Hz) |
|--------|-------|------------------|
| Loop period p50 | 104 ms | 125 ms |
| Loop period p95 | 204 ms | 125 ms |
| Loop period max | 775 ms (step 0) | 125 ms |
| Inference wait at boundary p50 | 0 ms (prefetch OK) | — |
| Inference wait at boundary max | 706 ms (step 0 only) | — |

Prefetch is working after the first chunk — boundary stalls are not the main jerk source in this run.

### Within-chunk vs at-boundary motion

| | Model EEF step (mm) | Executed joint step (rad) |
|--|---------------------|---------------------------|
| **Within chunk** (median) | 7.2 | 0.048 |
| **At chunk boundary** (median) | 53.7 | 0.559 |
| **At chunk boundary** (max) | 125.0 | 0.901 |

**Within a chunk**, the model produces smooth ~7 mm EEF steps and ~0.05 rad joint commands.  
**At every 16th step**, the new plan's `chunk[0]` can jump **~8× further** in EEF space, producing a **~10× larger** joint command delta. That is the jerk.

### Worst boundary events

| Step | Model EEF jump (mm) | Execute joint jump (rad) |
|------|---------------------|--------------------------|
| 80 | 125.0 | 0.901 |
| 112 | 123.8 | 0.808 |
| 96 | 94.4 | 0.594 |
| 16 | 53.7 | 0.559 |
| 48 | 52.0 | 0.401 |

Example at **step 16** (first replan after initial chunk):

- End of previous chunk (`step 15`) execute joints: `[0.12, 1.46, 1.55, -1.69, -1.33, -0.57]`
- New `chunk[0]` execute joints: `[0.09, 1.73, 1.10, -1.74, -1.42, -0.40]`
- Actual state at step 16: `[0.08, 1.58, 1.36, -1.74, -1.37, -0.50]` — arm has **not** reached step 15's target (tracking error **0.30 rad**)

The new plan pulls the arm back toward a `chunk[0]` EEF pose near `[0.30, -0.15, -0.22]` m — almost identical to step 0's `chunk[0]` — while the open-loop trajectory had progressed the target toward `[~0.28, -0.17, -0.17]` at `chunk[15]`.

### `chunk[0]` repeats across replans

| Inference step | Model `chunk[0]` right EEF xyz (m) |
|----------------|-------------------------------------|
| 0 | `[0.297, -0.146, -0.220]` |
| 16 | `[0.297, -0.146, -0.220]` ← identical |
| 32 | `[0.290, -0.148, -0.192]` |
| 48 | `[0.294, -0.139, -0.220]` |
| 80 | `[0.295, -0.131, -0.223]` |

The policy tends to **re-anchor** near a similar workspace pose at each replan rather than continuing smoothly from the end of the previous chunk. With `--open-loop-horizon 16`, that re-anchor becomes a visible snap.

### Gap: `chunk[15]` → next `chunk[0]`

| Boundary (infer step) | Model EEF gap (mm) |
|-----------------------|-------------------|
| 0 → 16 | 53.7 |
| 16 → 32 | 26.2 |
| 32 → 48 | 52.0 |
| 64 → 80 | **125.0** |
| 80 → 96 | 94.4 |
| 96 → 112 | **123.8** |

---

## Cross-run summary (2026-06-11 EEF runs)

All 19 EEF runs used `--open-loop-horizon 16`. Common pattern:

| Run | hz | Steps | IK failures | Inference max (ms) |
|-----|-----|-------|-------------|-------------------|
| `run_20260611_143136` | 8 | 123 | 0 | 706 |
| `run_20260611_143013` | 15 | 296 | 0 | 678 |
| `run_20260611_142815` | 15 | 339 | 0 | 704 |
| `run_20260611_135636` | 10 | 31 | 0 | 754 |
| … | 10–15 | … | **0** | 650–820 |

Recent runs: **IK is not the bottleneck.** Jerk correlates with chunk-boundary replanning and arm tracking lag, not IK divergence.

### Contrast: `run_20260609_181001` (earlier EEF run)

- **75 / 135 steps** had `ik_ok: false`
- Model predicted EEF targets with **11–48 mm FK residuals** after IK (unreachable poses)
- Arm **held position** on those steps → stuttering, not smooth boundary snaps
- Suggests the model can sometimes predict outside the calibrated IK workspace; this is a separate failure mode from boundary jerk

---

## Root-cause breakdown

### 1. Open-loop horizon 16 + chunk-boundary replan (primary)

Training uses a 16-step action horizon. Deploy replays all 16 steps before re-inferring. At step 16, 32, 48, …:

- The model sees **current** observation (images + EEF state)
- It outputs a **fresh** 16-step chunk whose `chunk[0]` is not constrained to match the previous chunk's `chunk[15]`
- IK converts that discontinuity into a large joint-space jump

This is **expected behavior** with open-loop chunk execution on hardware at Orin inference speeds, and matches the note in `commands.md` about a "brief jerk every ~16 steps."

### 2. Arm tracking lag during open-loop playback

The robot does not reach each intermediate joint target within one control period. At chunk boundaries, tracking error is often **0.07–0.30 rad**. The next `chunk[0]` is computed from current EEF state but commands a joint pose that may be far from both the **actual state** and the **previous command** — doubling the perceived jerk.

### 3. Inference latency (secondary)

Without TRT, inference is ~700 ms. The EEF client prefetches the next chunk in a background thread, so boundary `inference_ms` is usually 0 ms after step 0. However:

- Step 0 always blocks ~700 ms
- Loop periods still exceed target (p95 ~180–200 ms at hz=15 vs 67 ms target)
- Commands are issued faster than the arm can track, increasing open-loop drift

TRT DiT-only (~4–5 Hz, ~200 ms) would shrink but not eliminate boundary discontinuity.

### 4. Prefetch uses slightly stale observation

Prefetch submits inference with the observation captured at the **start** of the current chunk. After 16 steps (~1.3–2 s), the arm has moved. The prefetched chunk was planned from an older state. This adds replan error on top of the `chunk[0]` re-anchor effect.

### 5. IK / frame issues (not active in recent runs)

When the model predicts unreachable EEF poses (June 9 run), IK fails and the arm holds — a different "jerky" feel (rapid hold-release-hold). Frame calibration and QPIK are validated in tests; recent residuals are sub-mm.

---

## Recommendations (ordered by impact)

See [eef-deploy-umi-takeaways.md](eef-deploy-umi-takeaways.md) for UMI PD1 latency matching and stale-action skip (future code work).

### Quick experiments (no code changes)

1. **Try `--open-loop-horizon 6 --hz 8`** — default in `commands.md`; UMI-style execute horizon for quasi-static bottle pick.
2. **Try `--open-loop-horizon 4` or `8`** — compromise between reach and boundary frequency.
3. **Try `--open-loop-horizon 1`** — true closed-loop A/B test; too slow without TRT (~700 ms infer).
4. **Enable TRT** — rebuild engines for the EEF checkpoint, add `--trt-engine-path ... --trt-mode dit_only` to the server. Faster inference → less tracking drift per chunk.
5. **Reset to episode start pose** before each run (`reset_to_episode_start.py`) — reduces initial workspace mismatch.

### Medium-term (client / control)

6. **Chunk blending at boundaries** — interpolate last command toward new `chunk[0]` over 2–3 steps instead of stepping instantly.
7. **Re-infer from current obs at boundary** (disable prefetch for debugging) — isolates stale-obs effect from re-anchor effect.
8. **Execute only `chunk[1:]` after first infer** — skip `chunk[0]` if it is consistently a near-static "hold" in training data (validate against open-loop eval first).
9. **Lower IK rate limits on robot_api** — if the executor accepts velocity limits, cap joint delta per step.

### Model / training

10. **Open-loop eval on EEF checkpoint** — compare predicted chunk vs dataset on held-out trajectories; quantify `chunk[15]` → next-obs `chunk[0]` gap offline.
11. **DAgger / more data** near failure states if IK failures return (June 9 pattern).

---

## Corrected deploy commands (EEF)

```bash
# Policy server (EEF checkpoint)
python gr00t/eval/run_gr00t_server.py \
  --model-path /home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k \
  --embodiment-tag NEW_EMBODIMENT \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 5555
  # Optional TRT (rebuild engines for EEF checkpoint first):
  # --trt-engine-path /home/r2d3/checkpoints/gr00t-pick-place-bottle-eef-10k-trt/engines \
  # --trt-mode dit_only

# Deploy client — dry-run first
source .venv/bin/activate && source scripts/activate_orin.sh

python gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py \
  --task "pick up bottle" \
  --policy-host localhost \
  --policy-port 5555 \
  --robot-url http://localhost:5000 \
  --hz 8 \
  --open-loop-horizon 6 \
  --auto-close-grip \
  --dry-run --debug

# Smoother-motion experiments
# --open-loop-horizon 4
# --open-loop-horizon 16   # max reach; more boundary jerk
```

**Prerequisites:** `robot_api_server.py` on `:5000`, ROS drivers running, head camera positioned per `commands.md`.

---

## How to diagnose future runs

Check `./runs/run_YYYYMMDD_HHMMSS/`:

1. **`run.log`** — confirm `GR00T EEF CLOSED-LOOP DEPLOYMENT` and IK init; look for `Step N took XXXms` warnings and `IK failed` lines.
2. **`steps.jsonl`** — filter `is_infer_step: true` rows; compare `execute_action[7:13]` to previous step for joint jumps > 0.3 rad.
3. **`meta.json`** — verify `open_loop_horizon`, `hz`, and that you're not accidentally on the joint client (joint runs log `GR00T CLOSED-LOOP DEPLOYMENT` without "EEF" and use 14D `model_action` not 20D).

Quick analysis one-liner:

```bash
python3 -c "
import json, numpy as np
rows=[json.loads(l) for l in open('runs/run_YYYYMMDD_HHMMSS/steps.jsonl')]
inf=[r for r in rows if r['is_infer_step'] and r['step']>0]
jumps=[np.linalg.norm(np.array(rows[r['step']]['execute_action'][7:13])-np.array(rows[r['step']-1]['execute_action'][7:13])) for r in inf]
print(f'boundary joint jumps (rad): median={np.median(jumps):.3f} max={max(jumps):.3f}')
"
```

---

## Related docs

- [`commands.md`](commands.md) — operator runbook (terminals, flags, reset pose)
- [`gr00t-eef-robot-deploy.md`](gr00t-eef-robot-deploy.md) — EEF deploy architecture
- [`gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py`](gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py) — EEF client with prefetch + IK
