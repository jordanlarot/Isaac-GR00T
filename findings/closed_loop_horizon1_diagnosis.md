# Closed-Loop Deployment Diagnosis: `--open-loop-horizon 1`

**Date:** 2026-06-07  
**Run:** 30 steps, `--hz 15`, `--debug`, task `"pick up bottle"`  
**Policy server:** 192.168.104.105:5555 (ZMQ, remote GPU host)  
**Robot API:** http://localhost:5000

---

## Summary

`--open-loop-horizon 1` eliminates the periodic jerk seen with horizon=16, but the arm does not move at all. Root cause: inference latency (~660 ms avg) is 10× the training timestep (67 ms), causing `chunk[0]` to be a hold-position command at every re-infer cycle.

---

## Timing

| Metric | Value |
|---|---|
| Target step duration | 67 ms (15 Hz) |
| Actual average step duration | **659 ms** |
| Effective control rate | **1.52 Hz** |
| Robot API POST latency | 7–23 ms (not the bottleneck) |

Every one of the 30 steps fired the over-budget warning. Inference from the remote policy server consumed the entire step budget on each cycle.

---

## Arm Motion

`R_arm` delta was `[+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000]` across all 30 steps. The arm did not move. Only the gripper showed noise-level oscillations (±0.02 rad).

**Why chunk[0] = hold-position:** With `--open-loop-horizon 1`, only `chunk[0]` executes before the next inference. At the training rate of 15 Hz, `chunk[0]` represents ~67 ms of motion — a small first displacement. At 650 ms/inference the robot has barely moved by the time we re-query; the model sees an essentially unchanged state and again predicts "stay here" for `chunk[0]`. Later chunks (indices 8–15) show real planned motion toward the bottle, but they are never consumed.

Example from step 0:

```
observed R_arm : [+0.1429, +1.9525, +0.7489, -1.9259, -1.3494, -0.3687]
chunk[0] R_arm : [+0.1429, +1.9525, +0.7489, -1.9259, -1.3494, -0.3687]  ← exact match
chunk[8] R_arm : [+0.1744, +1.9270, +0.8013, -1.8800, -1.3217, -0.3918]  ← real motion planned
chunk[15] R_arm: [+0.2980, +1.7493, +1.1464, -1.6826, -1.3024, -0.6122]  ← significant reach
```

The policy is planning a coherent reach trajectory; the control loop can't consume it.

---

## Hypothesis Verdict

> *"With `--open-loop-horizon 1`, motion should be smoother — slower, but without the periodic jerk every ~16 steps."*

| Claim | Result |
|---|---|
| No periodic jerk at step 16 | ✓ Confirmed (no overshoot/reversal) |
| Smoother than horizon=16 | ✓ Trivially — the arm is stationary |
| Arm converges to bottle | ✗ Falsified — arm freezes at start pose |

The hypothesis is partially right about eliminating the jerk but wrong about achieving useful motion. The original horizon-16 jerkiness was an inference-stall artifact (750 ms pause + replay of 16 stale absolute targets); horizon-1 removes that artifact but introduces a new failure: the model cannot make incremental progress at 1.5 Hz.

---

## Root Cause

**Inference latency (620–660 ms) >> training timestep (67 ms).**

The policy was trained and expects to be queried at 15 Hz. Running it at 1.5 Hz via ZMQ to a remote host means `chunk[0]`'s displacement is effectively zero in wall-clock terms — the model plans for a robot that moves 10× faster than the actual loop.

---

## Recommended Next Steps

### 1. Reduce inference latency (prerequisite for any horizon to work)

- **Local Orin PyTorch:** Move `run_gr00t_server.py` to the Orin. PyTorch eager benchmarks ~340 ms on Orin (see `scripts/deployment/README.md`) — still 5× over budget, but better.
- **TRT DiT-only on Orin:** Build the TensorRT DiT engine (`--export-mode dit_only`). Estimated ~100–150 ms on Orin, which brings the loop to ~2–3 Hz. Not 15 Hz, but enough for visible motion.
- **Faster remote GPU:** If keeping inference remote, use a faster host or reduce model precision.

### 2. Intermediate horizon as a stopgap (before latency is fixed)

With inference at ~650 ms, `--open-loop-horizon 8` means each chunk runs for ~8 × 67 ms = ~530 ms of wall-clock, which roughly tracks the inference budget. The arm will make progress at the cost of periodic boundary jolts. A horizon of 4–8 is the practical range until latency is resolved.

```bash
python gr00t/eval/real_robot/realman/deploy_groot_realman.py \
  --task "pick up bottle" \
  --policy-host 192.168.104.105 \
  --policy-port 5555 \
  --robot-url http://localhost:5000 \
  --open-loop-horizon 8 \
  --hz 15 \
  --debug
```

### 3. Avoid horizon=16 with slow inference

The original jerk is a timing artifact, not a model defect. The policy's planned trajectories are smooth (confirmed by the chunk data above). Horizon=16 with 650 ms inference means a ~10-second stale open-loop replay that overshoots and corrects. Fix latency first, then return to horizon=16 or horizon=1 as appropriate.

---

## Raw Timing Sample (steps 1–10)

| Step | Latency (ms) |
|---|---|
| 1 | 831 |
| 2 | 672 |
| 3 | 628 |
| 4 | 726 |
| 5 | 641 |
| 6 | 654 |
| 7 | 672 |
| 8 | 643 |
| 9 | 634 |
| 10 | 669 |

Full log: `gr00t/eval/real_robot/realman/deploy_groot_realman.py` with `--debug --max-steps 30`.
