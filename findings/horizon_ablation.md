# Open-Loop Horizon Ablation: 1 vs 4 vs 8

**Date:** 2026-06-07  
**Task:** `"pick up bottle"`  
**Policy server:** 192.168.104.105:5555 (ZMQ, remote GPU host)  
**Robot API:** http://localhost:5000  
**Steps per run:** 30, `--hz 15`, `--debug`

---

## Summary Table

| Metric | Horizon 1 | Horizon 4 | Horizon 8 |
|---|---|---|---|
| Inferences per 30 steps | 30 | 8 | 4 |
| Avg inference latency | 659 ms | 697 ms | ~710 ms |
| Over-budget steps | 30 / 30 | 8 / 30 | 5 / 30 |
| Wall time for 30 steps | ~19 s | ~6.4 s | ~4.1 s |
| Effective overall rate | 1.52 Hz | 4.7 Hz | 7.3 Hz |
| Arm moves | No | Yes (small) | Yes (larger) |
| Max R_arm delta per step | 0.0000 rad | 0.0284 rad | 0.1218 rad |
| Reversal at infer boundary | None | Mild | Significant |
| Zero-motion infer steps | 30 | 8 | 4 |

---

## Horizon 1 (closed-loop, re-infer every step)

### Timing

- Every step exceeded budget. Avg: **659 ms**, range 602–831 ms.
- Effective control rate: **1.52 Hz** (10× below 15 Hz target).

### Motion

- `R_arm` delta was `[+0.0000, …]` on **all 30 steps**. The arm did not move.
- `chunk[0]` from each inference identically matched the observed joint state.

### Why the arm froze

The policy was trained at 15 Hz, where `chunk[0]` represents ~67 ms of motion. At 659 ms per cycle, the robot has barely moved by the time we re-query. The model sees an unchanged state and again outputs "hold current position" for `chunk[0]`. Later chunks (8–15) do contain real planned motion, but `horizon=1` never consumes them.

---

## Horizon 4 (re-infer every 4 steps)

### Timing

- 8 inferences triggered (at steps 0, 4, 8, 12, 16, 20, 24, 28).
- Each inference step was over-budget (avg **697 ms**); the 3 replay steps in between each ran in ~15–30 ms.
- Wall time for 30 steps: **~6.4 s** (3× faster than horizon=1).

### Warnings

```
Warning: step  1 took 732ms  (inference at step 0)
Warning: step  5 took 698ms  (inference at step 4)
Warning: step  9 took 692ms  (inference at step 8)
Warning: step 13 took 686ms  (inference at step 12)
Warning: step 17 took 720ms  (inference at step 16)
Warning: step 21 took 648ms  (inference at step 20)
Warning: step 25 took 714ms  (inference at step 24)
Warning: step 29 took 685ms  (inference at step 28)
```

Warnings fire exactly once per inference cycle — clean, predictable pattern.

### Motion

- Arm **does move**: non-zero `R_arm` deltas on 22 of 30 steps.
- Maximum per-step deltas by joint: J1=0.016, J2=0.017, J3=0.028, J4=0.013, J5=0.021, J6=0.025 rad.
- Observable J0 drift across run: 0.1424 → 0.1386 → 0.1574 (net ~+0.015 rad). Motion is visible but noisy.
- Reversal magnitude at re-infer boundaries: **mild** (~0.01–0.03 rad sign changes).

### 8 zero-motion steps

The first step of each 4-step block (the inference step itself) still shows `R_arm=[+0.0000, …]` because `chunk[0]` matches the observed state. Steps 1–3 of each block execute `chunk[1]`, `chunk[2]`, `chunk[3]` which carry real motion.

---

## Horizon 8 (re-infer every 8 steps)

### Timing

- 4 inferences triggered (at steps 0, 8, 16, 24).
- Wall time for 30 steps: **~4.1 s** (4.6× faster than horizon=1).
- 5 over-budget warnings: 4 are inference-driven (~710 ms each); 1 at step 22 was a minor spike (~117 ms, not an inference step).

### Warnings

```
Warning: step  1 took 752ms  (inference at step 0)
Warning: step  9 took 752ms  (inference at step 8)
Warning: step 17 took 712ms  (inference at step 16)
Warning: step 22 took 117ms  (minor API spike, not inference)
Warning: step 25 took 710ms  (inference at step 24)
```

### Motion

- Arm moves with **larger displacements** than horizon=4.
- Maximum per-step deltas by joint: J1=0.024, J2=0.065, J3=0.122, J4=0.040, J5=0.018, J6=0.046 rad.
- J3 reached +0.1218 rad in a single step — approximately 7° of elbow extension.

### Reversal at infer boundaries

The delta sign flips sharply at each re-infer event:

```
step 13 chunk_idx=5  delta J3: +0.0776   ← approaching
step 14 chunk_idx=6  delta J3: +0.1218   ← peak
step 15 chunk_idx=7  delta J3: +0.0521   ← decelerating
step 16 INFER        delta J3: +0.0000   ← inference (hold)
step 17 chunk_idx=1  delta J3: −0.0948   ← new chunk, reversal
step 18 chunk_idx=2  delta J3: −0.1081   ← deep reversal
step 19 chunk_idx=3  delta J3: −0.1121   ← deep reversal
```

This is the **jerk pattern** from the original horizon=16 diagnosis, now observed at 8-step intervals. The new chunk's absolute targets are inconsistent with where the arm actually arrived, causing corrective motion.

### 4 zero-motion steps

Same mechanism as horizon=4: `chunk[0]` from each inference is a hold-position command.

---

## Cross-Horizon Comparison

### Effective throughput vs motion quality

```
Horizon 1:  frozen arm,  cleanest "loop" (re-infers every step, useless)
Horizon 4:  arm moves,   mild reversals,  ~4.7 Hz effective rate
Horizon 8:  arm moves,   clear reversals, ~7.3 Hz effective rate, 7° per-step jumps
Horizon 16: (prior run)  arm moves,       heavy overshoot + corrective reversals
```

### The latency-horizon tradeoff

With inference fixed at ~700 ms, the "natural" horizon that keeps the loop in time budget is:

```
natural_horizon = inference_latency / target_step_dt
                = 700 ms / 67 ms
                ≈ 10.5
```

Horizon ≈ 10–11 steps lets the arm execute the cached chunk while the inference runs "for free" in the time that would otherwise be a stall. Below this value, the loop is slower than needed; above it, stale targets cause overshoot.

However, even at horizon=10, the reversal problem at infer boundaries remains unless inference latency is drastically reduced. The root cause is not the horizon value but the 700 ms gap between trajectory segments.

---

## Conclusions

1. **Horizon=1 is unviable at current latency.** The arm never moves. The training-rate mismatch (15 Hz trained, 1.5 Hz actual) makes `chunk[0]` a hold-position command.

2. **Horizon=4 produces useful motion** with mild boundary jolts and a 4.7 Hz effective rate. The small per-step displacements (≤0.03 rad) make the motion look hesitant but avoid violent overshoot.

3. **Horizon=8 produces larger motion** at 7.3 Hz effective rate, but introduces measurable reversals (~0.1 rad) at every re-infer boundary — the same artifact seen at horizon=16, just at half the frequency.

4. **The root fix is inference latency**, not horizon tuning. Until ZMQ inference drops below ~100 ms (TRT DiT-only on Orin, or a faster local GPU), no horizon value eliminates both failure modes simultaneously.

### Recommended operating point (given current latency)

`--open-loop-horizon 4` is the best available option: the arm makes progress, reversals are small enough that the arm controller absorbs them, and the inference stall is amortized over fewer cycles than horizon=1. Avoid horizon=8+ until inference latency is resolved.
