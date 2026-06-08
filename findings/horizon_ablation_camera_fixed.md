# Open-Loop Horizon Ablation — Camera Corrected

**Date:** 2026-06-07  
**Change from prior ablation:** Head camera lowered to match training-data framing (gripper visible at rest)  
**Task:** `"pick up bottle"` · **Steps per run:** 30 · **Target rate:** 15 Hz  
**Policy server:** 192.168.104.105:5555 · **Robot API:** http://localhost:5000  
**Model trained horizon:** 16

---

## Summary Table

| Horizon | Inferences | Wall time | Over-budget steps | Avg inference lat | Arm moves | Max \|ΔJ3\| | J0 net drift | Max J3 reversal |
|---|---|---|---|---|---|---|---|---|
| 1  | 30 | 18.5 s | 30 / 30 | 638 ms | No  | 0.0000 rad | −0.0004 rad | 0.0000 rad |
| 4  |  8 |  6.2 s |  9 / 30 | 622 ms | Yes | 0.0447 rad | +0.0149 rad | 0.0382 rad |
| 8  |  4 |  4.0 s |  5 / 30 | 598 ms | Yes | 0.2139 rad | +0.0234 rad | 0.0704 rad |
| 12 |  3 |  3.4 s |  4 / 30 | 568 ms | Yes | 0.0397 rad | +0.0014 rad | 0.0348 rad |
| **16** |  **2** |  **2.6 s** |  **2 / 30** | **707 ms** | **Yes** | **0.4377 rad** | **+0.1060 rad** | **0.4332 rad** |

---

## Per-Horizon Analysis

### Horizon 1 — No motion (same as before camera fix)

Inference latency (avg 638 ms) completely dominates every step. The arm does not move: all 30 steps show zero R_arm delta. `chunk[0]` still matches the observed state regardless of camera framing because the fundamental rate mismatch (1.54 Hz actual vs 15 Hz trained) is unchanged.

**Camera fix effect:** None measurable at this horizon. Latency is the binding constraint, not visual input quality.

---

### Horizon 4 — Useful motion, mild reversals

- 8 inferences; 22 of 30 steps produce non-zero deltas.
- Max per-step J3 displacement: **0.045 rad** (~2.6°).
- J0 net drift: **+0.015 rad** — the arm is making slow forward progress.
- Reversal at each re-infer boundary: **~0.038 rad** on J3 — felt as a mild hesitation.

The camera fix is visible here: in the pre-fix run, max J3 was 0.028 rad and J0 drift was +0.015 rad (similar). The model is producing slightly more consistent trajectories but the chunk is still cut off too early to develop meaningful reach.

---

### Horizon 8 — Larger motion, clear boundary jerk

- 4 inferences; 26 of 30 steps produce non-zero deltas.
- Max per-step J3: **0.214 rad** (~12°). Arm is visibly reaching.
- J0 net drift: **+0.023 rad** — best sustained forward progress after H=16.
- Reversal at re-infer boundary: **0.070 rad** on J3 — visible jolt when chunk transitions.

The camera fix produces noticeably larger max displacements vs the pre-fix run (0.214 vs 0.122 rad on J3), suggesting the better visual input gives the policy more confidence to commit to larger reaching motion.

---

### Horizon 12 — Unexpectedly conservative

- 3 inferences; 27 of 30 steps produce non-zero deltas.
- Max per-step J3: **0.040 rad** — lower than H=8 despite consuming more of each chunk.
- J0 net drift: **+0.001 rad** — almost no net displacement over 30 steps.
- Reversal: **0.035 rad**.

**Why less motion than H=8?** With H=12, we execute `chunk[0:12]` — covering the early, conservative part of the trajectory (acceleration and mid-reach) — then re-infer. The model's second inference starts from a mid-reach state and generates a fresh chunk that partially backtracks. The large displacements seen in H=8 at steps 13–15 (the late, high-velocity part of the chunk) are never reached before the next re-infer. This is a chunk-truncation artifact: H=12 misses the payoff region of the trained trajectory.

---

### Horizon 16 — Full trained trajectory, severe boundary reversal

This is the trained horizon. Executing all 16 steps before re-inferring produces the full planned reaching motion — and exposes the worst boundary artifact.

**J3 delta trace (the critical elbow joint):**

```
step  0  chunk_idx= 0   dJ3 = +0.0000  ← INFER (hold)
step  1  chunk_idx= 1   dJ3 = −0.0206
step  2  chunk_idx= 2   dJ3 = −0.0172
step  3  chunk_idx= 3   dJ3 = −0.0110
step  4  chunk_idx= 4   dJ3 = −0.0132  ← initial retraction / setup
step  5  chunk_idx= 5   dJ3 = +0.0281
step  6  chunk_idx= 6   dJ3 = +0.0219
step  7  chunk_idx= 7   dJ3 = +0.0502
step  8  chunk_idx= 8   dJ3 = +0.0808  ← arm begins reaching
step  9  chunk_idx= 9   dJ3 = +0.0989
step 10  chunk_idx=10   dJ3 = +0.1510
step 11  chunk_idx=11   dJ3 = +0.1726
step 12  chunk_idx=12   dJ3 = +0.2791
step 13  chunk_idx=13   dJ3 = +0.3364
step 14  chunk_idx=14   dJ3 = +0.3762
step 15  chunk_idx=15   dJ3 = +0.4332  ← peak reach (+0.43 rad, ~25° forward)
──────────────────────────────────────────
step 16  chunk_idx= 0   dJ3 = +0.0000  ← INFER — 707 ms stall, new chunk
step 17  chunk_idx= 1   dJ3 = −0.3299  ← REVERSAL: arm jerks back 0.43 rad
step 18  chunk_idx= 2   dJ3 = −0.2918
step 19  chunk_idx= 3   dJ3 = −0.2774
step 20  chunk_idx= 4   dJ3 = −0.2206
step 21  chunk_idx= 5   dJ3 = −0.1859
step 22  chunk_idx= 6   dJ3 = −0.1174
step 23  chunk_idx= 7   dJ3 = −0.0088
step 24  chunk_idx= 8   dJ3 = +0.0934  ← second reach begins
step 25  chunk_idx= 9   dJ3 = +0.1849
step 26  chunk_idx=10   dJ3 = +0.3245
step 27  chunk_idx=11   dJ3 = +0.3178
step 28  chunk_idx=12   dJ3 = +0.4012
step 29  chunk_idx=13   dJ3 = +0.4377  ← reaches again at step limit
```

The arm extends ~25° forward over 16 steps, then the 707 ms inference stall causes the robot controller to receive no new command (or hold). The new chunk starts from the arm's now-extended position but generates targets that pull it back — a **0.43 rad swing reversal** (roughly the same magnitude as the initial reach, in the opposite direction). This is the "jerk every ~16 steps" from the original complaint, now quantified.

J0 net drift of **+0.106 rad** over 30 steps shows that despite the reversal, H=16 produces the largest gross displacement of all horizons — the full trained trajectory is being executed.

---

## Effect of Camera Correction

Comparing H=8 across both ablations (same horizon, different camera framing):

| Metric | Pre-fix | Post-fix |
|---|---|---|
| Max \|ΔJ3\| | 0.1218 rad | 0.2139 rad |
| J0 net drift | +0.023 rad | +0.023 rad |
| Max reversal J3 | 0.070 rad | 0.070 rad |

The camera fix roughly doubles peak joint velocity (the model is more confident about where the arm is relative to the bottle), while reversal magnitude and net drift are unchanged. The policy's structure — when it moves, how far it reverses — is determined by the horizon/latency relationship, not visual input alone.

---

## Key Conclusions

### 1. Horizon 16 executes the full trained trajectory — and produces the worst jerk

The model plans a smooth 16-step reach (accelerating through steps 5–15 to +0.43 rad). Because inference takes 707 ms, the arm is mid-air when the stall hits; the next chunk's absolute targets pull it back violently. **The jerk is not a model failure; it is a timing artifact of replaying absolute-position chunks at a rate the inference cannot sustain.**

### 2. No horizon eliminates both problems simultaneously at current latency

| Problem | H=1 | H=4 | H=8 | H=12 | H=16 |
|---|---|---|---|---|---|
| Arm frozen | ✗ worst | ✓ | ✓ | ✓ | ✓ |
| Boundary reversal | ✓ none | ✓ mild | ~ moderate | ✓ mild | ✗ worst |
| Full trajectory executed | ✗ | ✗ | ✗ | ✗ | ✓ |
| Net forward progress | ✗ | ~ | ~ | ✗ | ✓ best |

### 3. Best operating point at current latency: H=8

H=8 provides the best balance of visible arm motion, moderate reversal magnitude, and a 4 Hz effective rate without the catastrophic 0.43 rad jerk of H=16. H=4 is smoother but too conservative; H=12 misses the late high-velocity region of the chunk and produces negligible net drift.

### 4. The real fix is inference latency

The "natural" horizon that amortizes one 700 ms inference over the step budget is:
```
natural_horizon = 700 ms / 67 ms ≈ 10 steps
```
At H=10, the arm would finish each chunk in roughly the same time it takes to compute the next one — eliminating the stall-and-jerk cycle. But even this only addresses timing; the absolute-target reversal at boundaries requires either relative-action decoding on the server or inference latency well below the step budget. **Target: < 100 ms inference (TRT DiT-only on Orin, or a faster local GPU).**

---

## Suggested Fixes and Solutions

The problems break down into two root causes that need separate fixes: **(A) inference latency** (700 ms, 10× too slow) and **(B) chunk boundary reversal** (new absolute targets don't match where the arm actually arrived). Some fixes address both; most address one.

---

### Fix 1 — TensorRT DiT-only engine on Orin (highest priority)

**Addresses:** Inference latency (A)

Build and wire the TRT DiT-only engine into `run_gr00t_server.py`. The backbone runs in PyTorch; only the DiT diffusion head is TRT-compiled (`--export-mode dit_only`). Expected latency on Orin: ~100–150 ms per call, bringing the effective rate to 6–10 Hz with H=1 and eliminating the H=16 stall entirely.

```bash
# Export
python scripts/deployment/export_onnx_n1d7.py \
    --model-path <checkpoint> --export-mode dit_only

# Build TRT engine
python scripts/deployment/build_trt_pipeline.py \
    --onnx-path <dit_only.onnx> --output-path <engine.trt>

# Benchmark before wiring in
python scripts/deployment/benchmark_inference.py --trt-engine-path <engine.trt>
```

Once latency drops below ~100 ms, H=16 becomes viable with manageable boundary artifacts, and H=1 (true closed-loop) becomes useful rather than frozen.

**Effort:** Medium — export + build scripts exist; wiring TRT into the ZMQ server path (`run_gr00t_server.py`) requires integration work, as noted in `scripts/deployment/README.md`.

---

### Fix 2 — Run the policy server locally on the Orin (immediate stopgap)

**Addresses:** Inference latency (A)

Currently the server runs on a remote GPU at 192.168.104.105. Moving it to the Orin itself eliminates network round-trip overhead and co-locates compute with the control loop. PyTorch eager on Orin benchmarks ~340 ms — still above the 67 ms target but roughly half the current 700 ms, bringing H=1 effective rate to ~2.9 Hz and reducing H=16 stall duration proportionally.

```bash
# On Orin, activate environment and run server locally
source scripts/activate_orin.sh
python gr00t/eval/run_gr00t_server.py \
    --model-path <checkpoint> \
    --embodiment-tag new_embodiment

# Then point deploy script at localhost
python gr00t/eval/real_robot/realman/deploy_groot_realman.py \
    --policy-host localhost --policy-port 5555 ...
```

**Effort:** Low — no code changes; just move where the server process runs.

---

### Fix 3 — Asynchronous inference (decouple inference from the control loop)

**Addresses:** Inference latency impact (A) — doesn't reduce raw latency but hides it

Run inference in a background thread. The control loop always executes from the most recently completed chunk at the target Hz, and fires a new inference request the moment it starts consuming a chunk. The stall disappears because the loop never waits — it either uses the fresh chunk (if ready) or continues replaying the previous one.

Concretely in `deploy_groot_realman.py`:

```python
import threading

class GR00TRealmanController:
    def __init__(self, ...):
        ...
        self._next_chunk = None
        self._infer_lock = threading.Lock()

    def _infer_background(self, raw):
        chunk = self._infer_action_chunk(raw)
        with self._infer_lock:
            self._next_chunk = chunk

    def run(self, max_steps):
        pred_chunk = None
        chunk_idx = 0
        infer_thread = None

        while step < max_steps:
            raw = self._fetch_obs()

            # Swap in the new chunk the moment inference finishes
            with self._infer_lock:
                if self._next_chunk is not None:
                    pred_chunk = self._next_chunk
                    self._next_chunk = None
                    chunk_idx = 0

            # Fire a new inference when the current chunk is exhausted
            if pred_chunk is None or chunk_idx >= self.open_loop_horizon:
                if infer_thread is None or not infer_thread.is_alive():
                    infer_thread = threading.Thread(
                        target=self._infer_background, args=(raw,), daemon=True)
                    infer_thread.start()
                    if pred_chunk is None:
                        infer_thread.join()  # block only on very first step

            action = pred_chunk[chunk_idx]
            ...
            chunk_idx += 1
```

With this change the loop runs at a steady ~15 Hz (limited only by robot API latency), and re-infer boundaries become smooth chunk swaps rather than stalls. The boundary reversal (Fix B below) is still present but no longer causes a timing gap.

**Effort:** Medium — contained change to `deploy_groot_realman.py`; needs testing for thread-safety around the observation used for inference vs. the observation at execution time.

---

### Fix 4 — Use H=10 as the timing-matched horizon (no code change)

**Addresses:** Latency impact (A), partially

At current 700 ms inference and 67 ms step target:

```
natural_horizon = 700 ms / 67 ms ≈ 10 steps
```

With H=10, the arm finishes its chunk in ~670 ms — just as the next inference completes. No stall, no frozen arm. The boundary reversal is still present but occurs at the most natural handoff point in the trained trajectory (the arm is decelerating as the new chunk arrives).

This requires no code changes. Run with `--open-loop-horizon 10` and adjust if inference latency shifts.

**Effort:** Zero — single flag change.

---

### Fix 5 — Relative action decoding to eliminate boundary reversal

**Addresses:** Chunk boundary reversal (B)

The reversal happens because the model outputs **absolute** joint targets: chunk[0] of a new inference targets joints at positions appropriate for the state seen *at inference time*, not at the state when the chunk finally executes (700 ms later, after 10+ steps of movement). The mismatch causes the corrective jerk.

The fix is to have the policy server decode actions as **deltas relative to the current observed state** rather than absolute targets, or to apply a correction offset:

```python
# In deploy_groot_realman.py, after getting a new chunk:
# Shift all absolute targets by (current_state - state_at_inference_time)
def _offset_chunk_to_current_state(chunk, state_at_infer, state_now):
    offset = state_now - state_at_infer  # (14,)
    return [action + offset for action in chunk]
```

This is an approximation (joint-space offsets don't perfectly account for arm kinematics) but eliminates the large systematic reversal. A cleaner solution is retraining with relative action targets, which `Gr00tPolicy.decode_action` already supports if the checkpoint was trained that way.

**Effort:** Low for the offset hack (a few lines in `deploy_groot_realman.py`); medium for verifying action space conventions with the checkpoint.

---

### Fix 6 — Increase `--hz` to match effective inference rate

**Addresses:** Loop timing mismatch (A) — indirect

Rather than fighting the 700 ms inference with a 67 ms target, set `--hz` to match the actual achievable rate at the chosen horizon:

| Horizon | Inferences per 30 steps | Effective Hz |
|---|---|---|
| 8 | 4 | ~7.5 Hz |
| 10 | 3 | ~8–9 Hz |
| 16 | 2 | ~11.5 Hz |

For H=8, `--hz 7` or `--hz 8` means the loop doesn't log over-budget warnings, the sleep budget is consumed correctly, and the robot controller sees commands at a rate it can track. This doesn't fix motion quality but cleans up timing and makes the log easier to interpret.

**Effort:** Zero — single flag change.

---

### Fix 7 — Retrain at lower Hz or with longer delta_indices (longer-term)

**Addresses:** Both root causes (A, B)

The model was trained at 15 Hz with a 16-step horizon (~1 s of action). If inference stays at 700 ms, retraining at **2–5 Hz** would make `chunk[0]` represent 200–500 ms of motion — enough for the arm to actually move between re-infers. This is the only fix that aligns the training distribution with the actual deployment rate.

Alternatively, retrain at 15 Hz but with **relative action targets** (joint deltas rather than absolute positions). This eliminates the boundary reversal regardless of inference latency, since each chunk's targets are always relative to wherever the arm currently is.

**Effort:** High — requires new data collection or data annotation changes, plus a full training run.

---

### Priority Order

| Priority | Fix | Effort | Impact |
|---|---|---|---|
| 1 | **Fix 4** — use `--open-loop-horizon 10` | Zero | Eliminates inference stall with no code change |
| 2 | **Fix 6** — match `--hz` to effective rate | Zero | Cleans up timing artifacts |
| 3 | **Fix 2** — run server on Orin locally | Low | Halves inference latency immediately |
| 4 | **Fix 3** — async inference thread | Medium | Hides latency; loop runs at full Hz |
| 5 | **Fix 5** — relative-offset chunk correction | Low–Medium | Kills boundary reversal |
| 6 | **Fix 1** — TRT DiT-only engine | Medium | Gets inference to ~100 ms; enables H=1 |
| 7 | **Fix 7** — retrain at deployment Hz | High | Permanent alignment; cleanest solution |
