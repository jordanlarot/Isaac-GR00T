# UMI Paper Takeaways — Applied to GR00T EEF Deploy (RealMan R2D3)

**Paper:** [Universal Manipulation Interface (UMI)](https://arxiv.org/pdf/2402.10329) — Chi et al., 2024  
**Context:** EEF checkpoint `gr00t-pick-place-bottle-eef-10k`, client `deploy_groot_realman_eef.py`  
**Related:** [`eef-deploy-jerk-report.md`](eef-deploy-jerk-report.md) (run-log jerk analysis)

---

## Executive summary

UMI’s main deploy insight is that **action representation and execution interface are equally important**. Your GR00T EEF stack already implements UMI’s **relative trajectory** representation correctly (train relative → decode to absolute → IK → absolute joint commands). The visible sway/jerk on hardware is **not** from executing absolute instead of relative, or from using the wrong representation.

The gap is mostly **UMI PD1 (inference-time latency matching)** and **execution horizon / speed** — how many chunk steps you play out, how fast you command them, and whether stale actions are discarded after inference delay.

---

## UMI action representations (Fig. 6, PD2.1)

UMI distinguishes three action spaces:

| Representation | Definition | UMI cup task result |
|----------------|------------|---------------------|
| **Absolute** | Targets in a global / robot-base frame | 25% success (calibration fragile) |
| **Delta** | Each step relative to the **previous** step | 80% success (error accumulates) |
| **Relative trajectory** | Each step in the chunk relative to the **same** current EE at inference \(t_0\) | Best (100% with mirrors) |

**Relative trajectory** means: for a chunk starting at \(t_0\), action step \(t\) is an \(SE(3)\) transform relative to the gripper pose at \(t_0\), **not** a chain of step-to-step deltas.

**Critical point:** UMI still **executes absolute pose targets** on the robot. Relative is the policy’s internal language; the controller receives composed absolute waypoints.

---

## What your code already matches

### 1. Relative trajectory (not delta, not global absolute)

EEF checkpoint uses `RELATIVE` EEF actions with `state_key: right_eef_9d`. On the policy server, `decode_action` converts the full chunk:

```
T_absolute[i] = T_ref @ T_relative[i]
```

where `T_ref` is the **same** current EEF from the observation at inference time. This matches UMI PD2.1. GR00T does **not** use `delta_chunking` (step-to-step) for deploy.

**Files:** `gr00t/data/state_action/state_action_processor.py`, `gr00t/data/state_action/action_chunking.py` (`to_absolute_chunking`), `gr00t/policy/gr00t_policy.py` (`decode_action`).

### 2. Absolute execution on hardware (correct)

Pipeline:

```
Model (relative EEF) → server decode (absolute EEF chunk)
    → deploy client IK (absolute joints) → robot_api Jointpos (absolute)
```

The deploy client does **not** integrate relative deltas on the robot. `eef_action_to_joint_command()` treats each `model_action` step as an absolute EEF target.

**Files:** `deploy_groot_realman_eef.py`, `pickup-objects/scripts/robot_api_server.py`, `ros2_bridge.execute_robot_actions()`.

### 3. Observation interface

Wrist/top cameras + EEF proprioception (`ee_pose` → `left_eef_9d` / `right_eef_9d` + grippers) aligns with UMI’s wrist-camera + relative EE proprioception design.

---

## What UMI emphasizes that you mostly lack

### PD1 — Inference-time latency matching

UMI treats latency as part of the policy interface:

- **PD1.1 Observation:** Align image, EE pose, and gripper streams to a common time base (interpolate proprio to camera timestamp).
- **PD1.2 Action:** Predict a chunk starting at last observation time; **discard** actions whose desired time is already in the past after `t_input + t_infer + t_exec`; send remaining commands ahead to compensate execution delay.

**UMI ablation (dynamic tossing):** Disabling latency matching dropped success from **87.5% → 57.5%** with visibly jerkier motion.

**Your client today:**

| UMI practice | Your EEF client |
|--------------|-----------------|
| Measure camera / proprio / execution latency | Partially via `inference_ms`, `loop_ms` in logs; no systematic calibration |
| Time-align observations | Fresh `/observation` each step; policy chunk anchored to obs at **inference** time |
| Discard stale chunk steps after latency | Always starts at `chunk[0]` |
| Schedule commands ahead of desired time | Fixed `--hz` loop; no execution-delay compensation |
| Prefetch next chunk | Yes (`ThreadPoolExecutor` in `deploy_groot_realman_eef.py`) |

Prefetch hides inference **wait** at chunk boundaries but does **not** replace stale-action discarding or timestamp scheduling.

### Shorter execution horizon than prediction horizon

UMI (Diffusion Policy, Table A1) uses **action horizon `Ta = 6`** for cup arrangement and similar quasi-static tasks, while the model may predict a longer sequence internally.

You predict **16** steps per inference; early hardware runs executed all **16** (`--open-loop-horizon 16`), which maximizes open-loop drift and boundary replan snaps (see jerk report: 50–125 mm EEF jumps every 16 steps). **`commands.md` now recommends `--open-loop-horizon 6 --hz 8`** (UMI `Ta≈6` for quasi-static tasks).

### Slower execution for quasi-static tasks

UMI Appendix E3: for quasi-static tasks they run at **0.5×** demonstration speed because imperfect latency compensation causes jitter at full speed. Bottle pick is quasi-static; dynamic tossing runs at 1.0×.

You command at **8–15 Hz** while the arm lags (median tracking error ~0.19 rad in recent runs).

---

## Relative vs absolute — direct answer

| Question | Answer |
|----------|--------|
| Is the model predicting relative trajectory? | **Yes** (UMI-style, not delta). |
| Does the server convert to absolute before the client? | **Yes** (`decode_action`). |
| Does the robot receive relative deltas? | **No** — absolute joint positions. |
| Is executing absolute wrong? | **No** — same as UMI. |
| Why does it sway? | Open-loop horizon, stale chunk anchor, tracking lag, boundary replans — not wrong representation. |

---

## How to apply UMI to your code

### Tier 1 — Try now (flags only, no code changes)

```bash
# UMI-ish: shorter execute horizon + slower rate (quasi-static bottle task)
python gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py \
  --task "pick up bottle" \
  --policy-host localhost --policy-port 5555 \
  --robot-url http://localhost:5000 \
  --open-loop-horizon 6 \
  --hz 8 \
  --auto-close-grip
```

| Experiment | Rationale |
|------------|-----------|
| `--open-loop-horizon 6` | Matches UMI `Ta=6`; fewer boundary snaps than 16 |
| `--open-loop-horizon 4` | Smoother replanning if 6 still jerky |
| `--hz 8` or `--hz 5` | UMI 0.5× speed idea; more time for arm to track |
| TRT on server (`--trt-mode dit_only`) | Shrinks inference → less stale-anchor error; enables tighter horizons |

**Prefetch timing at 10 Hz:**

| Horizon | Execution window | Fits ~700 ms infer (no TRT)? | Fits ~200 ms TRT? |
|---------|------------------|------------------------------|-------------------|
| 16 | 1600 ms | Yes | Yes |
| 8 | 800 ms | Tight | Yes |
| 6 | 600 ms | Borderline | Yes |
| 4 | 400 ms | No (boundary stalls) | Yes |
| 1 | 100 ms | No (every step blocks) | No |

### Tier 2 — Worth implementing in `deploy_groot_realman_eef.py`

1. **Stale-action skip (UMI PD1.2)**  
   After inference, skip `chunk[0:k]` where  
   `k = ceil((t_camera_latency + t_inference + t_robot_exec) / dt)`.  
   Only execute actions whose intended time is still in the future.

2. **Latency calibration in `meta.json`**  
   One-time measurement: camera pipeline delay, ZMQ inference, joint tracking lag. Log per run for tuning `k` and `--hz`.

3. **Optional `--execution-speed 0.5`**  
   Scale effective `dt` (UMI E3): e.g. command at 5 Hz while logging as 10 Hz equivalent.

4. **Prefetch from fresher obs at boundary**  
   Today prefetch submits inference with obs captured at **chunk start**; after 6–16 steps the arm has moved. Option: re-infer from latest obs at boundary (disable prefetch) or submit prefetch only ~1 horizon before boundary with fresh obs.

5. **Chunk-boundary blending (extension beyond UMI)**  
   Interpolate last command → new `chunk[k]` over 2–3 steps to reduce measured 0.4–0.9 rad joint jumps at replan.

### Tier 3 — Training / eval (offline validation)

- **Open-loop eval** on EEF dataset: if predicted chunks zigzag offline too, smoothness is a policy issue; if offline is smooth but hardware sways, it’s deploy/timing.
- Do **not** switch to delta execution on the robot; UMI’s own ablation favors relative trajectory over delta.

---

## UMI vs your deploy — checklist

```
[✓] Train with relative trajectory (not global absolute)
[✓] Decode to absolute EEF targets on server
[✓] Execute absolute joint commands (via IK)
[✓] Wrist cameras + EEF proprioception
[~] Observation time alignment (partial — fresh fetch, no explicit sync)
[✗] Discard stale actions after latency
[✗] Execution-delay compensation (send-ahead scheduling)
[✗] Execute subset of chunk (Ta=6 vs horizon=16)
[✗] Slower quasi-static execution (0.5× speed)
[✓] Prefetch inference (helps wait, not stale-action semantics)
```

---

## Mental model

```
UMI lesson 1:  Train with relative trajectory        →  you have this
UMI lesson 2:  Decode to absolute targets             →  server does this
UMI lesson 3:  Execute with latency awareness        →  main gap
UMI lesson 4:  Execute subset of chunk, slower rate  →  tune via flags first
```

**Do not** second-guess relative vs absolute representation.  
**Do** treat `--open-loop-horizon`, `--hz`, latency skip, and TRT as first-class deploy parameters — the UMI paper spends as much effort there as on the gripper hardware.

---

## Recommended experiment order

1. **`--open-loop-horizon 6 --hz 8`** — compare video to horizon-16 runs.  
2. **Enable TRT** on EEF checkpoint — re-test horizons 6, 8, 16.  
3. **Horizon 4** if boundary jerk persists (with TRT).  
4. **Implement stale-action skip** if flag tuning isn’t enough.  
5. **Measure latencies once** (camera, inference, execution) and record in run `meta.json`.

---

## References

| Resource | Path / link |
|----------|-------------|
| UMI paper | https://arxiv.org/pdf/2402.10329 |
| Jerk analysis (your runs) | [`eef-deploy-jerk-report.md`](eef-deploy-jerk-report.md) |
| EEF deploy client | [`gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py`](gr00t/eval/real_robot/realman/deploy_groot_realman_eef.py) |
| Operator runbook | [`commands.md`](commands.md) |
| EEF deploy guide | [`gr00t-eef-robot-deploy.md`](gr00t-eef-robot-deploy.md) |
