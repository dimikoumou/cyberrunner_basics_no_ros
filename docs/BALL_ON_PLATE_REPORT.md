# CyberRunner ball-on-plate — full report

*Branch `ball-on-plate`, work of 2026-09-26/27 (day 2 afternoon: §8). How to run everything: [`rl_hw/HOW_TO_RUN.md`](../rl_hw/HOW_TO_RUN.md).
Short project history in [`REPORT.md`](../REPORT.md); this document is the complete account of the
ball-on-plate work: what was found, what worked, what failed, and why.*

---

## 1. Goal and starting point

**Initial goal:** put the steel ball on the plate and hold it inside a red goal circle for at
least 10 s with `rl_hw/pd_balance.py` (no maze, no RL).
**Grew into:** a ball that goes to *any* red region, a clicked point, or along a drawn line —
in one smooth motion and then stays — with a web UI and a learned (RL) controller.

**The rig**

| Item | Fact |
|---|---|
| Tilt motors | Dynamixel **IDs 1 and 3**; ID 2 = ball-reload elevator (velocity mode, runs only while the ball is lost) |
| Surface | Glass removed; **paper** on the frame (the old bezel ledge that trapped the ball is gone) |
| Camera | See3CAM_24CUG, 1920×1080, processed at 640×360; **requested at 30 fps** |
| Markers | 4 outer (fixed frame) + 4 inner (tilting plate) blue dots |
| Plate half-extents | 0.1417 × 0.1192 m |
| Control loop | ~22 Hz originally, ~29 Hz after the camera fix (vision-bound) |
| Tilt range (measured) | beta (motor 3) −8.4…+10.6°, alpha (motor 1) +6.2…−5.6° |

---

## 2. Results at a glance

| Milestone | Result |
|---|---|
| First 10 s hold (Phase 3 goal) | **10.03 s**, 225/225 frames detected, max 4.27 mm inside a 45.5 mm disc |
| Resting in the middle (classic) | median **2.6 mm** from centre, **260 s** hold, 98.6 % in circle |
| Small 9 mm dot (open space) | 100 % in target, 0 exits after arrival |
| Smooth planned moves (classic) | most 10–14 cm trips in **one motion**, 98–100 % in target after arrival |
| Line following | 4 cm loop: **8.7 laps/min**, median **6.6 mm** off the line (carrot follower: 1.1 laps/min) |
| Draw-a-path (UI) | 22 cm L-path driven to the end, holding 6 mm from it |
| RL simulator calibration | 1-s replay error **3.4 / 4.3 mm** at 0.5 / 1 s (baseline “stays put”: 12.9 / 20.5 mm) |
| RL v3 in sim (200 episodes) | **92 %** reach vs **65 %** for PD; 1.00 s vs 1.34 s; 70 % vs 28 % inside afterwards |
| RL v3 on the rig (zero-shot) | 3/3 trips: 1.2–2.2 s, no hops, 100 % in target afterwards |
| ODIL v6 + stiction comp. in sim | **97 %** reach (RL 96 %), 69 % inside afterwards (RL 72 %), ~3× smoother than RL |
| ODIL on the rig (pure / + settle) | **20/20** each; + settle: 1.3 s, 77 % inside, 6.1 mm — on par with RL + settle |
| Hole + auto-reload | drop → elevator → ball back typically in 4–34 s, task resumes |

---

## 3. What we discovered (root causes, with evidence)

### Vision
- **Pipeline runs at 640×360**, not 1920×1080 — an early blob-area diagnosis was wrong by 9×
  because it was measured at full resolution (retracted; see §5).
- **Left outer markers are intermittently half-hidden** by a strip of the tilting frame
  (history-dependent). A half marker leaves ~5 px of blue, detection jumps to a wrong blob and the
  camera pose comes out ~30° off (rotation diagonal 0.86). → **Outer 4 corners cached once**
  (`state_est/fixed_corners_cache.json`, 4 clean frames within 1.3 px); inner 4 tracked live.
- **HSV retuned from sampled pixels:** corners hue 104–140; ball hue 80–108, sat ≥ 100. On bare
  paper the ball casts a **bluish shadow** that the old range merged into the ball; over the red
  disc the ball's rim mixes with red (hue tail to 120): maxHue 104 → 108 took detection on the disc
  from 50/52 to 52/52 snapshots and removed ball-lost resets.
- **Red-goal detection pitfalls:** a filled disc needs an outer-edge fit (all-pixel fit gave 33 px
  for a 55 px disc); **wooden-frame speckles** out-sized small dots → search restricted to the paper
  quad; **the ball covering a small dot** biased the fit (9.0 mm read as 5.9–7.4 mm, centre 3 mm
  off) → occluded views ignored, last clean goal reused.
- **Camera problems:** requesting **55 fps** gave ~1 s image delay after a USB re-plug (43 fps with
  internal buffering) — **30 fps gives ~0.15 s**. The camera index moves (0↔1) on re-plug →
  auto-select by resolution. The camera **froze twice** (listed by macOS, no frames) — needs a
  physical re-plug; a watchdog now reopens it while running.
- **Detection glitch:** a resting ball “jumped” 11 mm in one frame; the phantom ~0.3 m/s velocity
  made the D-term slam the plate and threw the ball into a corner → jump filter.

### Actuation / dynamics
- **Axes are swapped vs. the old code:** motor 1 (alpha) moves the ball in **y**, motor 3 (beta)
  in **x** (tilt probe: alpha −2.6° → dy +0.04–0.05 m, dx ≈ 0).
- **Open-loop tick→angle map is not repeatable** on motor 3 (same tick read +5.4° and −1.7° at
  different times) → **closed-loop tilt servo** on camera-measured angles (`_servo_tilt`).
- **Level point:** camera-0° ≠ gravity level; fitted from ball acceleration vs tilt (r ≈ 0.98) at
  beta ≈ +2.4…+3.0°, alpha ≈ −0.8…−1.4°, **drifting 0.3–0.5° between runs** → offset + integral.
- **Velocity bug:** velocity was divided by 1/55 s while the loop ran at ~22 Hz → D-term 2.4×
  too strong (drove a ~1.45 Hz limit cycle). Fixed with real dt: 52–67 % → **88 %** in circle.
- **Tilt-servo gain 0.5 was above the stability limit** (~0.445 for a 2–3 step lag): the plate
  shook ±5° at ~2 Hz. 0.25 → jitter 4.98° → 0.19°, ball still 67 % of frames, **89 s** hold;
  0.15 after the 30 fps change (loop ~29 Hz, lag 4–5 steps).
- **Delay misalignment:** the servo compared the current target with a 4–5-step-old camera tilt
  (a −0.9° brake arrived as −2.1…−2.4°). Comparing with the target active when the frame was taken
  (Smith-predictor idea): beta tracking error 0.60 → **0.29°**.
- **Paper stiction:** a still ball sinks into the paper — breakaway **~1.5–2.4°** vs rolling
  **~0.3–0.6°** — causing stick-slip hunting and hops.
- **Local slopes:** the paper isn't flat (~0.5° at typical spots; plate map: local level
  −0.8…+1.0°, breakaway 0.7…3.7°). **Near the frame** the paper grips much harder (top edge
  3.4–3.9°, sometimes > 4.5°).
- **Two controllers on one U2D2:** one crashed (“multiple access on port”) and its shutdown
  switched torque off → the other showed “running” with dead motors → single-instance lock.

---

## 4. What worked (in order)

| # | Change | Effect (measured) |
|---|---|---|
| 1 | Cached outer corners, HSV from pixels, closed-loop tilt servo, axis fix, level offset | first stable balancing; 0 → 2.11 s holds |
| 2 | Integral term (bias), ball hue 108 | **10.03 s** hold (Phase 3 success) |
| 3 | Real-dt velocity | 88 % in circle, median 26.6 mm |
| 4 | Servo gain 0.25 | 89 s hold, plate jitter 0.19° |
| 5 | **Ramp-kick & release** (integral learns only while rolling; kick dropped on motion) | **99.4 %**, median 4.4 mm |
| 6 | Done-zone 8 mm (don't kick a resting ball) | **median 2.6 mm, 260 s** |
| 7 | 30 fps camera + servo gain 0.15 | back to 98.4 %, 150 s after the camera delay problem |
| 8 | Auto goal detection, live sheet swap, any-shape regions (distance-transform deepest point) | new sheets without code edits |
| 9 | Capped approach speed (8 cm/s) | small dot: 22–28 % → 40 % in target |
| 10 | Delay-aligned servo | small dot: 40 → **58 %**, median 2.1 mm |
| 11 | **Timed pulse + planned brake** (Yang & Tomizuka adaptive pulse width, per 3 cm cell, persisted) | small dot: **82 %**, exits 11 → 2.2/min |
| 12 | Pulses anywhere (+ stronger ceiling 0.9) | big circle 100 %, 0 exits; small open-space target 100 %, 0 kicks |
| 13 | **Planned trapezoid moves** + friction feedforward | one smooth motion instead of hops |
| 14 | **Learned local level per cell** (`bias_table.json`) + glitch filter | 6 trips: 4 in one motion, 98–100 % in target after arrival in 5/6 |
| 15 | Web UI: sheet / click / line / draw-path modes, controller toggle | click → ball there in < 2 s |
| 16 | **Smooth line following** (moving reference, tangential + centripetal feedforward) | 8.7 laps/min, 6.6 mm (was 1.1 laps/min) |
| 17 | **RL**: calibrated sim, PPO v3 with policy rate limit | zero-shot on the rig (below) |
| 18 | **Hybrid**: RL approach, classic near-field settle (handover 2.5× radius, min 2.5 cm) | in testing with the user |

### RL vs classic on the rig — 8 random trips (seed 2027)

| Trip | Target r | Classic: arrived / in target | RL v3: arrived / in target (final) |
|---|---|---|---|
| 1 | 8 mm | 2.4 s / 100 % | 1.4 s / 98 % (7.0 mm) |
| 2 | 12 mm | 6.0 s / 37 % (6 moves) | 1.2 s / 1 % (13.7 mm, just outside) |
| 3 | 8 mm | 2.8 s / 100 % | 1.2 s / 100 % (3.4 mm) |
| 4 | 12 mm | 4.1 s / 91 % | 1.4 s / 100 % (2.6 mm) |
| 5 | 8 mm | 3.4 s / 96 % | 1.6 s / 12 % (8.5 mm, just outside) |
| 6 | 12 mm | 0.9 s / 68 % | 1.0 s / 100 % (3.0 mm) |
| 7 | 8 mm | 1.2 s / 95 % | never arrived (stuck 31 mm away) |
| 8 | 12 mm | 5.2 s / 66 % | 1.9 s / 54 % (12.4 mm) |

**Reading:** RL approaches faster and in one motion every time (1.0–1.9 s, no hops); classic
settles more precisely but needs several moves on half the trips. RL's weaknesses are the last
millimetres (no slope memory, can't nudge a stuck ball) and near-field jitter (action jerk
~0.015 vs ~0.001–0.01 classic) — hence the hybrid.

---

## 5. What failed or was reverted (and why)

| Tried | Outcome | Why / what we learned |
|---|---|---|
| Unstick/“violent shake” escapes | removed | only needed for the old glass ledge; vibration can drop the U2D2 |
| Symmetric open-loop tick map | replaced | tick→angle not repeatable → closed loop |
| `MAX_CORNER_AREA` diagnosis | **wrong**, reverted | areas measured at 1920×1080, pipeline runs at 640×360 |
| Blaming a slipping motor-3 linkage | **wrong call**, retracted | user: hardware is sound; the effect was handled in software (closed-loop tilt) |
| `I_MAX` 0.3 → 0.6 | reverted | stick-slip: breaks away at 1.5–2 deg and overshoots 50–80 mm; 88.8 → 76.8 % |
| Capture zone with KD 6 near target | reverted | no gain (23 % vs 30 %); overshoot came from kick energy, not damping |
| Plate-map feedforward | off by default | made the near-edge dot worse; map noisy near edges |
| Braking-curve speed limit alone | no fix | the overshoot wasn't arrival speed but edge-zone stiction + wall slams |
| Full-tilt (±1.0) wall guard | replaced | launched the ball at 140–160 mm/s → recurring ~46 mm overshoots |
| Gentle wall guard v1 | bug, fixed | it overrode stiction pulses → ball stuck at a sticky edge |
| Tilt-jump rejection filter | reverted | rejected real readings; servo chased a stale angle → side-to-side swinging |
| “Newest frame” drain | reverted | not the cause; the delay was the camera's 55 fps mode |
| Carrot line follower | replaced | stick-slip hops at ~5 mm/s |
| Delay-compensated Kalman filter | parked | better moving-ball prediction, but noisier at rest (6–8 vs 2.7 mm/s) — no stiction model |
| Small dot **near the frame** (≤ 5 cm) | unsolved | 0–93 % depending on start; paper grips 2–3× harder there → **scope: targets ≥ 5 cm from the frame** |
| First plate-map run | 4/30 points | simple positioning couldn't place a sticking ball; fixed by measuring where it stops (21 points) |
| RL v1 (smooth coef 0.05) | not deployed | 90 % success but bang-bang (jerk 0.16–0.28) |
| RL v2 (coef 1.0) | not deployed | 95 % success but still dithered (jerk ~0.16) — dithering beats stiction in sim |

---

## 6. Methods

- **Measure first, one change at a time**, each judged on a long run with the same metrics
  (in-circle %, median distance, longest hold, exits/kicks per minute, time to arrive).
- **Multi-agent workflows** (analysts + skeptic) on run logs chose several changes (level
  refit, servo gain, ramp-kick design panel, literature sweep).
- **Literature used:** Yang & Tomizuka 1988 (adaptive pulse width under stiction); van de Wouw &
  Leine 2012 (impulsive control with uncertain friction); Beerens et al. 2019 / Bisoffi et al. 2020
  (reset-integral against stick-slip); Armstrong-Hélouvry et al. 1994 (friction survey); Smith
  predictor / delayed-measurement alignment; Bi & D'Andrea, *CyberRunner* (ICRA 2024 — conditioning
  on past actions for delay).
- **RL:** `rl_sim/plate_sim.py` (tilt servo lag, camera delay/noise/dropouts, rolling friction,
  rest-dependent stiction, random slope field, walls; domain-randomised), fitted by replaying rig
  logs (`fit_plate_sim.py`); PPO (stable-baselines3, 8 envs) with the last 4 actions in the
  observation and a hard 0.1/step policy rate limit; exported to numpy so the rig needs no PyTorch.

---

## 7. Current state

- **Controllers:** classic (planned moves + near-field stiction handling + learned slopes);
  learned RL v3; **ODIL v6 + stiction compensation**; each learned controller optionally with the
  **classic near-field settle** (hybrid) — selectable in the UI.
- **UI modes:** red region on sheet, click to target, follow red line, draw path (runs in drawing
  order), **drop into hole** test; hole & reload panel; elevator panel (speed remembered, Max);
  green target circle for every target; view-only phone page (`rl_hw/remote_view.py`).
- **Lost ball:** plate levelled on the camera-measured angle, elevator (ID 2) forward at 328 until
  the ball is seen again, then off; alert (desktop + log → phone push) after 20 s.
- **Safety/robustness:** single-instance lock, camera watchdog, glitch filter, proportional wall
  guard, no shaking; tilt servo refuses to correct on a frame with a hidden plate marker and stops
  pushing a motor whose travel does not move the measured angle; verified elevator stop on exit.
- **Local learned files (not in git):** `bias_table.json`, `pulse_table.json`, `last_goal.json`,
  `last_level_position.json`, `holes.json`, `elevator_settings.json`. Policies in git:
  `rl_sim/runs/plate_goal_v3/policy.npz` (RL), `rl_sim/runs/odil_best/odil_policy.npz` (ODIL v6).

**Known limitations:** targets must be ≥ ~5 cm from the frame; motor 3's link has play (its
level point moved ~600–1000 ticks after the servo was driven into its end stop; the camera
closed loop absorbs it, but a stored motor position is never "level"); the ball can lie still
in the box under the plate after a hole drop (the alert catches it).

## 8. Day 2 (2026-09-27, afternoon): holes, reload, ODIL that works

### Holes and automatic reload
- **Hole detection** (`rl_hw/hole.py`): the plate's hole shows as the only very dark blob on the
  paper (V < 70 vs paper ~200); position/radius in plate metres, `holes.json`.
- **Keep-out zone** (hole + 8 mm): clicks inside are refused; moves and drawn paths go round it on
  an arc of via points (each chord provably clear of the zone).
- **Drop test** (UI): aims at the hole on purpose; the ball falls, the elevator reloads it
  (first test: back after 18.9 s), and the previous target resumes.
- **Lost-ball rule (user):** plate level + elevator on while the ball is missing, off when seen.

### The levelling incident — what it taught
A plate marker was briefly hidden → the pose (and so the tilt reading) was wrong by degrees → the
closed-loop "level" drove both tilt motors to their tick limits (~5° real tilt) → the ball rolled
onto a corner marker, keeping the reading wrong; the reload waited for a ball "away from the
edge" and the elevator kept running; the shutdown stop was lost mid-packet. Afterwards motor 3's
level point had moved by ~1000, then ~600 ticks between two sweeps (its link to the plate slips
under that load). **Lessons (all in the code now):** level on the camera angle, never on a stored
motor position (user); never correct on a frame with a missing marker; stop pushing a motor whose
travel does not change the measured angle (this is what makes a slipping link harmless); a reload
ends when the ball is seen anywhere; verify that the elevator really stopped.

### ODIL: from 56 % to on par with RL
| Version | Change | Sim: reached / inside after / exits / jerk |
|---|---|---|
| v4 (rig so far) | smooth static-friction model | 56 % / 43 % / 0.7 / 0.003 |
| v5 | + leaky integral of the goal error (state + input), near-goal starts, per-trajectory physics | 98 % / 46 % / 4.9 / 0.009 (limit cycle) |
| v6 | + randomised 3-stage delay, policy sees only an observer (output feedback), gentle near field | 84 % / 60 % / 2.6 / 0.006 |
| v7 | "rest anywhere inside" + radius input | 72 % / 44 % / 1.3 (stalls on stiction) |
| v9 | 5-stage delay (≈ pure delay) | 61 % / 45 % / 0.9 / 0.004 (too timid) |
| v10 (4-stage delay) + fc | between v6 and v9 | 93 % / 67 % / 2.1 / 0.004 |
| v6 far / v9 near, both + fc | two ODIL policies, switched near the target | 96 % / 67 % / 2.1 / 0.005 |
| **v6 + stiction compensation** | breakaway boost when the ball is still outside the target | **97 % / 69 % / 3.0 / 0.006** |
| RL v3 (reference) | PPO, trained in this simulator | 96 % / 72 % / 1.4 / 0.017 |

The trade-off that runs through all versions: robustness to the loop delay makes the policy gentle,
and a gentle policy cannot break paper stiction; a plain friction-compensation add-on (the classic
controller's pulse idea) resolves it. Velocity smoothing made it worse (the problem is delay, not
noise).

### Rig comparison (78 random targets, white paper, hole, taped right side)
| Setup | Reached | Taped side | Time to reach | Inside after (4 s) | Final distance |
|---|---|---|---|---|---|
| Classic | 18/18 | 5/5 | 1.5 s | 70 % | 7.5 mm |
| RL + settle | 20/20 | 6/6 | 1.3 s | 78 % | 6.0 mm |
| ODIL v6 + fc (pure) | 20/20 | 6/6 | 1.9 s | 67 % | 10.8 mm |
| ODIL v6 + fc + settle | 20/20 | 4/4 | 1.3 s | 77 % | 6.1 mm |

Before v6, pure ODIL (v4) reached 22/31 targets and 7/11 on the tape; RL alone 21/25 and 4/8 —
the tape (different friction) is where a controller without error memory fails.

**Smoothness on the rig** (second round, 40 trips, jerk = mean squared change of the applied
command per step, as in the simulator; one RL trip lost to the hole):

| Setup | Reached | Time to reach | Inside after | Final distance | Jerk (mean) |
|---|---|---|---|---|---|
| Classic | 10/10 | 1.8 s | 82 % | 6.0 mm | 0.0074 |
| RL + settle | 9/9 | 1.2 s | 77 % | 6.1 mm | 0.0074 |
| **ODIL v6 + fc (pure)** | 10/10 | 1.6 s | 70 % | 7.8 mm | **0.0039** |
| **ODIL v6 + fc + settle** | 10/10 | 1.6 s | **87 %** | 6.0 mm | 0.0059 |

Pure ODIL moves the plate about **half as jerkily** as the classic controller or RL + settle,
and ODIL + settle had the best time-in-target of this round. (10 trips per setup: indicative.)

### Large rig comparison (120 random targets, 30 per setup)
Same random targets, white paper with hole and taped side, after the marker-hold and
angle-guard fixes. "Inside after" = share of the 4 s after arrival spent inside the 12 mm
target (mean ± 95 % CI); jerk = median per trip.

| Setup | Reached | Taped side | Time to reach | Inside after | Final distance | Jerk (median) |
|---|---|---|---|---|---|---|
| Classic | 30/30 | 9/9 | 1.9 s | 0.56 ± 0.10 | 7.0 mm | 0.0095 |
| RL + settle | 30/30 | 6/6 | 1.2 s | 0.68 ± 0.10 | 7.1 mm | 0.0107 |
| **ODIL v6 + fc (pure)** | **29/29** | 6/6 | 1.6 s | **0.68 ± 0.05** | 9.2 mm | **0.0039** |
| ODIL v6 + fc + settle | 30/30 | 6/6 | 1.6 s | 0.68 ± 0.11 | 7.1 mm | 0.0083 |

**Result:** pure ODIL keeps the ball in the target as well as RL + settle and ODIL + settle
(and more consistently: the narrowest interval), and moves the plate **~2.5x more smoothly**
than every other setup. RL + settle is the fastest. (One ODIL trip excluded: ball lost.)

### Rig-fitted ODIL (v11) and drawing accuracy
- **System identification from ~30 min of rig logs** (`rl_sim/sysid_rig.py`): command -> plate angle =
  1-2 frames of pure delay + a 90-100 ms lag; tilt -> acceleration 0.113 m/s^2 per deg (as assumed);
  rolling friction 0.042 m/s^2 (70 % above the assumed 0.025).
- **ODIL v11** = v6 trained with these values. Rig, pure ODIL: v6 (29 trips) 0.68 ± 0.05 inside,
  9.2 mm, jerk 0.0039 -> **v11 (20 trips) 0.73 ± 0.07 inside, 7.8 mm, jerk 0.0036**; now the default.
- **Drawing (maze feasibility), classic path follower with corner braking:** circle at 4 / 2 /
  1.2 cm/s -> median 9.4 / 8.6 / 4.3 mm off the line, 90 % within 17.6 / 15.7 / 17.9 mm; square
  and star similar. Slower lowers the median but not the tail (stick-slip hops). Not yet maze-ready;
  next: ODIL trained to follow a moving reference, and routes that use the walls.

### ODIL path tracking (for drawing -> maze)
- **ODIL tracking policy** (`rl_sim/odil_track.py`): the ball follows a moving reference (the rig's
  PathTracker); trained on random arcs / straights / corners / waves at 1-4 cm/s with the rig-fitted
  model. Alone it fits its optimised trajectories (0.7 mm) but drifts in closed loop (even in its own
  model: median 7 mm, p90 30 mm) -- the discrete loss never shows it its own compounding errors.
- **Closed-loop refinement** (`rl_sim/finetune_track.py`): the ODIL policy is the initial policy and
  is refined by backpropagation through closed-loop rollouts in a differentiable model with the
  rig-identified timing (1-3 steps pure delay + 60-110 ms lag, 1 frame camera delay, noisy finite-
  difference velocity, rate limit, smooth static friction). Sim test: completes 93-100 % of drawings
  (classic line law 33-93 %, it stalls on stiction), p90 ~10 mm (8-25), max 26-36 mm (41-60).
- **Rig, 9 drawings each (circle / square / star, 3 cm/s, 3 rounds):**

| Follower | Median of medians | Mean p90 | Mean max | Worst max |
|---|---|---|---|---|
| Classic line law | 4.2 mm | 13.1 mm | 20.3 mm | 36.8 mm |
| **ODIL + closed-loop refinement** | **3.6 mm** | **8.4 mm** | **15.0 mm** | **19.1 mm** |

  Not yet maze-grade (goal: p90 <= 5 mm), but the tail -- what drops a ball into a hole -- is
  roughly halved.

## 9. Recommended next steps

1. **Pure ODIL near-field precision** (10.8 mm vs ~6 mm for the hybrids): v10 (4-stage delay,
   more decisive near field) and a two-policy ODIL (v6 far, v9 near: sim exits 3.0 → 2.1).
2. **Fit ODIL's model to measured trajectories** (rig logs) instead of hand-set ranges.
3. **The hole as a constraint inside the ODIL optimisation** (smooth planned paths round it).
4. Showcases: writing/drawing with the ball; automatic self-calibration; maze with ODIL.
5. User's feature list: coloured dots, UI extras, smoother line tracking on tight curves.
