# CyberRunner — Project Report

What this repo is and what's actually been done on it, reconstructed from git history and
the code itself.

## What CyberRunner is

A physical ball-balancing / maze-solving robot: a camera watches a board tilted by two
Dynamixel servos, and a controller (hand-tuned or learned) drives the tilt to move a
loose steel ball — either around a maze insert to a goal, or (a simpler sub-task) into
and holding it inside a drawn goal circle on a flat plate. This repo is a fork of
[`kro0l1k/cyberrunner_basics_no_ros`](https://github.com/kro0l1k/cyberrunner_basics_no_ros),
itself a simplified, ROS-free Python port of the original CyberRunner platform (a
real-world reinforcement-learning benchmark). The end goal, per `rl_sim/README.md`'s own
roadmap, is sim-to-real: train an RL agent to solve the maze in simulation, then transfer
the policy to the real board.

## Where this lives in git / GitHub

Three lines of work, on three branches of the fork (`dimikoumou/cyberrunner_basics_no_ros`):

| Branch | What's on it |
|---|---|
| `main` | Just the initial fork commit — kept clean/untouched. |
| `testCode` | The 2025 vision-pipeline work (below). |
| `phase1-python-port-fixes` | The 2026 port fixes, the RL simulator, and the hardware ball-balancing work (below). |

## Phase 1 — Vision pipeline (`testCode` branch, Jul–Sep 2025)

Nine commits, all yours, building the camera → ball/plate state pipeline from scratch on
top of the forked skeleton:

- Integrated **pyOCamCalib** (fisheye camera calibration) to replace the original MATLAB
  OCamCalib workflow, and adapted its calibration output into the format the existing
  state-estimation code expects (`todo.txt` documents working through this by hand —
  understanding the `cam2world`/`world2cam` polynomial model, bridging pyOCamCalib's JSON
  output to the row-based calibration file format).
- Got state estimation working end-to-end against the new calibration.
- Added a **geometric restriction** to ball detection — only search for the ball within
  the region bounded by the inner corner markers — cutting false-positive detections, plus
  calibration improvements.
- Tuned and re-tuned the HSV colour thresholds for the corner markers twice (measured
  against real captured frames, not guessed).
- Wrote `spin_motors.py` as a first hands-on pass at driving the Dynamixel motors from
  Python.
- Built and then cleaned up a substantial debug-image toolkit (per-corner mask/crop
  visualizations) while chasing detection failures, documented with its own markdown
  notes per script.

## Phase 2 — Port fixes + RL in simulation (`phase1-python-port-fixes`, Jul 2026)

**Phase 1 fix** (commit `29c3f95`): six real, previously-latent bugs found and fixed in
the no-ROS Python port of the state estimator:
- `capture_frame.py` was capturing at 720p while the OCamCalib model was calibrated at
  1920×1080 — the actual root cause of the recurring "Unable to find corner" failures
  from the 2025 debugging notes, not a detection-algorithm problem.
- A `mask is None` check had been written as `mask.all() == None` (always `False`),
  silently letting empty masks through.
- A missed corner detection called `exit()`, killing the whole process instead of
  reporting "not found" and continuing.
- An unconditional-but-actually-conditional variable definition (`radius_int`) crashed
  with `UnboundLocalError` on the non-verbose code path.
- Added resolution and `color_params=None` guards so a mismatched frame size fails
  loudly with a clear error instead of silently producing garbage state estimates.
- Added `run_state_estimation.py`, a clean hardware-free entry point for testing the
  pipeline against a saved image.

**RL simulation** (commits `8fb5f47`, `a84550e`), built from scratch specifically so an
agent could start learning before hardware access: a pure-NumPy ball-on-tilting-plate
physics model, a Gymnasium environment whose observation/action shapes are *designed to
match the real hardware interface* (so a policy trained here can transfer), and both SAC
and a multi-core PPO trainer. Result: the PPO agent went from 0% to 100% maze-solve
success by ~450k steps, then kept optimizing for speed down to **~279 steps per solve —
faster than the hand-scripted PD baseline's ~379 steps.** That's a learned policy beating
a designed controller, on a task built specifically to mirror the real board.

## Phase 3 — Hardware ball-balancing (`rl_hw/`, Aug–Sep 2026)

The most recent work, and the only phase that touches the real robot end-to-end. Goal:
validate the full camera → estimate → control → motors loop on the simplified
flat-plate-balancing task (no maze) before attempting the harder sim-to-real maze
transfer.

**Real bugs found and fixed** (each confirmed by direct measurement, not assumption —
repeatedly, what looked like "the hardware is broken" turned out to be config or logic):

- **Wrong motor mapping.** `state_est_control.py` assumed Dynamixel IDs `[1, 2]` were the
  two tilt axes; ID 2 is actually an unrelated ball-reload elevator motor that never
  moved the plate. Confirmed by checking whether position readback actually responded,
  found the real second tilt axis is ID 3.
- **Both servos' Position Integral Gain was 0**, causing a real steady-state droop under
  the plate's own weight that looked exactly like a stalling motor.
- `_write_action()` was mapping commanded tilt through a **stale, static calibration
  center** instead of the just-achieved level position — fighting its own corrections.
- **Motor3 (and Motor1, to a lesser extent) were pinned within ~5 ticks of the firmware's
  single-turn position ceiling** just to hold the plate level, silently eating almost all
  usable tilt range. Fixed with a homing-offset re-zero (a firmware register write, done
  with explicit approval, not a physical change) — recovered 1000+ ticks of real,
  previously inaccessible tilt authority.
- Corner-marker HSV threshold **re-measured directly** (pixel sampling on real frames)
  and widened after the old range turned out not to cover the real marker color.
- Camera calibration given a plausibility check + automatic retry (rejects and re-tries
  if the resulting pose doesn't look like a fixed overhead camera should).

**Control design.** `pd_balance.py` implements a *gain-scheduled* PD controller
(aggressive far from the goal, gentle near it) — not the obvious first guess. That came
from a real diagnosis: a single fixed-gain PD had a genuine bias problem, confirmed
directly — the ball would recover to near the goal, then drift straight back to the same
corner over ~30 steps under nominal control. That's a persistent bias steadily beating an
over-softened corrector, not ordinary overshoot, and it's why the gain schedule exists.

**Isolating the control law from the hardware.** Live testing kept getting interrupted by
two physical issues unrelated to the control algorithm (below), so `sim_verify.py` runs
the *exact same* control law against a simple, physically-grounded double-integrator
model (real measured 11° max tilt, friction, sensing noise) with zero hardware
dependency. Result: **20/20 simulated episodes reached sustained (2s+) in-circle
balance** — confirming the control logic itself is sound, independent of the rig's
current physical state.

**Known, unresolved physical defects (need hands, not more code):**
- A raised wooden bezel ledge at (at least) two corners traps the ball. Exhaustively
  tested — steady pushes up to 20s, oscillation at multiple frequencies, orbital shake,
  torque-cycling "drop and catch" — nothing reliably frees it. Needs the paper trimmed
  flush or the corners shimmed.
- The U2D2 USB-serial adapter can drop mid-run under the vibration from the aggressive
  "violent shake" unstick maneuver (`termios.error: Device not configured`). Needs a
  cable reseat when it happens; `wait_and_run.sh` polls for the device to reappear and
  auto-launches the run so the session doesn't need to be babysat, but the root cause is
  physical.

## Phase 3 – result

- **Success: 10.03 s continuous hold inside the red goal disc** (2026-09-26, glass removed, bare paper). All 225 frames in the hold had the ball detected, mean distance 2.7 cm and max 4.27 cm against a 4.55 cm radius. Evidence: `phase3_logs/success/pd_20260926_144455.csv`, snapshots, `phase3_logs/hold_10s_evidence.png`.
- **What worked:**
  - **Vision:** outer 4 markers measured once and cached (`state_est/fixed_corners_cache.json`), inner 4 tracked live. Marker/ball HSV set from sampled pixels: corner hue 104–140; ball hue 80–108, saturation ≥ 100, which drops the ball's shadow and keeps it detected over the red disc.
  - **Actuation:** closed-loop tilt control on camera-measured angles (`_servo_tilt`) instead of fixed tick maps. Axis mapping measured: ball x ← beta (motor 3), ball y ← alpha (motor 1).
  - **Bias:** level offset fitted from log data (ball acceleration vs tilt), plus a slow integral term.
- **Final gains:** PD unchanged (KP_FAR/KD_FAR = 5/2, KP_NEAR/KD_NEAR = 2.5/4, clip ±0.8, max_action_delta 0.5). New: KI = 1.0 (integrates only within 0.10 m of the goal, capped at ±0.3). TILT_MAX = 5° per unit action. LEVEL_OFFSET (alpha, beta) = (−1.1°, +2.55°).
- **Progression (one change at a time):** 0 s (axes swapped, open-loop ticks) → 2.11 s (closed loop + axis fix + level offset) → 4.84 s (offset refit) → 9.15 s (integral term) → 10.03 s (ball hue 104→108, which removed ball-lost resets on the red disc).
- **Anything odd:** the true level point moves 0.3–0.5° between runs; the integral term absorbs it, and the x integral often sits near its cap. Velocity is computed with dt = 1/55 s while the loop runs at ~21 Hz, so the effective D gain is ~2.4× nominal (a ~1.45 Hz near-goal oscillation remains). Violent shake removed; a lost ball is recovered with gentle tilts. In-circle fraction on the success run was 52%, so a 10 s hold is not yet reliable.

## Results summary

| Task | Result |
|---|---|
| Maze-solving, simulation (PPO) | 0% → 100% success by ~450k steps; converges to ~279 steps/solve, beating the scripted PD baseline's ~379 |
| Ball-balance control law, pure simulation (`sim_verify.py`) | 20/20 episodes reach sustained (2s+) in-circle balance |
| Ball-balance, real hardware | Control loop and vision pipeline both verified working; full success currently blocked by two *physical* defects (bezel ledge, USB dropout), not by code |
| Vision pipeline | Fisheye calibration (pyOCamCalib) integrated; marker detection made robust (geometric restriction, twice-tuned HSV, 6 latent bugs fixed in the port) |

## What's not done yet

- The maze layout used for RL training is still a placeholder serpentine, not a traced
  layout of the real physical board (`rl_sim/README.md`'s own roadmap item 3).
- The sim-to-real transfer for the *maze* task (as opposed to the simpler flat-plate
  balancing task) hasn't been attempted yet.
- Swapping SAC for DreamerV3 (what the original CyberRunner project uses, GPU-dependent)
  is still on the roadmap, not started.
- The two physical hardware defects above are blocking a clean end-to-end
  hardware success on the balancing task.

## Repo map

```
capture_frame.py, camera_calibration_realtime.py, board_detection.py, motors_control.py
                        — original vision/motor utility scripts (upstream + Phase 1 fixes)
state_est/              — the state-estimation pipeline (detection, calibration, pose)
rl_sim/                 — pure-simulation RL: maze physics, Gym env, SAC/PPO trainers
rl_hw/                  — real-hardware RL/control: HardwarePlateEnv, pd_balance.py,
                           sim_verify.py, calibration/probing scripts
motor_calibration/      — Dynamixel position-limit calibration tooling and data
todo.txt                — running personal debug log from the 2025 vision-pipeline phase
```
