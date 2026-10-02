# ODIL on the CyberRunner rig — everything we did

*Branch `ball-on-plate`. Covers 2026-09-26 to 2026-09-29. Written for future Claude Code sessions
and for the paper. Companion documents: [`BALL_ON_PLATE_REPORT.md`](BALL_ON_PLATE_REPORT.md)
(the rig, vision, classic controller, RL), [`PUBLICATION_ASSESSMENT.md`](PUBLICATION_ASSESSMENT.md),
[`../rl_hw/HOW_TO_RUN.md`](../rl_hw/HOW_TO_RUN.md). A formatted version with figures, videos and the
full mathematics is in Claude Docs: <https://claude.ai/code/artifact/df0c16d8-4fac-42ea-b28b-8884d6961e32>.*

---

## 1. One-paragraph summary

ODIL ("Optimizing a Discrete Loss") trains a controller by gradient descent through the ball's
physics equations instead of by trial and error. On this rig it is **as accurate as the RL (PPO)
controller and about 2.5x smoother** (120 random trips), with far less training and no GPU. Extended
to follow a moving reference, it draws shapes and follows the printed line of the real labyrinth
board: **the rig has completed the real maze 3 times** (first on 2026-09-28, 100 % of the 1.93 m
route in 172 s). A typical run still reaches about a quarter of the route. A learning loop
(drive -> learn a world model -> practise in it -> drive) exists but **has not yet shown an
improvement** on the real board.

## 2. What ODIL is here

- **Shared physics model.** A rolling ball accelerates at `k * tilt` (k ~ 0.113 m/s^2 per degree,
  5/7 g for a solid rolling ball), minus rolling friction, with static friction ("stiction") that
  holds a resting ball until about 1.5-2 degrees. The plate follows the command with a pure delay
  of 1-3 control steps plus a first-order lag (60-110 ms). Control runs at ~29 Hz.
- **ODIL.** The discretised physics is written as a loss; states, commands and a small policy
  network are optimised jointly so that the trajectory obeys the physics, reaches or tracks the
  target and moves smoothly. The gradient is exact (through the equations), unlike RL, which
  estimates it from sampled rewards.
- **Policy.** A small MLP. Balancing: inputs from an observer of ball error/velocity; tracking:
  error to the reference, velocity error, reference acceleration, two filter states; output
  `4 deg * tanh(.)`.
- **Closed-loop refinement.** An ODIL policy fits its own optimised trajectories but drifts in
  closed loop (it never saw its compounding errors). It is therefore used as the initial policy and
  refined by backpropagation through closed-loop rollouts in a differentiable model with the
  rig-identified delay, lag, camera noise, rate limit and friction.

Code: `rl_sim/odil_plate*.py` (balancing, versions v4-v12), `rl_sim/odil_friction_comp.py`,
`rl_sim/odil_track.py` (tracking policy `TrackPolicy`), `rl_sim/finetune_track.py` (refinement),
`rl_sim/sysid_rig.py` (fit the model to rig logs), `rl_sim/world_model.py` (learned world model),
`rl_sim/eval_*.py`. Policies: `rl_sim/runs/odil_best` (v6), `odil_v11` (rig-fitted, default for
balancing), `odil_track_best`, `odil_track_maze` (the one that solved the maze).

## 3. Balancing: ODIL vs RL vs classic

### Versions (simulation, 200 identical episodes)

| Version | Change | Reached / inside after / exits / jerk |
|---|---|---|
| v4 | smooth static-friction model | 56 % / 43 % / 0.7 / 0.003 |
| v5 | + leaky integral of the goal error, near-goal starts, per-trajectory physics | 98 % / 46 % / 4.9 / 0.009 (limit cycle) |
| v6 | + randomised 3-stage delay, policy sees only an observer, gentle near field | 84 % / 60 % / 2.6 / 0.006 |
| v7 | "rest anywhere inside" + radius input | 72 % / 44 % / 1.3 (stalls on stiction) |
| v9 | 5-stage delay | 61 % / 45 % / 0.9 / 0.004 (too timid) |
| v10 + fc | 4-stage delay | 93 % / 67 % / 2.1 / 0.004 |
| **v6 + stiction compensation (fc)** | breakaway boost when the ball is still outside the target | **97 % / 69 % / 3.0 / 0.006** |
| RL v3 (PPO, reference) | trained in the same simulator | 96 % / 72 % / 1.4 / 0.017 |

The trade-off through all versions: robustness to the loop delay makes the policy gentle, and a
gentle policy cannot break stiction. A plain friction-compensation add-on resolves it. Velocity
smoothing made it worse (the problem is delay, not noise).

### Rig, 120 random targets (30 per setup, white paper, hole, taped side)

| Setup | Reached | Time to reach | Inside after (4 s) | Final distance | Jerk (median) |
|---|---|---|---|---|---|
| Classic | 30/30 | 1.9 s | 0.56 +- 0.10 | 7.0 mm | 0.0095 |
| RL + settle | 30/30 | 1.2 s | 0.68 +- 0.10 | 7.1 mm | 0.0107 |
| **ODIL v6 + fc (pure)** | **29/29** | 1.6 s | **0.68 +- 0.05** | 9.2 mm | **0.0039** |
| ODIL v6 + fc + settle | 30/30 | 1.6 s | 0.68 +- 0.11 | 7.1 mm | 0.0083 |

Pure ODIL holds the target as well as RL + settle, more consistently, and moves the plate about
2.5x more smoothly than every other setup. RL + settle is the fastest.

### Rig-fitted ODIL (v11)

System identification from ~30 min of rig logs (`sysid_rig.py`): command -> plate = 1-2 frames
pure delay + 90-100 ms lag; 0.113 m/s^2 per degree; rolling friction 0.042 m/s^2 (70 % above the
assumed value). v11 = v6 trained with these values: 0.73 +- 0.07 inside, 7.8 mm, jerk 0.0036
(20 trips) -- the balancing default.

## 4. Path tracking and drawing

Rig, 9 drawings each (circle / square / star, 3 cm/s):

| Follower | Median of medians | Mean p90 | Mean max | Worst max |
|---|---|---|---|---|
| Classic line law | 4.2 mm | 13.1 mm | 20.3 mm | 36.8 mm |
| **ODIL + closed-loop refinement** | **3.6 mm** | **8.4 mm** | **15.0 mm** | **19.1 mm** |

The tail (what drops a ball into a hole) is roughly halved. In the controller
(`rl_hw/pd_balance.py`) the ODIL tracker is used in line / path / maze mode with a moving
reference from `rl_hw/line_path.py` (`PathTracker`: speed limited by curvature, waits when the
ball lags by more than 25 mm).

## 5. Maze on white paper (virtual holes and walls)

The printed route was extracted from a photo (`rl_hw/maze_capture.py`, `maze_route.py` ->
`maze/route.json`) and practised on paper with virtual holes and walls (a "fall" or wall crossing
restarts the run). 1259 runs: best 98 %, mean 18 %. Per-route iterative learning control (ILC)
diverged at gain 0.5 and did not help at 0.2; slow zones down to 0.3x caused stick-slip. **With
both off ("learning off") ODIL averaged about 33 %, best 98 %.**

## 6. The real maze board (from 2026-09-28)

- **Result:** 3 complete runs out of 167 real-maze runs: 172 s (no retries), 267 s (7 forward
  retries), and a third (duration not reliable, see logs). Videos in `maze/recordings/`
  (`run_20260928_230306_odil_100pct.mp4`, `run_20260929_105628_odil_100pct.mp4`,
  `run_20260929_113642_odil_100pct.mp4`); runs >= 70 % are recorded automatically
  (`rl_hw/run_recorder.py`). In the first finish the ball stayed 2.4 mm (median) / 5.0 mm (p90)
  from the line.
- **Typical run:** baseline tracker, 126 runs: mean 26 %, median 25 % of the route.
- **Why most runs fall** (`rl_hw/maze_falls.py`, 75 falls): outside the first hole the ball is
  2-3x too fast (50-70 mm/s vs the 25 mm/s reference) and overshoots bends by 11-17 mm into a
  hole. The route passes some holes within a few mm of the edge. Falls cluster near a handful of
  holes (around 7 %, 21 %, 25 %, 30 % of the route).

### Fixes the real walls and holes required (all in `pd_balance.py`, `line_path.py`, `maze_practice.py`)

| Problem | Fix |
|---|---|
| All runs stuck at 12 % (hairpin) | the reference stays in straight-line sight of the ball (`los_tol` 3 mm) |
| Ball pressed into a wall at full tilt | breakaway boost points from the ball to the reference, not along the tangent |
| Ball resting against a wall end | jolts: straight, then 50 deg to either side |
| Stall not detected | stall = moved < 3 mm in 4 s (the speed estimate jitters on a resting ball) |
| Ball thrown into a hole by a jolt | within 12 mm of a hole: no tilt-back, nothing towards the hole, softer |
| Driving back to the start through walls | removed: retries go forward only; the start is reached via the elevator |
| Acting on a frozen camera | no new frame -> no control: plate held, elevator off, alert |
| Tracker locked on the elevator bracket | a ball on the board beats a ball-coloured spot off it |
| False holes (bar ends, the ball in the capture photo) | `maze/false_holes.json`; route hole removed |

### Tried and dropped
Tracker gain x0.7 (all runs fell at 3-7 %); an "outlet rock" during reloads (user: no); wall
penalty in the refinement (walls are meant to be used).

## 7. Learning loop (`rl_hw/learn_loop.py`)

Round = drive 30 runs -> fit the world model on the logs -> refine the tracker in it -> drive.

- **World model** (`rl_sim/world_model.py`): the physics above with learnable parameters plus a
  small network that adds a bounded correction from position, velocity and the last 4 commands;
  multi-step loss over 10 frames. After one round of rig data: 0.35 s prediction error 7.5 -> 4.1 mm.
- **First refinement failed:** mean 29 % -> 5 %, and the policy drove the motors to their caps
  (it exploited errors of a model trained on one round).
- **Now:** a new policy is accepted only if its first 6 runs reach >= 70 % of the accepted
  policy's mean; 200 iterations, learning rate 1e-4, effort penalty 2, jerk weight 60, no wall
  penalty; the policy is swapped into the running controller (no restart).
- **Status:** no demonstrated improvement yet. This is the open research question.

## 8. Reliability tools that work for any maze

- `rl_hw/maze_walls.py` -- dense wall map (1 mm) from the board photo (glossy bars, board edge).
- `rl_hw/maze_plan.py` -- safe route planner (Dijkstra, cost rising near hole edges, targets the
  ball's centre can reach). Without `--corridor` it finds the board's physical shortcuts (0.46 m
  instead of 1.93 m); `--corridor 12` stays on the game's path but on its safest side
  (`maze/route_safe.json`, 1.58 m, with gentler bends than the printed line).
- `rl_hw/maze_falls.py` -- fall diagnosis per hole from the logs.
- Speed governor (`PD_MAZE_VCAPS`): brakes when the ball exceeds a cap (40 mm/s tested).
- A/B by alternation: `PD_MAZE_ROUTES`, `PD_MAZE_VCAPS`, `PD_MAZE_AB_T0` rotate route x cap run
  by run; each run logs its variant in `phase3_logs/maze_runs.jsonl`.
- **2x2 test (printed vs safe route, cap off vs 40 mm/s): inconclusive.** The ~25 runs collected
  overlap with a tilted plate and two motor runaways. All 3 finishes came from the printed line
  without a cap. Needs a clean rerun.

## 9. Safety lessons (read before touching the rig code)

1. **Never loosen or re-centre the motor-travel cap.** Two runaways on 2026-09-29: (a) a change
   that let the cap follow the camera during levelling wound the motors 3-9 turns on a bad
   reading; (b) the re-level handler re-centred the cap on a capped motor after every failed
   re-level (-29 deg, motor 1 overload shutdown). Both fixed: only a camera-confirmed level moves
   the cap reference.
2. **Never drive the motors to a stored or estimated position.** Startup reads the present
   positions and starts there (a cached "last level" undid a hand-levelled plate).
3. **No control on a stale frame**; no correction without a fresh camera pose.
4. **A hard kill leaves the elevator running** and the motors powered: stop the elevator and
   release torque explicitly.
5. Both tilt motors are in Dynamixel extended-position (multi-turn) mode; their absolute counts
   accumulate and mean nothing by themselves -- only the camera angle does.
6. Elevator: 15 s per reload, plate level, no tilt; it stops after 3 failures in a row.

## 10. User rules for this project

- Ask before writing Dynamixel registers, pushing to GitHub (pushes were authorised repeatedly
  in these sessions), or deleting files.
- Never assume a hardware problem; solve in software, and say plainly what is measured.
- Solve a stuck ball autonomously (jolts are fine), but never tilt while the elevator runs.
- On the real maze: no driving back to the start, walls are to be used, record runs >= 70 %.
- Phone alerts for anything that needs the user. Git author: `dimikoumou <dimi0330@icloud.com>`.

## 11. State when this was written

The rig is **stopped**; all three motors are released. Motor 1 reported an overload shutdown and
needs a power cycle or reboot; the plate must be levelled by hand before the controller starts
(start with a watchdog on motor travel). The learning loop and the 2x2 test are paused.

## 12. Next steps

1. Clean rerun of the 2x2 test (route x speed cap), 25+ runs per cell.
2. Learning loop with the winning setup; show (or rule out) an improvement over the hand-built
   tracker -- the headline result if it works.
3. Teach-and-repeat: use the tilts of the complete runs as a feedforward plan along the route.
4. Speed runs (the original CyberRunner claim) once finishes are frequent.
5. Free planning on a board without a printed line (the planner already finds the shortcuts).
