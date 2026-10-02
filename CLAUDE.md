# CyberRunner ball-on-plate / labyrinth rig

Read these before working here:

- **[`docs/ODIL_REPORT.md`](docs/ODIL_REPORT.md)** -- everything done with ODIL: balancing vs RL,
  path tracking, the real maze (solved 3 times), the learning loop, reliability tools, safety
  lessons, current state and next steps.
- [`docs/BALL_ON_PLATE_REPORT.md`](docs/BALL_ON_PLATE_REPORT.md) -- the rig, vision, classic
  controller, RL, root causes found.
- [`rl_hw/HOW_TO_RUN.md`](rl_hw/HOW_TO_RUN.md) -- how to run everything.

Branch: `ball-on-plate`. Git author for commits: `dimikoumou <dimi0330@icloud.com>`.

## Layout

- `rl_hw/` -- everything that touches the rig: `pd_balance.py` (controller + UI on :8000),
  `plate_env.py` (camera tilt servo, motor limits), `maze_*.py` (route, walls, planner, practice,
  fall diagnosis), `learn_loop.py`, `elevator.py`, `run_recorder.py`.
- `rl_sim/` -- training in simulation: `odil_*.py`, `finetune_track.py`, `world_model.py`,
  `sysid_rig.py`; policies in `rl_sim/runs/`.
- `state_est/` -- camera pipeline and Dynamixel helpers.
- `maze/` -- route files, wall map, recordings; `phase3_logs/` -- run logs.

## Hard rules (from the user)

1. Ask before writing Dynamixel registers, pushing to GitHub, or deleting files.
2. Never assume a hardware problem. Report what is measured and solve it in software.
3. **Never loosen or re-centre the motor-travel cap, and never drive the motors to a stored or
   estimated position.** Startup must begin from the motors' present positions. Two runaways on
   2026-09-29 came from breaking this (see the safety section of the ODIL report).
4. While the elevator runs the plate stays level: no tilt, no rocking. Elevator runs 15 s per reload.
5. On the real maze: never drive the ball back to the start, walls are to be used (no wall
   penalty), record runs of 70 % or more.
6. A stuck ball: solve it autonomously (jolts are fine, never towards a hole). Alert the user's
   phone for anything that needs them.
7. After a hard kill of the controller, stop the elevator (motor 2) and release motor torque
   explicitly -- they are left as they were.

## Before starting the controller

Check the camera delivers frames, that only one controller runs (one U2D2), and that the plate is
level. Start with a watchdog on motor travel (kill and release if a tilt motor moves more than
~1300 ticks from its start position).
