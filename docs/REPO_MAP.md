# Repository map

The project has three parts. Scripts stay where they are (`rl_sim/`, `rl_hw/`) because they find
each other through relative paths; this map says which part each one belongs to.

| Part | Question | Where |
|---|---|---|
| **1. Ball on plate: simulation -> rig** | Train controllers (classic, RL, ODIL) in a simulator of the plate, then run them on the real rig | `rl_sim/` (training) + `rl_hw/` (rig) |
| **2. Maze** | Follow lines, then the printed route of the real labyrinth board; learn from the rig | `rl_hw/maze_*`, `rl_sim/odil_track*`, `world_model.py`, `maze_world.py` |
| **3. Ball on plate: trained only on the rig** | Train ODIL and RL with nothing but real rig data; compare how much rig time each needs | `physical_training/` |

Reports: [`ODIL_REPORT.md`](ODIL_REPORT.md) (parts 1-2 and the ODIL story),
[`BALL_ON_PLATE_REPORT.md`](BALL_ON_PLATE_REPORT.md) (the rig, vision, classic and RL),
[`../physical_training/README.md`](../physical_training/README.md) (part 3).

## Shared rig infrastructure (all parts)

| File | Role |
|---|---|
| `rl_hw/plate_env.py` | camera pipeline -> ball and plate state; closed-loop tilt servo; motor caps and safety |
| `rl_hw/pd_balance.py` | the controller process: classic controller, policy runners, modes, web UI on :8000 |
| `rl_hw/ui_server.py`, `rl_hw/remote_view.py` | web UI and phone live view |
| `rl_hw/rl_policy.py` | runs exported RL and ODIL policies on the rig |
| `rl_hw/goal_circle.py`, `rl_hw/hole.py` | red goal regions, holes in the paper/board |
| `rl_hw/elevator.py` | ball-reload elevator (maze board only) |
| `rl_hw/calibrate.py`, `rl_hw/map_plate.py`, `rl_hw/ball_kf.py` | self-calibration, plate map, ball filter |
| `state_est/` | camera calibration, marker and ball detection, Dynamixel helpers |

## Part 1 -- ball on plate, trained in simulation, applied to the rig

| File | Role |
|---|---|
| `rl_sim/plate_sim.py`, `rl_sim/fit_plate_sim.py` | plate simulator, calibrated against rig logs |
| `rl_sim/plate_goal_env.py`, `rl_sim/train_plate_ppo.py` | RL (PPO) goal-reaching environment and training |
| `rl_sim/odil_plate.py`, `odil_plate_v5..v12.py` | ODIL controllers, version by version |
| `rl_sim/odil_friction_comp.py` | ODIL + stiction compensation (the version compared on the rig) |
| `rl_sim/sysid_rig.py` | system identification from rig logs (-> ODIL v11) |
| `rl_sim/eval_controllers.py`, `eval_hole.py` | controller comparisons in simulation |
| `rl_hw/sim_verify.py` | the classic controller's logic in simulation |
| results | `docs/ODIL_REPORT.md` sections 3-4, `docs/BALL_ON_PLATE_REPORT.md` |

## Part 2 -- maze (lines, drawing, paper maze, real board)

| File | Role |
|---|---|
| `rl_hw/line_path.py`, `rl_hw/shapes.py` | path following (PathTracker), shapes and letters to draw |
| `rl_sim/odil_track.py`, `rl_sim/finetune_track.py`, `rl_sim/eval_track.py` | ODIL path tracker and its closed-loop refinement |
| `rl_hw/maze_capture.py`, `maze_route.py`, `maze_walls.py` | board photo -> route, holes, wall map |
| `rl_hw/maze_practice.py` | maze runs (paper with virtual holes, or the real board) |
| `rl_hw/maze_plan.py`, `rl_hw/maze_falls.py` | safe route planner, fall diagnosis |
| `rl_hw/learn_loop.py`, `rl_sim/world_model.py` | drive -> world model -> practise -> drive |
| `rl_sim/maze_world.py`, `rl_sim/maze_eval.py` | simulator of the real board (walls, holes, variation) and whole-run evaluation |
| `rl_hw/run_recorder.py` | videos of good runs (`maze/recordings/`) |
| data | `maze/` (route, walls, recordings, report media), `phase3_logs/maze_runs.jsonl` |

## Part 3 -- ball on plate, trained only on the physical rig

| File | Role |
|---|---|
| `physical_training/rig_learn.py` | one process, one night: ODIL rounds (drive -> fit -> train -> test) and SAC from scratch, the same 30-target test for both |
| results | `phase3_logs/rig_learn.jsonl`, `physical_training/data/` |

## Older or one-off material

| File | What it was |
|---|---|
| `rl_sim/maze_sim.py`, `maze_env.py`, `maze_layout.py`, `train_ppo.py`, `train_sac.py`, `test_sim.py`, `record_baseline.py` | the first RL attempt on a generic placeholder maze (before the real board) |
| `rl_hw/train_sac_hw.py`, `smoke_test.py`, `preflight_check.py` | the first SAC-on-hardware attempt (manual resets); superseded by `physical_training/rig_learn.py` |
| `rl_hw/side_to_side.py`, `characterize_tilt.py`, `tilt_probe.py`, `diag_level.py`, `probe_m*_range.py`, `read_position_limits.py`, `motor_jog*.py` | early diagnostics and manual tools |
| `test_NN_approach/`, `failed_OpenCV_WorldModel/`, top-level `*.py` | the original course material and early experiments |
