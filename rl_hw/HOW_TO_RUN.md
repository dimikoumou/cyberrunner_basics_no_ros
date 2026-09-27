# Running the CyberRunner plate

## Start

```bash
cd ~/cyberrunner_basics_no_ros/rl_hw
PD_UI=1 ../.venv/bin/python3 pd_balance.py 2000000 100000 0 2000000
```

Open **http://localhost:8000**. `Ctrl+C` stops it cleanly and switches the motors off.
Only one controller can run at a time; a second one exits without touching the motors.

## Modes (UI)

| Mode | What it does |
|---|---|
| **Red region on sheet** | Finds the red shape on the paper (circle, square, blob…) and balances the ball at its deepest point. Swap the sheet mid-run: the new region is picked up within ~1.5 s. |
| **Click to target** | Click anywhere on the live view; the ball goes there (one smooth planned move) and stays. Clicking also starts balancing. |
| **Follow red line** | Finds the red line on the paper (open line or closed loop) and rolls the ball along it. Open lines: starts at the nearer end, stops at the other. |
| **Draw path** | Click waypoints on the live view, press **Go**; the ball drives the route. **Clear path** to start over. |

**Controller: Classic / Learned (RL).** *Classic* is the hand-built controller (planned moves,
stiction handling, learned local slopes). *Learned* uses the policy trained in simulation
(`rl_sim/runs/plate_goal_v1/policy.npz`) for the click and region modes.

## Preparing sheets

- Red marker, clear strokes. Other colours are ignored for now.
- Keep shapes and lines **at least ~5 cm from the wooden frame**: near the frame the paper grips
  the ball much harder, and results there are unreliable.
- Keep the blue corner dots uncovered.

## What it learns and stores (local files, not in git)

| File | Contents |
|---|---|
| `bias_table.json` | Local slope of the paper per 3 cm cell, learned when the ball rests in a target. Makes repeat targets smoother. |
| `pulse_table.json` | Stiction-pulse settings per cell. |
| `last_goal.json` | Last cleanly seen red region (used when the ball covers it at startup). |
| `last_level_position.json` | Motor positions of the last levelling. |
| `rl_hw/pd_logs/*.csv` | One CSV per run (ball, tilt, actions, goal...). |

## Troubleshooting

- **UI says running but the motors don't move**: another controller was started and one of them
  switched torque off. Stop everything (`pkill -INT -f pd_balance.py`) and start once.
- **Plate swings side to side**: the camera image is delayed. It must run at 30 fps (the code
  requests this); after re-plugging, check it's on a direct USB-3 connection.
- **"no camera delivering 1920x1080"**: the rig camera isn't connected (or is frozen: unplug/replug).
  The controller also reopens a frozen camera by itself after 2 s.
- **Ball jumps to a corner**: see the run's CSV; ball-position glitches are filtered, but a real
  disturbance (hand, sheet) will make it re-approach.

## Training the learned controller

```bash
cd ~/cyberrunner_basics_no_ros/rl_sim
../.venv-rl/bin/python3 train_plate_ppo.py 20e6 plate_goal_v1    # separate venv with torch
../.venv/bin/python3 fit_plate_sim.py ../rl_hw/pd_logs/*.csv      # re-check the sim against the rig
```

The simulator (`plate_sim.py`) is calibrated from rig logs and domain-randomised; the trained
network is exported to numpy (`policy.npz`), so the rig itself needs no PyTorch.
