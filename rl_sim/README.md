# CyberRunner — RL in simulation (pure Python, no ROS)

A self-contained reinforcement-learning setup that lets an agent **learn to solve
the labyrinth by trial and error** — before the physical hardware is available.
Trains in a simulator, then the same policy transfers to the real board
(sim-to-real), because the simulator's observations and actions are identical to
what the real state-estimation + motor code use.

## Why a simulator

RL learns from *consequences*: act → the world changes → observe a reward →
repeat, millions of times. That needs an environment that reacts to the agent's
actions. A photo or a recorded video can't (they don't respond to what the agent
does). So the only way to train before the hardware is available is to simulate
the board. The simulator also de-risks the hardware phase: you arrive with an
agent that already knows roughly how to play.

## Files

| File | What it is |
|---|---|
| `maze_sim.py` | Pure-numpy physics: ball on a plane tilted by (α, β), rolling-ball dynamics, wall collisions, holes, goal. |
| `maze_layout.py` | The maze geometry as **swappable data** (walls, holes, path checkpoints, goal). Placeholder serpentine layout; replace with a trace of the real BRIO board later. |
| `maze_env.py` | Gymnasium environment. Observation and action match the real system (see below). Potential-based reward on progress along the path. |
| `test_sim.py` | Sanity checks: validates the env, renders the maze, and runs a scripted PD controller to prove the maze is solvable. |
| `train_sac.py` | Trains a model-free agent (SAC) and saves a learning curve + a rollout video. |

## The interface (this is what makes sim-to-real work)

```
observation (8,):  [ x, y, vx, vy, alpha, beta, dx_to_next_cp, dy_to_next_cp ]
action (2,):       [ target_alpha, target_beta ]   in [-1, 1]  ->  plate tilt
```

- On **hardware**, `x, y, vx, vy` come from `state_est` (the estimator we fixed),
  `alpha, beta` from the motor feedback, and the checkpoint vector from the known
  maze path. The action maps to the two Dynamixel motors.
- In **sim**, the same vector is produced by the physics. So a policy trained in
  sim can be dropped onto the real board and fine-tuned.

## Run it

```bash
pip install numpy opencv-python-headless scipy gymnasium stable-baselines3 torch
cd rl_sim
python test_sim.py          # sanity check + maze_layout.png + pd_rollout.avi
python train_sac.py 250000  # train; writes runs/learning_curve.png + runs/trained_rollout.avi
```

## Roadmap

1. **[done]** Physics sim + Gym env + solvable placeholder maze.
2. **[done]** Model-free agent (SAC) learns to solve it.
3. **[next]** Replace the placeholder maze with an exact trace of the real board.
4. **[GPU]** Swap SAC for DreamerV3 (what the original CyberRunner used) — model-based, more sample-efficient.
5. **[hardware, August]** Sim-to-real: run the trained policy on the board, fine-tune on the real dynamics, close the loop with the motors.
