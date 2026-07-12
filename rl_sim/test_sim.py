"""Sanity checks: env validity, render the maze, and a scripted PD controller
to confirm the maze is solvable (sanity floor for what RL should beat)."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import cv2
from maze_env import MazeEnv
from stable_baselines3.common.env_checker import check_env

env = MazeEnv(render_mode="rgb_array")
check_env(env)  # raises if the Gym API is malformed
print("check_env: OK")

# save the maze layout image
env.reset()
cv2.imwrite("maze_layout.png", env.render())
print("saved maze_layout.png")

def pd_controller(env, kp=180.0, kd=35.0, gain=1.0, krep=0.9, d0=0.030):
    """Steer toward the next checkpoint (PD) + repulsion from nearby holes.
    Proves the maze is solvable by *some* controller (a floor for RL)."""
    s = env.sim.state()
    pos, vel = s["pos"], s["vel"]
    nxt = env.cfg.checkpoints[env.sim.checkpoint_idx]
    err = nxt - pos
    cmd = kp * err - kd * vel              # desired acceleration (2,)
    # potential-field repulsion from holes within d0
    for h in env.cfg.holes:
        d = pos - h
        dist = np.linalg.norm(d)
        if 1e-6 < dist < d0:
            cmd += (d / dist) * (kp * krep) * (d0 - dist) / d0
    # beta drives +x accel; alpha drives +y accel via -sin(alpha)
    alpha = -cmd[1] * gain
    beta = cmd[0] * gain
    return np.clip([alpha, beta], -1, 1)

# run several episodes, record one as video
successes, results = 0, []
frames = []
for ep in range(20):
    obs, _ = env.reset()
    done = False
    total = 0.0
    steps = 0
    while not done:
        a = pd_controller(env)
        obs, r, term, trunc, info = env.step(a)
        total += r
        steps += 1
        done = term or trunc
        if ep == 0:
            frames.append(env.render())
    results.append((info["status"], steps, round(total, 2), info["checkpoint_idx"]))
    if info["status"] == "goal":
        successes += 1

print(f"\nPD controller: {successes}/20 reached goal")
for i, (st, steps, tot, cp) in enumerate(results):
    print(f"  ep{i:2d}: {st:5s} steps={steps:4d} reward={tot:7.2f} checkpoint={cp}")

# write the first-episode video
if frames:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter("pd_rollout.avi", cv2.VideoWriter_fourcc(*"XVID"), 60, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"saved pd_rollout.avi ({len(frames)} frames)")
