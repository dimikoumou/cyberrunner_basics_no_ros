"""Record an UNTRAINED (random) agent for the before/after contrast: it flails
and falls into a hole almost immediately. Saves runs/baseline_rollout.avi."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import cv2
from maze_env import MazeEnv

os.makedirs("runs", exist_ok=True)
env = MazeEnv(render_mode="rgb_array")

# pick an episode that lasts long enough to be watchable (>=30 frames) but still
# fails, so the clip clearly shows the untrained agent losing.
best = None
for seed in range(50):
    obs, _ = env.reset(seed=seed)
    frames, done = [], False
    while not done:
        a = env.action_space.sample()          # random policy = untrained
        obs, r, term, trunc, info = env.step(a)
        frames.append(env.render())
        done = term or trunc
    if info["status"] == "hole" and 30 <= len(frames) <= 200:
        best = (frames, info)
        break
    if best is None:
        best = (frames, info)

frames, info = best
h, w = frames[0].shape[:2]
vw = cv2.VideoWriter("runs/baseline_rollout.avi", cv2.VideoWriter_fourcc(*"XVID"), 60, (w, h))
for f in frames:
    vw.write(f)
vw.release()
print(f"saved runs/baseline_rollout.avi ({len(frames)} frames, status={info['status']}, max_cp={info['max_cp']})")
