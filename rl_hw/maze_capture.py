#!/usr/bin/env python3
"""
Capture the maze board for route extraction (2026-09-27): level the plate on the camera, save a
full-resolution frame (1920x1080) and the 640x360 frame the pipeline uses, the detected plate
markers, and a pixel -> plate-metres lookup grid (every 2 px of the 640x360 frame) computed
with the same camera model as the ball position, so a route traced in the image lands at the
right place on any sheet.

  python3 maze_capture.py [out_dir]
"""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_env import HardwarePlateEnv  # noqa: E402
from goal_circle import pixel_to_plate  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "maze")
os.makedirs(out, exist_ok=True)
env = HardwarePlateEnv(fixed_goal=(0, 0), goal_tolerance=0.02, max_action_delta=0.5, max_episode_steps=100,
                       allow_unstick=False)
env._servo_level(max_s=8.0)
for _ in range(10):
    env._read_state()
time.sleep(0.3)
env._read_state()
with env._frame_lock:
    full = None if env._latest_frame is None else env._latest_frame.copy()
small = env._last_frame.copy()
if full is not None:
    cv2.imwrite(os.path.join(out, "maze_full.png"), full)
cv2.imwrite(os.path.join(out, "maze_640.png"), small)
rows, cols = np.arange(0, 360, 2), np.arange(0, 640, 2)
grid = np.full((len(rows), len(cols), 2), np.nan)
for i, r in enumerate(rows):
    for j, c in enumerate(cols):
        p = pixel_to_plate(env, r, c)
        if p is not None:
            grid[i, j] = p
np.savez(os.path.join(out, "pixel_to_plate_grid.npz"), rows=rows, cols=cols, grid=grid)
json.dump({"t": time.strftime("%Y-%m-%d %H:%M:%S"), "full_shape": None if full is None else list(full.shape),
           "inner_corners_row_col": np.asarray(env.pipeline.measurements.detector.corners).tolist(),
           "tilt_deg": env._meas_tilt}, open(os.path.join(out, "capture.json"), "w"), indent=1)
print("saved to", os.path.abspath(out), "full:", None if full is None else full.shape, "tilt", env._meas_tilt)
env.close()
