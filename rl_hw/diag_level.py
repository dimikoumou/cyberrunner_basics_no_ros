#!/usr/bin/env python3
"""One-off: isolate _level_plate()'s convergence behavior, no control loop, no
unstick escalation -- just repeated leveling calls to see where/why it plateaus."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_env import HardwarePlateEnv  # noqa: E402

env = HardwarePlateEnv(max_episode_steps=1)
try:
    for i in range(3):
        print(f"\n--- level attempt {i+1} ---")
        env._level_plate(max_iters=40)
finally:
    env.close()
