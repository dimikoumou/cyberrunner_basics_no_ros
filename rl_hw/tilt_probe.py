#!/usr/bin/env python3
"""
Phase-3 diagnostic (2026-09-26): apply one slow, sustained single-axis tilt at a
time from the leveled plate and record the camera-measured plate angles and the
ball's displacement. Answers two questions directly from data: does a positive
action[i] roll the ball toward +x / +y (sign), and how much tilt does a given
|action| actually produce (magnitude). No shaking: each tilt is held, then the
plate returns to level before the next one.

Usage:
    python3 tilt_probe.py [magnitude] [hold_s]
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from plate_env import HardwarePlateEnv  # noqa: E402


def read(env, n=5):
    xs, ys, al, be = [], [], [], []
    for _ in range(n):
        xb, yb, a, b, found = env._read_state()
        al.append(np.degrees(a))
        be.append(np.degrees(b))
        if found:
            xs.append(xb)
            ys.append(yb)
    ball = (np.median(xs), np.median(ys)) if xs else (np.nan, np.nan)
    return ball, float(np.median(al)), float(np.median(be))


def main():
    mag = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    hold = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    env = HardwarePlateEnv(max_episode_steps=1)
    try:
        env._level_plate()
        print("commanded ticks after leveling:", env._cmd_ticks, "measured tilt:", env._meas_tilt)
        seq = [[mag, 0], [-mag, 0], [0, mag], [0, -mag], [mag, mag], [-mag, -mag]]
        if len(sys.argv) > 3:  # e.g. "x" -> only +x/-x pairs, repeated
            seq = {"x": [[mag, 0], [-mag, 0]] * 2, "y": [[0, mag], [0, -mag]] * 2}[sys.argv[3]]
        for action in seq:
            env._hold_tilt(np.zeros(2, dtype=np.float32), 1.5)
            (x0, y0), a0, b0 = read(env)
            env._hold_tilt(np.array(action, dtype=np.float32), hold)
            (x1, y1), a1, b1 = read(env)
            print(f"action=({action[0]:+.2f},{action[1]:+.2f}): tilt alpha {a0:+.2f}->{a1:+.2f}deg "
                  f"beta {b0:+.2f}->{b1:+.2f}deg | ball ({x0:+.3f},{y0:+.3f}) -> ({x1:+.3f},{y1:+.3f}) "
                  f"d=({x1 - x0:+.3f},{y1 - y0:+.3f})")
        env._hold_tilt(np.zeros(2, dtype=np.float32), 1.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
