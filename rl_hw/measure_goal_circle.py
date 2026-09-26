#!/usr/bin/env python3
"""
Detects the red goal circle (filled disc or outline) on the plate's paper and
reports its center/radius in the plate frame (meters) that HardwarePlateEnv's
xb/yb use. pd_balance.py does the same detection automatically at startup
(rl_hw/goal_circle.py); this script is for checking a new sheet by hand.

Usage:
    python3 measure_goal_circle.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_env import HardwarePlateEnv  # noqa: E402
from goal_circle import detect_goal_circle, save_debug  # noqa: E402


def main():
    env = HardwarePlateEnv()
    try:
        res = detect_goal_circle(env)
        (gx, gy), r = res["center"], res["radius"]
        print(f"source: {res['source']}  (clean = ball not covering the circle; last_goal = ball on it, "
              f"reused the last clean measurement)")
        if res["px"] is not None:
            cx, cy, r_px = res["px"]
            print(f"detected circle (pixel, this frame's resolution): center=({cx:.1f},{cy:.1f}) radius={r_px:.1f}")
        print(f"\ncircle center (world meters): ({gx:.4f}, {gy:.4f})")
        print(f"circle radius (world meters): {r:.4f}")
        print(f"median of {res['n_frames']} frames, centre spread {res['spread_m'] * 1000:.1f} mm")
        print(f"plate half-extents for reference: x_half={env._x_half:.4f} y_half={env._y_half:.4f}")
        print(f"\nfixed_goal=({gx:.4f}, {gy:.4f}), goal_tolerance={r:.4f}")
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goal_circle_debug.jpg")
        save_debug(res, out_path)
        print(f"\ndebug overlay saved to {out_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
