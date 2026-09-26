#!/usr/bin/env python3
"""
Phase-3 diagnostic (2026-09-26): map each tilt motor's full usable range against
the camera-measured plate angle. From a start position, step one motor outward in
fixed increments in each direction (the other motor held), recording the
measured alpha/beta at every step, and stop a direction once the angle has
plateaued (no longer changes with more ticks) or a firmware limit is reached --
so the plate is never forced further than it actually travels.

Writes a CSV per motor to ../phase3_logs/ and prints a summary (angle range,
ticks-per-degree slope, and the tick where each angle crosses zero).

Usage:
    python3 characterize_tilt.py
"""
import sys
import os
import time
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from plate_env import HardwarePlateEnv  # noqa: E402
from state_est_control import set_position  # noqa: E402

LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "phase3_logs"))
STEP = 100
FW_MIN, FW_MAX = 50, 4045        # stay inside Position Control's 0..4095 range
PLATEAU_DEG = 0.25               # change per step below which the angle has stopped moving
PLATEAU_STEPS = 2


def measure(env, n=6):
    al, be = [], []
    for _ in range(n):
        _, _, a, b, _ = env._read_state()
        al.append(np.degrees(a))
        be.append(np.degrees(b))
    return float(np.median(al)), float(np.median(be))


def present(env, dxl_id):
    v = env.packet_handler.read4ByteTxRx(env.port_handler, dxl_id, 132)[0]
    return v - 2**32 if v >= 2**31 else v


def sweep(env, motor, start, other, other_pos, idx):
    rows = []
    for direction in (-1, +1):
        set_position(env.port_handler, env.packet_handler, other, other_pos)
        set_position(env.port_handler, env.packet_handler, motor, start)
        time.sleep(1.2)
        pos, flat, last = start, 0, None
        while FW_MIN <= pos <= FW_MAX:
            set_position(env.port_handler, env.packet_handler, motor, pos)
            time.sleep(0.9)
            ang = measure(env)
            rows.append((direction, pos, present(env, motor), ang[0], ang[1]))
            print(f"  motor{motor} dir={direction:+d} goal={pos} present={rows[-1][2]} "
                  f"alpha={ang[0]:+.2f} beta={ang[1]:+.2f}")
            if last is not None and abs(ang[idx] - last) < PLATEAU_DEG:
                flat += 1
                if flat >= PLATEAU_STEPS:
                    print(f"  motor{motor} dir={direction:+d}: angle plateaued -- stopping this direction")
                    break
            else:
                flat = 0
            last = ang[idx]
            pos += direction * STEP
    set_position(env.port_handler, env.packet_handler, motor, start)
    time.sleep(1.0)
    return rows


def summarize(name, rows, idx):
    ang = np.array([r[3 + idx] for r in rows])
    ticks = np.array([r[2] for r in rows], dtype=float)
    lo, hi = ang.min(), ang.max()
    # slope from the non-plateaued middle part (angles within 80% of the range)
    mid = (ang > lo + 0.1 * (hi - lo)) & (ang < hi - 0.1 * (hi - lo))
    slope = np.polyfit(ticks[mid], ang[mid], 1)[0] if mid.sum() >= 3 else np.nan
    order = np.argsort(ticks)
    zero = np.interp(0.0, ang[order], ticks[order]) if lo < 0 < hi and np.all(np.diff(ang[order]) * np.sign(slope) >= -0.5) else np.nan
    print(f"{name}: angle range {lo:+.2f}..{hi:+.2f} deg over ticks {ticks.min():.0f}..{ticks.max():.0f}, "
          f"slope {slope:+.5f} deg/tick ({1 / slope if slope else np.nan:+.1f} ticks/deg), zero near tick {zero:.0f}")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    env = HardwarePlateEnv(max_episode_steps=1)
    try:
        m1_start, m3_start = 2712, 2300
        print("--- motor 3 (beta) ---")
        r3 = sweep(env, 3, m3_start, 1, m1_start, idx=1)
        print("--- motor 1 (alpha) ---")
        r1 = sweep(env, 1, m1_start, 3, m3_start, idx=0)
        stamp = time.strftime("%H%M%S")
        for name, rows in (("m3", r3), ("m1", r1)):
            with open(os.path.join(LOG_DIR, f"characterize_{name}_{stamp}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["direction", "goal", "present", "alpha_deg", "beta_deg"])
                w.writerows(rows)
        print()
        summarize("motor3 -> beta", r3, 1)
        summarize("motor1 -> alpha", r1, 0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
