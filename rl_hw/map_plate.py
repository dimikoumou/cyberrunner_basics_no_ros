#!/usr/bin/env python3
"""
Map the plate surface (2026-09-26): the controller works well on targets in open
space but not on a small dot near the right edge, where the paper holds the ball
harder and pushes often fail. This measures, on a grid of points, what the surface
does locally, instead of tuning against it blind:

  - breakaway tilt: from rest, ramp a tilt slowly (0.5 deg/s) in 4 directions and
    record the tilt at which the ball starts to move (> 2 mm);
  - local level bias + rolling friction: from the same ramps, the tilt at which it
    starts rolling in +d vs -d (the midpoint is the local "level", half the gap is
    the static friction).

The ball is brought to each grid point with the normal controller (virtual goal),
then left to settle before measuring. Results go to ../plate_map.json.

Usage:
    python3 map_plate.py [step_m]      # default grid step 0.04 m
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from plate_env import HardwarePlateEnv  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plate_map.json"))
RAMP_DEG_PER_S = 0.5
RAMP_MAX_DEG = 4.5
MOVE_M = 0.002


def ball(env, n=3):
    pts = []
    for _ in range(n):
        xb, yb, _, _, found = env._read_state()
        if found:
            pts.append((xb, yb))
    return np.median(np.array(pts), axis=0) if pts else None


def go_to(env, target, timeout_s=25.0):
    """Bring the ball near `target` with a plain PD + a gentle push if stuck."""
    t0, last, still_t = time.time(), None, time.time()
    while time.time() - t0 < timeout_s:
        p = ball(env, 1)
        if p is None:
            env._hold_tilt(np.zeros(2, dtype=np.float32), 0.05)
            continue
        e = np.asarray(target) - p
        if np.hypot(*e) < 0.006:
            break
        v = np.zeros(2) if last is None else (p - last) / env.dt
        last = p
        stuck_boost = 0.0
        if np.hypot(*v) < 0.01:
            stuck_boost = min(0.8, 0.4 * (time.time() - still_t))  # grows while stuck
        else:
            still_t = time.time()
        a = np.clip(4.0 * e - 2.0 * v + stuck_boost * e / max(np.hypot(*e), 1e-6), -0.7, 0.7)
        env._write_action(a.astype(np.float32))
        time.sleep(env.dt)
    env._hold_tilt(np.zeros(2, dtype=np.float32), 1.5)  # settle at level
    return ball(env)


def breakaway(env, direction):
    """Ramp tilt along `direction` (unit, in action space = ball x/y) from level until
    the ball moves MOVE_M; return (tilt_deg, start_pos) or (None, start_pos)."""
    env._hold_tilt(np.zeros(2, dtype=np.float32), 1.0)
    p0 = ball(env)
    if p0 is None:
        return None, None
    t0 = time.time()
    while True:
        tilt = RAMP_DEG_PER_S * (time.time() - t0)
        if tilt > RAMP_MAX_DEG:
            return None, p0
        env._write_action((direction * tilt / 5.0).astype(np.float32))
        time.sleep(env.dt)
        p = ball(env, 1)
        if p is not None and np.hypot(*(p - p0)) > MOVE_M:
            # the camera sees ~0.15 s late: report the tilt commanded then
            return max(0.0, tilt - RAMP_DEG_PER_S * 0.15), p0


def main():
    step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
    env = HardwarePlateEnv(max_episode_steps=1)
    results = []
    try:
        env._level_plate()
        xs = np.arange(-0.10, 0.1001, step)
        ys = np.arange(-0.08, 0.0801, step)
        for gy in ys:
            for gx in xs:
                pos = go_to(env, (gx, gy))
                # Stiction often stops the ball short of the grid point: measure where it
                # actually is (that's still a valid map sample); skip only near-duplicates.
                if pos is None or any(np.hypot(*(pos - np.array(r["pos"]))) < 0.015 for r in results):
                    print(f"point ({gx:+.2f},{gy:+.2f}): ball at {pos}, too close to an existing sample -- skipped")
                    continue
                row = {"target": [float(gx), float(gy)], "pos": [float(v) for v in pos]}
                for name, d in (("+x", (1, 0)), ("-x", (-1, 0)), ("+y", (0, 1)), ("-y", (0, -1))):
                    tb, p0 = breakaway(env, np.array(d, dtype=float))
                    row[name] = None if tb is None else round(float(tb), 2)
                    go_to(env, (gx, gy), timeout_s=10.0)
                for ax, (pl, mi) in (("x", ("+x", "-x")), ("y", ("+y", "-y"))):
                    if row[pl] is not None and row[mi] is not None:
                        # rolls toward +d at +t1, toward -d at -t2 -> local level at (t1 - t2)/2
                        row[f"level_{ax}_deg"] = round((row[pl] - row[mi]) / 2, 2)
                        row[f"static_{ax}_deg"] = round((row[pl] + row[mi]) / 2, 2)
                print(f"point ({gx:+.2f},{gy:+.2f}) at ({pos[0]:+.3f},{pos[1]:+.3f}): breakaway +x {row['+x']} "
                      f"-x {row['-x']} +y {row['+y']} -y {row['-y']} deg | level offset x {row.get('level_x_deg')} "
                      f"y {row.get('level_y_deg')} | static x {row.get('static_x_deg')} y {row.get('static_y_deg')}")
                results.append(row)
                with open(OUT, "w") as f:
                    json.dump({"step_m": step, "points": results}, f, indent=1)
    finally:
        env.close()
    print(f"saved {len(results)} points to {OUT}")


if __name__ == "__main__":
    main()
