"""
Preflight check for HardwarePlateEnv / train_sac_hw.py -- run this BEFORE committing
to a long unattended training run. Verifies the hardware, calibration, coordinate
frame, and control loop all actually work, runs a handful of real steps to measure
your actual real-world step rate, and uses that to give a real (not guessed) time
estimate for how long a given training budget will take.

Usage:
    python3 preflight_check.py [total_timesteps_you_plan_to_run]   # default 200000
"""
import sys
import time

import numpy as np

from plate_env import HardwarePlateEnv

PASS, FAIL, WARN = [], [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}")
    else:
        FAIL.append((name, detail))
        print(f"  [FAIL] {name}  {detail}")


def warn(name, detail=""):
    WARN.append((name, detail))
    print(f"  [WARN] {name}  {detail}")


def main():
    planned_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000

    print("=" * 70)
    print("PREFLIGHT CHECK")
    print("=" * 70)

    print("\n[1/5] Opening camera + motors + estimation pipeline...")
    try:
        env = HardwarePlateEnv()
        check("env constructs (camera opens, motor port opens, pipeline loads)", True)
    except Exception as e:
        check("env constructs (camera opens, motor port opens, pipeline loads)", False, str(e))
        print("\nCannot continue -- fix the above before running training.")
        sys.exit(1)

    try:
        print(f"\n[2/5] Coordinate frame sanity (x_half={env._x_half:.4f} y_half={env._y_half:.4f})")
        check("x_half is a sane plate half-width (0.05-0.30m)", 0.05 < env._x_half < 0.30, str(env._x_half))
        check("y_half is a sane plate half-height (0.05-0.30m)", 0.05 < env._y_half < 0.30, str(env._y_half))
        if env.fixed_goal == "center":
            expected = np.array([0.0, 0.0])
        elif env.fixed_goal is not None:
            expected = np.array(env.fixed_goal)
        else:
            expected = None
        if expected is not None:
            check("configured fixed_goal is within the reachable area",
                  abs(expected[0]) <= env._x_half and abs(expected[1]) <= env._y_half,
                  f"goal={expected}, reachable=+-({env._x_half:.3f},{env._y_half:.3f})")

        print("\n[3/5] Corner + ball detection (place the ball somewhere visible now)...")
        input("      Press Enter once the ball is placed and visible...")
        xb, yb, alpha, beta, ball_found = env._read_state()
        check("corners detected", env.pipeline.measurements.detector.corners is not None)
        check("ball detected", ball_found, f"xb={xb} yb={yb}")
        if ball_found:
            check("ball position is within the reachable coordinate range",
                  abs(xb) <= env._x_half + 0.02 and abs(yb) <= env._y_half + 0.02,
                  f"ball=({xb:.3f},{yb:.3f}) reachable=+-({env._x_half:.3f},{env._y_half:.3f})")
            dist = float(np.hypot(xb, yb))
            print(f"      ball is {dist:.3f}m from plate center")

        print("\n[4/5] Running 60 real control steps to measure actual step rate "
              "(motors will move -- watch the plate)...")
        obs, info = env.reset()
        t0 = time.time()
        n = 60
        rewards = []
        for _ in range(n):
            action = env.action_space.sample() * 0.3  # gentle, not full-random-extreme
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                obs, info = env.reset()
        elapsed = time.time() - t0
        steps_per_sec = n / elapsed
        check("completed 60 real steps without error", True)
        check("rewards are finite numbers (not NaN/inf)", all(np.isfinite(r) for r in rewards),
              f"sample={rewards[:3]}")
        print(f"      measured rate: {steps_per_sec:.2f} steps/sec ({elapsed/n*1000:.0f}ms/step)")

        print("\n[5/5] Action smoothing check...")
        check("max_action_delta is set (rate-limits violent action jumps)",
              env.max_action_delta is not None and env.max_action_delta > 0,
              str(env.max_action_delta))

    finally:
        print("\nClosing (motors should de-torque)...")
        env.close()

    print("\n" + "=" * 70)
    print(f"{len(PASS)} passed, {len(WARN)} warnings, {len(FAIL)} failed")
    if FAIL:
        print("\nFAILURES -- do not start the long run until these are fixed:")
        for name, detail in FAIL:
            print(f"  - {name}: {detail}")

    if not FAIL:
        est_seconds = planned_steps / steps_per_sec
        est_hours = est_seconds / 3600
        learning_starts_seconds = 500 / steps_per_sec  # matches train_sac_hw.py's current value
        print(f"\nAt this measured rate ({steps_per_sec:.2f} steps/sec):")
        print(f"  - learning_starts=500 reached in ~{learning_starts_seconds/60:.1f} minutes")
        print(f"  - full run of {planned_steps} steps: ~{est_hours:.1f} hours")
        print("\nLooks good to run.")
    print("=" * 70)

    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
