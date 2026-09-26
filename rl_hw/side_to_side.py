#!/usr/bin/env python3
"""
Simplest possible proof that the hardware/vision loop works end to end: no precise
goal, no leveling perfection required. Just:
  - ball in the left half (xb < 0)  -> tilt hard toward the right
  - ball in the right half (xb > 0) -> tilt hard toward the left
  - ball not detected               -> go flat (last known-good level position, or
                                        the static calibrated center if none cached)
  - ball not moving despite a real push -> run the same escalating unstick sequence
                                        HardwarePlateEnv uses (targeted push, sweep,
                                        fast shake)

Usage:
    python3 side_to_side.py [duration_seconds]
"""
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from plate_env import HardwarePlateEnv, set_position, LAST_LEVEL_CACHE_PATH  # noqa: E402

TARGET_MARGIN = 0.10  # aim for a point this far into the opposite half, not the far wall itself
KX, KY = 4.0, 4.0     # proportional gain -- eases off as the ball nears the target, doesn't
                      # keep slamming it at full power once it's crossed center. Lowered from
                      # an earlier 6.0: even with damping, that was still enough initial force
                      # to send the ball past TARGET_MARGIN and into the far edge zone.
KD = 2.5  # velocity damping -- without this, a strong push has nothing counteracting
          # momentum once the ball is moving fast, so it doesn't settle near the
          # target, it just keeps rolling on through to the far wall (a ballistic
          # "gravity did all the work" crossing, not real closed-loop control -- and
          # exactly the degenerate case where the ball ends up wedged at the new
          # corner instead of actually being controlled to a stop).
# Plate half-extents are ~0.142/0.119m, and EVERY corner has turned out to have the
# same real physical trap (confirmed: a ball parked at any of them barely budges
# even under a violent max-tilt shake). Rather than rely on the PD gains alone to
# never overshoot there -- confirmed directly that they can, even with damping --
# hard-override with a strong direct brake once the ball crosses into the outer
# ~25% of either axis, regardless of which "half" logic currently wants.
EDGE_X, EDGE_Y = 0.10, 0.085
FLAT_HOLD_S = 1.5   # how long to sit flat before checking for the ball again
STATUS_EVERY_S = 3.0
STUCK_CHECK_S = 4.0       # how often to check for zero progress
STUCK_THRESHOLD = 0.01    # meters of movement below which it's "not moving"


def go_flat(env, m1_id, m3_id):
    try:
        with open(LAST_LEVEL_CACHE_PATH) as f:
            cached = json.load(f)
        m1p, m3p = int(cached["m1"]), int(cached["m3"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        _, m1p, _ = env.calibration[m1_id]
        _, m3p, _ = env.calibration[m3_id]
    set_position(env.port_handler, env.packet_handler, m1_id, m1p)
    set_position(env.port_handler, env.packet_handler, m3_id, m3p)


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    env = HardwarePlateEnv()
    m1_id, m3_id = sorted(env.calibration.keys())
    try:
        go_flat(env, m1_id, m3_id)
        time.sleep(FLAT_HOLD_S)
        t_end = time.time() + duration
        last_side = None
        last_status = 0.0
        stuck_check_at = time.time() + STUCK_CHECK_S
        stuck_ref_xb = None
        prev_xb = prev_yb = prev_t = None

        while time.time() < t_end:
            now = time.time()
            xb, yb, alpha, beta, found = env._read_state()
            if not found:
                if last_side is not None:
                    print("ball not detected -- going flat")
                go_flat(env, m1_id, m3_id)
                last_side = None
                stuck_ref_xb = None
                prev_xb = prev_yb = prev_t = None
                time.sleep(FLAT_HOLD_S)
                continue

            if prev_t is not None and now > prev_t:
                vx = (xb - prev_xb) / (now - prev_t)
                vy = (yb - prev_yb) / (now - prev_t)
            else:
                vx = vy = 0.0
            prev_xb, prev_yb, prev_t = xb, yb, now

            # Full 2D proportional-derivative push, not X-only and not just P: a ball
            # wedged in a corner is pinned by BOTH axes, so leaving Y uncontrolled
            # can't help free it even if X is exactly right (position term). The
            # velocity term (KD) is just as important -- without it, a strong push
            # has nothing to counteract momentum once the ball is rolling, so it
            # doesn't settle near the target, it just keeps going to the far wall on
            # gravity alone (a ballistic crossing, not real control -- and exactly
            # how it ends up wedged in a new corner instead of actually stopping).
            # Aim for a point inside the opposite half (TARGET_MARGIN), not the far
            # wall itself, so there's a real target short of the edge to settle at.
            target_x = TARGET_MARGIN if xb < 0 else -TARGET_MARGIN
            action = np.array([
                np.clip(KX * (target_x - xb) - KD * vx, -1.0, 1.0),
                np.clip(KY * (0.0 - yb) - KD * vy, -1.0, 1.0),
            ], dtype=np.float32)
            # Hard override once near an edge: full brake toward center on whichever
            # axis is in danger, ignoring the target-crossing logic above. This is a
            # safety backstop, not fine control -- the PD gains alone have been
            # observed to overshoot past TARGET_MARGIN and into this zone even with
            # damping, and every corner here is a real trap once reached.
            if abs(xb) > EDGE_X:
                action[0] = -1.0 if xb > 0 else 1.0
            if abs(yb) > EDGE_Y:
                action[1] = -1.0 if yb > 0 else 1.0

            side = "left" if xb < 0 else "right"
            target = "right" if side == "left" else "left"
            if side != last_side:
                print(f"ball in {side} half (xb={xb:+.4f} yb={yb:+.4f}) -> pushing toward "
                      f"{target}, action=({action[0]:+.2f},{action[1]:+.2f})")
                last_side = side
                last_status = now
            elif now - last_status >= STATUS_EVERY_S:
                print(f"  still in {side} half, xb={xb:+.4f} yb={yb:+.4f} -> "
                      f"action=({action[0]:+.2f},{action[1]:+.2f})")
                last_status = now

            if stuck_ref_xb is None:
                stuck_ref_xb = xb
                stuck_check_at = now + STUCK_CHECK_S
            elif now >= stuck_check_at:
                if abs(xb - stuck_ref_xb) < STUCK_THRESHOLD:
                    print(f"  ball hasn't moved (xb {stuck_ref_xb:+.4f} -> {xb:+.4f}) -- unsticking")
                    env._attempt_unstick(stuck_pos=(xb, yb))
                    # The unstick maneuver just made large deliberate jumps -- the
                    # next real reading isn't a continuation of normal rolling, so
                    # don't compute a velocity from the gap across it.
                    prev_xb = prev_yb = prev_t = None
                stuck_ref_xb = None

            env._write_action(action)
            time.sleep(env.dt)
    finally:
        env.close()


if __name__ == "__main__":
    main()
