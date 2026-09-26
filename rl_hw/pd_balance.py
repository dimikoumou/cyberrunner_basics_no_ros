#!/usr/bin/env python3
"""
Hand-tuned PD controller to balance the ball inside the drawn goal circle. Used as
a fast, direct proof that the hardware/vision stack actually works end to end,
independent of (and much faster to validate than) full SAC training.

The plate is a double-integrator plant for the ball: tilt (the action) drives ball
ACCELERATION via gravity (a ~ g*sin(tilt) ~ tilt for small angles), not position or
velocity directly. So full state feedback -- proportional on position error, plus a
derivative term on velocity for damping -- is the physically-motivated controller
structure here, not just a heuristic.

Runs across multiple episodes automatically: if one ends early (ball_wedged,
ball_lost), that's a single bad attempt, not the run being over -- it resets and
keeps going until it either racks up solid in-circle time or the episode budget
runs out.

Usage:
    python3 pd_balance.py [total_steps] [max_episodes]
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from plate_env import HardwarePlateEnv  # noqa: E402

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pd_snapshots")
SNAPSHOT_EVERY_S = 2.0

# Single fixed (KP, KD) was a real tuning mistake, not just imprecise: lowering
# KP to stop overshoot AT the goal also weakens correction EVERYWHERE else,
# including wherever the plate's residual leveling bias is strongest (confirmed
# directly: the ball recovered to dist=0.128 from a corner, then drifted straight
# back to that exact same corner over the next ~30 steps under uninterrupted
# normal control -- a persistent bias steadily winning against a now-too-gentle
# correction, not overshoot). Gain-schedule instead: aggressive far from the
# goal (needs to win against real corner bias / reach across the plate), gentle
# only once close enough that overshoot-through-the-circle is the actual risk.
KP_FAR, KD_FAR = 5.0, 2.0
KP_NEAR, KD_NEAR = 2.5, 4.0
GAIN_BLEND_DIST = 0.09  # start easing toward the gentle gains at this distance
GAIN_BLEND_WIDTH = 0.05  # over this much additional distance
GOAL = (-0.0093, 0.0080)
GOAL_TOLERANCE = 0.048
# Direct feedback while watching this live: correcting X and Y together lets the
# combined vector point diagonally, straight at a corner, instead of going through
# the middle of an edge on the way back. Once either axis is out near the edge,
# stop blending -- fully prioritize bringing THAT axis back first before resuming
# normal 2D control, so the path runs through edge-middles, not corners.
EDGE_X, EDGE_Y = 0.10, 0.085
# One specific point (-0.15, +0.125) keeps recurring as a trap: a real V-notch
# where the frame's two inner walls meet at a right angle -- UPDATE: a closer
# zoomed photo showed this is actually a raised wooden bezel LEDGE where the
# paper meets the frame wall, not a corner notch; the ball perches slightly
# above the play surface there. Confirmed directly, in isolation: the full
# escalation (targeted push + tilt sweep + violent shake, ~15s combined)
# displaced it only ~6mm from a cold stop at this spot -- a real, severe
# structural defect that no shake pattern tried so far (linear, biased-linear,
# orbital) reliably clears. That fix needs a physical correction (built up
# paper/shim height to remove the step) -- it's not something more code can
# solve. What code CAN do: don't let one truly-stuck episode burn the whole
# step budget on futile escapes when a fresh reset() elsewhere on the plate has
# a real chance (confirmed: several fresh starts this session reached
# dist<0.1). See max_episode_steps below.


def main():
    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    # max_episode_steps default (600) let one truly-stuck episode (the ledge
    # defect, see note above) burn its entire step budget on repeated futile
    # escape attempts. 150 steps (~2.7s of real control time at 55Hz, though
    # wedge-detection/escalation stretches wall-clock well beyond that) is
    # still generous for genuine balancing but ends a hopeless episode fast
    # enough to let the remaining total_steps budget go toward fresh starts,
    # several of which land clear of the defect entirely.
    env = HardwarePlateEnv(
        fixed_goal=GOAL, goal_tolerance=GOAL_TOLERANCE, max_action_delta=0.5, max_episode_steps=150
    )
    try:
        total_in_circle = 0
        total_taken = 0
        episode = 0
        last_snapshot = 0.0
        snap_i = 0
        while total_taken < total_steps and episode < max_episodes:
            episode += 1
            print(f"\n=== episode {episode} ===")
            obs, info = env.reset()
            ep_in_circle = 0
            ep_steps = 0
            for i in range(total_steps - total_taken):
                xb, yb, vx, vy, alpha, beta, gx, gy = obs
                dist_now = float(np.hypot(gx, gy))
                # 0 = fully "far" gains, 1 = fully "near" gains, smoothly blended
                # over GAIN_BLEND_WIDTH so there's no gain discontinuity right at
                # the switch point (which would itself show up as a small kick).
                near_frac = np.clip(
                    (GAIN_BLEND_DIST - dist_now) / GAIN_BLEND_WIDTH, 0.0, 1.0
                )
                kp = KP_FAR + near_frac * (KP_NEAR - KP_FAR)
                kd = KD_FAR + near_frac * (KD_NEAR - KD_FAR)
                action = np.clip(
                    np.array([kp * gx - kd * vx, kp * gy - kd * vy], dtype=np.float32),
                    -0.8, 0.8,
                )
                # Hard override once near an edge -- full brake straight back toward
                # center on whichever axis is in danger, overriding the blended 2D
                # goal-seeking above. Without this the combined X+Y correction can
                # point diagonally at a corner instead of cutting back through the
                # middle of the edge it's near.
                if abs(xb) > EDGE_X:
                    action[0] = -1.0 if xb > 0 else 1.0
                if abs(yb) > EDGE_Y:
                    action[1] = -1.0 if yb > 0 else 1.0
                # A dedicated trap-zone override was tried here (both a steady
                # push and an in-loop oscillation toward the goal) and REMOVED --
                # confirmed directly to be counterproductive, not just ineffective.
                # env.step()'s own max_action_delta rate limit smooths any in-loop
                # oscillation enough that it keeps the ball moving JUST enough to
                # never trip plate_env.py's wedge detector, without ever moving it
                # enough to actually escape -- silently burning entire episodes
                # (600/600 steps, 0% in-circle) vibrating in place instead of ever
                # escalating to the real, un-rate-limited shake escape in
                # _attempt_unstick(), which is the only thing that has ever
                # actually freed the ball from this specific spot. Let the wedge
                # detector fire and do its job instead of masking it.
                obs, reward, terminated, truncated, info = env.step(action)
                dist = float(np.hypot(obs[6], obs[7]))
                in_circle = dist < GOAL_TOLERANCE
                ep_in_circle += int(in_circle)
                ep_steps += 1

                now = time.time()
                if now - last_snapshot >= SNAPSHOT_EVERY_S:
                    frame = env._grab_frame()
                    if frame is not None:
                        path = os.path.join(SNAPSHOT_DIR, f"snap_{snap_i:04d}.jpg")
                        cv2.imwrite(path, frame)
                        snap_i += 1
                    last_snapshot = now
                if i % 10 == 0 or info["status"] not in ("running", "in_circle"):
                    print(f"  step {i:4d}: xb={obs[0]:+.4f} yb={obs[1]:+.4f} dist={dist:.4f} "
                          f"{'IN' if in_circle else '  '}  action=({action[0]:+.2f},{action[1]:+.2f})  "
                          f"status={info['status']}")
                if terminated or truncated:
                    print(f"  episode {episode} ended at step {i}: {info['status']}")
                    break
            total_in_circle += ep_in_circle
            total_taken += ep_steps
            pct = 100 * ep_in_circle / max(1, ep_steps)
            print(f"  episode {episode} summary: {ep_in_circle}/{ep_steps} steps in circle ({pct:.1f}%)")

        overall_pct = 100 * total_in_circle / max(1, total_taken)
        print(f"\n=== overall: {total_in_circle}/{total_taken} steps in circle ({overall_pct:.1f}%) "
              f"across {episode} episode(s) ===")
    finally:
        env.close()


if __name__ == "__main__":
    main()
