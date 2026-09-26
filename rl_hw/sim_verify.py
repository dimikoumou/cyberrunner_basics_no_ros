#!/usr/bin/env python3
"""
Standalone physics simulation of the ball-on-plate control logic in pd_balance.py,
decoupled from the real rig entirely -- no camera, no motors, no USB.

Purpose: this session's live hardware testing kept getting interrupted by real
electromechanical issues (a USB-serial adapter dropping under vibration, a
corner-ledge defect trapping the ball) that have nothing to do with whether the
CONTROL ALGORITHM (gain-scheduled PD + edge braking) is actually capable of
reliably balancing the ball. This runs the identical control law against a
simple but physically-grounded double-integrator model (tilt drives ball
acceleration via gravity, exactly as documented in pd_balance.py's own
docstring) so the algorithm itself can be verified/tuned independent of the
rig's hardware state.

This is NOT a substitute for the real hardware -- real friction, backlash,
detection noise, and the specific corner defect aren't modeled. It answers a
narrower, useful question: assuming clean sensing and an unobstructed plate,
does this control law reliably settle the ball in the goal circle and hold it
there, or does the algorithm itself have a flaw?

Usage:
    python3 sim_verify.py [n_episodes]
"""
import sys

import numpy as np

# ---- Physical model -----------------------------------------------------
# a = g * sin(tilt) ~ g * tilt for small angles (see pd_balance.py's own
# docstring). MAX_TILT_RAD matches this session's directly-measured real
# achievable tilt range on the actual rig (alpha saturates ~-11deg to +some
# positive bound before the linkage's own kinematic limit) -- using the real
# measured ceiling, not an idealized full +-90deg, so this sim's action
# authority matches what the hardware can actually deliver.
G = 9.81
MAX_TILT_RAD = np.radians(11.0)
DT = 1.0 / 55.0  # matches control_hz in plate_env.py
FRICTION = 0.15  # simple linear damping coefficient (rolling resistance), 1/s
PLATE_HALF_X, PLATE_HALF_Y = 0.14, 0.12  # matches C2C_X/2, C2C_Y/2 on the real rig

# ---- Control law: IDENTICAL to pd_balance.py's gain schedule -------------
KP_FAR, KD_FAR = 5.0, 2.0
KP_NEAR, KD_NEAR = 2.5, 4.0
GAIN_BLEND_DIST = 0.09
GAIN_BLEND_WIDTH = 0.05
GOAL = np.array([-0.0093, 0.0080])
GOAL_TOLERANCE = 0.048
EDGE_X, EDGE_Y = 0.10, 0.085
MAX_ACTION_DELTA = 0.5  # matches plate_env.py's rate limit


def pd_action(pos, vel, last_action):
    gx, gy = GOAL[0] - pos[0], GOAL[1] - pos[1]
    dist_now = float(np.hypot(gx, gy))
    near_frac = np.clip((GAIN_BLEND_DIST - dist_now) / GAIN_BLEND_WIDTH, 0.0, 1.0)
    kp = KP_FAR + near_frac * (KP_NEAR - KP_FAR)
    kd = KD_FAR + near_frac * (KD_NEAR - KD_FAR)
    action = np.clip(
        np.array([kp * gx - kd * vel[0], kp * gy - kd * vel[1]]), -0.8, 0.8
    )
    if abs(pos[0]) > EDGE_X:
        action[0] = -1.0 if pos[0] > 0 else 1.0
    if abs(pos[1]) > EDGE_Y:
        action[1] = -1.0 if pos[1] > 0 else 1.0
    # Same rate limit env.step() applies on the real rig.
    action = np.clip(action, last_action - MAX_ACTION_DELTA, last_action + MAX_ACTION_DELTA)
    return action


def run_episode(start_pos, max_steps=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.array(start_pos, dtype=float)
    vel = np.zeros(2)
    last_action = np.zeros(2)
    in_circle_steps = 0
    max_consecutive_in_circle = 0
    consecutive = 0
    trajectory = []
    for step in range(max_steps):
        action = pd_action(pos, vel, last_action)
        last_action = action
        tilt = action * MAX_TILT_RAD
        accel = G * np.sin(tilt) - FRICTION * vel
        # Small sensing noise, matching the real camera's practical jitter
        # (a few mm of pixel-to-world noise on this rig).
        vel = vel + accel * DT
        pos = pos + vel * DT + rng.normal(0, 0.0005, size=2)
        # Contain within the plate -- a real ball can't pass through the frame.
        for i, half in enumerate((PLATE_HALF_X, PLATE_HALF_Y)):
            if pos[i] > half:
                pos[i] = half
                vel[i] = -0.3 * vel[i]  # inelastic bounce off the frame
            elif pos[i] < -half:
                pos[i] = -half
                vel[i] = -0.3 * vel[i]
        dist = float(np.hypot(*(GOAL - pos)))
        in_circle = dist < GOAL_TOLERANCE
        in_circle_steps += int(in_circle)
        consecutive = consecutive + 1 if in_circle else 0
        max_consecutive_in_circle = max(max_consecutive_in_circle, consecutive)
        trajectory.append((pos[0], pos[1], dist))
        # "Sustained" success: held inside the circle continuously for 2+
        # real seconds (110 steps at 55Hz) -- this is the actual bar, not
        # just touching the circle once.
        if consecutive >= 110:
            return step + 1, in_circle_steps, max_consecutive_in_circle, trajectory
    return max_steps, in_circle_steps, max_consecutive_in_circle, trajectory


def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rng = np.random.default_rng(42)
    successes = 0
    results = []
    for ep in range(n_episodes):
        start = rng.uniform(-0.13, 0.13, size=2)
        steps, in_circle, max_consec, _ = run_episode(start, seed=ep)
        sustained = max_consec >= 110
        successes += int(sustained)
        pct = 100 * in_circle / steps
        print(f"episode {ep+1:2d}: start=({start[0]:+.3f},{start[1]:+.3f}) "
              f"steps={steps:4d} in_circle={in_circle:4d} ({pct:5.1f}%) "
              f"max_consecutive={max_consec:4d} ({max_consec/55:.1f}s) "
              f"sustained={'YES' if sustained else 'no'}")
        results.append((sustained, pct, max_consec))

    n_sustained = sum(r[0] for r in results)
    mean_pct = np.mean([r[1] for r in results])
    print(f"\n=== {n_sustained}/{n_episodes} episodes achieved sustained (2s+) "
          f"in-circle balance ({100*n_sustained/n_episodes:.0f}%) ===")
    print(f"=== mean in-circle time across all episodes: {mean_pct:.1f}% ===")


if __name__ == "__main__":
    main()
