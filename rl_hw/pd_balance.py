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
import json
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from plate_env import HardwarePlateEnv  # noqa: E402
from goal_circle import detect_goal_circle, detect_on_frame, save_debug, save_last_goal  # noqa: E402

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pd_snapshots")
SNAPSHOT_EVERY_S = float(os.environ.get("PD_SNAPSHOT_S", "2.0"))  # e.g. 30 for long unattended runs
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pd_logs")
# Success criterion (Phase 3): ball held continuously inside the goal circle for
# this long, measured in wall-clock seconds, not steps -- the real step rate is
# well below the nominal 55Hz, so a step count would overstate the hold.
HOLD_TARGET_S = 10.0

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
# 2026-09-26: slow integral on position error. The rig's true level point, fit
# from each run's own data (ball acceleration vs measured tilt, r~0.98), is
# stable within a run but moves 0.3-0.5deg between runs (beta0 3.00 -> 2.55 ->
# 2.98, alpha0 -0.82 -> -1.13 -> -1.41), so no constant LEVEL_OFFSET_DEG stays
# right; with KP_NEAR the residual bias parked the ball 25-50mm off-centre in a
# 45.5mm disc. Integrate only near the goal (anti-windup), real wall-clock dt,
# clamped to +-0.3 action (+-1.5deg), carried across episodes (the bias is).
KI = 1.0
I_ZONE = 0.10
I_MAX = 0.3  # 0.6 tried 2026-09-26: stick-slip (ball stuck ~1.5-2deg breakaway, then 50-80mm overshoot), in-circle 88.8% -> 76.8%; reverted
# 2026-09-26 ramp-kick & release (design panel + judge on the logs): on bare paper
# a still ball sinks into a slight dimple -> static breakaway ~1.5-2.4deg >> rolling
# ~0.3-0.6deg. With a plain integral the stored push was never released after
# breakaway (action unchanged 10 frames later), so the ball rolled through the goal
# and re-stuck 60-80mm away. Now: the integral learns only while the ball rolls;
# a separate kick ramps toward the goal while it is stuck and is dumped the moment
# the ball moves. Stuck is detected from POSITION (logged speed of a still ball is
# 7-16 mm/s, useless as a stillness test).
STUCK_WIN = 15          # frames (~0.66 s) for the stationarity test
STUCK_PTP = 0.0015      # m, per-axis position range => stationary (rest noise ptp <= 1.4 mm)
BREAK_DISP = 0.002      # m from anchor => broke free -> dump kick
R_DONE = 0.008          # m, stationary this close = success, never kick (0.005: a ball resting at 4.5 mm crossed it on noise and got kicked out 33 mm)
KICK_RATE = 0.6         # action/s (3 deg/s) ramp along the unit vector to the goal
KICK_MAX = 0.6          # action vector cap (3 deg); one stick held 87 s at |a|~0.47
KICK_START_FRAC = 0.6   # restart the ramp at 60% of the last successful breakaway kick
KICK_HOLD_S = 0.5       # give up if still stuck this long at the cap
KICK_LOCKOUT_S = 1.0    # after a release (2x after a give-up)
# The goal is detected from the red circle on the sheet at every startup
# (goal_circle.py), so a new sheet needs no edits. These are only a reference: the
# centre disc used on 2026-09-26. Set PD_FIXED_GOAL=1 to use them instead.
# Live goal tracking (swap the sheet mid-run): every GOAL_CHECK_S the red circle
# is re-detected on the frame the current plate pose came from. A new goal is
# adopted only after GOAL_CONFIRM consecutive checks agree (within GOAL_AGREE_M) on
# a circle clearly different from the current one (moved > GOAL_MOVE_M or radius
# changed > GOAL_RADIUS_CHANGE_M), so hands in view or a half-swapped sheet can't
# create a false goal. No visible circle -> keep the current goal.
# (A target-sized capture zone with KD 6 was tried 2026-09-26 on a 9 mm dot: no
# improvement, 23% vs 30% in the dot -- reverted.)
# Speed-reference approach (user's step 2): PD rewritten as kd*(v_ref - v) with
# v_ref = (kp/kd)*e, i.e. identical to kp*e - kd*v -- except that |v_ref| is capped,
# so a ball far from the goal is brought in at a limited speed instead of being
# flung at it (far gains alone ask for 2.5 m/s per m of error, 0.375 m/s at 15 cm).
V_REF_MAX = 0.08  # m/s
# Braking curve (research experiment #6): the allowed approach speed also shrinks
# with the distance left, v <= sqrt(2*A_BRAKE*(dist - r/2)), so the ball can stop
# at the target. With only the 8 cm/s cap a ball arriving at a 9 mm dot coasted
# ~10 cm on paper (rolling decel ~0.03 m/s^2) and overshot it by 42-52 mm.
A_BRAKE = 0.05    # m/s^2 -- gentle tilt-braking on top of rolling friction
# Timed pulse + planned brake (research workflow 2026-09-26; Yang & Tomizuka 1988
# adaptive pulse-width control, van de Wouw & Leine 2012 impulsive control under
# uncertain friction). Replaces the ramp-kick: the ramp pushed until motion was SEEN,
# 0.15 s late, so the freed ball carried too much energy and rolled past a small dot
# (82% of exits followed a kick). Now: a pulse just above breakaway for N frames,
# then an open-loop brake for N frames, then coast; afterwards the advance p toward
# the goal is measured and N adapted (N <- N*sqrt(d0/p)), per 3 cm plate cell. A
# pulse that doesn't free the ball makes the next one stronger.
PULSE_A0, PULSE_A_MAX = 0.40, 0.90   # pulse amplitude (action, 1.0 = 5 deg); 0.6 often failed to free the ball on the wall side of a near-edge dot
PULSE_N0, PULSE_N_MIN, PULSE_N_MAX = 3, 2, 8   # pulse width in frames
BRAKE_A = 0.12                       # ~0.6 deg: above rolling friction, below breakaway
COAST_MAX_S = 1.5
PULSE_LOCKOUT_S = 0.5
PULSE_CELL_M = 0.03
# learned per-cell pulse settings persist across runs (smooth from the first second)
PULSE_TABLE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pulse_table.json"))
# Plate map (rl_hw/map_plate.py): per-area local level and breakaway tilt measured on
# the rig. The paper is not uniform: the local "level" varies by ~1 deg across the
# plate and breakaway between ~1.2 and 2.3 deg, which a single global offset and a
# fixed first-pulse strength can't serve. Off by default if the file is missing.
PLATE_MAP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plate_map.json"))
USE_PLATE_MAP = os.environ.get("PD_PLATE_MAP", "0") == "1"  # off: made the near-edge dot worse (2026-09-26)


class PlateMap:
    """Inverse-distance-weighted interpolation of the measured map samples."""

    def __init__(self, path):
        with open(path) as f:
            pts = json.load(f)["points"]
        self.pos = np.array([p["pos"] for p in pts])
        self.level = np.array([[p.get("level_x_deg") or 0.0, p.get("level_y_deg") or 0.0] for p in pts])
        self.static = np.array([np.nanmean([v for v in (p.get("static_x_deg"), p.get("static_y_deg")) if v is not None]
                                           or [np.nan]) for p in pts])

    def _w(self, xy):
        d = np.hypot(*(self.pos - np.asarray(xy)).T)
        return 1.0 / np.maximum(d, 0.01) ** 2

    def level_deg(self, xy):
        w = self._w(xy)
        return (w[:, None] * self.level).sum(0) / w.sum()

    def static_deg(self, xy):
        ok = np.isfinite(self.static)
        w = self._w(xy)[ok]
        return float((w * self.static[ok]).sum() / w.sum()) if ok.any() else None
GOAL_CHECK_S = 0.5
GOAL_CONFIRM = 3
GOAL_AGREE_M = 0.004
GOAL_MOVE_M = 0.008
GOAL_RADIUS_CHANGE_M = 0.005
GOAL = (-0.0078, 0.0060)
GOAL_TOLERANCE = 0.0455
# Direct feedback while watching this live: correcting X and Y together lets the
# combined vector point diagonally, straight at a corner, instead of going through
# the middle of an edge on the way back. Once either axis is out near the edge,
# stop blending -- fully prioritize bringing THAT axis back first before resuming
# normal 2D control, so the path runs through edge-middles, not corners.
EDGE_X, EDGE_Y = 0.10, 0.085
EDGE_PUSH_MIN, EDGE_PUSH_GAIN, EDGE_PUSH_MAX = 0.2, 20.0, 0.4  # action: 0.2 at the line, +0.1 per 5 mm, max 0.4 (2 deg)
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
    total_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    max_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    # optional: hold target in s (0 = never stop early), episode length in steps
    hold_target = float(sys.argv[3]) if len(sys.argv) > 3 else HOLD_TARGET_S
    episode_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, time.strftime("pd_%Y%m%d_%H%M%S.csv"))
    log = open(log_path, "w")
    log.write("t,episode,step,xb,yb,vx,vy,alpha,beta,dist,in_circle,ball_found,a0,a1,status,i0,i1,k0,k1,kicking,"
              "c0r,c0c,c1r,c1c,c2r,c2c,c3r,c3c,goal_x,goal_y,goal_r\n")
    print(f"logging to {log_path}")
    # max_episode_steps default (600) let one truly-stuck episode (the ledge
    # defect, see note above) burn its entire step budget on repeated futile
    # escape attempts. 150 steps (~2.7s of real control time at 55Hz, though
    # wedge-detection/escalation stretches wall-clock well beyond that) is
    # still generous for genuine balancing but ends a hopeless episode fast
    # enough to let the remaining total_steps budget go toward fresh starts,
    # several of which land clear of the defect entirely.
    # 2026-09-26: glass removed, ledge gone -- the 150-step cap above (and the
    # unstick escalation) only existed for the ledge. 150 steps is ~3s of real
    # time, too short to ever contain a 10s hold: truncation -> reset() ->
    # re-level + probe tilt throws the ball out of the circle. No shaking at all.
    env = HardwarePlateEnv(
        fixed_goal=GOAL, goal_tolerance=GOAL_TOLERANCE, max_action_delta=0.5,
        max_episode_steps=episode_steps, allow_unstick=False,
    )
    live_goal = True
    goal_provisional = False
    if os.environ.get("PD_GOAL"):
        # virtual goal for testing: PD_GOAL="x,y,r" in metres; disables live re-detection
        gx_, gy_, gr_ = (float(v) for v in os.environ["PD_GOAL"].split(","))
        goal, goal_tol, live_goal = (gx_, gy_), gr_, False
        env._level_plate()
        print(f"using virtual goal ({goal[0]:+.4f}, {goal[1]:+.4f}) r={goal_tol:.4f} (PD_GOAL, live tracking off)")
    elif os.environ.get("PD_FIXED_GOAL") == "1":
        goal, goal_tol = GOAL, GOAL_TOLERANCE
        print(f"using fixed goal ({goal[0]:+.4f}, {goal[1]:+.4f}) r={goal_tol:.4f} (PD_FIXED_GOAL=1)")
    else:
        # Level the plate first (closed loop): the cached startup posture can leave it
        # several degrees tilted, which rolls the ball away and gives a moving pose.
        env._level_plate()
        try:
            res = detect_goal_circle(env)
        except RuntimeError as e:
            env.close()
            log.close()
            sys.exit(f"could not find the goal circle on the sheet: {e}")
        goal, goal_tol = res["center"], res["radius"]
        save_debug(res, os.path.join(LOG_DIR, os.path.basename(log_path).replace(".csv", "_goal.jpg")))
        print(f"goal circle detected ({res['source']}): centre ({goal[0]:+.4f}, {goal[1]:+.4f}) m, radius "
              f"{goal_tol * 1000:.1f} mm (median of {res['n_frames']} frames, spread {res['spread_m'] * 1000:.1f} mm)")
        goal_provisional = res["source"] == "occluded"
    env.fixed_goal = goal
    env.goal_tolerance = goal_tol
    # Edge guard only where the ball is past the wall threshold AND further out than
    # the goal disc on that side -- a disc drawn near an edge stays reachable.
    def edge_limits(g, r):
        # Guard only well past the goal circle: with it right at the circle's edge
        # (e.g. at 10.4 cm for a 9 mm dot at x=9.5 cm) every small overshoot got a
        # full-tilt shove back -> overshoot the other way -> repeating pattern.
        # 3 cm beyond the circle, capped 1 cm short of the plate's edge.
        ex = max(EDGE_X, min(abs(g[0]) + r + 0.03, env._x_half - 0.01))
        ey = max(EDGE_Y, min(abs(g[1]) + r + 0.03, env._y_half - 0.01))
        if ex > EDGE_X or ey > EDGE_Y:
            print(f"goal is near an edge: edge guard moved out to |x|>{ex:.3f}, |y|>{ey:.3f}")
        return ex, ey

    edge_x, edge_y = edge_limits(goal, goal_tol)
    goal_cands = []
    last_goal_check = 0.0
    n_goal_changes = 0
    t_start = time.time()
    integ = np.zeros(2)
    pos_buf = deque(maxlen=STUCK_WIN)
    kick = np.zeros(2)
    kicking = False
    anchor = None
    cap_t = None
    lockout_until = 0.0
    brk_mag = 0.0
    last_found = True
    phase, ph_steps, u, d0, start_pos, cell, t_coast = "idle", 0, np.zeros(2), 0.0, None, None, 0.0
    pulse_table = {}
    plate_map = None
    if USE_PLATE_MAP and os.path.exists(PLATE_MAP_PATH):
        plate_map = PlateMap(PLATE_MAP_PATH)
        print(f"using plate map with {len(plate_map.pos)} samples ({PLATE_MAP_PATH})")
    try:
        with open(PULSE_TABLE_PATH) as f:
            pulse_table = {tuple(int(v) for v in k.split(",")): list(val) for k, val in json.load(f).items()}
        print(f"loaded learned pulse settings for {len(pulse_table)} plate cells")
    except (OSError, ValueError):
        pass
    PHASE_CODE = {"idle": 0, "pulse": 1, "brake": 2, "coast": 3}
    t_prev = None
    best_hold = 0.0
    hold_start = None
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
            pos_buf.clear()
            kick[:] = 0
            kicking, anchor, cap_t, last_found = False, None, None, True
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
                t_now = time.time()
                dt_real = 0.0 if t_prev is None else min(t_now - t_prev, 0.2)
                t_prev = t_now
                e = np.array([gx, gy])
                stationary = False
                if last_found:  # frozen obs during not-found frames must not read as "still"
                    pos_buf.append((xb, yb))
                    P = np.array(pos_buf)
                    stationary = (len(P) == STUCK_WIN and np.ptp(P[:, 0]) < STUCK_PTP
                                  and np.ptp(P[:, 1]) < STUCK_PTP)
                    # Pulses work anywhere on the plate: with the approach speed capped, a
                    # STILL ball far from the goal only gets kd*V_REF_MAX = 0.16 (0.8 deg),
                    # below the ~2 deg breakaway -- it sat 11 cm away for a whole run.
                    if phase == "idle":
                        if stationary and dist_now > R_DONE and t_now >= lockout_until:
                            u = e / max(dist_now, 1e-6)
                            d0, start_pos = dist_now, np.array([xb, yb])
                            cell = (int(round(xb / PULSE_CELL_M)), int(round(yb / PULSE_CELL_M)))
                            a0 = PULSE_A0
                            if plate_map is not None and plate_map.static_deg((xb, yb)) is not None:
                                a0 = float(np.clip(1.1 * plate_map.static_deg((xb, yb)) / 5.0, 0.3, PULSE_A_MAX))
                            pulse_table.setdefault(cell, [a0, PULSE_N0])
                            phase, ph_steps = "pulse", 0
                    elif phase == "pulse":
                        ph_steps += 1
                        if ph_steps >= pulse_table[cell][1]:
                            phase, ph_steps = "brake", 0
                    elif phase == "brake":
                        ph_steps += 1
                        if ph_steps >= pulse_table[cell][1]:
                            phase, t_coast = "coast", t_now
                            pos_buf.clear()
                    elif phase == "coast" and (stationary or t_now - t_coast > COAST_MAX_S):
                        adv = float(np.dot(np.array([xb, yb]) - start_pos, u))
                        A, N = pulse_table[cell]
                        if adv < 0.001:
                            A = min(A + 0.10, PULSE_A_MAX)  # didn't break free -> stronger
                        else:
                            N = int(np.clip(round(N * np.sqrt(d0 / max(adv, 0.001))), PULSE_N_MIN, PULSE_N_MAX))
                        pulse_table[cell] = [A, N]
                        try:
                            with open(PULSE_TABLE_PATH, "w") as f:
                                json.dump({f"{k[0]},{k[1]}": v for k, v in pulse_table.items()}, f)
                        except OSError:
                            pass
                        print(f"  pulse: d0={d0 * 1000:.1f}mm advance={adv * 1000:.1f}mm -> cell {cell} A={A:.2f} N={N}")
                        phase = "idle"
                        lockout_until = t_now + PULSE_LOCKOUT_S
                        pos_buf.clear()
                if phase == "pulse":
                    kick = pulse_table[cell][0] * u
                elif phase == "brake":
                    kick = -BRAKE_A * u
                else:
                    kick = np.zeros(2)
                kicking = phase != "idle"
                # the integral learns the level bias only from a rolling ball, never stiction
                if dist_now < I_ZONE and not kicking and not stationary:
                    integ = np.clip(integ + KI * e * dt_real, -I_MAX, I_MAX)
                v_ref = (kp / kd) * e
                v_ref_mag = float(np.hypot(*v_ref))
                v_allow = min(V_REF_MAX, float(np.sqrt(2 * A_BRAKE * max(dist_now - 0.5 * goal_tol, 0.0))))
                if v_ref_mag > v_allow:
                    v_ref *= v_allow / max(v_ref_mag, 1e-9)
                pd = kd * (v_ref - np.array([vx, vy]))
                # per-area level feedforward from the plate map (action units: 5 deg = 1)
                ff = plate_map.level_deg((xb, yb)) / 5.0 if plate_map is not None else np.zeros(2)
                if phase in ("pulse", "brake"):
                    pd = pd - np.dot(pd, u) * u  # the pulse/brake own the goal direction
                action = np.clip(
                    np.array([pd[0] + integ[0] + kick[0] + ff[0], pd[1] + integ[1] + kick[1] + ff[1]], dtype=np.float32),
                    -0.8, 0.8,
                )
                # Hard override once near an edge -- full brake straight back toward
                # center on whichever axis is in danger, overriding the blended 2D
                # goal-seeking above. Without this the combined X+Y correction can
                # point diagonally at a corner instead of cutting back through the
                # middle of the edge it's near.
                # Gentle, proportional wall guard (was a full +-1.0 = 5 deg slam, which
                # shot the ball back across the plate at 140-160 mm/s every time it
                # overshot toward the wall -- the recurring ~46 mm overshoots).
                # It guarantees a MINIMUM push away from the wall but never caps a
                # stronger one: a stiction pulse must still get through (overriding it
                # left a ball stuck at the sticky top edge for a whole run).
                for ax, pos, edge in ((0, xb, edge_x), (1, yb, edge_y)):
                    if abs(pos) > edge:
                        push = min(EDGE_PUSH_MAX, EDGE_PUSH_MIN + EDGE_PUSH_GAIN * (abs(pos) - edge))
                        away = -np.sign(pos)
                        action[ax] = away * max(push, away * action[ax])
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
                # Only a frame where the ball was actually detected counts: during
                # step()'s not-found grace frames obs holds the frozen last position.
                ball_found = bool(info.get("ball_found", True))
                last_found = ball_found
                in_circle = ball_found and dist < goal_tol
                ep_in_circle += int(in_circle)
                ep_steps += 1

                now = time.time()
                # A lone missed detection neither extends nor breaks a hold (the
                # detector drops single frames on a stationary ball); a detected
                # frame outside the circle, or a real ball_lost, does break it.
                if in_circle:
                    if hold_start is None:
                        hold_start = now
                    best_hold = max(best_hold, now - hold_start)
                elif ball_found or info["status"] == "ball_lost":
                    hold_start = None
                log.write(f"{now - t_start:.3f},{episode},{i},{obs[0]:.5f},{obs[1]:.5f},{obs[2]:.4f},"
                          f"{obs[3]:.4f},{obs[4]:.5f},{obs[5]:.5f},{dist:.5f},{int(in_circle)},{int(ball_found)},"
                          f"{action[0]:.3f},{action[1]:.3f},{info['status']},"
                          f"{integ[0]:.4f},{integ[1]:.4f},{kick[0]:.4f},{kick[1]:.4f},{PHASE_CODE[phase]},"
                          + (",".join(f"{v:.1f}" for v in env.last_inner_corners.ravel())
                             if getattr(env, "last_inner_corners", None) is not None else ",,,,,,,")
                          + f",{goal[0]:.5f},{goal[1]:.5f},{goal_tol:.5f}\n")

                if live_goal and now - last_goal_check >= GOAL_CHECK_S and getattr(env, "_last_frame", None) is not None:
                    last_goal_check = now
                    cand = detect_on_frame(env, env._last_frame)
                    # a provisional goal (fitted with the ball covering part of it) is
                    # refined by the first clean view, however small the difference
                    mv, rc = (0.0005, 0.0005) if goal_provisional else (GOAL_MOVE_M, GOAL_RADIUS_CHANGE_M)
                    differs = cand is not None and (
                        np.hypot(cand[0][0] - goal[0], cand[0][1] - goal[1]) > mv
                        or abs(cand[1] - goal_tol) > rc)
                    if not differs:
                        goal_cands.clear()
                    else:
                        if goal_cands and np.hypot(*(cand[0] - goal_cands[-1][0])) > GOAL_AGREE_M:
                            goal_cands.clear()
                        goal_cands.append(cand)
                        if len(goal_cands) >= GOAL_CONFIRM:
                            c = np.median([g for g, _ in goal_cands], axis=0)
                            goal = (float(c[0]), float(c[1]))
                            goal_tol = float(np.median([r for _, r in goal_cands]))
                            env.fixed_goal = goal
                            env.goal = np.array(goal, dtype=np.float32)
                            env.goal_tolerance = goal_tol
                            edge_x, edge_y = edge_limits(goal, goal_tol)
                            # new target: restart the hold and any kick; keep the learned
                            # level bias (it belongs to the plate, not the sheet)
                            hold_start = None
                            kicking, cap_t, lockout_until = False, None, 0.0
                            kick[:] = 0
                            pos_buf.clear()
                            goal_cands.clear()
                            n_goal_changes += 1
                            goal_provisional = False
                            save_last_goal(goal, goal_tol)
                            print(f"\n>>> NEW GOAL circle: centre ({goal[0]:+.4f}, {goal[1]:+.4f}) m, "
                                  f"radius {goal_tol * 1000:.1f} mm <<<\n")

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
                          f"I=({integ[0]:+.3f},{integ[1]:+.3f}) K=({kick[0]:+.2f},{kick[1]:+.2f}){'*' if kicking else ''}  "
                          f"status={info['status']}")
                if hold_target > 0 and best_hold >= hold_target:
                    print(f"  reached {hold_target:.0f}s continuous hold -- stopping")
                    break
                if terminated or truncated:
                    print(f"  episode {episode} ended at step {i}: {info['status']}")
                    break
            total_in_circle += ep_in_circle
            total_taken += ep_steps
            pct = 100 * ep_in_circle / max(1, ep_steps)
            print(f"  episode {episode} summary: {ep_in_circle}/{ep_steps} steps in circle ({pct:.1f}%)")
            hold_start = None  # a hold never spans a reset()
            if hold_target > 0 and best_hold >= hold_target:
                break

        overall_pct = 100 * total_in_circle / max(1, total_taken)
        print(f"\n=== overall: {total_in_circle}/{total_taken} steps in circle ({overall_pct:.1f}%) "
              f"across {episode} episode(s) ===")
        rate = total_taken / max(1e-6, time.time() - t_start)
        print(f"=== goal changes during run: {n_goal_changes} ===")
        print(f"=== longest continuous hold inside goal circle: {best_hold:.2f} s "
              f"(target {HOLD_TARGET_S:.0f} s, {'SUCCESS' if best_hold >= HOLD_TARGET_S else 'not reached'}), "
              f"~{rate:.1f} steps/s overall, log: {log_path} ===")
    finally:
        log.close()
        env.close()


if __name__ == "__main__":
    main()
