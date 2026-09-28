"""
Self-calibration (2026-09-27): what the rig measures about itself at the press of a button.

  1. level      -- servo to 0 deg on the CAMERA-measured angle, note the motor positions
  2. motors     -- step each tilt motor +-STEP ticks from level (slowly), measure with the
                   camera how far the plate really tilts: deg per tick each way + play
                   (hysteresis). A motor whose travel barely moves the plate is reported --
                   that is what motor 3's slipping link looked like.
  3. compare    -- against the previous calibration: level position moved / response changed
  4. tour       -- (pd_balance) the ball visits a 3x3 grid with the classic controller, which
                   learns the local slope ("bias table") where it comes to rest
Results: calibration.json (+ calibration_history.jsonl), warnings as readable text.
"""
import json
import os
import time

import numpy as np

CAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "calibration.json"))
HIST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "calibration_history.jsonl"))
STEP = 150
EXPECTED_DEG_PER_TICK = {1: 1 / 140.0, 3: 1 / 100.0}
TOUR = [(x, y) for y in (0.06, 0.0, -0.06) for x in (-0.08, 0.0, 0.08)]


def _measure(env, n=15):
    a, b = [], []
    for _ in range(n):
        env._read_state()
        if env._meas_tilt is not None and getattr(env, "_pose_ok", True):
            a.append(env._meas_tilt[0])
            b.append(env._meas_tilt[1])
        time.sleep(0.03)
    if len(a) < n // 3:
        return None
    return float(np.median(a)), float(np.median(b))


def _goto(env, dxl_id, target, setp):
    cur = env._cmd_ticks.get(dxl_id, target)
    while cur != target:
        cur += int(np.clip(target - cur, -40, 40))
        setp(env.port_handler, env.packet_handler, dxl_id, cur)
        env._cmd_ticks[dxl_id] = cur
        time.sleep(0.04)
    time.sleep(1.0)


def tilt_calibration(env, setp, log=print):
    """steps 1-3; returns (result dict, list of warning strings). Leaves the plate level."""
    warnings = []
    env._servo_level(max_s=8.0)
    lvl = {1: int(env._cmd_ticks[1]), 3: int(env._cmd_ticks[3])}
    m0 = _measure(env)
    if m0 is None:
        return None, ["could not see all plate markers -- calibration skipped"]
    res = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "level_ticks": lvl,
           "level_meas_deg": {"alpha": m0[0], "beta": m0[1]}, "motors": {}}
    for dxl_id, ax in ((1, 0), (3, 1)):
        pts = []
        for tgt in (lvl[dxl_id] + STEP, lvl[dxl_id], lvl[dxl_id] - STEP, lvl[dxl_id]):
            _goto(env, dxl_id, tgt, setp)
            m = _measure(env)
            pts.append(None if m is None else m[ax])
        if any(p is None for p in pts):
            warnings.append(f"motor {dxl_id}: plate markers hidden during the test -- not measured")
            continue
        up, back1, down, back2 = pts
        gain_up = abs(up - back1) / STEP
        gain_down = abs(back2 - down) / STEP
        play = abs(back1 - back2)
        rel = 0.5 * (gain_up + gain_down) / EXPECTED_DEG_PER_TICK[dxl_id]
        res["motors"][str(dxl_id)] = {"deg_per_tick_up": gain_up, "deg_per_tick_down": gain_down,
                                      "play_deg": play, "response_vs_expected": rel}
        log(f"  motor {dxl_id}: {gain_up * 100:.2f} / {gain_down * 100:.2f} deg per 100 ticks (up/down), "
            f"play {play:.2f} deg, {rel * 100:.0f} % of the expected response")
        if min(gain_up, gain_down) < 0.5 * EXPECTED_DEG_PER_TICK[dxl_id]:
            warnings.append(f"motor {dxl_id} responds more weakly in one direction "
                            f"({min(gain_up, gain_down) * 100:.2f} deg per 100 ticks) -- the camera loop compensates")
        if play > 1.5:
            warnings.append(f"motor {dxl_id}: {play:.1f} deg of play -- the camera loop covers it")
    env._servo_level(max_s=5.0)
    try:
        with open(CAL_PATH) as f:
            prev = json.load(f)
        for k in ("1", "3"):
            d = res["level_ticks"][int(k)] - prev["level_ticks"][k]
            if abs(d) > 150:
                warnings.append(f"motor {k}: level position moved {d:+d} ticks since {prev['t']}")
            if k in prev.get("motors", {}) and k in res["motors"]:
                r0, r1 = prev["motors"][k]["response_vs_expected"], res["motors"][k]["response_vs_expected"]
                if abs(r1 - r0) > 0.25 * max(r0, 1e-3):
                    warnings.append(f"motor {k}: tilt response changed {r0 * 100:.0f} % -> {r1 * 100:.0f} %")
    except (OSError, ValueError, KeyError):
        pass
    res["warnings"] = warnings
    res["level_ticks"] = {str(k): v for k, v in res["level_ticks"].items()}
    with open(CAL_PATH, "w") as f:
        json.dump(res, f, indent=1)
    with open(HIST_PATH, "a") as f:
        f.write(json.dumps(res) + "\n")
    return res, warnings
