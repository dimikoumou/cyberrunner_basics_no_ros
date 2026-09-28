#!/usr/bin/env python3
"""
System identification of the real plate from rig logs (2026-09-27), for ODIL's model:
  1. command -> measured plate angle: pure delay (steps) + first-order time constant, per axis
  2. measured angle -> ball acceleration: gain k_acc (m/s^2 per deg) + rolling friction a_roll
  3. breakaway: the tilt at which a resting ball starts to roll
Frames are used only when the ball is detected and the controller was actually running a
command (not holding level for a lost ball). Axes: ball x <- beta - 2.55, ball y <- -(alpha + 1.1).

  python3 sysid_rig.py log1.csv [log2.csv ...]
"""
import csv
import json
import sys

import numpy as np

LEVEL = (-1.1, 2.55)       # (alpha, beta) deg
TILT_MAX = 5.0


def load(fn):
    r = list(csv.DictReader(open(fn)))
    g = lambda k: np.array([float(q[k]) if q.get(k) not in (None, "") else np.nan for q in r])
    d = {k: g(k) for k in ("t", "xb", "yb", "alpha", "beta", "a0", "a1", "ball_found", "episode")}
    d["tilt_x"] = np.degrees(d["beta"]) - LEVEL[1]            # plate tilt driving ball x (deg)
    d["tilt_y"] = -(np.degrees(d["alpha"]) - LEVEL[0])
    d["cmd_x"], d["cmd_y"] = d["a0"] * TILT_MAX, d["a1"] * TILT_MAX
    return d


def fit_servo(cmd, meas, ok):
    """meas[t] ~ first-order lag of cmd[t - D]; grid over delay D and alpha = dt/tau"""
    best = None
    for D in range(0, 9):
        for al in np.linspace(0.1, 1.0, 19):
            y = np.zeros_like(cmd)
            for t in range(1, len(cmd)):
                y[t] = y[t - 1] + al * (cmd[t - D] if t >= D else 0.0) - al * y[t - 1]
            m = ok.copy()
            m[:40] = False
            # compare changes only (removes the constant offset of a slightly wrong level)
            e = (np.diff(meas) - np.diff(y))[m[1:]]
            e = e[np.isfinite(e)]
            if len(e) < 200:
                continue
            c = float(np.mean(e ** 2))
            if best is None or c < best[0]:
                best = (c, D, al)
    return best


def main():
    files = sys.argv[1:]
    out = {"files": files}
    servo, accs = {"x": [], "y": []}, []
    brk = []
    for fn in files:
        d = load(fn)
        t = d["t"]
        dt = np.median(np.diff(t))
        ok = (d["ball_found"] > 0) & np.isfinite(d["tilt_x"]) & np.isfinite(d["tilt_y"])
        # 1. servo per axis, on a stretch of up to 6000 frames with real commands
        n = min(len(t), 6000)
        for ax in ("x", "y"):
            b = fit_servo(d[f"cmd_{ax}"][:n], d[f"tilt_{ax}"][:n], ok[:n])
            if b:
                servo[ax].append((b[1], b[2]))
        # 2. acceleration from positions (central differences on a 5-frame smoothed track)
        for ax, p in (("x", d["xb"]), ("y", d["yb"])):
            ps = np.convolve(p, np.ones(5) / 5, mode="same")
            v = np.gradient(ps, t)
            a = np.gradient(v, t)
            tilt = d[f"tilt_{ax}"]
            m = ok & (np.abs(v) > 0.02) & (np.abs(v) < 0.3) & (np.abs(a) < 3) & (np.abs(tilt) < 8)
            m &= np.abs(p) < 0.11                          # away from the walls
            accs.append(np.column_stack([tilt[m], np.sign(v[m]), a[m]]))
        # 3. breakaway: a ball at rest >= 0.5 s that starts moving: |tilt| at that moment
        sp = np.hypot(np.gradient(d["xb"], t), np.gradient(d["yb"], t))
        rest = 0
        for i in range(len(t)):
            if not ok[i]:
                rest = 0
                continue
            if sp[i] < 0.004:
                rest += 1
            else:
                if rest * dt > 0.5 and sp[i] > 0.01:
                    brk.append(float(np.hypot(d["tilt_x"][i], d["tilt_y"][i])))
                rest = 0
    A = np.vstack(accs)
    X = np.column_stack([A[:, 0], -A[:, 1], np.ones(len(A))])     # a = k*tilt - a_roll*sign(v) + c
    coef, *_ = np.linalg.lstsq(X, A[:, 2], rcond=None)
    k_acc, a_roll = float(coef[0]), float(coef[1])
    for ax in ("x", "y"):
        if servo[ax]:
            D = int(np.median([s[0] for s in servo[ax]]))
            al = float(np.median([s[1] for s in servo[ax]]))
            out[f"servo_{ax}"] = {"delay_steps": D, "alpha": al, "tau_s": (1 / al - 1) / 29.0 if al < 1 else 0.0}
    out.update(k_acc=k_acc, a_roll=a_roll, n_acc=int(len(A)),
               breakaway_deg={"n": len(brk), "median": float(np.median(brk)) if brk else None,
                              "p25": float(np.percentile(brk, 25)) if brk else None,
                              "p75": float(np.percentile(brk, 75)) if brk else None})
    print(json.dumps(out, indent=1))
    json.dump(out, open("rig_sysid.json", "w"), indent=1)


if __name__ == "__main__":
    main()
