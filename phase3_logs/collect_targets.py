#!/usr/bin/env python3
"""
Unattended data collection through the running UI controller (2026-09-27): random
targets on the plate, sent as scripted plate-position targets (same routing as a click),
in blocks of BLOCK trips per controller (classic / ODIL / RL, the policies pure, i.e.
without the classic settle). The controller's own CSV log (rl_hw/pd_logs) has the full
trajectories; this script writes one summary line per trip to a .jsonl.

  python3 collect_targets.py [n_trips] [out.jsonl] [controllers, e.g. classic,odil,learned]
Stop early: touch the STOP file named below.
"""
import json
import math
import os
import random
import sys
import time
import urllib.request

URL = "http://localhost:8000"
BLOCK = 10
REACH_TIMEOUT_S, HOLD_S, AFTER_S = 12.0, 2.0, 4.0
X_MAX, Y_MAX, R = 0.10, 0.08, 0.012
STOP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STOP_COLLECT")


def get():
    with urllib.request.urlopen(URL + "/state", timeout=2) as r:
        return json.load(r)


def cmd(**c):
    req = urllib.request.Request(URL + "/cmd", data=json.dumps(c).encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=2).read()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    out = sys.argv[2] if len(sys.argv) > 2 else time.strftime("collect_%Y%m%d_%H%M%S.jsonl")
    ctrls = (sys.argv[3] if len(sys.argv) > 3 else "classic,odil,learned").split(",")
    rng = random.Random(int(time.time()))
    holes = []
    try:
        holes = json.load(open(os.path.join(os.path.dirname(__file__), "..", "holes.json")))
    except (OSError, ValueError):
        pass
    f = open(out, "a")
    for i in range(n):
        if os.path.exists(STOP):
            print("STOP file found -- ending")
            break
        ctrl = ctrls[(i // BLOCK) % len(ctrls)]
        if i % BLOCK == 0:
            # "odil+settle" / "learned+settle" = the policy with the classic near-field settle
            base = ctrl.split("+")[0]
            cmd(cmd="controller", which=base)
            cmd(cmd="hybrid", on=(base == "classic" or ctrl.endswith("+settle")))
            time.sleep(0.5)
        s = get()
        while s.get("ball") is None:          # ball lost / being reloaded: wait
            time.sleep(1.0)
            s = get()
        bx, by = s["ball"]
        for _ in range(200):
            tx, ty = rng.uniform(-X_MAX, X_MAX), rng.uniform(-Y_MAX, Y_MAX)
            if math.hypot(tx - bx, ty - by) < 0.04:
                continue
            if any(math.hypot(tx - h["center"][0], ty - h["center"][1]) < h["radius"] + 0.02 for h in holes):
                continue
            break
        start = math.hypot(tx - bx, ty - by)
        cmd(cmd="target_xy", x=tx, y=ty, r=R)
        t0 = time.time()
        j0 = (s.get("jerk_sum", 0.0), s.get("jerk_n", 0))
        t_reach, hold_t, lost, inside, samples, dists = None, None, False, 0, 0, []
        while True:
            time.sleep(0.1)
            s = get()
            now = time.time() - t0
            if s.get("ball") is None:
                lost = True
            if s.get("in_target"):
                t_reach = t_reach if t_reach is not None else now
            if t_reach is not None:
                samples += 1
                inside += bool(s.get("in_target"))
                if s.get("dist") is not None:
                    dists.append(s["dist"])
            if (t_reach is None and now > REACH_TIMEOUT_S) or (t_reach is not None and now - t_reach > AFTER_S) \
                    or (lost and now > REACH_TIMEOUT_S):
                break
        jerk = ((s.get("jerk_sum", 0.0) - j0[0]) / max(1, s.get("jerk_n", 0) - j0[1])) if "jerk_sum" in s else None
        rec = dict(i=i, t=time.time(), ctrl=ctrl, jerk=jerk, target=[tx, ty], start=[bx, by], start_dist=start,
                   reached=t_reach is not None, t_reach=t_reach, inside_after=(inside / samples) if samples else 0.0,
                   final_dist=dists[-1] if dists else None, max_after=max(dists) if dists else None, lost=lost)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"{i:3d} {ctrl:8s} {start * 1000:5.0f} mm  reached={rec['reached']!s:5s} "
              f"t={t_reach if t_reach is None else round(t_reach, 1)}  inside_after={rec['inside_after']:.2f}  "
              f"final={None if rec['final_dist'] is None else round(rec['final_dist'] * 1000, 1)} mm  "
              f"jerk={None if jerk is None else round(jerk, 4)}", flush=True)
    cmd(cmd="controller", which="classic")
    cmd(cmd="hybrid", on=True)


if __name__ == "__main__":
    main()
