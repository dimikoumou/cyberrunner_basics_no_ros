#!/usr/bin/env python3
"""
Videos for sharing (2026-10-03), drawn top-down (no camera footage):

  sim_odil_vs_ppo.mp4      simulation: ODIL v11 (+ stiction comp.) vs PPO v3, same starts, targets, physics
  rig_replay_odil_vs_sac.mp4  REAL recorded ball paths from session 1 (rig-only learning): ODIL after 60 rig
                           min vs SAC after 120 rig min on the same test targets (camera measurements replayed)

  ../../.venv-rl/bin/python3 make_videos.py [n_sim_episodes=6] [n_rig_trips=10]
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIM = os.path.join(ROOT, "rl_sim")
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "video")
sys.path.insert(0, SIM)
FPS, PX = 29, 2.4                      # frames/s, pixels per mm
PW, PH = 0.26, 0.216                   # plate (m)
W, H = int(PW * 1000 * PX), int(PH * 1000 * PX)
HEAD = 70
BG, PLATE, INK, MUTED = (245, 247, 250), (255, 255, 255), (40, 24, 17), (128, 114, 107)
COL = {"ODIL": (235, 99, 37), "PPO": (38, 38, 220), "SAC": (38, 38, 220)}     # BGR: blue, red


def to_px(p):
    return int(W / 2 + p[0] * 1000 * PX), int(HEAD + H / 2 - p[1] * 1000 * PX)


def panel(name, sub, pos, trail, goal, radius, tilt_deg, t, reached):
    img = np.full((H + HEAD + 40, W, 3), BG, np.uint8)
    cv2.rectangle(img, (0, HEAD), (W - 1, HEAD + H - 1), PLATE, -1)
    cv2.rectangle(img, (0, HEAD), (W - 1, HEAD + H - 1), (200, 200, 200), 2)
    col = COL[name.split()[0]]
    cv2.putText(img, name, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)
    cv2.putText(img, sub, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1, cv2.LINE_AA)
    g = to_px(goal)
    cv2.circle(img, g, int(radius * 1000 * PX), (60, 170, 60) if reached else (120, 120, 120), 2, cv2.LINE_AA)
    cv2.drawMarker(img, g, (60, 170, 60), cv2.MARKER_CROSS, 10, 1)
    for a, b in zip(trail[:-1], trail[1:]):
        cv2.line(img, to_px(a), to_px(b), tuple(int(0.45 * c + 0.55 * 255) for c in col), 2, cv2.LINE_AA)
    cv2.circle(img, to_px(pos), int(6.5 * PX), (90, 90, 90), -1, cv2.LINE_AA)
    cv2.circle(img, to_px(pos), int(6.5 * PX), (30, 30, 30), 1, cv2.LINE_AA)
    # tilt command as an arrow from the plate centre (4 deg = 60 px)
    c = (W - 60, HEAD + H + 20)
    tip = (int(c[0] + tilt_deg[0] / 4 * 30), int(c[1] - tilt_deg[1] / 4 * 30))
    cv2.circle(img, c, 30, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.arrowedLine(img, c, tip, col, 2, cv2.LINE_AA, tipLength=0.3)
    d = 1000 * float(np.hypot(*(np.asarray(goal) - pos)))
    cv2.putText(img, f"t {t:4.1f} s   distance {d:5.1f} mm   tilt cmd", (12, HEAD + H + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK, 1, cv2.LINE_AA)
    return img


def writer(name, w, h):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"avc1"), FPS, (w, h))
    if not vw.isOpened():
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    return vw, p


def title_card(w, h, lines, n=FPS * 3):
    img = np.full((h, w, 3), BG, np.uint8)
    for i, (txt, sc, col) in enumerate(lines):
        cv2.putText(img, txt, (40, 90 + 48 * i), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2 if sc > 0.8 else 1, cv2.LINE_AA)
    return [img] * n


# ---- simulation ---------------------------------------------------------------------------
def sim_video(n_eps):
    from plate_goal_env import PlateGoalEnv
    from eval_controllers import MLP
    from odil_friction_comp import ODILFrictionComp
    odil = ODILFrictionComp(os.path.join(SIM, "runs", "odil_v11", "odil_policy.npz"))
    pp = os.path.join(SIM, "runs", "plate_goal_v3", "policy_final.npz")
    ppo_net = MLP(pp if os.path.exists(pp) else os.path.join(SIM, "runs", "plate_goal_v3", "policy.npz"))
    ctrls = [("ODIL v11", "simulation | gradient through the physics", odil),
             ("PPO v3", "simulation | reinforcement learning", lambda o: np.clip(ppo_net(o), -1, 1))]
    frames = title_card(2 * W + 10, H + HEAD + 40, [
        ("Ball on plate in simulation", 1.1, INK),
        ("Left: ODIL (physics-based optimisation)   Right: PPO (reinforcement learning)", 0.6, INK),
        ("Same start, target and physics for both. Circle = target (radius varies), arrow = tilt command.", 0.5, MUTED),
        ("Simulator with stiction, motor delay and camera noise; real time.", 0.5, MUTED)])
    for ep in range(n_eps):
        runs = []
        for name, sub, c in ctrls:
            env = PlateGoalEnv(seed=777 + ep)
            o, _ = env.reset(seed=777 + ep)
            env.switch_at = -1
            if hasattr(c, "reset"):
                c.reset()
            rec, reached = [], False
            for k in range(300):                      # ~10 s
                o, r, te, tr, info = env.step(c(o))
                reached = reached or info["inside"]
                rec.append((env.sim.pos.copy(), 5.0 * env.applied, k / FPS, reached))
            runs.append((name, sub, env.goal.copy(), env.radius, rec))
        for k in range(300):
            ims = []
            for name, sub, goal, rad, rec in runs:
                pos, tilt, t, rch = rec[k]
                trail = [q[0] for q in rec[max(0, k - 45):k + 1]]
                ims.append(panel(name, f"{sub} | episode {ep + 1}", pos, trail, goal, rad, tilt, t, rch))
            frames.append(np.hstack([ims[0], np.full((ims[0].shape[0], 10, 3), 220, np.uint8), ims[1]]))
    vw, p = writer("sim_odil_vs_ppo.mp4", frames[0].shape[1], frames[0].shape[0])
    for f in frames:
        vw.write(f)
    vw.release()
    return p


# ---- rig replay ---------------------------------------------------------------------------
def trips_from_rows(rows):
    st = [i for i, q in enumerate(rows) if q["a0"] == "0.0000" and q["a1"] == "0.0000"]
    trips = [rows[a:b] for a, b in zip(st, st[1:] + [len(rows)])]
    return [t for t in trips if len(t) > 2]          # a lone zero-command frame is not a trip start


def rig_trips(csv_rows, k_block=None):
    if k_block is None:
        return trips_from_rows(csv_rows)
    blocks, cur = [], []                     # one test = everything between two SAC training stretches
    for q in csv_rows:                       # (a lost ball inserts 'wait_ball' frames inside a test)
        if q["tag"] == "test":
            cur.append(q)
        elif q["tag"] == "sac" and cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    return trips_from_rows(blocks[k_block])


def rig_video(n_trips):
    odil_rows = list(csv.DictReader(open(os.path.join(DATA, "test_odil1_odil_60min.csv"))))
    sac_rows = list(csv.DictReader(open(os.path.join(DATA, "sac1.csv"))))
    T_odil, T_sac = rig_trips(odil_rows), rig_trips(sac_rows, k_block=3)          # 4th SAC test = 120 min
    J_odil = json.load(open(os.path.join(DATA, "test_odil1_odil_60min.json")))["trips"]
    J_sac = json.load(open(os.path.join(DATA, "test_sac1_sac_120min.json")))["trips"]
    assert len(T_odil) == 30 and len(T_sac) == 30, (len(T_odil), len(T_sac))
    sides = [("ODIL rig-only", "real rig recording | after 60 rig minutes of data", T_odil, J_odil),
             ("SAC rig-only", "real rig recording | after 120 rig minutes of data", T_sac, J_sac)]
    frames = title_card(2 * W + 10, H + HEAD + 40, [
        ("Learning on the real rig only: replay of recorded data", 1.0, INK),
        ("Left: ODIL after 60 rig minutes   Right: SAC after 120 rig minutes", 0.6, INK),
        ("Ball positions as measured by the camera, drawn top-down; same 30 test targets (12 mm).", 0.5, MUTED),
        ("Each trip starts where the previous one ended. Real time.", 0.5, MUTED)])
    for i in range(n_trips):
        recs = []
        for name, sub, T, J in sides:
            tr = T[i]
            pos = [np.array([float(q["xb"]), float(q["yb"])]) for q in tr]
            tilt = [5.0 * 0.8 * np.array([float(q["a0"]), float(q["a1"])]) for q in tr]
            t0 = float(tr[0]["t"])
            ts = [float(q["t"]) - t0 for q in tr]
            g = np.array(J[i]["target"])
            reached = [bool(np.hypot(*(g - p)) < 0.012) for p in pos]
            reached = list(np.maximum.accumulate(reached))
            recs.append((name, sub, g, pos, tilt, ts, reached))
        n = max(len(r[3]) for r in recs)
        for k in range(n):
            ims = []
            for name, sub, g, pos, tilt, ts, reached in recs:
                j = min(k, len(pos) - 1)
                ims.append(panel(name, f"{sub} | target {i + 1}", pos[j], pos[max(0, j - 45):j + 1], g, 0.012,
                                 tilt[j], ts[j], reached[j]))
            frames.append(np.hstack([ims[0], np.full((ims[0].shape[0], 10, 3), 220, np.uint8), ims[1]]))
    vw, p = writer("rig_replay_odil_vs_sac.mp4", frames[0].shape[1], frames[0].shape[0])
    for f in frames:
        vw.write(f)
    vw.release()
    return p


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    nr = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if "--rig-only" not in sys.argv:
        print(sim_video(ns))
    print(rig_video(nr))
