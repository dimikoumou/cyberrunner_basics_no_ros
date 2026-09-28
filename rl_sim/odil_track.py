#!/usr/bin/env python3
"""
ODIL for PATH TRACKING (2026-09-27): the ball follows a moving reference r(t) along a line
(the rig's PathTracker: corner-aware speed profile), for precise drawing -> maze tracing.

Same discrete-loss idea as odil_plate_v9 (trajectory points and the policy optimised together,
physics as a midpoint residual), with the rig-fitted plate model (3-stage tilt lag, rolling
friction 0.042, static "hold" friction, per-trajectory gain/stiction/slope) but:
  - no goal/end point: the loss asks the ball to STAY ON the moving reference,
        W_TRACK * sum_n |p^n - r(t_n)|^2  (+ smooth commands, + physics),
    over fixed 3 s windows of random reference pieces (arcs, straights, corners, waves) at
    1-4 cm/s, starting a few mm off the reference;
  - policy inputs (14): tracking error (r - p)/0.02, speed error (v_r - v)/0.05, the
    reference's acceleration as a feed-forward tilt a_r/(0.113*5), own speed v/0.1, the
    observer's tilt states /5 (as on the rig), and a leaky integral of the tracking error.
TrackPolicy (numpy, no torch) runs it in the simulator eval and on the rig.

  ../.venv-rl/bin/python3 odil_track.py [iters_per_round] [run_name]
"""
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAU_OBS, K_NOM = 0.045, 0.113
TZ, Z_SCALE = 1.0, 0.01
U_MAX_DEG = 4.0


class TrackPolicy:
    """numpy runtime of the tracking policy; call(ref, pos, vel) -> env action (deg/5)"""
    DT = 1.0 / 29.0

    def __init__(self, path):
        w = np.load(path)
        n = sum(1 for k in w.files if k.startswith("W") and k[1:].isdigit())
        self.layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(n)]
        self.reset()

    def reset(self):
        self.o1, self.o2, self.z = np.zeros(2), np.zeros(2), np.zeros(2)

    def __call__(self, ref, pos, vel, dt=None):
        dt = dt or self.DT
        e = np.asarray(ref["p"]) - np.asarray(pos)
        self.z += (e / Z_SCALE - self.z / TZ) * dt
        feat = np.concatenate([e / 0.02, (np.asarray(ref["v"]) - vel) / 0.05, np.asarray(ref["a"]) / (K_NOM * 5.0),
                               np.asarray(vel) / 0.1, self.o1 / 5.0, self.o2 / 5.0, self.z])
        h = feat
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if i < len(self.layers) - 1:
                h = np.tanh(h)
        u = U_MAX_DEG * np.tanh(h)                     # commanded tilt, deg
        self.o1 += (u - self.o1) * min(1.0, dt / TAU_OBS)
        self.o2 += (self.o1 - self.o2) * min(1.0, dt / TAU_OBS)
        return u / 5.0


def _references(rng, m, n_pts, T):
    """m reference windows: arrays p, v, a of shape (m, n_pts, 2) sampled every T/(n_pts-1)"""
    sys.path.insert(0, os.path.join(HERE, "..", "rl_hw"))
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    gc = types.ModuleType("goal_circle")
    gc.red_mask = gc.pixel_to_plate = None
    sys.modules.setdefault("goal_circle", gc)
    from line_path import PathTracker
    import shapes
    dt = T / (n_pts - 1)
    P, V, A = np.zeros((m, n_pts, 2)), np.zeros((m, n_pts, 2)), np.zeros((m, n_pts, 2))
    for j in range(m):
        kind = rng.integers(0, 4)
        c = rng.uniform(-0.05, 0.05, 2)
        th0 = rng.uniform(0, 2 * np.pi)
        if kind == 0:        # arc
            R = rng.uniform(0.02, 0.10)
            t = np.linspace(0, rng.uniform(0.6, 1.8) * np.pi, 120)
            pts = c + R * np.column_stack([np.cos(th0 + t), np.sin(th0 + t)])
        elif kind == 1:      # straight
            d = np.array([np.cos(th0), np.sin(th0)])
            pts = c + np.outer(np.linspace(-0.08, 0.08, 60), d)
        elif kind == 2:      # corner
            ang = rng.uniform(np.radians(30), np.radians(150))
            d1 = np.array([np.cos(th0), np.sin(th0)])
            d2 = np.array([np.cos(th0 + ang), np.sin(th0 + ang)])
            pts = np.vstack([c - d1 * 0.06, c, c + d2 * 0.06])
        else:                # wave
            d = np.array([np.cos(th0), np.sin(th0)])
            nrm = np.array([-d[1], d[0]])
            s = np.linspace(-0.08, 0.08, 120)
            pts = c + np.outer(s, d) + np.outer(rng.uniform(0.01, 0.03) * np.sin(s * rng.uniform(20, 60)), nrm)
        pts = shapes.resample(np.clip(pts, -0.11, 0.09))
        tr = PathTracker(pts, False, pts[0], keep_direction=True, v=rng.uniform(0.01, 0.04))
        # start somewhere along the path (already moving) or at its start
        warm = int(rng.integers(0, 40))
        for _ in range(warm):
            tr.update(tr.point(tr.s), dt)
        for k in range(n_pts):
            ref = tr.update(tr.point(tr.s), dt)
            P[j, k], V[j, k], A[j, k] = ref["p"], ref["v"], ref["a"]
    return P, V, A


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("ODIL_THREADS", "6")))
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    name = sys.argv[2] if len(sys.argv) > 2 else "odil_track_v1"
    out = os.path.join(HERE, "runs", name)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(3)
    torch.manual_seed(3)
    M, N, T = int(os.environ.get("ODIL_PAIRS", "512")), 61, 3.0
    dt = T / (N - 1)
    NST = 3
    A_ROLL, EPS_V, V_STRIB = 0.042, 0.004, 0.01
    W_TRACK = float(os.environ.get("ODIL_W_TRACK", "1.0"))
    Pr, Vr, Ar = (torch.tensor(a, dtype=torch.float32) for a in _references(rng, M, N, T))
    # state: p(2) v(2) chain(2*NST) o1(2) o2(2) z(2)
    NS = 4 + 2 * NST + 6
    CH = [slice(4 + 2 * k, 6 + 2 * k) for k in range(NST)]
    O1, O2, Z = slice(4 + 2 * NST, 6 + 2 * NST), slice(6 + 2 * NST, 8 + 2 * NST), slice(8 + 2 * NST, 10 + 2 * NST)
    SCALE = torch.tensor([0.1] * 4 + [5.0] * (2 * NST + 4) + [1.0, 1.0])
    k_acc = torch.tensor(rng.uniform(0.100, 0.125, (M, 1, 1)), dtype=torch.float32)
    a_st = k_acc * torch.tensor(rng.uniform(1.2, 2.6, (M, 1, 1)), dtype=torch.float32)
    bd = rng.normal(0, 1, (M, 2))
    bd = bd / np.linalg.norm(bd, axis=1, keepdims=True) * rng.uniform(0, 0.8, (M, 1))
    bias = torch.tensor(bd[:, None, :], dtype=torch.float32)
    taus = torch.tensor(rng.uniform(0.025, 0.07, (M, 1, NST)), dtype=torch.float32)
    # start: a few mm off the reference, roughly at its speed, tilted for its acceleration
    x0 = torch.zeros(M, NS)
    x0[:, 0:2] = Pr[:, 0] + torch.tensor(rng.normal(0, 0.004, (M, 2)), dtype=torch.float32)
    x0[:, 2:4] = Vr[:, 0] * torch.tensor(rng.uniform(0.5, 1.2, (M, 1)), dtype=torch.float32)
    tilt0 = Ar[:, 0] / k_acc[:, 0]
    for s_ in CH + [O1, O2]:
        x0[:, s_] = tilt0
    x_in = torch.zeros(M, N - 1, NS)
    x_in[..., 0:2], x_in[..., 2:4] = Pr[:, 1:], Vr[:, 1:]
    for s_ in CH + [O1, O2]:
        x_in[..., s_] = Ar[:, 1:] / k_acc
    x_in.requires_grad_(True)

    class Pol(torch.nn.Module):
        def __init__(self, h=128):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(14, h), torch.nn.Tanh(), torch.nn.Linear(h, h),
                                           torch.nn.Tanh(), torch.nn.Linear(h, 2))

        def forward(self, x, r, vr, ar):
            e = r - x[..., 0:2]
            feat = torch.cat([e / 0.02, (vr - x[..., 2:4]) / 0.05, ar / (K_NOM * 5.0), x[..., 2:4] / 0.1,
                              x[..., O1] / 5.0, x[..., O2] / 5.0, x[..., Z]], -1)
            return U_MAX_DEG * torch.tanh(self.net(feat))

    def f(x, u, r):
        v = x[..., 2:4]
        ch = [x[..., s_] for s_ in CH]
        o1, o2, z = x[..., O1], x[..., O2], x[..., Z]
        speed = torch.sqrt((v ** 2).sum(-1, keepdim=True) + EPS_V ** 2)
        drive = k_acc * (ch[-1] + bias)
        w = torch.exp(-(speed / V_STRIB) ** 2)
        dmag = torch.sqrt((drive ** 2).sum(-1, keepdim=True) + 1e-8)
        kk = 0.02 * a_st
        held = -kk * torch.log(torch.exp(-dmag / kk) + torch.exp(-a_st.expand_as(dmag) / kk))
        acc = drive - w * held * drive / dmag - (1 - w) * A_ROLL * v / speed
        chd = [((u if k == 0 else ch[k - 1]) - ch[k]) / taus[..., k:k + 1] for k in range(NST)]
        zd = (r - x[..., 0:2]) / Z_SCALE - z / TZ
        return torch.cat([v, acc] + chd + [(u - o1) / TAU_OBS, (o1 - o2) / TAU_OBS, zd], -1)

    pol = Pol()
    opt_p = torch.optim.Adam(pol.parameters(), lr=1e-3)
    opt_x = torch.optim.Adam([x_in], lr=3e-3)
    hist, t0 = [], time.time()
    rm, vm, am = 0.5 * (Pr[:, 1:] + Pr[:, :-1]), 0.5 * (Vr[:, 1:] + Vr[:, :-1]), 0.5 * (Ar[:, 1:] + Ar[:, :-1])
    for rnd in range(6):
        for it in range(iters):
            x = torch.cat([x0[:, None], x_in], 1)
            xm = 0.5 * (x[:, 1:] + x[:, :-1])
            u = pol(xm, rm, vm, am)
            res = (x[:, 1:] - x[:, :-1] - f(xm, u, rm) * dt) / SCALE
            phys = (res ** 2).sum(-1).sum(-1).mean()
            track = (((x[:, 1:, 0:2] - Pr[:, 1:]) / 0.005) ** 2).sum(-1).mean()
            smooth = ((u[:, 1:] - u[:, :-1]) / U_MAX_DEG).pow(2).sum(-1).sum(-1).mean()
            loss = 100.0 * phys + W_TRACK * track + 0.02 * smooth
            opt_p.zero_grad()
            opt_x.zero_grad()
            loss.backward()
            opt_p.step()
            opt_x.step()
        err = (x[:, 1:, 0:2] - Pr[:, 1:]).norm(dim=-1) * 1000
        rec = dict(round=rnd, phys=float(phys), track_mm_median=float(err.median()),
                   track_mm_p90=float(torch.quantile(err.flatten(), 0.9)), smooth=float(smooth),
                   minutes=(time.time() - t0) / 60)
        hist.append(rec)
        print(rec, flush=True)
    lin = [mm for mm in pol.net if isinstance(mm, torch.nn.Linear)]
    np.savez(os.path.join(out, "odil_track_policy.npz"), **{f"W{i}": l.weight.detach().numpy() for i, l in enumerate(lin)},
             **{f"b{i}": l.bias.detach().numpy() for i, l in enumerate(lin)}, n_in=np.array(14))
    json.dump(hist, open(os.path.join(out, "history.json"), "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
