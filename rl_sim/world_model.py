#!/usr/bin/env python3
"""
Learned world model of the real plate (2026-09-28): physics + a small neural correction.

    acceleration = k_acc * (tilt + bias) - rolling friction - static hold        (physics)
                 + corr_net(position, velocity, last commands)                   (learned)

The tilt follows the commanded tilt through a pure delay + first-order lag (as in
finetune_track.py). The correction network gets the ball POSITION, so it can learn
place-specific effects (taped paper, a dip, the edge of a hole), and the recent commands,
so it can learn motor quirks. Trained on the rig's own logs with a multi-step loss: start
at a logged state, replay the logged commands for H frames, match the logged positions.

  ../.venv-rl/bin/python3 world_model.py out.pt log1.csv [log2.csv ...]
"""
import csv
import os
import sys
import time

import numpy as np
import torch

DT = 1.0 / 29.0
H = int(os.environ.get("WM_HORIZON", "10"))           # frames predicted per window (~0.35 s)
LEVEL = (-1.1, 2.55)
N_HIST = 4
TILT_MAX = 5.0


class WorldModel(torch.nn.Module):
    def __init__(self, delay=3, tau=0.025, k_acc=0.095, a_roll=0.033, a_static_deg=1.8, h=64):
        super().__init__()
        self.delay = delay
        # physics parameters are learnable too (log-parametrised, positive)
        self.log_tau = torch.nn.Parameter(torch.tensor(float(np.log(tau))))
        self.log_k = torch.nn.Parameter(torch.tensor(float(np.log(k_acc))))
        self.log_roll = torch.nn.Parameter(torch.tensor(float(np.log(a_roll))))
        self.log_static = torch.nn.Parameter(torch.tensor(float(np.log(a_static_deg))))
        self.bias = torch.nn.Parameter(torch.zeros(2))        # global level error (deg)
        self.net = torch.nn.Sequential(torch.nn.Linear(4 + 2 * N_HIST, h), torch.nn.Tanh(),
                                       torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Linear(h, 2))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)               # starts as pure physics

    def acc(self, p, v, tilt, u_hist):
        """p, v (B,2) m, m/s; tilt (B,2) deg (actual plate); u_hist (B,N_HIST,2) deg commanded"""
        k = torch.exp(self.log_k)
        a_roll = torch.exp(self.log_roll)
        a_st = k * torch.exp(self.log_static)
        speed = torch.sqrt((v ** 2).sum(-1, keepdim=True) + 0.004 ** 2)
        drive = k * (tilt + self.bias)
        w = torch.exp(-(speed / 0.01) ** 2)
        dmag = torch.sqrt((drive ** 2).sum(-1, keepdim=True) + 1e-8)
        kk = 0.02 * a_st
        held = -kk * torch.log(torch.exp(-dmag / kk) + torch.exp(-a_st.expand_as(dmag) / kk))
        phys = drive - w * held * drive / dmag - (1 - w) * a_roll * v / speed
        feat = torch.cat([p / 0.14, v / 0.1, (u_hist / TILT_MAX).flatten(1)], -1)
        return phys + 0.05 * torch.tanh(self.net(feat))       # correction up to 0.05 m/s^2 (~0.5 deg)

    def step(self, p, v, tilt, u_hist, u_now, dt=DT, sub=4):
        """advance one control step: u_now = commanded tilt (deg) reaching the plate now"""
        tau = torch.exp(self.log_tau)
        h = dt / sub
        for _ in range(sub):
            tilt = tilt + (u_now - tilt) * (h / tau)
            a = self.acc(p, v, tilt, u_hist)
            v = v + a * h
            p = p + v * h
        return p, v, tilt

    def export(self, path):
        lin = [m for m in self.net if isinstance(m, torch.nn.Linear)]
        np.savez(path, **{f"W{i}": l.weight.detach().numpy() for i, l in enumerate(lin)},
                 **{f"b{i}": l.bias.detach().numpy() for i, l in enumerate(lin)},
                 delay=self.delay, tau=float(torch.exp(self.log_tau)), k_acc=float(torch.exp(self.log_k)),
                 a_roll=float(torch.exp(self.log_roll)), a_static_deg=float(torch.exp(self.log_static)),
                 bias=self.bias.detach().numpy())


def load_windows(files, max_rows=None):
    """logged frames -> training windows: start state + commands + positions over H frames.
    Only stretches where the ball is seen, the controller runs, and dt is regular."""
    P0, V0, U, POS, UH = [], [], [], [], []
    for fn in files:
        r = list(csv.DictReader(open(fn)))
        if max_rows:
            r = r[-max_rows:]
        g = lambda k: np.array([float(q[k]) if q.get(k) not in (None, "") else np.nan for q in r])
        t, x, y, a0, a1 = g("t"), g("xb"), g("yb"), g("a0"), g("a1")
        vx, vy, found = g("vx"), g("vy"), g("ball_found")
        st = [q.get("status", "") for q in r]
        # commanded tilt (deg, ball axes) after the env's rate limit (0.5 action per step)
        a = np.column_stack([a0, a1])
        app = np.zeros_like(a)
        prev = np.zeros(2)
        for i in range(len(a)):
            prev = np.clip(np.clip(a[i], prev - 0.5, prev + 0.5), -1, 1) if np.all(np.isfinite(a[i])) else prev
            app[i] = prev
        u = app * TILT_MAX
        ok = (found > 0) & np.isfinite(x) & np.isfinite(y) & np.array([s in ("running", "in_circle") for s in st])
        dt = np.diff(t, prepend=t[0])
        ok &= (dt > 0.02) & (dt < 0.06)
        for i in range(N_HIST + 5, len(t) - H - 1, 3):
            if not ok[i - N_HIST - 5:i + H + 1].all():
                continue
            P0.append([x[i], y[i]])
            V0.append([vx[i], vy[i]])
            U.append(u[i - 5:i + H])                  # commands from 5 frames back (delay) on
            UH.append(u[i - N_HIST:i])
            POS.append(np.column_stack([x[i + 1:i + H + 1], y[i + 1:i + H + 1]]))
    T = lambda z: torch.tensor(np.array(z), dtype=torch.float32)
    return T(P0), T(V0), T(U), T(UH), T(POS)


def rollout_loss(m, P0, V0, U, UH, POS):
    """replay the logged commands from the logged start; error in the predicted positions (m)"""
    p, v = P0, V0
    d = m.delay
    tilt = U[:, 5 - d]                                 # plate tilt ~ the command d frames ago
    uh = UH
    err = []
    for k in range(H):
        u_now = U[:, 5 + k - d]
        p, v, tilt = m.step(p, v, tilt, uh, u_now)
        uh = torch.cat([uh[:, 1:], U[:, 5 + k][:, None]], 1)
        err.append((p - POS[:, k]).norm(dim=-1))
    return torch.stack(err, 1)


def main():
    out, files = sys.argv[1], sys.argv[2:]
    torch.set_num_threads(int(os.environ.get("WM_THREADS", "6")))
    data = load_windows(files, int(os.environ.get("WM_MAX_ROWS", "0")) or None)
    n = len(data[0])
    print(f"{n} windows of {H} frames from {len(files)} log(s)")
    idx = np.random.default_rng(0).permutation(n)
    tr, va = idx[: int(0.85 * n)], idx[int(0.85 * n):]
    m = WorldModel(delay=int(os.environ.get("WM_DELAY", "3")))
    if os.path.exists(out) and os.environ.get("WM_WARM", "1") == "1":
        m.load_state_dict(torch.load(out))
        print("warm start from", out)
    with torch.no_grad():
        e0 = rollout_loss(m, *(d[va] for d in data))
    print(f"before: validation error after {H} frames median {1000 * e0[:, -1].median():.1f} mm, "
          f"mean over window {1000 * e0.mean():.1f} mm")
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    t0 = time.time()
    for it in range(int(os.environ.get("WM_ITERS", "600"))):
        b = torch.tensor(np.random.default_rng(it).choice(tr, min(512, len(tr)), replace=False))
        e = rollout_loss(m, *(d[b] for d in data))
        loss = (e / 0.005).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        e1 = rollout_loss(m, *(d[va] for d in data))
    print(f"after ({(time.time() - t0) / 60:.1f} min): validation error after {H} frames median "
          f"{1000 * e1[:, -1].median():.1f} mm, mean over window {1000 * e1.mean():.1f} mm")
    print("physics: tau %.3f s, k_acc %.3f, a_roll %.3f, static %.2f deg, level bias %s deg" % (
        float(torch.exp(m.log_tau)), float(torch.exp(m.log_k)), float(torch.exp(m.log_roll)),
        float(torch.exp(m.log_static)), m.bias.detach().numpy().round(2)))
    torch.save(m.state_dict(), out)


if __name__ == "__main__":
    main()
