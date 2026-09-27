#!/usr/bin/env python3
"""
ODIL for the CyberRunner plate (2026-09-27): "Optimal Navigation in Microfluidics via
the Optimization of a Discrete Loss" (arXiv 2506.15902) applied to the ball.

Idea (from the paper): write M trajectories (random start -> target pairs) as unknowns
x^n on a time grid and minimise ONE discrete loss jointly over the trajectory points
and the weights of a closed-loop neural policy a_theta(x):

    L = sum_n | x^{n+1} - x^n - f(x^{n+1/2}, a_theta(x^{n+1/2})) dt |^2      (physics, midpoint rule)
        + lambda * T                                                     (travel time)
        + mu * sum_n | a^{n+1} - a^n |^2                                 (smooth commands; ours)

with the boundary points fixed: start = ball at rest, plate level; END = at the target,
velocity 0, plate level  ->  rest-to-rest: "one smooth motion, then stillness".

Model f (differentiable version of plate_sim, fitted on rig logs):
  state x = [p(2), v(2), th(2), th_lag(2)]  (tilt in deg; two first-order stages model the
            servo + camera delay of ~0.1 s)
  p' = v
  v' = K_ACC * th_lag - A_ROLL * v / sqrt(|v|^2 + eps^2)          (smooth rolling friction)
  th' = (u - th) / TAU,   th_lag' = (th - th_lag) / TAU,   u = 5 deg * 0.8 * tanh(net)
Policy input (relative, so it works for any target): [(goal - p)/0.1, v/0.1, th/5, th_lag/5].
Stiction is NOT in the differentiable model; the result is evaluated in the full sticky,
noisy, delayed simulator (plate_sim) and compared with PD and RL.

  ../.venv-rl/bin/python3 odil_plate.py [iters_per_round] [run_name]
"""
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(4)
DEV = "cpu"
K_ACC, A_ROLL, TAU, EPS_V = 0.11, 0.025, 0.045, 0.004
# v2: smooth Stribeck friction as a differentiable stand-in for paper stiction -- v1's
# model had none, so it planned gentle ~2.8 s motions whose tilts (< breakaway ~1.8 deg)
# never freed a resting ball in the full sticky simulator (26% success).
A_STATIC = K_ACC * float(os.environ.get("ODIL_STATIC_DEG", "1.6"))   # m/s^2 at ~zero speed
V_STRIBECK = float(os.environ.get("ODIL_V_STRIBECK", "0.01"))       # m/s
LAM0 = float(os.environ.get("ODIL_LAM0", "0.2"))
# v3: the policy was only ever trained along its own optimal rest-to-rest paths, so in
# the real (sticky, delayed, noisy) loop it met states it had never seen. Start states
# now include moving balls and tilted plates, and every round draws a fresh set of pairs
# (the policy carries over), so it learns feedback over much more of the state space.
RANDOM_START = os.environ.get("ODIL_RANDOM_START", "0") == "1"   # v3 (on) collapsed to ~zero tilts
RESAMPLE = os.environ.get("ODIL_RESAMPLE", "0") == "1"
STICTION_MODEL = os.environ.get("ODIL_STICTION", "hold")
LAM_DECAY = float(os.environ.get("ODIL_LAM_DECAY", "0.7"))
T_MAX = float(os.environ.get("ODIL_T_MAX", "3.0"))
U_MAX_DEG = 5.0 * 0.8
M_PAIRS, N_PTS = int(os.environ.get("ODIL_PAIRS", "512")), 41
GOAL_X, GOAL_Y, START_X, START_Y = 0.09, 0.07, 0.12, 0.10
SCALE = torch.tensor([0.1, 0.1, 0.1, 0.1, 5.0, 5.0, 5.0, 5.0])     # residual normalisation


class Policy(torch.nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(8, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(),
                                       torch.nn.Linear(h, 2))

    def features(self, x, goal):
        return torch.cat([(goal - x[..., 0:2]) / 0.1, x[..., 2:4] / 0.1, x[..., 4:6] / 5.0, x[..., 6:8] / 5.0], -1)

    def forward(self, x, goal):
        return U_MAX_DEG * torch.tanh(self.net(self.features(x, goal)))   # commanded tilt (deg)


def f(x, u):
    v, th, thl = x[..., 2:4], x[..., 4:6], x[..., 6:8]
    speed = torch.sqrt((v ** 2).sum(-1, keepdim=True) + EPS_V ** 2)
    drive = K_ACC * thl
    if STICTION_MODEL == "hold":
        # v4: smooth STATIC friction. A slow ball is held against the drive up to A_STATIC
        # (friction opposes the DRIVE, not the velocity), so it only starts when the tilt
        # beats breakaway -- and at the end, level + slow = held in place. v2's Stribeck
        # term scaled with v/|v| and vanished at rest, so the model let the ball creep
        # from rest under tiny tilts; on the rig ODIL v2 then stalled 14-33 mm short.
        w = torch.exp(-(speed / V_STRIBECK) ** 2)                         # 1 when slow
        dmag = torch.sqrt((drive ** 2).sum(-1, keepdim=True) + 1e-8)
        k = 0.02 * A_STATIC
        held = -k * torch.log(torch.exp(-dmag / k) + torch.exp(-torch.full_like(dmag, A_STATIC) / k))  # smooth min
        acc = drive - w * held * drive / dmag - (1 - w) * A_ROLL * v / speed
        return torch.cat([v, acc, (u - th) / TAU, (th - thl) / TAU], -1)
    a_fric = A_ROLL + (A_STATIC - A_ROLL) * torch.exp(-(speed / V_STRIBECK) ** 2)
    return torch.cat([v, drive - a_fric * v / speed, (u - th) / TAU, (th - thl) / TAU], -1)


def sample_pairs(rng, m):
    s = np.column_stack([rng.uniform(-START_X, START_X, m), rng.uniform(-START_Y, START_Y, m)])
    g = np.column_stack([rng.uniform(-GOAL_X, GOAL_X, m), rng.uniform(-GOAL_Y, GOAL_Y, m)])
    keep = np.hypot(*(s - g).T) > 0.03
    return s[keep], g[keep]


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    name = sys.argv[2] if len(sys.argv) > 2 else "odil_v1"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", name)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    pol = Policy()
    opt_pol = torch.optim.Adam(pol.parameters(), lr=1e-3)
    lam, mu = LAM0, 0.02
    t0 = time.time()
    hist = []

    def build():
        s, g = sample_pairs(rng, M_PAIRS)
        m = len(s)
        start = torch.zeros(m, 8)
        start[:, 0:2] = torch.tensor(s, dtype=torch.float32)
        if RANDOM_START:
            k = torch.rand(m, 1) < 0.6                       # 60% start moving/tilted
            start[:, 2:4] = torch.where(k, 0.1 * (2 * torch.rand(m, 2) - 1), start[:, 2:4])
            tilt = torch.where(k, 3.0 * (2 * torch.rand(m, 2) - 1), torch.zeros(m, 2))
            start[:, 4:6], start[:, 6:8] = tilt, tilt
        end = torch.zeros(m, 8)
        end[:, 0:2] = torch.tensor(g, dtype=torch.float32)
        tt = torch.linspace(0, 1, N_PTS)[1:-1]
        blend = (3 * tt ** 2 - 2 * tt ** 3)[None, :, None]
        x_in = (start[:, None, :] * (1 - blend) + end[:, None, :] * blend).clone()
        x_in[..., 2:] += 0.001 * torch.randn_like(x_in[..., 2:])
        x_in.requires_grad_(True)
        log_T = torch.full((m, 1), float(np.log(1.5)), requires_grad=True)
        return start, end, x_in, log_T, torch.optim.Adam([x_in, log_T], lr=3e-3)

    start, end, x_in, log_T, opt_traj = build()
    for rnd in range(6):
        if RESAMPLE and rnd > 0:
            start, end, x_in, log_T, opt_traj = build()
        goal = end[:, None, 0:2]
        for it in range(iters):
            x = torch.cat([start[:, None, :], x_in, end[:, None, :]], 1)           # (M, N, 8)
            T = torch.exp(log_T)
            dt = (T / (N_PTS - 1))[:, :, None]
            xm = 0.5 * (x[:, 1:] + x[:, :-1])
            u = pol(xm, goal.expand(-1, N_PTS - 1, -1))
            res = (x[:, 1:] - x[:, :-1] - f(xm, u) * dt) / SCALE
            phys = (res ** 2).sum(-1).sum(-1).mean()
            smooth = ((u[:, 1:] - u[:, :-1]) / U_MAX_DEG).pow(2).sum(-1).sum(-1).mean()
            loss = phys * 100.0 + lam * T.mean() + mu * smooth
            opt_pol.zero_grad()
            opt_traj.zero_grad()
            loss.backward()
            opt_pol.step()
            opt_traj.step()
            with torch.no_grad():
                log_T.clamp_(np.log(0.3), np.log(T_MAX))
        rec = dict(round=rnd, lam=lam, phys=float(phys), T_med=float(T.median()), smooth=float(smooth),
                   minutes=(time.time() - t0) / 60)
        hist.append(rec)
        print(rec, flush=True)
        lam *= LAM_DECAY
    torch.save(pol.state_dict(), os.path.join(out, "odil_policy.pt"))
    lin = [m for m in pol.net if isinstance(m, torch.nn.Linear)]
    np.savez(os.path.join(out, "odil_policy.npz"), **{f"W{i}": l.weight.detach().numpy() for i, l in enumerate(lin)},
             **{f"b{i}": l.bias.detach().numpy() for i, l in enumerate(lin)})
    json.dump(hist, open(os.path.join(out, "odil_history.json"), "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
