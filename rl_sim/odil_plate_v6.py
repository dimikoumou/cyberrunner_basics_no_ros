#!/usr/bin/env python3
"""
ODIL v6 (2026-09-27): v5 + a realistic, randomised delay and output feedback.

v5 reached 98% of targets in the full simulator but then limit-cycled round them (+-2-3
deg, ~1.3 s period, ~5 exits per run): its model plate responded faster (2 x 45 ms lag)
than the real loop (servo delay + lag + camera: ~0.1-0.2 s), so its near-field gain was
too high. v6:
  - true tilt = 3 first-order stages with per-trajectory time constants (20-60 ms each);
  - the policy does NOT see the true tilt, only what the rig can compute: a nominal
    observer (2 x 45 ms stages driven by its own commands, exactly the rig's
    ODILRigController), so its feedback must be stable for every delay in the range;
  - near-goal trajectories carry only 20% of the travel-time penalty (settle, don't snap).
State: p(2) v(2) true tilt chain(6) observer(4) z(2) = 16. Policy input (10) unchanged, so
the rig/eval controllers run v6 as they are.

v5 notes:

v4 (odil_plate.py) reached targets smoothly but drifted out again (rig: ~57% of the time
inside after arriving; sim: 68% exits). Two causes:
  1. the policy had no memory -> a constant disturbance (level error, paper slope) left
     the ball off target or creeping; the classic controller cancels it with its
     integral term;
  2. it was only ever trained along ideal rest-to-rest paths from far away, never on
     "nearly there but slightly off / moving / tilted" states.
v5 changes (same discrete-loss ODIL otherwise):
  - state gains a leaky integral of the goal error, z' = (goal - p)/Z_SCALE - z/TZ, in the
    model AND as a policy input, so the optimiser learns to use it;
  - every trajectory has its own hidden physics: tilt->acceleration gain, static friction
    and a constant disturbance tilt (level error + slope); the policy must work for all
    of them (it only sees p, v, tilt states and z);
  - NEAR_FRAC of trajectories start near the goal: offset <= 3 cm, moving, tilted;
  - end: at the goal, at rest; end tilt and z are free (it may hold a compensating tilt).
Output npz carries n_in=10 and tz so the rig/eval controllers run the same integrator.

  ../.venv-rl/bin/python3 odil_plate_v5.py [iters_per_round] [run_name]
"""
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("ODIL_THREADS", "3")))  # leave cores for the rig
TAU, A_ROLL, EPS_V, V_STRIBECK = 0.045, 0.025, 0.004, 0.01
K_RANGE = (0.085, 0.125)
STATIC_DEG_RANGE = (1.0, 2.2)
BIAS_DEG_MAX = float(os.environ.get("ODIL_BIAS_DEG", "0.8"))
TZ = 1.5                                # s, integrator leak
Z_SCALE = 0.02                          # m: 2 cm error held 1 s -> z ~ 1
NEAR_FRAC = float(os.environ.get("ODIL_NEAR_FRAC", "0.4"))
U_MAX_DEG = 5.0 * 0.8
M_PAIRS, N_PTS = int(os.environ.get("ODIL_PAIRS", "768")), 41
T_MAX = 3.0
LAM0, LAM_DECAY = 0.2, 0.7
GOAL_X, GOAL_Y, START_X, START_Y = 0.09, 0.07, 0.12, 0.10
SCALE = torch.tensor([0.1] * 4 + [5.0] * 10 + [1.0, 1.0])
TAU_RANGE = tuple(float(v) for v in os.environ.get("ODIL_TAU_RANGE", "0.02,0.06").split(","))
NEAR_LAM = float(os.environ.get("ODIL_NEAR_LAM", "0.2"))
# state layout
P, V, C1, C2, C3, O1, O2, Z = (slice(0, 2), slice(2, 4), slice(4, 6), slice(6, 8), slice(8, 10), slice(10, 12),
                               slice(12, 14), slice(14, 16))
NS = 16


class Policy(torch.nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(10, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(),
                                       torch.nn.Linear(h, 2))

    def forward(self, x, goal):
        feat = torch.cat([(goal - x[..., P]) / 0.1, x[..., V] / 0.1, x[..., O1] / 5.0, x[..., O2] / 5.0,
                          x[..., Z]], -1)
        return U_MAX_DEG * torch.tanh(self.net(feat))


def f(x, u, goal, k_acc, a_static, bias, taus):
    """k_acc (M,1,1), a_static (M,1,1), bias (M,1,2) deg, taus (M,1,3) -- per-trajectory hidden physics"""
    v, c1, c2, c3, o1, o2, z = (x[..., s_] for s_ in (V, C1, C2, C3, O1, O2, Z))
    speed = torch.sqrt((v ** 2).sum(-1, keepdim=True) + EPS_V ** 2)
    drive = k_acc * (c3 + bias)
    w = torch.exp(-(speed / V_STRIBECK) ** 2)
    dmag = torch.sqrt((drive ** 2).sum(-1, keepdim=True) + 1e-8)
    kk = 0.02 * a_static
    held = -kk * torch.log(torch.exp(-dmag / kk) + torch.exp(-a_static.expand_as(dmag) / kk))
    acc = drive - w * held * drive / dmag - (1 - w) * A_ROLL * v / speed
    zdot = (goal - x[..., P]) / Z_SCALE - z / TZ
    return torch.cat([v, acc, (u - c1) / taus[..., 0:1], (c1 - c2) / taus[..., 1:2], (c2 - c3) / taus[..., 2:3],
                      (u - o1) / TAU, (o1 - o2) / TAU, zdot], -1)


def build(rng, m):
    g = np.column_stack([rng.uniform(-GOAL_X, GOAL_X, m), rng.uniform(-GOAL_Y, GOAL_Y, m)])
    s = np.column_stack([rng.uniform(-START_X, START_X, m), rng.uniform(-START_Y, START_Y, m)])
    near = rng.random(m) < NEAR_FRAC
    ang, rad = rng.uniform(0, 2 * np.pi, m), rng.uniform(0.003, 0.03, m)
    s[near] = g[near] + np.column_stack([np.cos(ang), np.sin(ang)])[near] * rad[near, None]
    far_ok = np.hypot(*(s - g).T) > 0.03
    keep = near | far_ok
    s, g, near = s[keep], g[keep], near[keep]
    m = len(s)
    start = torch.zeros(m, NS)
    start[:, P] = torch.tensor(s, dtype=torch.float32)
    nt = torch.tensor(near)[:, None]
    start[:, V] = torch.where(nt, 0.06 * (2 * torch.rand(m, 2) - 1), torch.zeros(m, 2))
    tilt = torch.where(nt, 1.5 * (2 * torch.rand(m, 2) - 1), torch.zeros(m, 2))
    for s_ in (C1, C2, C3, O1, O2):
        start[:, s_] = tilt
    start[:, Z] = torch.where(nt, 0.5 * (2 * torch.rand(m, 2) - 1), torch.zeros(m, 2))
    goal = torch.tensor(g, dtype=torch.float32)
    k_acc = torch.tensor(rng.uniform(*K_RANGE, (m, 1, 1)), dtype=torch.float32)
    a_static = k_acc * torch.tensor(rng.uniform(*STATIC_DEG_RANGE, (m, 1, 1)), dtype=torch.float32)
    bd = rng.normal(0, 1, (m, 2))
    bd = bd / np.maximum(np.linalg.norm(bd, axis=1, keepdims=True), 1e-9) * rng.uniform(0, BIAS_DEG_MAX, (m, 1))
    bias = torch.tensor(bd[:, None, :], dtype=torch.float32)
    taus = torch.tensor(rng.uniform(*TAU_RANGE, (m, 1, 3)), dtype=torch.float32)
    # unknowns: interior points, the free part of the end state (tilt, z), duration
    tt = torch.linspace(0, 1, N_PTS)[1:-1]
    blend = (3 * tt ** 2 - 2 * tt ** 3)[None, :, None]
    end0 = torch.zeros(m, NS)
    end0[:, P] = goal
    x_in = (start[:, None, :] * (1 - blend) + end0[:, None, :] * blend).clone()
    x_in[..., 2:] += 0.001 * torch.randn_like(x_in[..., 2:])
    x_in.requires_grad_(True)
    end_free = torch.zeros(m, NS - 4, requires_grad=True)     # tilt chain, observer, z at the end
    T0 = torch.where(torch.tensor(near), torch.full((m,), 1.0), torch.full((m,), 1.5))[:, None]
    log_T = torch.log(T0).clone().requires_grad_(True)
    return dict(start=start, goal=goal, k=k_acc, s=a_static, b=bias, taus=taus, x_in=x_in, end_free=end_free,
                log_T=log_T, near=torch.tensor(near))


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    name = sys.argv[2] if len(sys.argv) > 2 else "odil_v6"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", name)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    pol = Policy()
    opt_pol = torch.optim.Adam(pol.parameters(), lr=1e-3)
    lam, mu = LAM0, 0.02
    t0, hist = time.time(), []
    B = build(rng, M_PAIRS)
    opt_traj = torch.optim.Adam([B["x_in"], B["end_free"], B["log_T"]], lr=3e-3)
    m = len(B["start"])
    for rnd in range(6):
        goal = B["goal"][:, None, :]
        for it in range(iters):
            end = torch.cat([B["goal"], torch.zeros(m, 2), B["end_free"]], -1)
            x = torch.cat([B["start"][:, None, :], B["x_in"], end[:, None, :]], 1)
            T = torch.exp(B["log_T"])
            dt = (T / (N_PTS - 1))[:, :, None]
            xm = 0.5 * (x[:, 1:] + x[:, :-1])
            gx = goal.expand(-1, N_PTS - 1, -1)
            u = pol(xm, gx)
            res = (x[:, 1:] - x[:, :-1] - f(xm, u, gx, B["k"], B["s"], B["b"], B["taus"]) * dt) / SCALE
            phys = (res ** 2).sum(-1).sum(-1).mean()
            smooth = ((u[:, 1:] - u[:, :-1]) / U_MAX_DEG).pow(2).sum(-1).sum(-1).mean()
            # the end state must be a true rest: the policy's command there = the held tilt,
            # and every tilt stage / the observer agree with it
            u_end = pol(end[:, None, :], goal)[:, 0]
            chain = torch.stack([end[:, s_] for s_ in (C1, C2, C3, O1, O2)], 1)
            hold = (((u_end[:, None, :] - chain) / U_MAX_DEG) ** 2).sum(-1).mean()
            w_T = torch.where(B["near"][:, None], torch.full_like(T, NEAR_LAM), torch.ones_like(T))
            loss = phys * 100.0 + lam * (w_T * T).mean() + mu * smooth + 10.0 * hold
            opt_pol.zero_grad()
            opt_traj.zero_grad()
            loss.backward()
            opt_pol.step()
            opt_traj.step()
            with torch.no_grad():
                B["log_T"].clamp_(np.log(0.3), np.log(T_MAX))
                B["end_free"][:, 0:10].clamp_(-U_MAX_DEG, U_MAX_DEG)
        rec = dict(round=rnd, lam=lam, phys=float(phys), hold=float(hold), T_med=float(T.median()),
                   T_near=float(T[B["near"]].median()), smooth=float(smooth), minutes=(time.time() - t0) / 60)
        hist.append(rec)
        print(rec, flush=True)
        lam *= LAM_DECAY
    torch.save(pol.state_dict(), os.path.join(out, "odil_policy.pt"))
    lin = [mm for mm in pol.net if isinstance(mm, torch.nn.Linear)]
    np.savez(os.path.join(out, "odil_policy.npz"), **{f"W{i}": l.weight.detach().numpy() for i, l in enumerate(lin)},
             **{f"b{i}": l.bias.detach().numpy() for i, l in enumerate(lin)},
             n_in=np.array(10), tz=np.array(TZ), z_scale=np.array(Z_SCALE))
    json.dump(hist, open(os.path.join(out, "odil_history.json"), "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
