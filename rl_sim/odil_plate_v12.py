#!/usr/bin/env python3
"""
ODIL v12 (2026-09-27): the hole as a constraint INSIDE the ODIL optimisation.

Every trajectory gets a hole (centre H, keep-out radius RK = hole radius + margin), half of
them placed on or near the straight start->goal line. The discrete loss gets one more term,
    w_hole * sum_n relu(RK - |p^n - H|)^2,
so the optimiser plans its own smooth curve round the hole (no via points), and the policy
gets the hole as input: direction to the hole times a closeness factor
exp(-(|H - p| - RK) / 0.03) (0 when far or when there is no hole) and RK / 0.03 -> 14 inputs.
Rig/eval controllers pass the hole the same way (hole_features()).

ODIL v9 (2026-09-27): v8 (v7 code, end at the centre) with an N-stage tilt chain (default
5 x 15-40 ms): a longer chain of first-order lags approximates the rig's PURE delay (servo
delay + camera frame) much better than 3 stages -- more phase lag for the same mean delay,
which is what destabilises feedback near the target.

ODIL v7 (2026-09-27): v6 + "stay inside" -- the target's radius is an input and the goal
is to come to REST anywhere safely inside it, not at its exact centre.

v6 (sim: 84% reached, 60% inside after arrival, 2.6 exits) still hunted round the centre
(stick, integral builds, break free, overshoot 5-18 mm) or stopped 8-12 mm short -- it
chased the exact centre and did not know how big the target was. v7:
  - per trajectory a target radius R ~ U(8, 30) mm, fed to the policy as R/0.03 (11 inputs);
  - the end point is free inside the target: penalty on max(0, |end - goal| - 0.5 R)^2;
  - a quarter of the near-goal starts begin INSIDE the target (slow, small tilt) -> learn
    to leave a ball that is already in alone;
  - static friction range widened to the simulator's (1.2-2.6 deg).

v6 notes:

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
TAU, EPS_V, V_STRIBECK = 0.045, 0.004, 0.01
A_ROLL = float(os.environ.get("ODIL_A_ROLL", "0.025"))           # rig fit (sysid_rig.py): 0.042
K_RANGE = tuple(float(v) for v in os.environ.get("ODIL_K_RANGE", "0.085,0.125").split(","))   # rig: 0.113
STATIC_DEG_RANGE = tuple(float(v) for v in os.environ.get("ODIL_STATIC_RANGE", "1.2,2.6").split(","))
R_RANGE = (0.008, 0.03)
INSIDE_FRAC = float(os.environ.get("ODIL_INSIDE_FRAC", "0.25"))    # of the near starts
END_FREE = float(os.environ.get("ODIL_END_FREE", "0.0"))   # end may rest this x R off centre (v8: 0 = at the centre)
BIAS_DEG_MAX = float(os.environ.get("ODIL_BIAS_DEG", "0.8"))
TZ = 1.5                                # s, integrator leak
Z_SCALE = 0.02                          # m: 2 cm error held 1 s -> z ~ 1
NEAR_FRAC = float(os.environ.get("ODIL_NEAR_FRAC", "0.4"))
U_MAX_DEG = 5.0 * 0.8
M_PAIRS, N_PTS = int(os.environ.get("ODIL_PAIRS", "768")), 41
T_MAX = 3.0
LAM0, LAM_DECAY = 0.2, 0.7
GOAL_X, GOAL_Y, START_X, START_Y = 0.09, 0.07, 0.12, 0.10
NST = int(os.environ.get("ODIL_N_STAGES", "5"))
SCALE = torch.tensor([0.1] * 4 + [5.0] * (2 * NST + 4) + [1.0, 1.0])
TAU_RANGE = tuple(float(v) for v in os.environ.get("ODIL_TAU_RANGE", "0.015,0.04").split(","))
NEAR_LAM = float(os.environ.get("ODIL_NEAR_LAM", "0.2"))
# state layout
P, V = slice(0, 2), slice(2, 4)
CH = [slice(4 + 2 * k, 6 + 2 * k) for k in range(NST)]           # true tilt chain
O1, O2 = slice(4 + 2 * NST, 6 + 2 * NST), slice(6 + 2 * NST, 8 + 2 * NST)
Z = slice(8 + 2 * NST, 10 + 2 * NST)
NS = 10 + 2 * NST


class Policy(torch.nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(14, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(),
                                       torch.nn.Linear(h, 2))

    def forward(self, x, goal, radius, hole, rk):
        feat = torch.cat([(goal - x[..., P]) / 0.1, x[..., V] / 0.1, x[..., O1] / 5.0, x[..., O2] / 5.0,
                          x[..., Z], (radius / 0.03).expand(*x.shape[:-1], 1),
                          hole_features(x[..., P], hole, rk)], -1)
        return U_MAX_DEG * torch.tanh(self.net(feat))


def f(x, u, goal, k_acc, a_static, bias, taus):
    """k_acc (M,1,1), a_static (M,1,1), bias (M,1,2) deg, taus (M,1,3) -- per-trajectory hidden physics"""
    v, o1, o2, z = (x[..., s_] for s_ in (V, O1, O2, Z))
    ch = [x[..., s_] for s_ in CH]
    speed = torch.sqrt((v ** 2).sum(-1, keepdim=True) + EPS_V ** 2)
    drive = k_acc * (ch[-1] + bias)
    w = torch.exp(-(speed / V_STRIBECK) ** 2)
    dmag = torch.sqrt((drive ** 2).sum(-1, keepdim=True) + 1e-8)
    kk = 0.02 * a_static
    held = -kk * torch.log(torch.exp(-dmag / kk) + torch.exp(-a_static.expand_as(dmag) / kk))
    acc = drive - w * held * drive / dmag - (1 - w) * A_ROLL * v / speed
    zdot = (goal - x[..., P]) / Z_SCALE - z / TZ
    chd = [((u if k == 0 else ch[k - 1]) - ch[k]) / taus[..., k:k + 1] for k in range(NST)]
    return torch.cat([v, acc] + chd + [(u - o1) / TAU, (o1 - o2) / TAU, zdot], -1)


HOLE_W = float(os.environ.get("ODIL_HOLE_W", "50.0"))
HOLE_R_RANGE = (0.006, 0.010)
HOLE_MARGIN = 0.008
NO_HOLE_FRAC = 0.2


def hole_features(p, hole, rk):
    """(dir to hole) * closeness, and RK/0.03; hole = nan -> zeros (no hole). torch or numpy."""
    if isinstance(p, np.ndarray):
        d = np.asarray(hole) - p
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        if not np.all(np.isfinite(d)):
            return np.zeros(p.shape[:-1] + (3,))
        close = np.exp(-np.clip(n - rk, 0, None) / 0.03)
        return np.concatenate([d / np.maximum(n, 1e-6) * close, np.full(p.shape[:-1] + (1,), rk / 0.03)], -1)
    d = hole - p
    n = torch.sqrt((d ** 2).sum(-1, keepdim=True) + 1e-10)
    close = torch.exp(-torch.relu(n - rk) / 0.03)
    has = (rk > 0).to(p.dtype)
    return torch.cat([d / n * close * has, (rk / 0.03).expand(*p.shape[:-1], 1)], -1)


def build(rng, m):
    g = np.column_stack([rng.uniform(-GOAL_X, GOAL_X, m), rng.uniform(-GOAL_Y, GOAL_Y, m)])
    s = np.column_stack([rng.uniform(-START_X, START_X, m), rng.uniform(-START_Y, START_Y, m)])
    near = rng.random(m) < NEAR_FRAC
    R = rng.uniform(*R_RANGE, m)
    inside = near & (rng.random(m) < INSIDE_FRAC)
    ang, rad = rng.uniform(0, 2 * np.pi, m), rng.uniform(0.003, 0.03, m)
    rad = np.where(inside, rng.uniform(0.0, 0.7, m) * R, rad)
    s[near] = g[near] + np.column_stack([np.cos(ang), np.sin(ang)])[near] * rad[near, None]
    far_ok = np.hypot(*(s - g).T) > 0.03
    keep = near | far_ok
    s, g, near, R, inside = s[keep], g[keep], near[keep], R[keep], inside[keep]
    m = len(s)
    start = torch.zeros(m, NS)
    start[:, P] = torch.tensor(s, dtype=torch.float32)
    nt = torch.tensor(near)[:, None]
    it = torch.tensor(inside)[:, None]
    start[:, V] = torch.where(nt, torch.where(it, 0.015, 0.06) * (2 * torch.rand(m, 2) - 1), torch.zeros(m, 2))
    tilt = torch.where(nt, torch.where(it, 0.7, 1.5) * (2 * torch.rand(m, 2) - 1), torch.zeros(m, 2))
    for s_ in CH + [O1, O2]:
        start[:, s_] = tilt
    start[:, Z] = torch.where(nt, 0.5 * (2 * torch.rand(m, 2) - 1), torch.zeros(m, 2))
    goal = torch.tensor(g, dtype=torch.float32)
    # holes: on/near the straight line for most, random for some, none for NO_HOLE_FRAC
    rh = rng.uniform(*HOLE_R_RANGE, m)
    RKn = rh + HOLE_MARGIN
    tl = rng.uniform(0.3, 0.7, m)
    nrm = np.column_stack([-(g - s)[:, 1], (g - s)[:, 0]])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    on_line = s + tl[:, None] * (g - s) + nrm * rng.uniform(-1.0, 1.0, (m, 1)) * RKn[:, None]
    rand_h = np.column_stack([rng.uniform(-GOAL_X, GOAL_X, m), rng.uniform(-GOAL_Y, GOAL_Y, m)])
    H = np.where((rng.random(m) < 0.7)[:, None], on_line, rand_h)
    clear = (np.hypot(*(H - s).T) > RKn + 0.012) & (np.hypot(*(H - g).T) > RKn + R + 0.012)
    none = (rng.random(m) < NO_HOLE_FRAC) | ~clear | near
    RKn = np.where(none, 0.0, RKn)
    H = np.where(none[:, None], 1.0, H)                      # far away and rk = 0 -> no effect
    hole = torch.tensor(H[:, None, :], dtype=torch.float32)
    rk = torch.tensor(RKn[:, None, None], dtype=torch.float32)
    k_acc = torch.tensor(rng.uniform(*K_RANGE, (m, 1, 1)), dtype=torch.float32)
    a_static = k_acc * torch.tensor(rng.uniform(*STATIC_DEG_RANGE, (m, 1, 1)), dtype=torch.float32)
    bd = rng.normal(0, 1, (m, 2))
    bd = bd / np.maximum(np.linalg.norm(bd, axis=1, keepdims=True), 1e-9) * rng.uniform(0, BIAS_DEG_MAX, (m, 1))
    bias = torch.tensor(bd[:, None, :], dtype=torch.float32)
    taus = torch.tensor(rng.uniform(*TAU_RANGE, (m, 1, NST)), dtype=torch.float32)
    # unknowns: interior points, the free part of the end state (tilt, z), duration
    tt = torch.linspace(0, 1, N_PTS)[1:-1]
    blend = (3 * tt ** 2 - 2 * tt ** 3)[None, :, None]
    end0 = torch.zeros(m, NS)
    end0[:, P] = goal
    x_in = (start[:, None, :] * (1 - blend) + end0[:, None, :] * blend).clone()
    # start the interior bent round the hole (a straight guess through its centre has no
    # gradient direction); side = the side the hole is NOT on
    side = -np.sign(np.sum((H - s) * nrm, axis=1))
    side = np.where(side == 0, 1.0, side)
    bump = torch.sin(np.pi * tt)[None, :, None] * torch.tensor(
        (nrm * (side * 1.6 * RKn)[:, None])[:, None, :], dtype=torch.float32)
    x_in[..., 0:2] += bump
    x_in[..., 2:] += 0.001 * torch.randn_like(x_in[..., 2:])
    x_in.requires_grad_(True)
    end_free = torch.zeros(m, NS - 4, requires_grad=True)     # tilt chain, observer, z at the end
    # end position: free, but inside the target (penalised beyond 0.5 R); start at the goal
    # for balls coming from outside, at the start point for balls already inside
    e0 = torch.where(torch.tensor(inside)[:, None], torch.tensor(s - g, dtype=torch.float32), torch.zeros(m, 2))
    end_off = e0.clone().requires_grad_(True)
    T0 = torch.where(torch.tensor(near), torch.full((m,), 1.0), torch.full((m,), 1.5))[:, None]
    log_T = torch.log(T0).clone().requires_grad_(True)
    return dict(start=start, goal=goal, k=k_acc, s=a_static, b=bias, taus=taus, x_in=x_in, end_free=end_free,
                log_T=log_T, near=torch.tensor(near), R=torch.tensor(R, dtype=torch.float32)[:, None, None],
                end_off=end_off, hole=hole, rk=rk)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    name = sys.argv[2] if len(sys.argv) > 2 else "odil_v12"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", name)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    pol = Policy()
    opt_pol = torch.optim.Adam(pol.parameters(), lr=1e-3)
    lam, mu = LAM0, 0.02
    t0, hist = time.time(), []
    B = build(rng, M_PAIRS)
    opt_traj = torch.optim.Adam([B["x_in"], B["end_free"], B["log_T"], B["end_off"]], lr=3e-3)
    m = len(B["start"])
    for rnd in range(6):
        goal = B["goal"][:, None, :]
        for it in range(iters):
            end = torch.cat([B["goal"] + B["end_off"], torch.zeros(m, 2), B["end_free"]], -1)
            x = torch.cat([B["start"][:, None, :], B["x_in"], end[:, None, :]], 1)
            T = torch.exp(B["log_T"])
            dt = (T / (N_PTS - 1))[:, :, None]
            xm = 0.5 * (x[:, 1:] + x[:, :-1])
            gx = goal.expand(-1, N_PTS - 1, -1)
            u = pol(xm, gx, B["R"], B["hole"], B["rk"])
            res = (x[:, 1:] - x[:, :-1] - f(xm, u, gx, B["k"], B["s"], B["b"], B["taus"]) * dt) / SCALE
            phys = (res ** 2).sum(-1).sum(-1).mean()
            smooth = ((u[:, 1:] - u[:, :-1]) / U_MAX_DEG).pow(2).sum(-1).sum(-1).mean()
            # the end state must be a true rest: the policy's command there = the held tilt,
            # and every tilt stage / the observer agree with it
            u_end = pol(end[:, None, :], goal, B["R"], B["hole"], B["rk"])[:, 0]
            chain = torch.stack([end[:, s_] for s_ in CH + [O1, O2]], 1)
            hold = (((u_end[:, None, :] - chain) / U_MAX_DEG) ** 2).sum(-1).mean()
            w_T = torch.where(B["near"][:, None], torch.full_like(T, NEAR_LAM), torch.ones_like(T))
            off = torch.relu(B["end_off"].norm(dim=-1) - END_FREE * B["R"][:, 0, 0]) / 0.005
            inside_pen = (off ** 2).mean()
            # the hole: every trajectory point (and midpoint) outside the keep-out radius
            pts = torch.cat([x[..., P], xm[..., P]], 1)
            dh = torch.sqrt(((pts - B["hole"]) ** 2).sum(-1) + 1e-10)
            hole_pen = ((torch.relu(B["rk"][:, :, 0] - dh) / 0.005) ** 2).sum(-1).mean()
            loss = (phys * 100.0 + lam * (w_T * T).mean() + mu * smooth + 10.0 * hold + 10.0 * inside_pen
                    + HOLE_W * hole_pen)
            opt_pol.zero_grad()
            opt_traj.zero_grad()
            loss.backward()
            opt_pol.step()
            opt_traj.step()
            with torch.no_grad():
                B["log_T"].clamp_(np.log(0.3), np.log(T_MAX))
                B["end_free"][:, 0:2 * NST + 4].clamp_(-U_MAX_DEG, U_MAX_DEG)
        rec = dict(round=rnd, lam=lam, phys=float(phys), hold=float(hold), hole_pen=float(hole_pen),
                   inside_pen=float(inside_pen), T_med=float(T.median()),
                   T_near=float(T[B["near"]].median()), smooth=float(smooth), minutes=(time.time() - t0) / 60)
        hist.append(rec)
        print(rec, flush=True)
        lam *= LAM_DECAY
    torch.save(pol.state_dict(), os.path.join(out, "odil_policy.pt"))
    lin = [mm for mm in pol.net if isinstance(mm, torch.nn.Linear)]
    np.savez(os.path.join(out, "odil_policy.npz"), **{f"W{i}": l.weight.detach().numpy() for i, l in enumerate(lin)},
             **{f"b{i}": l.bias.detach().numpy() for i, l in enumerate(lin)},
             n_in=np.array(14), tz=np.array(TZ), z_scale=np.array(Z_SCALE))
    json.dump(hist, open(os.path.join(out, "odil_history.json"), "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
