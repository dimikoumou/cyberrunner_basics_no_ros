#!/usr/bin/env python3
"""
Closed-loop refinement of the ODIL tracking policy (2026-09-27).

The ODIL tracking policy (odil_track.py) fits its optimised trajectories (0.7 mm) but, rolled
out in closed loop -- even in its own model -- drifts (median 7 mm, p90 30 mm): the discrete
loss never shows it its own compounding errors. Here the ODIL policy is the INITIAL policy and is
refined by backpropagation through closed-loop rollouts in a differentiable plate model with the
rig-identified timing: pure delay of 1-3 control steps + a first-order lag (60-110 ms), one frame
of camera delay, velocity from noisy finite differences, the rig's rate limit, smooth static
("hold") friction, rolling friction, per-rollout gain / stiction / slope. Loss: squared distance to
the moving reference (+ small smoothness term).

  ../.venv-rl/bin/python3 finetune_track.py <init_policy.npz> <out_name> [iters]
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import odil_track as ot  # noqa: E402

torch.set_num_threads(int(os.environ.get("FT_THREADS", "6")))
DT, SUB = 1.0 / 29.0, 4
B, STEPS = int(os.environ.get("FT_BATCH", "96")), int(os.environ.get("FT_STEPS", "120"))   # 120 steps ~ 4 s
A_ROLL, EPS_V, V_STRIB = 0.042, 0.004, 0.01
POS_NOISE = 0.0004
JERK_W = float(os.environ.get("FT_JERK_W", "20"))   # smoothness weight (ft1: 20 -> accurate but jerky on the rig)
# "the board swerved" (user): penalise tilt BEYOND what the line needs -- the reference's own
# acceleration a_ref / k_acc is the tilt a perfect follower would use
EFFORT_W = float(os.environ.get("FT_EFFORT_W", "0"))


class Pol(torch.nn.Module):
    def __init__(self, npz):
        super().__init__()
        w = np.load(npz)
        n = sum(1 for k in w.files if k.startswith("W") and k[1:].isdigit())
        self.lins = torch.nn.ModuleList()
        for i in range(n):
            lin = torch.nn.Linear(w[f"W{i}"].shape[1], w[f"W{i}"].shape[0])
            lin.weight.data = torch.tensor(w[f"W{i}"])
            lin.bias.data = torch.tensor(w[f"b{i}"])
            self.lins.append(lin)

    def forward(self, feat):
        h = feat
        for i, l in enumerate(self.lins):
            h = l(h)
            if i < len(self.lins) - 1:
                h = torch.tanh(h)
        return ot.U_MAX_DEG * torch.tanh(h)


def rollout(pol, P, V, A, rng, train=True):
    """closed-loop rollout on references (B, STEPS+1, 2) -> tracking errors (B, STEPS)"""
    b = P.shape[0]
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    k_acc = t(rng.uniform(0.100, 0.125, (b, 1)))
    a_st = k_acc * t(rng.uniform(1.2, 2.6, (b, 1)))
    bd = rng.normal(0, 1, (b, 2))
    bias = t(bd / np.linalg.norm(bd, axis=1, keepdims=True) * rng.uniform(0, 0.8, (b, 1)))
    dly = torch.tensor(rng.integers(1, 4, b))           # pure delay, control steps
    ar = torch.arange(b)
    tau = t(rng.uniform(0.06, 0.11, (b, 1)))
    # start: behind / off the reference, 40 % at rest
    that = V[:, 0] / torch.clamp(V[:, 0].norm(dim=-1, keepdim=True), min=1e-6)
    p = P[:, 0] - that * t(rng.uniform(0, 0.02, (b, 1))) + t(rng.normal(0, 0.005, (b, 2)))
    v = torch.where(t(rng.random((b, 1)) < 0.4) > 0, torch.zeros(b, 2), V[:, 0] * t(rng.uniform(0.3, 1.2, (b, 1))))
    tilt = A[:, 0] / k_acc
    o1, o2, z = tilt.clone(), tilt.clone(), torch.zeros(b, 2)
    ubuf = [tilt.clone() for _ in range(4)]             # commands on their way to the plate
    applied = tilt / 5.0
    meas_prev = p.clone()
    p_seen, v_seen = p.clone(), v.clone()
    pend_p, pend_v = p.clone(), v.clone()
    errs, jerk, effort = [], [], []
    for n in range(STEPS):
        e = P[:, n] - p_seen
        z = z + (e / ot.Z_SCALE - z / ot.TZ) * DT
        feat = torch.cat([e / ot.E_SCALE, (V[:, n] - v_seen) / 0.05, A[:, n] / (ot.K_NOM * 5.0), v_seen / 0.1,
                          o1 / 5.0, o2 / 5.0, z], -1)
        u = pol(feat)                                    # deg
        effort.append((((u - A[:, n] / ot.K_NOM) / ot.U_MAX_DEG) ** 2).sum(-1))
        a_cmd = torch.clamp(u / 5.0, -0.8, 0.8)
        a_new = applied + torch.clamp(a_cmd - applied, -0.5, 0.5)
        jerk.append(((a_new - applied) ** 2).sum(-1))
        applied = a_new
        u_ap = 5.0 * applied
        o1 = o1 + (u_ap - o1) * min(1.0, DT / ot.TAU_OBS)
        o2 = o2 + (o1 - o2) * min(1.0, DT / ot.TAU_OBS)
        ubuf = ubuf[1:] + [u_ap]
        u_plate = torch.stack(ubuf)[len(ubuf) - 1 - dly, ar]
        h = DT / SUB
        for _ in range(SUB):
            tilt = tilt + (u_plate - tilt) * (h / tau)
            speed = torch.sqrt((v ** 2).sum(-1, keepdim=True) + EPS_V ** 2)
            drive = k_acc * (tilt + bias)
            w = torch.exp(-(speed / V_STRIB) ** 2)
            dmag = torch.sqrt((drive ** 2).sum(-1, keepdim=True) + 1e-8)
            kk = 0.02 * a_st
            held = -kk * torch.log(torch.exp(-dmag / kk) + torch.exp(-a_st.expand_as(dmag) / kk))
            acc = drive - w * held * drive / dmag - (1 - w) * A_ROLL * v / speed
            v = v + acc * h
            p = p + v * h
        errs.append((p - P[:, n + 1]).norm(dim=-1))
        # camera: noisy, one frame late (the policy acts on the previous frame's measurement);
        # velocity = finite difference of the noisy positions
        meas = p + (t(rng.normal(0, POS_NOISE, (b, 2))) if train else 0.0)
        p_seen, v_seen = pend_p, pend_v
        pend_p, pend_v = meas, (meas - meas_prev) / DT
        meas_prev = meas
    return torch.stack(errs, 1), torch.stack(jerk, 1), torch.stack(effort, 1)


def main():
    init, name = sys.argv[1], sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    out = os.path.join(HERE, "runs", name)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)
    pol = Pol(init)
    opt = torch.optim.Adam(pol.parameters(), lr=float(os.environ.get("FT_LR", "3e-4")))
    t0, hist = time.time(), []
    Pv, Vv, Av = (torch.tensor(a, dtype=torch.float32) for a in ot._references(np.random.default_rng(99), 64, STEPS + 1, STEPS * DT))
    for it in range(iters):
        if it % 25 == 0:      # fresh references every 25 iterations
            P, V, A = (torch.tensor(a, dtype=torch.float32) for a in ot._references(rng, B, STEPS + 1, STEPS * DT))
        err, jerk, eff = rollout(pol, P, V, A, rng)
        loss = ((err / 0.005) ** 2).mean() + JERK_W * jerk.mean() + EFFORT_W * eff.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        if it % 100 == 0 or it == iters - 1:
            with torch.no_grad():
                ev, jv, efv = rollout(pol, Pv, Vv, Av, np.random.default_rng(123), train=False)
            e = ev.flatten().numpy() * 1000
            rec = dict(it=it, median_mm=float(np.median(e)), p90_mm=float(np.percentile(e, 90)),
                       max_mm=float(e.max()), jerk=float(jv.mean()), extra_tilt=float(efv.mean()),
                       minutes=(time.time() - t0) / 60)
            hist.append(rec)
            print(rec, flush=True)
            np.savez(os.path.join(out, "odil_track_policy.npz"),
                     **{f"W{i}": l.weight.detach().numpy() for i, l in enumerate(pol.lins)},
                     **{f"b{i}": l.bias.detach().numpy() for i, l in enumerate(pol.lins)}, n_in=np.array(14))
    json.dump(hist, open(os.path.join(out, "history.json"), "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
