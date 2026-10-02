#!/usr/bin/env python3
"""
Simulator of the REAL maze board (2026-10-02): the plate physics of world_model.py plus the two
things the open-plate model lacks -- walls that stop the ball and holes that end the run.

Walls: the dense wall map (rl_hw/maze_walls.py -> maze/walls_grid.npz) becomes a signed distance
field; a ball whose centre comes closer to a wall than its (effective) radius is pushed back along
the field's gradient by a spring + damper, with sliding friction along the wall. Everything is
smooth, so gradients flow through contact (for ODIL refinement in finetune_track.py).
Holes: the centre crossing a hole's edge = a fall (hard flag for evaluation, soft penalty for
training).

The contact parameters (effective radius, stiffness, damping, wall friction) are fitted, together
with the world model, on the rig's real-maze logs with the multi-step loss; the check that
matters is the prediction error NEAR walls with and without the wall model.

  ../.venv-rl/bin/python3 maze_world.py fit  out.pt log1.csv [log2.csv ...]
  ../.venv-rl/bin/python3 maze_world.py eval model.pt log1.csv [...]
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("WM_HORIZON", "12")           # 0.41 s windows (longer ones rarely survive the late frames)
os.environ.setdefault("WM_DT_MAX", "0.075")         # ~4 % of maze frames arrive later than 0.06 s
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import world_model as wm                             # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
GRID = os.path.join(ROOT, "maze", "walls_grid.npz")
ROUTE = os.path.join(ROOT, "maze", "route.json")


class Walls(torch.nn.Module):
    """signed distance to the nearest wall (m, > 0 in free space) and the contact acceleration"""

    def __init__(self, grid=GRID, r_eff=0.0050, k=600.0, c=25.0, mu=0.15):
        super().__init__()
        g = np.load(grid)
        lo, cell = g["lo"].astype(float), float(g["cell"])
        if "sdf" not in g.files:
            raise RuntimeError("walls_grid.npz has no distance field -- run rl_hw/maze_walls.py again")
        sdf = g["sdf"].astype(np.float32)             # signed distance to the nearest wall, smoothed
        self.register_buffer("sdf", torch.tensor(sdf)[None, None])
        self.lo, self.cell = lo, cell
        self.ny, self.nx = sdf.shape
        # learnable, positive (log-parametrised)
        self.log_r = torch.nn.Parameter(torch.tensor(float(np.log(r_eff))))
        self.log_k = torch.nn.Parameter(torch.tensor(float(np.log(k))))
        self.log_c = torch.nn.Parameter(torch.tensor(float(np.log(c))))
        self.log_mu = torch.nn.Parameter(torch.tensor(float(np.log(mu))))

    def _sample(self, p):
        gx = 2 * (p[:, 0] - self.lo[0]) / (self.cell * (self.nx - 1)) - 1
        gy = 2 * (p[:, 1] - self.lo[1]) / (self.cell * (self.ny - 1)) - 1
        grid = torch.stack([gx, gy], -1)[None, :, None, :]
        return F.grid_sample(self.sdf, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0, :, 0]

    def dist(self, p):
        return self._sample(p)

    def normal(self, p, eps=0.0015):
        ex, ey = torch.tensor([eps, 0.0]), torch.tensor([0.0, eps])
        n = torch.stack([self._sample(p + ex) - self._sample(p - ex),
                         self._sample(p + ey) - self._sample(p - ey)], -1)
        return n / torch.sqrt((n ** 2).sum(-1, keepdim=True) + 1e-12)

    def acc(self, p, v):
        """contact acceleration (B, 2): spring + damper along the normal, friction along the wall"""
        r, k, c, mu = (torch.exp(q) for q in (self.log_r, self.log_k, self.log_c, self.log_mu))
        pen = F.softplus((r - self.dist(p)) * 2000.0) / 2000.0          # smooth relu, ~0.35 mm knee
        n = self.normal(p)
        vn = (v * n).sum(-1)
        f_n = k * pen + c * torch.relu(-vn) * torch.tanh(pen / 0.0005)   # damping only when in contact, closing
        vt = v - vn[:, None] * n
        f_t = -mu * f_n[:, None] * vt / torch.sqrt((vt ** 2).sum(-1, keepdim=True) + 0.003 ** 2)
        return f_n[:, None] * n + f_t


class MazeWorld(torch.nn.Module):
    """world model + walls; same step() signature as world_model.WorldModel"""

    def __init__(self, delay=3, walls=True):
        super().__init__()
        self.base = wm.WorldModel(delay=delay)
        self.walls = Walls() if walls else None
        self.delay = delay
        r = json.load(open(ROUTE))
        self.hole_c = torch.tensor([h["center"] for h in r["holes"]], dtype=torch.float32)
        self.hole_r = torch.tensor([h["radius"] for h in r["holes"]], dtype=torch.float32)

    def set_variation(self, rng=None, b=1, level_deg=0.3, slope_deg=0.4, slope_len=0.03, n_bumps=40,
                      static_sigma=0.3, k_sigma=0.08, acc_noise=0.004):
        """Run-to-run variation as measured on the rig (level drifts 0.3-0.5 deg between runs, local
        slopes up to ~1 deg, stiction varies a lot from place to place): per run a global level error,
        a smooth random slope field, a stiction and a gain factor, and a little random acceleration.
        Without it every simulated run ends at the same one or two holes. rng=None switches it off."""
        if rng is None:
            self.var = None
            return
        t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
        self.var = {"bias": t(rng.normal(0, level_deg, (b, 1, 2))),
                    "c": t(rng.uniform([-0.15, -0.13], [0.15, 0.13], (b, n_bumps, 2))),
                    "amp": t(rng.normal(0, slope_deg, (b, n_bumps, 2))), "len": slope_len,
                    "static": t(np.exp(rng.normal(0, static_sigma, (b, 1)))),
                    "k": t(np.exp(rng.normal(0, k_sigma, (b, 1)))), "noise": acc_noise}

    def _slope(self, p):
        """extra tilt (deg) the ball feels at p: global level error + the local slope field"""
        V = self.var
        w = torch.exp(-((p[:, None, :] - V["c"]) ** 2).sum(-1) / (2 * V["len"] ** 2))        # (B, K)
        return V["bias"][:, 0] + (w[..., None] * V["amp"]).sum(1) / 2.0

    def step(self, p, v, tilt, u_hist, u_now, dt=wm.DT, sub=4):
        tau = torch.exp(self.base.log_tau)
        h = dt / sub
        V = getattr(self, "var", None)
        for _ in range(sub):
            tilt = tilt + (u_now - tilt) * (h / tau)
            if V is None:
                a = self.base.acc(p, v, tilt, u_hist)
            else:
                a = self.base.acc(p, v, tilt + self._slope(p), u_hist, V["k"], V["static"])
                a = a + V["noise"] * torch.randn_like(a)
            if self.walls is not None:
                a = a + self.walls.acc(p, v)
            v = v + a * h
            p = p + v * h
        return p, v, tilt

    def hole_edge(self, p):
        """(B,) distance of the centre to the nearest hole's edge (m); < 0 = over the hole = falls"""
        return (torch.cdist(p, self.hole_c) - self.hole_r[None]).min(-1).values


def near_wall_mask(model, P0, POS, margin=0.010):
    """windows in which the ball comes within `margin` of a wall (start or any logged position)"""
    w = model.walls if model.walls is not None else Walls()
    with torch.no_grad():
        d0 = w.dist(P0)
        dmin = torch.stack([w.dist(POS[:, k]) for k in range(0, POS.shape[1], 4)], 1).min(1).values
    return torch.minimum(d0, dmin) < margin


def errors(model, data):
    with torch.no_grad():
        e = wm.rollout_loss(model, *data)
    return e


def report(tag, e, near):
    H = e.shape[1]
    print(f"{tag}: after {H} frames ({H * wm.DT:.2f} s)  all {1000 * e[:, -1].median():.1f} mm median / "
          f"{1000 * e[:, -1].mean():.1f} mean | near walls ({int(near.sum())} windows) "
          f"{1000 * e[near, -1].median():.1f} / {1000 * e[near, -1].mean():.1f} | "
          f"open ({int((~near).sum())}) {1000 * e[~near, -1].median():.1f} / {1000 * e[~near, -1].mean():.1f}", flush=True)


def fit(out, files):
    torch.set_num_threads(int(os.environ.get("WM_THREADS", "6")))
    data = wm.load_windows(files, int(os.environ.get("WM_MAX_ROWS", "0")) or None)
    n = len(data[0])
    print(f"{n} windows of {wm.H} frames from {len(files)} log(s)")
    idx = np.random.default_rng(0).permutation(n)
    tr, va = idx[: int(0.85 * n)], idx[int(0.85 * n):]
    vdat = tuple(d[va] for d in data)
    iters = int(os.environ.get("WM_ITERS", "500"))
    results = {}
    for name, use_walls in (("no walls", False), ("with walls", True)):
        m = MazeWorld(delay=int(os.environ.get("WM_DELAY", "3")), walls=use_walls)
        near = near_wall_mask(m, vdat[0], vdat[4])
        report(f"{name} / before", errors(m, vdat), near)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        # near-wall windows are the minority: sample them as half of every batch
        tr_near = near_wall_mask(m, data[0][tr], data[4][tr]).numpy()
        tn, to = tr[tr_near], tr[~tr_near]
        for it in range(iters):
            rng = np.random.default_rng(it)
            b = np.concatenate([rng.choice(tn, min(256, len(tn)), replace=False),
                                rng.choice(to, min(256, len(to)), replace=False)])
            e = wm.rollout_loss(m, *(d[torch.tensor(b)] for d in data))
            loss = (e / 0.005).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        e1 = errors(m, vdat)
        report(f"{name} / after {iters} its", e1, near)
        results[name] = (m, e1, near)
        if use_walls:
            w = m.walls
            print("contact: effective radius %.1f mm, stiffness %.0f /s^2, damping %.1f /s, wall friction %.2f" % (
                1000 * float(torch.exp(w.log_r)), float(torch.exp(w.log_k)), float(torch.exp(w.log_c)),
                float(torch.exp(w.log_mu))))
            torch.save(m.state_dict(), out)
            print("saved", out)
    b = m.base
    print("physics: tau %.3f s, k_acc %.3f, a_roll %.3f, static %.2f deg" % (
        float(torch.exp(b.log_tau)), float(torch.exp(b.log_k)), float(torch.exp(b.log_roll)),
        float(torch.exp(b.log_static))))


def main():
    mode, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    if mode == "fit":
        fit(out, files)
    else:
        m = MazeWorld()
        m.load_state_dict(torch.load(out))
        data = wm.load_windows(files)
        report("eval", errors(m, data), near_wall_mask(m, data[0], data[4]))


if __name__ == "__main__":
    main()
