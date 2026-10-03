#!/usr/bin/env python3
"""Figures for the visual guide (physical_training/report/). Run with ../../.venv-rl/bin/python3."""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon   # noqa: E402
import numpy as np                                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SIM = os.path.join(ROOT, "rl_sim")
FIG = os.path.join(HERE, "fig")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, SIM)

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
C = {"odil": "#2563eb", "ppo": "#dc2626", "sac": "#d97706", "classic": "#6b7280", "data": "#059669"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#9ca3af",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, title, lines=(), color="#9ca3af", fill="white", fs=10):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fill, ec=color, lw=1.6))
    ax.text(x, y + (0.12 if lines else 0), title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK)
    for i, l in enumerate(lines):
        ax.text(x, y - 0.12 - 0.17 * i, l, ha="center", va="center", fontsize=fs - 2, color=MUTED)


def arrow(ax, p, q, color="#6b7280", text=None, rad=0.0, off=(0, 0.08)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", fontsize=8, color=MUTED)


def blank(w=10, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_axis_off()
    return fig, ax


# 1 -- the control loop -------------------------------------------------------------------------
def fig_loop():
    fig, ax = blank(10, 3.4)
    ax.set_xlim(0, 10), ax.set_ylim(0, 3.4)
    ax.text(0.1, 3.2, "Thirty times a second: see the ball, decide a tilt, move the motors", fontsize=12,
            fontweight="bold", color=INK)
    pts = [(1.4, 2.2), (4.4, 2.2), (7.4, 2.2), (7.4, 0.7), (4.4, 0.7), (1.4, 0.7)]
    labels = [("1. Camera", ["30 photos / s"]), ("2. Find plate + ball", ["markers -> tilt", "colour -> ball"]),
              ("3. Position + speed", ["mm on the plate", "from 2 photos"]),
              ("4. Controller", ["ODIL / SAC / PPO", "-> wanted tilt"]),
              ("5. Tilt servo", ["camera angle vs wanted", "nudges 2 motors"]), ("6. Plate tilts", ["ball rolls"])]
    for (x, y), (t, l) in zip(pts, labels):
        box(ax, x, y, 2.3, 0.95, t, l, color=C["odil"] if t.startswith("4") else "#9ca3af",
            fill="#eff6ff" if t.startswith("4") else "white")
    for a, b in [(0, 1), (1, 2)]:
        arrow(ax, (pts[a][0] + 1.15, 2.2), (pts[b][0] - 1.15, 2.2))
    arrow(ax, (7.4, 1.72), (7.4, 1.18))
    for a, b in [(3, 4), (4, 5)]:
        arrow(ax, (pts[a][0] - 1.15, 0.7), (pts[b][0] + 1.15, 0.7))
    arrow(ax, (1.4, 1.18), (1.4, 1.72), text="33 ms later", off=(0.55, 0))
    save(fig, "01_loop.png")


# 2 -- ball physics + real plate response --------------------------------------------------------
def fig_physics():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.3})
    a.set_axis_off(), a.set_xlim(-1.2, 1.2), a.set_ylim(-0.6, 0.9), a.set_aspect("equal")
    th = np.radians(12)
    p0, p1 = np.array([-1, 0.1]), np.array([1, 0.1 - 2 * np.tan(th)])
    a.add_patch(Polygon([p0, p1, p1 + [0, -0.08], p0 + [0, -0.08]], fc="#e5e7eb", ec="#9ca3af"))
    c = np.array([0.15, 0.1 - 1.15 * np.tan(th) + 0.16])
    a.add_patch(Circle(c, 0.15, fc="#0f766e", ec="#134e4a"))
    d = np.array([np.cos(th), -np.sin(th)])
    arrow(a, tuple(c), tuple(c + 0.55 * d), color=C["odil"])
    a.text(-0.95, 0.45, "acceleration = 5/7 g sin(tilt)", color=C["odil"], fontsize=9)
    a.text(-1.1, 0.75, "Rolling ball on a tilted plate", fontweight="bold", color=INK)
    a.text(-1.1, -0.45, "1 deg of tilt -> ~0.12 m/s^2;\nfriction: rolling + 'stiction' (~1.8 deg to start)",
           fontsize=8.5, color=MUTED)
    # real response: commanded vs measured tilt (dry run, random tilts)
    fn = sorted(glob.glob(os.path.join(ROOT, "physical_training", "data", "odil1_dry_r0.csv")))
    if fn:
        r = list(csv.DictReader(open(fn[0])))[400:700]
        t = np.array([float(q["t"]) for q in r]); t -= t[0]
        cmd = np.array([float(q["a0"]) for q in r]) * 5
        beta = np.degrees([float(q["beta"]) for q in r]) - 2.55
        b.plot(t, cmd, color=MUTED, lw=1.4, label="commanded tilt")
        b.plot(t, beta, color=C["odil"], lw=1.6, label="measured by the camera")
        b.set_xlabel("time (s)"), b.set_ylabel("tilt (deg)")
        b.set_title("The real plate follows the command late (delay + lag)")
        b.legend(frameon=False, fontsize=8)
    save(fig, "02_physics.png")


# 3 -- simulated trajectories of the trained controllers -----------------------------------------
def rollouts():
    from plate_goal_env import PlateGoalEnv
    from eval_controllers import MLP, pd
    from odil_friction_comp import ODILFrictionComp
    runs = os.path.join(SIM, "runs")
    rl = MLP(os.path.join(runs, "plate_goal_v3", "policy.npz"))
    ctrls = {"Classic": (pd, C["classic"]),
             "PPO (RL)": (lambda o: np.clip(rl(o), -1, 1), C["ppo"]),
             "ODIL": (ODILFrictionComp(os.path.join(runs, "odil_best", "odil_policy.npz")), C["odil"])}
    seeds = [11, 23, 37, 58]
    out = {}
    for name, (ctl, col) in ctrls.items():
        eps = []
        for s in seeds:
            env = PlateGoalEnv(seed=s)
            o, _ = env.reset(seed=s)
            env.switch_at = -1
            if hasattr(ctl, "reset"):
                ctl.reset()
            P, U, D = [env.sim.pos.copy()], [], []
            for k in range(150):
                o, r, te, tr, info = env.step(ctl(o))
                P.append(env.sim.pos.copy()), U.append(env.applied.copy() * 5), D.append(info["dist"])
            eps.append({"P": np.array(P), "U": np.array(U), "D": np.array(D), "goal": env.goal.copy(), "r": env.radius})
        out[name] = (eps, col)
    return out


def fig_trajectories(R):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharex=True, sharey=True)
    for ax, (name, (eps, col)) in zip(axes, R.items()):
        for e in eps:
            ax.add_patch(Circle(e["goal"] * 1000, e["r"] * 1000, fc="#fee2e2", ec="#ef4444", lw=1))
            ax.plot(e["P"][:, 0] * 1000, e["P"][:, 1] * 1000, color=col, lw=1.6)
            ax.plot(*(e["P"][0] * 1000), "o", color=col, ms=4)
        ax.set_title(name, color=col)
        ax.set_aspect("equal"), ax.set_xlim(-140, 140), ax.set_ylim(-120, 120)
        ax.set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    fig.suptitle("Same 4 starts (dots), same targets (red): paths of the trained controllers (simulation)",
                 fontsize=11, fontweight="bold", color=INK)
    save(fig, "03_trajectories.png")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2))
    for name, (eps, col) in R.items():
        e = eps[0]
        t = np.arange(len(e["D"])) / 29
        a.plot(t, e["D"] * 1000, color=col, lw=1.6, label=name)
        b.plot(t, e["U"][:, 0], color=col, lw=1.3, label=name)
    a.axhline(eps[0]["r"] * 1000, color="#ef4444", lw=1, ls="--")
    a.text(0.05, eps[0]["r"] * 1000 - 7, "target edge", color="#ef4444", fontsize=8)
    a.set_title("Distance to the target (one start)"), a.set_xlabel("time (s)"), a.set_ylabel("mm")
    b.set_title("Tilt command over time: smoothness"), b.set_xlabel("time (s)"), b.set_ylabel("deg (x axis)")
    a.legend(frameon=False, fontsize=8), b.legend(frameon=False, fontsize=8)
    save(fig, "04_distance_tilt.png")


# 4 -- ODIL worked example: optimise the whole trajectory at once --------------------------------
def fig_odil_demo():
    """1-D ODIL: positions x_0..x_N and tilts u_n are ALL unknowns; the loss asks that every step obey
    x'' = k u (finite differences), that the ball starts at rest at 0 and stops at 1, smoothly"""
    import torch
    N, dt, k = 40, 0.05, 1.2
    t = torch.linspace(0, 1, N + 1)
    guess = t + 0.18 * torch.sin(2 * 3.14159 * t)                      # a wrong first guess
    xi = guess[1:-1].clone().requires_grad_()
    u = torch.zeros(N - 1, requires_grad=True)
    opt = torch.optim.Adam([xi, u], lr=0.02)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.998)
    snaps, hist = {}, []
    for it in range(2001):
        x = torch.cat([torch.zeros(2), xi, torch.ones(2)])             # start at rest at 0, stop at rest at 1
        acc = (x[2:] - 2 * x[1:-1] + x[:-2]) / dt ** 2
        resid = acc[1:-1] - k * u                                      # physics: x'' = k u at every step
        loss = (resid ** 2).mean() + 0.002 * ((u[1:] - u[:-1]) ** 2).sum()
        hist.append(float((resid ** 2).mean()))
        if it in (0, 100, 400, 2000):
            snaps[it] = (x.detach().numpy()[1:-1].copy(), u.detach().numpy().copy(), hist[-1])
        opt.zero_grad(), loss.backward(), opt.step(), sched.step()
    tt = np.arange(N + 1) * dt
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 3.3))
    cols = ["#cbd5e1", "#93c5fd", "#3b82f6", "#1e3a8a"]
    for (it, (xs, us, r)), col in zip(snaps.items(), cols):
        a.plot(tt, xs, color=col, lw=2, label=f"iteration {it}")
        b.plot(tt[1:-1], us, color=col, lw=2)
    a.set_title("Path: all positions are unknowns"), a.set_xlabel("time (s)"), a.set_ylabel("ball position")
    a.legend(frameon=False, fontsize=8, loc="lower right")
    b.set_title("Tilt: every command is an unknown"), b.set_xlabel("time (s)"), b.set_ylabel("tilt")
    c.semilogy(hist, color=C["odil"], lw=1.8)
    c.set_title("Physics error falls together"), c.set_xlabel("iteration"), c.set_ylabel("mean squared residual")
    fig.tight_layout()
    save(fig, "05_odil_demo.png")


# 5 -- method diagrams ---------------------------------------------------------------------------
def fig_method_odil():
    fig, ax = blank(10, 2.9)
    ax.set_xlim(0, 10), ax.set_ylim(-0.4, 2.3)
    box(ax, 1.2, 1.3, 2.0, 1.1, "Rig recordings", ["ball + tilt,", "30 min"], C["data"], "#ecfdf5")
    box(ax, 3.7, 1.3, 2.0, 1.1, "Fit the physics", ["delay, lag, gain,", "friction, stiction"], C["odil"], "#eff6ff")
    box(ax, 6.3, 1.3, 2.3, 1.1, "Optimise (ODIL)", ["paths + tilts + network", "obey the physics"], C["odil"], "#eff6ff")
    box(ax, 8.8, 1.3, 1.9, 1.1, "Controller", ["small network", "-> test on rig"], C["odil"], "white")
    for a_, b_ in [(2.2, 2.7), (4.7, 5.15), (7.45, 7.85)]:
        arrow(ax, (a_, 1.3), (b_, 1.3))
    arrow(ax, (8.8, 0.75), (1.2, 0.75), rad=-0.25, text="next round drives with the new controller", off=(0, -0.75))
    save(fig, "06_method_odil.png")


def fig_method_ppo():
    fig, ax = blank(10, 2.9)
    ax.set_xlim(0, 10), ax.set_ylim(-0.4, 2.3)
    box(ax, 1.5, 1.3, 2.4, 1.1, "Drive 2048 steps", ["~70 s on the rig", "current controller"], C["ppo"], "#fef2f2")
    box(ax, 4.6, 1.3, 2.4, 1.1, "Score the batch", ["rewards -> which", "actions were good"], C["ppo"], "white")
    box(ax, 7.6, 1.3, 2.4, 1.1, "Small update", ["then THROW the", "batch away"], C["ppo"], "white")
    arrow(ax, (2.7, 1.3), (3.4, 1.3)), arrow(ax, (5.8, 1.3), (6.4, 1.3))
    arrow(ax, (7.6, 0.75), (1.5, 0.75), rad=-0.25, text="repeat with fresh data: on-policy", off=(0, -0.75))
    save(fig, "07_method_ppo.png")


def fig_method_sac():
    fig, ax = blank(10, 3.3)
    ax.set_xlim(0, 10), ax.set_ylim(-0.5, 2.9)
    box(ax, 1.4, 1.5, 2.2, 1.1, "Drive 15 s", ["actor + a bit", "of randomness"], C["sac"], "#fffbeb")
    box(ax, 4.4, 1.5, 2.4, 1.3, "Memory", ["every step kept", "(up to 300 000)"], C["sac"], "white")
    box(ax, 7.6, 2.25, 2.3, 0.9, "Critic", ["how good is this tilt here?"], C["sac"], "white", fs=9)
    box(ax, 7.6, 0.75, 2.3, 0.9, "Actor", ["the controller"], C["sac"], "white", fs=9)
    arrow(ax, (2.5, 1.5), (3.2, 1.5)), arrow(ax, (5.6, 1.7), (6.45, 2.2)), arrow(ax, (5.6, 1.3), (6.45, 0.8))
    arrow(ax, (7.6, 1.8), (7.6, 1.2), text="actor follows the critic", off=(1.0, 0))
    arrow(ax, (6.45, 0.6), (1.4, 0.95), rad=-0.2, text="replays old steps many times: off-policy", off=(0, -0.75))
    save(fig, "08_method_sac.png")


# 6 -- PPO learning curve and experience needed --------------------------------------------------
def fig_ppo_curve():
    h = json.load(open(os.path.join(SIM, "runs", "plate_goal_v3", "eval_history.json")))
    steps = np.array([r["steps"] for r in h]) / 1e6
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(steps, [100 * r["success"] for r in h], color=C["ppo"], lw=2, label="targets reached (%)")
    ax.plot(steps, [100 * r["inside_after"] for r in h], color=C["ppo"], lw=1.5, ls="--", label="inside afterwards (%)")
    ax.set_xlabel("PPO training steps (millions)"), ax.set_ylabel("%"), ax.set_ylim(0, 105)
    sec = ax.secondary_xaxis("top", functions=(lambda s: s * 1e6 / 29 / 3600, lambda h_: h_ * 3600 * 29 / 1e6))
    sec.set_xlabel("= hours of driving if this happened on the rig (29 steps/s)", color=MUTED)
    ax.axvline(2.5, color=MUTED, lw=1, ls=":")
    ax.text(2.6, 15, "best: 2.5 M steps = 24 h", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("PPO in simulation: how much experience it needed", pad=28)
    save(fig, "09_ppo_curve.png")


def fig_experience():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"wspace": 0.45})
    names = ["ODIL", "PPO to its best", "PPO full run"]
    hours = [38 / 60, 24, 192]
    a.barh(names, hours, color=[C["odil"], C["ppo"], "#fca5a5"])
    a.set_xscale("log"), a.set_xlabel("hours of ball time learned from (log scale)")
    for i, v in enumerate(hours):
        a.text(v * 1.15, i, f"{v * 60:.0f} min" if v < 1 else f"{v:.0f} h", va="center", fontsize=9, color=INK)
    a.set_title("Experience needed\n(simulation, ball time learned from)"), a.grid(axis="y", visible=False)
    setups = ["Classic", "RL + settle", "ODIL", "ODIL + settle"]
    jerk = [0.0095, 0.0107, 0.0039, 0.0083]
    inside = [0.56, 0.68, 0.68, 0.68]
    x = np.arange(4)
    b.bar(x - 0.2, inside, 0.4, color="#cbd5e1", label="inside after arrival (share)")
    b2 = b.twinx()
    b2.bar(x + 0.2, jerk, 0.4, color=[C["classic"], C["ppo"], C["odil"], "#93c5fd"])
    b.set_xticks(x, setups, fontsize=8), b.set_ylim(0, 1), b2.set_ylim(0, 0.014)
    b2.spines["right"].set_visible(True), b2.grid(False)
    b.set_title("Rig, 120 random targets:\nsame accuracy, ODIL 2.5x smoother")
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color="#cbd5e1", label="inside after arrival (left axis)"),
                      Patch(color="#64748b", label="jerk, lower = smoother (right axis)")],
             frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "10_experience_rig.png")


# 7 -- the physical-training session ------------------------------------------------------------
def fig_session():
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(10, 2.9))
    rows = [("References", [(0, 0.25, "#9ca3af")]),
            ("ODIL", [(0.25, 0.75, C["data"]), (0.75, 1.1, C["odil"]), (1.1, 1.18, INK),
                      (1.18, 1.68, C["data"]), (1.68, 2.03, C["odil"]), (2.03, 2.11, INK),
                      (2.11, 2.61, C["data"]), (2.61, 2.96, C["odil"]), (2.96, 3.04, INK)]),
            ("SAC", [(3.04, 13.04, C["sac"])])]
    for i, (name, segs) in enumerate(rows):
        y = 2 - i
        for s_, e, col in segs:
            ax.barh(y, e - s_, left=s_, color=col, height=0.5)
        ax.text(-0.2, y, name, ha="right", va="center", fontsize=9, fontweight="bold", color=INK)
    for t in np.arange(3.54, 13.1, 0.5):
        ax.plot([t, t], [-0.25, 0.25], color=INK, lw=1.2)
    ax.set_yticks([]), ax.set_xlim(-0.1, 13.2), ax.set_xlabel("hours into the session")
    ax.legend(handles=[Patch(color="#9ca3af", label="sim-trained references"), Patch(color=C["data"], label="ODIL drives (data)"),
                       Patch(color=C["odil"], label="fit + train, plate level"), Patch(color=C["sac"], label="SAC learns on the rig"),
                       Patch(color=INK, label="30-target test")], frameon=False, fontsize=7.5, ncol=5,
              loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title("One session (about 13 h)")
    save(fig, "11_session.png")


def fig_random_tilts():
    fn = os.path.join(ROOT, "physical_training", "data", "odil1_dry_r0.csv")
    if not os.path.exists(fn):
        return
    r = list(csv.DictReader(open(fn)))
    x = np.array([float(q["xb"]) for q in r]) * 1000
    y = np.array([float(q["yb"]) for q in r]) * 1000
    ok = np.array([q["ball_found"] == "1" for q in r]) & (np.abs(np.diff(x, prepend=x[0])) < 30)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    sc = ax.scatter(x[ok], y[ok], c=np.arange(len(x))[ok] / 29, s=3, cmap="viridis")
    ax.set_aspect("equal"), ax.set_xlim(-142, 142), ax.set_ylim(-120, 120)
    ax.add_patch(plt.Rectangle((-141.7, -119.2), 283.4, 238.4, fill=False, ec=INK, lw=1.5))
    plt.colorbar(sc, ax=ax, label="seconds")
    ax.set_title("Round-0 data on the real rig:\nrandom tilts (dry run, 2 min)")
    ax.set_xlabel("x (mm)"), ax.set_ylabel("y (mm)")
    save(fig, "12_random_tilts.png")


if __name__ == "__main__":
    fig_loop(); fig_physics(); fig_odil_demo(); fig_method_odil(); fig_method_ppo(); fig_method_sac()
    fig_ppo_curve(); fig_experience(); fig_session(); fig_random_tilts()
    R = rollouts()
    fig_trajectories(R)
    print("figures:", sorted(os.listdir(FIG)))
