#!/usr/bin/env python3
"""
Compare controllers in the full plate simulator (sticky, noisy, delayed, random slopes)
on identical episodes (2026-09-27): PD, RL (PPO v3), ODIL.

Metrics per episode (target radius from the env, goal fixed, 300 steps ~10 s):
  success       reached the target within 6 s
  t_reach       time to first enter the target
  inside_after  share of time inside after first arrival
  exits_after   how often it leaves the target again after arriving ("readjustments")
  jerk          mean squared change of the applied command per step

  ../.venv-rl/bin/python3 eval_controllers.py [n_episodes]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_goal_env import PlateGoalEnv, ACTION_SCALE  # noqa: E402

DT = 1.0 / 29.0
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


class MLP:
    def __init__(self, path, out_key=None):
        w = np.load(path)
        n = sum(1 for k in w.files if k.startswith("W") and k[1:].isdigit())
        self.layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(n)]
        self.head = (w["Wout"], w["bout"]) if "Wout" in w.files else None

    def __call__(self, x):
        h = np.asarray(x, dtype=np.float64)
        for i, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if self.head is not None or i < len(self.layers) - 1:
                h = np.tanh(h)
        if self.head is not None:
            h = self.head[0] @ h + self.head[1]
        return h


class ODILController:
    """ODIL policy on measured position/velocity + model-observed tilt states."""
    TAU = 0.045

    def __init__(self, path):
        self.net = MLP(path)
        w = np.load(path)
        self.n_in = int(w["n_in"]) if "n_in" in w.files else 8      # v5: + leaky goal-error integral
        self.tz = float(w["tz"]) if "tz" in w.files else 1.5
        self.z_scale = float(w["z_scale"]) if "z_scale" in w.files else 0.02
        self.reset()

    def reset(self):
        self.th = np.zeros(2)
        self.thl = np.zeros(2)
        self.z = np.zeros(2)

    def __call__(self, obs):
        rel, v = obs[0:2] * 0.1, obs[2:4] * 0.1
        applied_deg = obs[-2:] * 5.0                    # last applied command (deg)
        # observer: advance the model's tilt states with the command actually applied
        self.th += (applied_deg - self.th) * DT / self.TAU
        self.thl += (self.th - self.thl) * DT / self.TAU
        feat = np.concatenate([rel / 0.1, v / 0.1, self.th / 5.0, self.thl / 5.0])
        if self.n_in >= 10:
            self.z += (rel / self.z_scale - self.z / self.tz) * DT
            feat = np.concatenate([feat, self.z])
        if self.n_in >= 11:
            feat = np.concatenate([feat, [obs[8]]])            # target radius / 0.03 (build_obs)
        if self.n_in == 14:                                     # v12: hole features (set by the caller)
            feat = np.concatenate([feat, getattr(self, "_extra", np.zeros(3))])
        u_deg = 5.0 * 0.8 * np.tanh(self.net(feat))
        return np.clip(u_deg / 5.0 / ACTION_SCALE, -1, 1)   # env multiplies by ACTION_SCALE


def pd(obs):
    e, v = obs[0:2] * 0.1, obs[2:4] * 0.1
    return np.clip((5.0 * e - 2.0 * v) / 0.8, -1, 1)


def run(ctrl, n, seed=424242):
    env = PlateGoalEnv(seed=seed)
    R = {k: [] for k in ("success", "t_reach", "inside_after", "exits_after", "jerk", "final_dist")}
    for ep in range(n):
        o, _ = env.reset(seed=seed + ep)
        env.switch_at = -1
        if hasattr(ctrl, "reset"):
            ctrl.reset()
        reached, ins, exits, was_in, prev, dj = None, [], 0, False, np.zeros(2), []
        for k in range(300):
            o, r, te, tr, info = env.step(ctrl(o))
            dj.append(float(np.sum((env.applied - prev) ** 2)))
            prev = env.applied.copy()
            if reached is None and info["inside"]:
                reached = k
            if reached is not None:
                ins.append(info["inside"])
                if was_in and not info["inside"]:
                    exits += 1
            was_in = info["inside"]
        R["success"].append(reached is not None and reached < 180)
        R["t_reach"].append((reached if reached is not None else 300) * DT)
        R["inside_after"].append(np.mean(ins) if ins else 0.0)
        R["exits_after"].append(exits)
        R["jerk"].append(np.mean(dj))
        R["final_dist"].append(info["dist"])
    return {k: (float(np.mean(v)) if k != "t_reach" else float(np.median(v))) for k, v in R.items()}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ctrls = {"PD": pd}
    rl_path = os.path.join(RUNS, "plate_goal_v3", "policy.npz")
    if os.path.exists(rl_path):
        rl = MLP(rl_path)
        ctrls["RL v3"] = lambda o: np.clip(rl(o), -1, 1)
    for name in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, name, "odil_policy.npz")
        only = os.environ.get("EVAL_ONLY")
        if only and name not in only.split(","):
            continue
        if name.startswith("odil") and os.path.exists(p) and "smoke" not in name:
            ctrls[name] = ODILController(p)
    print(f"{'controller':12s} success  t_reach  inside_after  exits_after  jerk     final_dist")
    for name, c in ctrls.items():
        r = run(c, n)
        print(f"{name:12s} {r['success']:.2f}     {r['t_reach']:.2f}s    {r['inside_after']:.2f}          "
              f"{r['exits_after']:.2f}         {r['jerk']:.4f}   {1000 * r['final_dist']:.1f}mm")


if __name__ == "__main__":
    main()
