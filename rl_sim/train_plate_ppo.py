#!/usr/bin/env python3
"""
Train a goal-reaching policy for the real plate rig in simulation (2026-09-27).

  ../.venv-rl/bin/python3 train_plate_ppo.py [total_steps] [run_name]

PPO (stable-baselines3) on PlateGoalEnv (domain-randomised, calibrated to the rig),
8 environments in parallel. Every EVAL_EVERY steps the policy is evaluated on a fixed
set of test episodes; the best one is kept and exported to plain numpy weights
(runs/<name>/policy.npz) so the rig controller needs no PyTorch.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_goal_env import PlateGoalEnv  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
EVAL_EPISODES = 40
EVAL_EVERY = 500_000


def make_env(i):
    def _f():
        return PlateGoalEnv(seed=1000 + i)
    return _f


def evaluate(act_fn, n=EVAL_EPISODES, seed=12345):
    """Fixed test episodes (no goal switch counted separately): success = reached the
    target within 6 s; also time to reach, share of time inside after reaching,
    and action smoothness."""
    env = PlateGoalEnv(seed=seed)
    succ, t_reach, inside_after, jerk = [], [], [], []
    for ep in range(n):
        o, _ = env.reset(seed=seed + ep)
        env.switch_at = -1
        reached, ins, prev, dj = None, [], np.zeros(2), []
        for k in range(300):
            a = act_fn(o)
            o, r, te, tr, info = env.step(a)
            dj.append(float(np.sum((env.applied - prev) ** 2)))
            prev = env.applied.copy()
            if reached is None and info["inside"]:
                reached = k
            if reached is not None:
                ins.append(info["inside"])
        succ.append(reached is not None and reached < 180)
        t_reach.append((reached if reached is not None else 300) / 29.0)
        inside_after.append(np.mean(ins) if ins else 0.0)
        jerk.append(np.mean(dj))
    return {"success": float(np.mean(succ)), "t_reach_med": float(np.median(t_reach)),
            "inside_after": float(np.mean(inside_after)), "jerk": float(np.mean(jerk))}


def pd_baseline(o):
    """Plain PD in the same observation space, as a reference point."""
    e, v = o[0:2] * 0.1, o[2:4] * 0.1
    return np.clip((5.0 * e - 2.0 * v) / 0.8, -1, 1)


def export_numpy(model, path):
    p = model.policy
    layers = [m for m in p.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    w = {f"W{i}": l.weight.detach().numpy() for i, l in enumerate(layers)}
    w.update({f"b{i}": l.bias.detach().numpy() for i, l in enumerate(layers)})
    w["Wout"] = p.action_net.weight.detach().numpy()
    w["bout"] = p.action_net.bias.detach().numpy()
    w["n_hidden"] = np.array(len(layers))
    np.savez(path, **w)


class EvalCallback(BaseCallback):
    def __init__(self, run_dir):
        super().__init__()
        self.run_dir, self.best, self.next_eval, self.t0 = run_dir, -1e9, EVAL_EVERY, time.time()
        self.history = []

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval += EVAL_EVERY
            res = evaluate(lambda o: self.model.predict(o, deterministic=True)[0])
            score = res["success"] + 0.5 * res["inside_after"] - 0.02 * res["t_reach_med"]
            res.update(steps=self.num_timesteps, minutes=(time.time() - self.t0) / 60, score=score)
            self.history.append(res)
            print(f"[eval] {res}", flush=True)
            json.dump(self.history, open(os.path.join(self.run_dir, "eval_history.json"), "w"), indent=1)
            if score > self.best:
                self.best = score
                self.model.save(os.path.join(self.run_dir, "best_model"))
                export_numpy(self.model, os.path.join(self.run_dir, "policy.npz"))
        return True


def main():
    total = int(float(sys.argv[1])) if len(sys.argv) > 1 else 20_000_000
    name = sys.argv[2] if len(sys.argv) > 2 else "plate_goal"
    run_dir = os.path.join(RUNS, name)
    os.makedirs(run_dir, exist_ok=True)
    torch.set_num_threads(4)
    print("PD baseline in sim:", evaluate(pd_baseline), flush=True)
    venv = SubprocVecEnv([make_env(i) for i in range(8)])
    model = PPO("MlpPolicy", venv, n_steps=1024, batch_size=4096, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                learning_rate=3e-4, clip_range=0.2, ent_coef=0.0, verbose=0,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh,
                                   log_std_init=-0.5))
    model.learn(total_timesteps=total, callback=EvalCallback(run_dir))
    model.save(os.path.join(run_dir, "final_model"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
