"""
Train a PPO agent to solve the maze, using ALL cpu cores.

PPO is on-policy and embarrassingly parallel: we run N environments in separate
subprocesses (one per core) that collect experience simultaneously, then do a
batched policy update. This saturates the CPU and gives ~N x the data throughput
of the single-env SAC trainer.

Usage:
    python train_ppo.py [total_timesteps] [n_envs]
    # defaults: 2_000_000 timesteps, n_envs = cpu_count
"""
import os
import sys
os.environ.setdefault("MPLBACKEND", "Agg")
# Env subprocesses use pure-light numpy -> cap their BLAS to 1 thread each so they
# don't fight each other. The heavy compute (the PPO network update) runs in the
# MAIN process and is told to use all cores via torch.set_num_threads() below.
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from maze_env import MazeEnv

OUT = "runs"
os.makedirs(OUT, exist_ok=True)
os.makedirs("models", exist_ok=True)


def make_env():
    return Monitor(MazeEnv())


def evaluate(model, n=30):
    env = MazeEnv()
    n_cp = len(env.cfg.checkpoints) - 1
    goals, steps_to_goal, max_cps = 0, [], []
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            a = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            done = term or trunc
        max_cps.append(info["max_cp"])
        if info["status"] == "goal":
            goals += 1
            steps_to_goal.append(steps)
    sr = goals / n
    ms = float(np.mean(steps_to_goal)) if steps_to_goal else float("nan")
    return sr, ms, float(np.mean(max_cps)), n_cp


class ProgressCallback(BaseCallback):
    def __init__(self, eval_every=50_000, verbose=0):
        super().__init__(verbose)
        self.eval_every = eval_every
        self._next = eval_every
        self.history = []

    def _on_step(self):
        if self.num_timesteps >= self._next:
            self._next += self.eval_every
            sr, ms, mcp, ncp = evaluate(self.model, n=20)
            self.history.append((self.num_timesteps, sr, mcp))
            print(f"  [{self.num_timesteps:>8d} steps]  success={sr*100:5.1f}%  "
                  f"progress={mcp:.1f}/{ncp} checkpoints  steps_to_goal={ms:.0f}", flush=True)
        return True


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else (os.cpu_count() or 8)
    n_cores = os.cpu_count() or 8
    torch.set_num_threads(n_cores)   # let the PPO update use all cores
    print(f"PPO on {n_envs} parallel envs, torch threads={n_cores}, target {total} timesteps")

    sr0, _, mcp0, ncp = evaluate(None, n=30)
    print(f"baseline (random): success={sr0*100:.1f}%  progress={mcp0:.1f}/{ncp}\n", flush=True)

    venv = SubprocVecEnv([make_env for _ in range(n_envs)])
    model = PPO(
        "MlpPolicy", venv,
        n_steps=1024, batch_size=2048, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, learning_rate=3e-4,
        use_sde=True, sde_sample_freq=4,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0, device="cpu",
    )
    cb = ProgressCallback(eval_every=50_000)
    model.learn(total_timesteps=total, callback=cb, progress_bar=False)
    model.save("models/ppo_maze")
    venv.close()

    sr, ms, mcp, ncp = evaluate(model, n=50)
    print(f"\nFINAL: success={sr*100:.1f}%  progress={mcp:.1f}/{ncp}  mean_steps_to_goal={ms:.0f}")

    if cb.history:
        t, s, cp = zip(*cb.history)
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(t, np.array(s) * 100, marker="o", color="tab:blue")
        ax1.set_xlabel("training timesteps"); ax1.set_ylabel("success rate (%)", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(t, cp, marker="s", color="tab:green")
        ax2.set_ylabel(f"mean checkpoints reached (/{ncp})", color="tab:green")
        plt.title("PPO learning to solve the maze"); ax1.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{OUT}/learning_curve.png", dpi=110)
        print(f"saved {OUT}/learning_curve.png")

    venv2 = MazeEnv(render_mode="rgb_array")
    obs, _ = venv2.reset()
    frames, done = [], False
    while not done:
        a = model.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = venv2.step(a)
        frames.append(venv2.render())
        done = term or trunc
    if frames:
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(f"{OUT}/trained_rollout.avi", cv2.VideoWriter_fourcc(*"XVID"), 60, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"saved {OUT}/trained_rollout.avi ({len(frames)} frames, status={info['status']})")


if __name__ == "__main__":
    main()
