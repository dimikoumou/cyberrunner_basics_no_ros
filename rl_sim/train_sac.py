"""
Train a model-free RL agent (SAC) to solve the maze.

Runs on CPU. Logs a baseline (untrained) success rate, trains, periodically
reports success rate so you can watch it learn, then saves the model, a learning
curve, and a rollout video of the trained agent.

Usage:
    python train_sac.py [total_timesteps]   # default 250000
"""
import os
import sys
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import cv2
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from maze_env import MazeEnv

OUT = "runs"
os.makedirs(OUT, exist_ok=True)
os.makedirs("models", exist_ok=True)


def evaluate(model, n=30, deterministic=True):
    env = MazeEnv()
    goals, steps_to_goal = 0, []
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            a = model.predict(obs, deterministic=deterministic)[0] if model else env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            done = term or trunc
        if info["status"] == "goal":
            goals += 1
            steps_to_goal.append(steps)
    sr = goals / n
    mean_steps = float(np.mean(steps_to_goal)) if steps_to_goal else float("nan")
    return sr, mean_steps


class ProgressCallback(BaseCallback):
    """Every eval_every steps, report success rate; log for the learning curve."""
    def __init__(self, eval_every=10000, verbose=0):
        super().__init__(verbose)
        self.eval_every = eval_every
        self.history = []  # (timesteps, success_rate, mean_steps)

    def _on_step(self):
        if self.num_timesteps % self.eval_every == 0:
            sr, ms = evaluate(self.model, n=20)
            self.history.append((self.num_timesteps, sr, ms))
            print(f"  [{self.num_timesteps:>7d} steps]  success={sr*100:5.1f}%  "
                  f"mean_steps_to_goal={ms:.0f}", flush=True)
        return True


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 250_000
    env = MazeEnv()

    print("baseline (random policy):")
    sr0, _ = evaluate(None, n=30)
    print(f"  success={sr0*100:.1f}%\n")

    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, buffer_size=300_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        learning_starts=5_000, policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0, device="cpu",
    )
    cb = ProgressCallback(eval_every=10_000)
    print(f"training SAC for {total} timesteps...")
    model.learn(total_timesteps=total, callback=cb, progress_bar=False)
    model.save("models/sac_maze")

    sr, ms = evaluate(model, n=50)
    print(f"\nFINAL: success={sr*100:.1f}%  mean_steps_to_goal={ms:.0f}")

    # learning curve
    if cb.history:
        t, s, _ = zip(*cb.history)
        plt.figure(figsize=(7, 4))
        plt.plot(t, np.array(s) * 100, marker="o")
        plt.xlabel("training timesteps"); plt.ylabel("success rate (%)")
        plt.title("SAC learning to solve the maze"); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{OUT}/learning_curve.png", dpi=110)
        print(f"saved {OUT}/learning_curve.png")

    # rollout video of the trained agent
    venv = MazeEnv(render_mode="rgb_array")
    obs, _ = venv.reset()
    frames, done = [], False
    while not done:
        a = model.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = venv.step(a)
        frames.append(venv.render())
        done = term or trunc
    if frames:
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(f"{OUT}/trained_rollout.avi",
                             cv2.VideoWriter_fourcc(*"XVID"), 60, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        print(f"saved {OUT}/trained_rollout.avi ({len(frames)} frames, status={info['status']})")


if __name__ == "__main__":
    main()
