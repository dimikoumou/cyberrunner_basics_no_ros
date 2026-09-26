"""
rl_hw/train_sac_hw.py

Train a goal-conditioned SAC policy online, directly against the real hardware
(camera + motors) via HardwarePlateEnv -- no simulator. Mirrors rl_sim/train_sac.py's
SAC hyperparameters; what's different here is real-hardware concerns: manual resets
(you'll be prompted to place the ball), periodic checkpointing (a crash shouldn't cost
hours of real training), and a clean Ctrl+C path that always de-torques the motors.

Usage:
    python train_sac_hw.py [total_timesteps]   # default 200000
"""
import os
import sys
import signal

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# Resolve output paths before plate_env.py chdirs the process into state_est/.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "sac_plate_hw")
os.makedirs(MODELS_DIR, exist_ok=True)

from plate_env import HardwarePlateEnv  # noqa: E402


class RollingStatsCallback(BaseCallback):
    """No cheap way to spin up a second real env for a separate eval rollout, so this
    reports rolling stats (success rate, mean reward) from the training episodes
    themselves, and checkpoints the model periodically."""

    def __init__(self, checkpoint_every=2_000, window=20, verbose=0):
        super().__init__(verbose)
        self.checkpoint_every = checkpoint_every
        self.window = window
        self.recent_status = []
        self.recent_reward = []
        self.recent_in_circle_frac = []
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_in_circle_steps = 0

    def _on_step(self):
        info = self.locals["infos"][0]
        self._ep_reward += float(self.locals["rewards"][0])
        self._ep_steps += 1
        if info.get("status") == "in_circle":
            self._ep_in_circle_steps += 1
        if bool(self.locals["dones"][0]):
            self.recent_status.append(info.get("status", "unknown"))
            self.recent_reward.append(self._ep_reward)
            # Balancing task: what matters is how much of the episode was spent
            # inside the goal circle, not whether the final/terminating step
            # happened to be "in_circle" (episodes now only end via ball_lost or
            # a timeout, not by reaching the circle).
            self.recent_in_circle_frac.append(self._ep_in_circle_steps / max(1, self._ep_steps))
            self._ep_reward = 0.0
            self._ep_steps = 0
            self._ep_in_circle_steps = 0
            self.recent_status = self.recent_status[-self.window:]
            self.recent_reward = self.recent_reward[-self.window:]
            self.recent_in_circle_frac = self.recent_in_circle_frac[-self.window:]
            n = len(self.recent_status)
            in_circle_pct = float(np.mean(self.recent_in_circle_frac)) * 100
            mean_r = float(np.mean(self.recent_reward))
            print(
                f"  [{self.num_timesteps:>7d} steps] episode done ({info.get('status')})  "
                f"rolling time-in-circle={in_circle_pct:5.1f}%  rolling mean reward={mean_r:7.2f}  "
                f"(last {n} episodes)",
                flush=True,
            )

        if self.num_timesteps % self.checkpoint_every == 0:
            self.model.save(CHECKPOINT_PATH)
            print(f"  [{self.num_timesteps:>7d} steps] checkpoint saved -> {CHECKPOINT_PATH}.zip", flush=True)
        return True


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000

    # Fixed single-goal task: balance inside the circle drawn by hand on the paper.
    # Center/radius measured 2026-08-29 by detecting the drawn circle in a camera
    # frame and backprojecting it through the same undistort+ball_pos_backproject
    # path used for the ball itself, so it's in the identical world frame as xb/yb.
    # Re-measure (see rl_hw/measure_goal_circle.py) if the paper is ever replaced.
    env = HardwarePlateEnv(fixed_goal=(-0.0093, 0.0080), goal_tolerance=0.048)

    def handle_sigint(signum, frame):
        print("\n[train_sac_hw] interrupted, disabling motors before exit...")
        env.close()
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_sigint)

    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4, buffer_size=300_000, batch_size=256,
        gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1,
        # Lower than the sim default (5000) -- real steps are slow and precious
        # (dominated by camera+estimation time, not the nominal 55Hz), so waiting
        # through a multi-minute pure-random phase before any learning-driven
        # behavior shows up isn't worth the extra replay-buffer diversity here.
        learning_starts=500, policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0, device="cpu",
    )
    cb = RollingStatsCallback(checkpoint_every=2_000)

    print(f"training SAC on real hardware for {total} timesteps...")
    try:
        model.learn(total_timesteps=total, callback=cb, progress_bar=False)
    finally:
        model.save(CHECKPOINT_PATH)
        print(f"final model saved -> {CHECKPOINT_PATH}.zip")
        env.close()


if __name__ == "__main__":
    main()
