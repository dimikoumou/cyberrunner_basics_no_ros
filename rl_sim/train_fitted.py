"""
Model-based baseline for the paper ("is it ODIL, or just having a model?", 2026-10-05): SAC / PPO with the
rig runs' exact settings (physical_training/rig_learn.py rl()), trained in PlateGoalEnv -- the same
observation, action and reward as on the rig -- whose physics is set to ODIL's OWN fit from the same rig
data (the "odil fitted+trained" event of one ODIL round), with ODIL's randomisation widths:
k_acc 0.9-1.1x, servo tau 0.7*min-1.3*max, breakaway 0.7-1.3x, a_roll fixed (rig_learn.odil_rounds).
Everything else keeps PlateSim's default randomisation.

Saved as runs/fit_<odil run><rig min>_<algo>1_rig/<algo>_<rig min>min.zip, so rig_learn's retest phase
tests it on the rig (--plan retest --retest-runs fit_s2c30_sac1,... --retest-mm 12 / 5).

  python train_fitted.py s2c_odil1 0 sac 300000
  python train_fitted.py s2c_odil1 2 ppo 5000000
"""
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)                          # runs next to the rig loop (niced): one core at most
from stable_baselines3 import SAC, PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from plate_goal_env import PlateGoalEnv  # noqa: E402
from plate_sim import PlateSim  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "phase3_logs", "rig_learn.jsonl")


def odil_fit(run, rnd):
    for line in open(LOG):
        r = json.loads(line)
        if r.get("event") == "odil fitted+trained" and r.get("run") == run and r.get("round") == rnd:
            return r
    raise SystemExit(f"no ODIL fit for {run} round {rnd}")


class FittedEnv(PlateGoalEnv):
    def __init__(self, fit, seed):
        super().__init__(randomize=True, seed=seed)
        self.fit = fit

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        f, u = self.fit, self.rng.uniform
        taus = f["tau_s"]
        p = dict(k_acc=u(0.9 * f["k_acc"], 1.1 * f["k_acc"]), a_roll=f["a_roll"],
                 servo_tau=u(max(0.01, 0.7 * min(taus)), max(0.02, 1.3 * max(taus))),
                 static_rest_deg=u(0.7 * f["breakaway_deg"], 1.3 * f["breakaway_deg"]))
        self.sim = PlateSim(self.rng, params=p, randomize=True)
        self.sim.reset(self.pos.copy())
        return obs, info


def main():
    run, rnd, algo, steps = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(float(sys.argv[4]))
    fit = odil_fit(run, rnd)
    mins = int(round(fit["rig_minutes"]))
    name = f"fit_{run.split('_odil')[0]}{mins}_{algo}1_rig"
    out = os.path.join(HERE, "runs", name)
    if os.path.isdir(out) and any(n.endswith(".zip") for n in os.listdir(out)):
        raise SystemExit(f"{out} already holds a model -- not overwriting")
    os.makedirs(out, exist_ok=True)
    seed = 4242 + rnd
    if algo == "sac":
        env = FittedEnv(fit, seed)
        model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=1_000_000, batch_size=256,
                    gamma=0.99, tau=0.005, learning_starts=5_000, train_freq=(1, "episode"), gradient_steps=-1,
                    policy_kwargs=dict(net_arch=[256, 256]), verbose=0, device="cpu", seed=seed)
    else:
        env = DummyVecEnv([lambda i=i: FittedEnv(fit, seed + 100 * i) for i in range(4)])
        model = PPO("MlpPolicy", env, n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4,
                    gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    policy_kwargs=dict(net_arch=[256, 256]), verbose=0, device="cpu", seed=seed)
    json.dump({"odil_fit": fit, "algo": algo, "steps": steps, "seed": seed, "started": time.time()},
              open(os.path.join(out, "fitted_meta.json"), "w"), indent=1)
    t0 = time.time()
    model.learn(total_timesteps=steps)
    model.save(os.path.join(out, f"{algo}_{mins}min"))
    meta = json.load(open(os.path.join(out, "fitted_meta.json")))
    meta["train_minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(meta, open(os.path.join(out, "fitted_meta.json"), "w"), indent=1)
    print("saved", out, meta["train_minutes"], "min", flush=True)


if __name__ == "__main__":
    main()
