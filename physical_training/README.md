# Part 3 -- ball on plate, trained only on the physical rig

**Question:** how much real rig time do ODIL and RL need to learn ball-on-plate when nothing comes
from a hand-made simulator?

- **ODIL:** drive the rig (round 0: random tilts, no prior controller) -> fit the physics to the
  recordings (`rl_sim/sysid_rig.py`) -> train ODIL in that fitted model (`rl_sim/odil_plate_v9.py`,
  the v6/v11 recipe) -> test; the next round drives with the new policy. Default 3 rounds x 30 min.
- **RL:** Soft Actor-Critic from scratch on the rig, with the same observation, action limits and
  reward as the simulated RL (`rl_sim/plate_goal_env.py`). Network updates happen between episodes
  with the plate level, so the 29 Hz loop is never slowed.
- **Test:** the same 30 fixed random targets for both (12 mm, away from the hole): reached within
  8 s, time to reach, share of the next 4 s inside, final distance, jerk. Every result is logged with
  the rig minutes of training data -> `phase3_logs/rig_learn.jsonl`.

Expected from earlier measurements (to be confirmed by this experiment): ODIL about 30 min of rig
data per round; RL (SAC) about 2-5 h; PPO about 10-30 h (not run: too long for a night).

## Running it

Needs: the white paper plate on the rig, the plate levelled by hand, no `pd_balance.py` running.
Uses `.venv-rl` (PyTorch + stable-baselines3; OpenCV 4.10 and the Dynamixel SDK were added to it).

```
cd physical_training
../.venv-rl/bin/python3 rig_learn.py --dry          # ~10 min supervised check: every phase briefly
../.venv-rl/bin/python3 rig_learn.py                # the night: ODIL 3 x 30 min, then SAC 6 h
```

Safety: motor caps fixed around the start position; the script aborts and releases the motors if
a tilt motor leaves +-1300 ticks of its start; a frozen camera holds the plate level; a ball that
is not visible -> plate level until it is seen again (no elevator on the plate rig).
