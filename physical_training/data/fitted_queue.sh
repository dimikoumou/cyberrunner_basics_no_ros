#!/bin/zsh
# model-based baseline (2026-10-05): SAC/PPO trained in ODIL's fitted physics, lowest CPU priority next to the rig
cd ~/cyberrunner_basics_no_ros/rl_sim
for job in "s2c_odil1 0 sac 500000" "s2c_odil1 2 sac 500000" "s2c_odil1 0 ppo 5000000" "s2c_odil1 2 ppo 5000000"; do
  echo "$(date '+%a %H:%M') start $job"
  nice -n 20 ../.venv-rl/bin/python3 -u train_fitted.py ${=job} 2>&1 | grep -v Warn
done
echo "$(date '+%a %H:%M') all fitted baselines done"
