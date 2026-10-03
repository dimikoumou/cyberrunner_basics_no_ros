#!/usr/bin/env python3
"""
Reuse test (2026-10-03): ODIL learns the plate's physics once -- can the SAME rig data train a
controller for a NEW task with no further rig data?

From the ODIL rounds' rig recordings only (physical_training/data/<run>_r*.csv):
  1. fit the physics (rl_sim/sysid_rig.py)
  2. train an ODIL path-tracking policy in that fitted model (rl_sim/odil_track.py, 3-stage delay)
  3. refine it in closed loop in the same fitted model (rl_sim/finetune_track.py)
-> rl_sim/runs/reuse_<run>_track/odil_track_policy.npz, then test it on the rig drawing shapes
(rl_hw/pd_balance.py draw mode, PD_ODIL_TRACK=<policy>) -- no rig minutes spent on this task.

Run it when the rig is NOT driving (training load slows the camera loop).

  ../.venv-rl/bin/python3 reuse_tracker.py [run=odil1]
"""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SIM = os.path.join(ROOT, "rl_sim")
PY = os.path.join(ROOT, ".venv-rl", "bin", "python3")


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "odil1"
    files = sorted(glob.glob(os.path.join(HERE, "data", f"{run}_r*.csv")))
    if not files:
        raise SystemExit(f"no rig data for {run}")
    t0 = time.time()
    subprocess.run([PY, "sysid_rig.py", *files], cwd=SIM, check=True, capture_output=True)
    fit = json.load(open(os.path.join(SIM, "rig_sysid.json")))
    k, ar = float(fit["k_acc"]), float(fit["a_roll"])
    taus = [fit[f"servo_{a}"]["tau_s"] for a in ("x", "y")]
    delays = [fit[f"servo_{a}"]["delay_steps"] for a in ("x", "y")]
    print(f"fitted from {len(files)} rig files: k {k:.3f}, a_roll {ar:.3f}, tau {taus}, delay {delays}")
    base = f"reuse_{run}_odiltrack"
    env = dict(os.environ, ODIL_THREADS="6", ODIL_N_STAGES="3",
               ODIL_TAU_RANGE=f"{max(0.01, 0.7 * min(taus)):.3f},{max(0.02, 1.3 * max(taus)):.3f}")
    subprocess.run([PY, "odil_track.py", "1500", base], cwd=SIM, env=env, check=True)
    env = dict(os.environ, FT_THREADS="6", FT_K_RANGE=f"{0.9 * k:.4f},{1.1 * k:.4f}", FT_A_ROLL=f"{ar:.4f}",
               FT_TAU_RANGE=f"{max(0.01, 0.7 * min(taus)):.3f},{max(0.02, 1.3 * max(taus)):.3f}",
               FT_DELAY_RANGE=f"{max(1, min(delays) - 1)},{max(delays)}")
    out = f"reuse_{run}_track"
    subprocess.run([PY, "finetune_track.py", os.path.join(SIM, "runs", base, "odil_track_policy.npz"), out, "600"],
                   cwd=SIM, env=env, check=True)
    pol = os.path.join(SIM, "runs", out, "odil_track_policy.npz")
    rec = {"t": time.time(), "event": "reuse tracker trained", "run": run, "rig_files": len(files),
           "k_acc": k, "a_roll": ar, "tau_s": taus, "delay_steps": delays,
           "compute_minutes": round((time.time() - t0) / 60, 1), "policy": pol, "ok": os.path.exists(pol)}
    with open(os.path.join(ROOT, "phase3_logs", "rig_learn.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
