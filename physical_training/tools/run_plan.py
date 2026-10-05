#!/usr/bin/env python3
"""
Run the rest of the plan unattended (user-approved 2026-10-04: "set up in a way where you have minimal
things to do"). Steps run in order; each step is rig_learn.py with its arguments.

Recoverable stops (the camera re-level failed, the 3000-tick total limit) -> keep the recording, level on
the camera in a FRESH process (tools/level_plate.py: starts where the motors are, torque stays on), resume
the run from its last saved state (SAC: last test checkpoint + its replay memory; PPO: latest batch) --
logged as "auto-resume", at most MAX_RECOVER per step. Anything else (motors released by the watchdog,
an error, a failed levelling) ends the plan; the plate is held.

  ../.venv-rl/bin/python3 tools/run_plan.py > data/run_plan.log 2>&1
"""
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PT = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PT, ".."))
PY = os.path.join(ROOT, ".venv-rl", "bin", "python3")
LOG = os.path.join(ROOT, "phase3_logs", "rig_learn.jsonl")
RUNS = os.path.join(ROOT, "rl_sim", "runs")
DATA = os.path.join(PT, "data")
MAX_RECOVER = 10
RECOVERABLE = ("relevel failed", "3000 ticks from the session start")

# step = (name, base args, the RL run it may resume, algo)
STEPS = [
    ("session2c", ["--plan", "sac,odil", "--sac-hours", "10", "--tag", "s2c"], "s2c_sac1", "sac"),
    # 2026-10-05 (user: "after odil it runs like that in that order"): 5 mm precision retest of session 2
    # while the rig is as it was in training; no checkpoint to resume -> a recoverable stop reruns it
    ("retest_s2c", ["--plan", "retest", "--retest-mm", "5", "--tag", "s2c",
                    "--retest-runs", "s2c_sac1,s2c_odil1", "--retest-min", "60,120,300,600"], None, None),
    ("ppo_extension", ["--plan", "ppo", "--ppo-hours", "24"], "ppo1", "ppo"),
]


def log(rec):
    rec = {"t": time.time(), **rec}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def events(since):
    out = []
    for line in open(LOG):
        r = json.loads(line)
        if r["t"] >= since:
            out.append(r)
    return out


def latest_checkpoint(run, algo):
    """SAC: the last test checkpoint (its replay memory is saved with it); PPO: the latest batch."""
    d = os.path.join(RUNS, f"{run}_rig")
    if algo == "ppo" and os.path.exists(os.path.join(d, "ppo_latest.json")):
        m = json.load(open(os.path.join(d, "ppo_latest.json")))["rig_minutes"]
        return os.path.join(d, "ppo_latest.zip"), round(float(m), 2)
    best = None
    for p in glob.glob(os.path.join(d, f"{algo}_*min.zip")):
        mm = re.search(r"_(\d+)min\.zip$", p)
        if mm and (best is None or int(mm.group(1)) > best[1]):
            best = (p, int(mm.group(1)))
    return best


def running():
    return subprocess.run(["pgrep", "-f", "rig_learn.py"], capture_output=True).stdout.strip() != b""


def last_start():
    t = 0.0
    for line in open(LOG):
        r = json.loads(line)
        if r.get("event") == "start":
            t = r["t"]
    return t


def run_step(name, args, run, algo, first_attempt_running=False, resume=None):
    n_rec, odil_retry = 0, 0
    while True:
        t0 = time.time()
        if first_attempt_running:                       # a session already running when the plan started
            t0 = last_start()
            while running():
                time.sleep(20)
            first_attempt_running = False
        else:
            cmd = [PY, "-u", "rig_learn.py"] + args + (["--resume", resume[0], str(resume[1])] if resume else [])
            log({"event": "plan step", "step": name, "cmd": " ".join(cmd[2:])})
            with open(os.path.join(DATA, f"plan_{name}.log"), "a") as out:
                subprocess.run(cmd, cwd=PT, stdout=out, stderr=subprocess.STDOUT)
        ev = events(t0)
        stops = [r for r in ev if r.get("event") in ("shutdown",)]
        why = stops[-1].get("why", "") if stops else "no shutdown logged"
        released = bool(stops[-1].get("released")) if stops else False
        if why == "finished":
            log({"event": "plan step done", "step": name})
            return True
        if released or not any(k in why for k in RECOVERABLE) or n_rec >= MAX_RECOVER:
            log({"event": "plan stopped", "step": name, "why": why, "released": released, "recoveries": n_rec})
            return False
        n_rec += 1
        lv = subprocess.run([PY, "-u", os.path.join(HERE, "level_plate.py")], cwd=PT, capture_output=True, text=True)
        if lv.returncode != 0:                           # a second try from where the first one stopped
            lv = subprocess.run([PY, "-u", os.path.join(HERE, "level_plate.py")], cwd=PT, capture_output=True, text=True)
        res = [l for l in lv.stdout.splitlines() if l.startswith(("RESULT", "STOP", "STILL"))]
        if lv.returncode != 0:
            log({"event": "plan stopped", "step": name, "why": f"levelling failed {res}", "recoveries": n_rec})
            return False
        in_odil = any(r.get("event") == "phase" and "odil" in str(r.get("run")) for r in ev)
        if run is None:                                  # a retest: run it again (recordings are never overwritten)
            resume = None
        elif in_odil:                                    # ODIL rounds cannot resume: redo ODIL under a new tag
            odil_retry += 1
            args = ["--plan", "odil", "--tag", f"s2c_odilretry{odil_retry}"]
            resume = None
        else:
            resume = latest_checkpoint(run, algo)
            if resume is None:
                log({"event": "plan stopped", "step": name, "why": f"no checkpoint for {run}", "recoveries": n_rec})
                return False
        log({"event": "auto-resume", "step": name, "n": n_rec, "after": why, "level": res,
             "resume": None if resume is None else {"checkpoint": os.path.basename(resume[0]), "rig_minutes": resume[1]},
             "args": args})


def main():
    already = running()
    for i, (name, args, run, algo) in enumerate(STEPS):
        if i > 0:
            lv = subprocess.run([PY, "-u", os.path.join(HERE, "level_plate.py")], cwd=PT, capture_output=True, text=True)
            if lv.returncode != 0:
                log({"event": "plan stopped", "step": name, "why": "levelling before the step failed"})
                return
        first = (os.path.join(RUNS, "ppo1_rig", "ppo_600min.zip"), 600) if name == "ppo_extension" else None
        ok = run_step(name, args, run, algo, first_attempt_running=(i == 0 and already), resume=first)
        if not ok:
            return
    log({"event": "plan finished"})


if __name__ == "__main__":
    main()
