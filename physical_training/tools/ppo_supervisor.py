#!/usr/bin/env python3
"""
Keep a PPO run going across the session's 3000-tick total limit (user-approved 2026-10-04).

PPO's hard tilting makes motor 3's level position creep (measured play ~0.4-0.85 deg, REPORT 13a);
the camera re-levels absorb it, but a session stops when the level position is > 3000 ticks from that
session's start. ONLY for that stop: keep the recording, free the ball / level on the camera
(tools/level_plate.py), and resume PPO from its latest state (saved after every batch). Any other
stop (error, failed re-level, hard watchdog) ends the supervisor -- nothing is retried.

  ../.venv-rl/bin/python3 tools/ppo_supervisor.py [first_resume_checkpoint first_resume_min]
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PT = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(PT, ".."))
PY = os.path.join(ROOT, ".venv-rl", "bin", "python3")
LOG = os.path.join(ROOT, "phase3_logs", "rig_learn.jsonl")
RUNS = os.path.join(ROOT, "rl_sim", "runs", "ppo1_rig")
DATA = os.path.join(PT, "data")
MAX_RESUMES = 25


def log(rec):
    rec = {"t": time.time(), **rec}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def last_shutdown(since):
    why = None
    for line in open(LOG):
        r = json.loads(line)
        if r["t"] >= since and r.get("event") in ("shutdown", "done"):
            why = r.get("why", "done") if r.get("event") == "shutdown" else "done"
    return why


def main():
    resume = sys.argv[1:3] if len(sys.argv) >= 3 else None
    for n in range(MAX_RESUMES + 1):
        t0 = time.time()
        cmd = [PY, "-u", "rig_learn.py", "--plan", "ppo,retest", "--ppo-hours", "30", "--retest-mm", "5"]
        if resume:
            cmd += ["--resume", resume[0], str(resume[1])]
        with open(os.path.join(DATA, "session_ppo.log"), "a") as out:
            subprocess.run(cmd, cwd=PT, stdout=out, stderr=subprocess.STDOUT)
        why = last_shutdown(t0) or "unknown"
        if "3000 ticks from the session start" not in why:
            log({"event": "supervisor end", "why": why, "resumes": n})
            return
        part = os.path.join(DATA, f"ppo1_part{n + 2}_{int(time.time())}.csv")
        if os.path.exists(os.path.join(DATA, "ppo1.csv")):
            os.replace(os.path.join(DATA, "ppo1.csv"), part)
        lv = subprocess.run([PY, "-u", os.path.join(HERE, "level_plate.py")], cwd=PT, capture_output=True, text=True)
        res = [l for l in lv.stdout.splitlines() if l.startswith(("RESULT", "STOP", "STILL"))]
        if lv.returncode != 0:
            log({"event": "supervisor end", "why": f"levelling failed (exit {lv.returncode}) {res}", "resumes": n})
            return
        latest = json.load(open(os.path.join(RUNS, "ppo_latest.json")))
        resume = (os.path.join(RUNS, "ppo_latest.zip"), round(latest["rig_minutes"], 2))
        log({"event": "auto-resume", "n": n + 1, "after": why, "level": res, "recording": os.path.basename(part),
             "resume_rig_minutes": resume[1]})
    log({"event": "supervisor end", "why": f"{MAX_RESUMES} resumes reached"})


if __name__ == "__main__":
    main()
