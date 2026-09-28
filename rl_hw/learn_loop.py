#!/usr/bin/env python3
"""
Learning loop on the rig (2026-09-28): drive -> learn the world model -> practise -> drive.

Each round:
  1. the rig drives the maze route with the current ODIL tracking policy (no per-route ILC),
     RUNS_PER_ROUND runs (fall/stall -> back to the start; on the real maze the elevator reloads);
  2. the rig pauses (plate level) -- training load must not slow the camera;
  3. the world model (rl_sim/world_model.py: physics + learned correction) is trained on the
     logs of the recent rounds, warm-started from the previous round;
  4. the policy practises the route in that world model (rl_sim/finetune_track.py with
     FT_WORLD, holes/walls penalised), starting from the current policy;
  5. the new policy drives the next round.
Round results (runs, mean/best progress, finishes) go to phase3_logs/learn_loop.jsonl -- the
learning curve. Stop: touch phase3_logs/STOP_LOOP.

  python3 learn_loop.py [rounds] [--real]        (--real: on the maze board)
"""
import glob
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HW, SIM = os.path.join(ROOT, "rl_hw"), os.path.join(ROOT, "rl_sim")
RUNS_LOG = os.path.join(ROOT, "phase3_logs", "maze_runs.jsonl")
LOOP_LOG = os.path.join(ROOT, "phase3_logs", "learn_loop.jsonl")
STOP = os.path.join(ROOT, "phase3_logs", "STOP_LOOP")
CTRL_LOG = os.environ.get("LOOP_CTRL_LOG", "/tmp/cyberrunner_ctrl.log")
RUNS_PER_ROUND = int(os.environ.get("LOOP_RUNS", "30"))
MAX_ROUND_S = float(os.environ.get("LOOP_MAX_ROUND_S", "1500"))
URL = "http://localhost:8000"


def cmd(**c):
    req = urllib.request.Request(URL + "/cmd", data=json.dumps(c).encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=3).read()


def state():
    with urllib.request.urlopen(URL + "/state", timeout=3) as r:
        return json.load(r)


def controller_pids():
    out = subprocess.run(["pgrep", "-f", "pd_balance.py"], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out]


def stop_controller():
    for p in controller_pids():
        os.kill(p, signal.SIGTERM)
    for _ in range(60):
        if not controller_pids():
            return
        time.sleep(0.5)


def start_controller(policy, real):
    env = dict(os.environ, PD_UI="1", PD_SNAPSHOT_S="60", PD_MAZE_LEARN="0", PD_MAZE_CONTROLLERS="odil",
               PD_ODIL_TRACK=policy)
    if real:
        env["PD_MAZE_REAL"] = "1"
    subprocess.Popen([os.path.join(ROOT, ".venv", "bin", "python3"), "-u", "pd_balance.py", "2000000", "100000",
                      "0", "2000000"], cwd=HW, env=env, stdout=open(CTRL_LOG, "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        time.sleep(1)
        try:
            state()
            if "=== episode 1" in open(CTRL_LOG).read():
                return True
        except (OSError, ValueError):
            pass
    return False


def runs_since(t0):
    try:
        return [json.loads(l) for l in open(RUNS_LOG) if json.loads(l)["t"] > t0]
    except OSError:
        return []


def drive_round(policy, real):
    stop_controller()
    if not start_controller(policy, real):
        raise RuntimeError("controller did not start")
    time.sleep(2)
    cmd(cmd="maze_start", alternate=True, controller="odil")
    t0 = time.time()
    while True:
        time.sleep(10)
        R = runs_since(t0)
        if len(R) >= RUNS_PER_ROUND or time.time() - t0 > MAX_ROUND_S or os.path.exists(STOP):
            break
    cmd(cmd="maze_stop")
    cmd(cmd="stop")
    newest_log = max(glob.glob(os.path.join(HW, "pd_logs", "pd_*.csv")), key=os.path.getmtime)
    return runs_since(t0), newest_log


def train(round_no, policy, logs):
    py = os.path.join(ROOT, ".venv-rl", "bin", "python3")
    wm = os.path.join(SIM, "runs", "world_model_loop.pt")
    env = dict(os.environ, WM_ITERS="400", WM_MAX_ROWS="150000", WM_THREADS="6")
    r1 = subprocess.run([py, "world_model.py", wm] + logs[-3:], cwd=SIM, env=env, capture_output=True, text=True)
    wm_msg = [l for l in r1.stdout.splitlines() if l.startswith(("before", "after", "physics"))]
    out = f"odil_track_loop_r{round_no}"
    env = dict(os.environ, FT_WORLD=wm, FT_ROUTE=os.path.join(ROOT, "maze", "route.json"), FT_THREADS="6",
               WM_DELAY=os.environ.get("WM_DELAY", "3"))
    r2 = subprocess.run([py, "finetune_track.py", policy, out, os.environ.get("LOOP_FT_ITERS", "600")], cwd=SIM,
                        env=env, capture_output=True, text=True)
    ft_last = [l for l in r2.stdout.splitlines() if l.startswith("{")][-1:] or [r2.stderr[-300:]]
    new_policy = os.path.join(SIM, "runs", out, "odil_track_policy.npz")
    return (new_policy if os.path.exists(new_policy) else policy), wm_msg, ft_last


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    real = "--real" in sys.argv
    policy = os.environ.get("LOOP_POLICY", os.path.join(SIM, "runs", "odil_track_maze", "odil_track_policy.npz"))
    logs = []
    for rnd in range(rounds):
        if os.path.exists(STOP):
            print("STOP file -- ending the loop")
            break
        R, log = drive_round(policy, real)
        logs.append(log)
        p = [r["progress"] for r in R]
        rec = {"t": time.time(), "round": rnd, "policy": policy, "real": real, "runs": len(R),
               "mean_progress": sum(p) / len(p) if p else None, "best_progress": max(p) if p else None,
               "finished": sum(r["result"] == "finished" for r in R), "log": log}
        print(json.dumps(rec), flush=True)
        policy, wm_msg, ft_last = train(rnd, policy, logs)
        rec.update(world_model=wm_msg, refine=ft_last, next_policy=policy)
        with open(LOOP_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("  world model:", wm_msg, "\n  refine:", ft_last, flush=True)
    stop_controller()
    start_controller(policy, real)       # leave the rig idle-ready with the latest policy
    print("loop done; latest policy", policy)


if __name__ == "__main__":
    main()
