#!/usr/bin/env python3
"""
Screen recordings of good maze runs (2026-09-28): keeps the last few minutes of the live
view (the controller UI's /frame.jpg, with route, holes and target drawn in) in memory,
and whenever phase3_logs/maze_runs.jsonl gets a run with progress >= MIN_PROGRESS, writes
that run (plus a second before and after) to maze/recordings/*.mp4.
Runs in its own process, so it never slows the control loop.

  python3 run_recorder.py [min_progress=0.7]
"""
import collections
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_LOG = os.path.join(ROOT, "phase3_logs", "maze_runs.jsonl")
OUT_DIR = os.path.join(ROOT, "maze", "recordings")
MIN_PROGRESS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
FPS = 12
KEEP_S = 300
URL = "http://127.0.0.1:8000/frame.jpg"


def runs_count():
    try:
        with open(RUNS_LOG) as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def new_runs(n_seen):
    try:
        with open(RUNS_LOG) as f:
            lines = f.readlines()
    except OSError:
        return n_seen, []
    return len(lines), [json.loads(l) for l in lines[n_seen:] if l.strip()]


def save(frames, run):
    t_end = run["t"]
    t_start = t_end - run.get("duration_s", 60) - 1.0
    clip = [(t, j) for t, j in frames if t_start <= t <= t_end + 1.0]
    if len(clip) < 5:
        print("too few frames for run", run.get("run"), flush=True)
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(t_end))
    path = os.path.join(OUT_DIR, f"run_{stamp}_{run.get('controller', '')}_{100 * run['progress']:.0f}pct.mp4")
    first = cv2.imdecode(np.frombuffer(clip[0][1], np.uint8), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), FPS, (w, h))
    if not vw.isOpened():
        path = path[:-4] + ".avi"
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
    # resample to a steady frame rate (repeat the latest frame) so playback is real time
    k = 0
    for i in range(int((clip[-1][0] - clip[0][0]) * FPS) + 1):
        t = clip[0][0] + i / FPS
        while k + 1 < len(clip) and clip[k + 1][0] <= t:
            k += 1
        img = cv2.imdecode(np.frombuffer(clip[k][1], np.uint8), cv2.IMREAD_COLOR)
        if img is not None and img.shape[:2] == (h, w):
            vw.write(img)
    vw.release()
    with open(os.path.join(OUT_DIR, "index.jsonl"), "a") as f:
        f.write(json.dumps({"file": os.path.basename(path), **run}) + "\n")
    print(f"saved {path} ({100 * run['progress']:.0f} %, {run.get('result')})", flush=True)


def main():
    frames = collections.deque()
    n_seen = runs_count()
    last, t_check = None, 0.0
    print(f"recording runs >= {100 * MIN_PROGRESS:.0f} % to {OUT_DIR}", flush=True)
    while True:
        t0 = time.time()
        try:
            with urllib.request.urlopen(URL, timeout=2) as r:
                jpg = r.read()
            if jpg != last:
                frames.append((t0, jpg))
                last = jpg
        except OSError:
            pass
        while frames and frames[0][0] < t0 - KEEP_S:
            frames.popleft()
        if t0 - t_check > 1.0:
            t_check = t0
            n_seen, runs = new_runs(n_seen)
            for run in runs:
                if run.get("progress", 0) >= MIN_PROGRESS:
                    time.sleep(1.0)                # the second after the run
                    save(list(frames), run)
        time.sleep(max(0.0, 1 / FPS - (time.time() - t0)))


if __name__ == "__main__":
    main()
