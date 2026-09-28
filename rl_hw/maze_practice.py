"""
Maze practice on white paper (2026-09-27): the ball follows the real maze's route
(maze/route.json from maze_route.py) with VIRTUAL holes and walls.

  - a run starts at the route's start; the ball "falls" if its centre enters a hole's radius,
    and "hits a wall" if its centre is more than WALL_TOL inside a wall outline -> the run ends
    and the next one starts again from the beginning (user: like the real maze);
  - iterative learning control: after every run the lateral error along the route (per route
    point, where the ball got to) is added to that controller's reference offset, shifted a
    little earlier to account for the loop delay, so repeated runs follow the route more tightly;
  - every run is logged to phase3_logs/maze_runs.jsonl; the live view shows route, holes and walls.
"""
import json
import os
import time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROUTE_PATH = os.path.join(ROOT, "maze", "route.json")
LOG_PATH = os.path.join(ROOT, "phase3_logs", "maze_runs.jsonl")
WALL_TOL = 0.0015            # m: inside a wall outline by more than this = crossed it
ILC_GAIN = 0.2              # 0.5 overcorrected: ODIL went from 21-65 % to failing at 4 % every run
ILC_LEAD_S = 0.15            # the loop reacts ~0.15 s late: correct that much earlier
ILC_MAX = 0.008              # m, never shift the reference further than this (15 mm let it drift into walls)
LOS_TOL_M = float(os.environ.get("PD_MAZE_LOS_MM", "3")) / 1000   # target stays in sight (no corner cutting)
MAZE_SPEED = float(os.environ.get("PD_MAZE_SPEED", "0.025"))
STALL_S = 15.0
LEARN = os.environ.get("PD_MAZE_LEARN", "1") == "1"   # 0: no ILC / slow zones (diagnostic baseline)
# the real maze board: holes and walls are physical -- a fall is the ball disappearing (the
# controller then reloads it), walls may be leaned on; only the stall timeout is checked here
REAL = os.environ.get("PD_MAZE_REAL") == "1"               # no progress along the route for this long -> the run counts as stuck
CONTROLLERS = tuple(os.environ.get("PD_MAZE_CONTROLLERS", "odil,blend").split(","))   # blend = mean of ODIL and classic (classic alone dropped: stuck at ~8 %)


class MazePractice:
    def __init__(self, env, plate_to_pixel):
        r = json.load(open(ROUTE_PATH))
        self.route = np.array(r["route_m"], dtype=float)
        self.holes = [(np.array(h["center"]), float(h["radius"])) for h in r["holes"]]
        self.walls = [(np.array(w, dtype=float) * 1e4).astype(np.float32) for w in r.get("walls_m", [])]
        self.L = float(r["length_m"])
        self.n = len(self.route)
        self.ilc = {}                                  # controller -> (n, 2) reference offset
        self.slow = {}                                 # controller -> (n,) speed factor (learned slow zones)
        self.fail_at = {}                              # controller -> list of route indices where runs failed
        for ctl in ("classic", "odil", "blend"):
            fn = os.path.join(ROOT, "maze", f"ilc_{ctl}.npy")
            self.ilc[ctl] = np.load(fn) if os.path.exists(fn) else np.zeros((self.n, 2))
            fs = os.path.join(ROOT, "maze", f"slow_{ctl}.npy")
            self.slow[ctl] = np.load(fs) if os.path.exists(fs) else np.ones(self.n)
            self.fail_at[ctl] = []
        self.run_no, self.best = 0, {c: 0.0 for c in ("classic", "odil", "blend")}
        self.last = ""
        self.fail_counts = {}
        self._pixels(env, plate_to_pixel)
        self.active = False

    def _pixels(self, env, p2p):
        """route / holes / walls in image pixels (level plate) for the live view"""
        def px(q, guess=None):
            rc = p2p(env, q, guess=guess) if guess is not None else p2p(env, q)
            return None if rc is None else (float(rc[1]), float(rc[0]))
        pts, g = [], (180.0, 320.0)
        for q in self.route[::4]:
            rc = p2p(env, q, guess=g)
            if rc is not None:
                g = rc
                pts.append((float(rc[1]), float(rc[0])))
        self.route_px = np.array(pts)
        self.route_px_idx = np.arange(0, self.n, 4)[:len(pts)]
        self.holes_px = []
        for c, r in self.holes:
            a, b = px(c), px((c[0] + r, c[1]))
            if a is not None and b is not None:
                self.holes_px.append((a[0], a[1], float(np.hypot(b[0] - a[0], b[1] - a[1]))))
        self.walls_px = []
        for w in self.walls:
            poly = [px(q / 1e4) for q in w]
            poly = [q for q in poly if q is not None]
            if len(poly) >= 3:
                self.walls_px.append(np.array(poly))

    # ------------------------------------------------------------------ a run
    def start_run(self, ctl, tracker_cls, ball_xy):
        self.ctl = ctl
        self.run_no += 1
        self.t0 = time.time()
        self.samples = []                              # (route index, ball xy)
        self.max_idx = 0
        self.t_progress = None
        self.follower = tracker_cls(self.route, False, np.asarray(ball_xy), keep_direction=True, v=MAZE_SPEED,
                                    ref_offset=self.ilc[ctl] if LEARN else None,
                                    speed_scale=self.slow[ctl] if LEARN else None,
                                    los_tol=LOS_TOL_M)
        self.active = True
        return self.follower

    def step(self, ball_xy, following):
        """call every frame the ball is seen; -> None or a failure string"""
        if not following:
            return None                                # joining the route: no checks yet
        b = np.asarray(ball_xy, dtype=float)
        idx = int(np.searchsorted(self.follower.S, self.follower._project(b, self.follower.s)))
        idx = min(idx, self.n - 1)
        self.samples.append((idx, b.copy()))
        now = time.time()
        if self.t_progress is None or idx > self.max_idx + 2:
            self.t_progress = now
        self.max_idx = max(self.max_idx, idx)
        if now - self.t_progress > STALL_S:
            return "stuck (no progress for 15 s)"
        if REAL:
            return None
        for k, (c, r) in enumerate(self.holes):
            if np.hypot(*(b - c)) < r:
                return f"fell into hole {k}"
        bp = (float(b[0] * 1e4), float(b[1] * 1e4))
        for k, w in enumerate(self.walls):
            if cv2.pointPolygonTest(w, bp, True) / 1e4 > WALL_TOL:
                return f"went through wall {k}"
        return None

    def end_run(self, result, jerk=None):
        """log the run, update this controller's learning (ILC), return a status line"""
        self.active = False
        progress = self.max_idx / max(1, self.n - 1)
        if result == "finished":
            progress = 1.0
        self.best[self.ctl] = max(self.best[self.ctl], progress)
        acc = self.follower.accuracy() or {}
        # ILC: mean lateral error per route point over this run, lead-shifted and smoothed
        if LEARN and len(self.samples) > 20:
            err = np.zeros((self.n, 2))
            cnt = np.zeros(self.n)
            T = self.follower.T[:self.n]
            for idx, b in self.samples:
                e_ = self.route[idx] - b
                err[idx] += e_ - np.dot(e_, T[idx]) * T[idx]          # lateral part only
                cnt[idx] += 1
            seen = cnt > 0
            err[seen] /= cnt[seen, None]
            ds = self.L / max(1, self.n - 1)
            lead = int(round(ILC_LEAD_S * MAZE_SPEED / ds))
            e = np.zeros_like(err)
            e[:self.n - lead] = err[lead:]
            m = np.zeros(self.n)
            m[:self.n - lead] = seen[lead:]
            k = np.ones(9) / 9
            e_s = np.column_stack([np.convolve(e[:, 0], k, "same"), np.convolve(e[:, 1], k, "same")])
            self.ilc[self.ctl] = np.clip(self.ilc[self.ctl] + ILC_GAIN * e_s * m[:, None], -ILC_MAX, ILC_MAX)
            np.save(os.path.join(ROOT, "maze", f"ilc_{self.ctl}.npy"), self.ilc[self.ctl])
        if result != "finished":
            self.fail_counts[result] = self.fail_counts.get(result, 0) + 1
        # learned slow zones: 3 failures within +-3 % of the route at one spot -> slow the stretch
        # +-5 % around it (x0.7, down to x0.3 over repeats); e.g. the taped paper at ~65 %
        if LEARN and result.startswith(("fell into", "went through")) and self.max_idx > 0:
            fa = self.fail_at[self.ctl]
            fa.append(self.max_idx)
            w3, w5 = int(0.03 * self.n), int(0.05 * self.n)
            near = [i for i in fa if abs(i - self.max_idx) <= w3]
            if len(near) >= 3:
                c = int(np.median(near))
                lo_, hi_ = max(0, c - w5), min(self.n, c + w5)
                self.slow[self.ctl][lo_:hi_] = np.maximum(0.6, self.slow[self.ctl][lo_:hi_] * 0.85)   # min 0.3 made it crawl and stick-slip
                self.fail_at[self.ctl] = [i for i in fa if abs(i - c) > w3]
                np.save(os.path.join(ROOT, "maze", f"slow_{self.ctl}.npy"), self.slow[self.ctl])
                print(f"  maze: {self.ctl} slows down around {100 * c / self.n:.0f} % of the route "
                      f"(now x{self.slow[self.ctl][c]:.2f})")
        rec = {"t": time.time(), "run": self.run_no, "controller": self.ctl, "learn": LEARN, "real": REAL,
               "policy": os.environ.get("PD_ODIL_TRACK", "default"), "result": result,
               "progress": progress, "duration_s": time.time() - self.t0, "jerk": jerk, **acc}
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.last = f"run {self.run_no} ({self.ctl}): {result} at {progress * 100:.0f} %"
        return self.last

    def status(self):
        worst = sorted(self.fail_counts.items(), key=lambda kv: -kv[1])[:2]
        return (f"Maze practice | {self.last or 'starting'} | best: ODIL {self.best['odil'] * 100:.0f} %, "
                f"ODIL+classic {self.best['blend'] * 100:.0f} %"
                + (" | most fails: " + ", ".join(f"{k} x{v}" for k, v in worst) if worst else ""))

    def overlay(self, fell=None):
        prog = int(np.searchsorted(self.route_px_idx, self.max_idx)) if self.active or self.run_no else 0
        return {"route_px": self.route_px, "done": prog, "holes_px": self.holes_px, "walls_px": self.walls_px,
                "fell": fell}
