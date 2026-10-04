"""level the plate on the camera, starting from where the motors ARE (never a stored position).
A ball resting on a plate marker blocks the camera pose (marker guard): it is first rolled off with an
open-loop away-tilt ramping to 5 deg (+ rocking). Stops with torque ON if a motor moves > 900 ticks.
Exit 0 = level (|alpha|, |beta| < 0.3 deg); 2 = ball still stuck; 3 = not level. Torque stays on.

  ../.venv-rl/bin/python3 tools/level_plate.py
"""
import os, sys, time
import numpy as np
os.environ["PLATE_CAP_TICKS"] = "1000"; os.environ["PLATE_MARKER_GUARD_M"] = "0.045"
ROOT = os.path.expanduser("~/cyberrunner_basics_no_ros")
sys.path[:0] = [os.path.join(ROOT, "rl_hw"), os.path.join(ROOT, "state_est")]
os.chdir(os.path.join(ROOT, "rl_hw"))
from plate_env import HardwarePlateEnv, LEVEL_OFFSET_DEG, LEVEL_TOL_DEG
env = HardwarePlateEnv(fixed_goal=(0.0, 0.0), max_action_delta=0.5, max_episode_steps=10 ** 9, allow_unstick=False)
start = dict(env._cmd_ticks); print("start", start)
zero = np.zeros(2, np.float32)
def guard():
    if any(abs(env._cmd_ticks[k] - start[k]) > 900 for k in start):
        print("STOP: motor moved > 900 ticks", dict(env._cmd_ticks)); os._exit(1)
def at_marker():
    xb, yb, _, _, f = env._read_state()
    return (f and getattr(env, "ball_at_marker", False)), (xb, yb)
stuck, p = at_marker()
t0 = time.time()
while stuck and time.time() - t0 < 40:
    away = -np.sign(p)
    # ramp 1.4 -> 5 deg over 2 s, hold 1 s
    t1 = time.time()
    while time.time() - t1 < 3.0:
        mag = min(1.0, 0.35 + 0.33 * (time.time() - t1))
        env._write_action((away * mag).astype(np.float32)); time.sleep(env.dt); guard()
        stuck, q = at_marker()
        if not stuck: break
    if stuck:   # rock: alternate the two axes around the away tilt
        for k in range(6):
            w = away * 0.9 + (np.array([0.4, -0.4]) if k % 2 else np.array([-0.4, 0.4]))
            t2 = time.time()
            while time.time() - t2 < 0.25:
                env._write_action(np.clip(w, -1, 1).astype(np.float32)); time.sleep(env.dt); guard()
            stuck, q = at_marker()
            if not stuck: break
    print(f"{time.time() - t0:4.1f}s ball {np.round(1000 * np.array(q))} still at marker: {stuck}")
    # back to the commanded level (open loop) before the next try / the levelling
    t1 = time.time()
    while time.time() - t1 < 1.0:
        env._write_action(zero); time.sleep(env.dt); env._read_state()
    stuck, p = at_marker()
if stuck:
    print("STILL STUCK"); os._exit(2)
env._read_state(); t0 = time.time(); ok = 0
while time.time() - t0 < 25:
    env._write_action(zero); time.sleep(env.dt); env._read_state()
    if any(abs(env._cmd_ticks[k] - start[k]) > 900 for k in start):
        print("STOP: motor moved > 900 ticks", dict(env._cmd_ticks)); os._exit(1)
    a, b = env._meas_tilt if env._meas_tilt is not None else (99, 99)
    da, db = a - LEVEL_OFFSET_DEG[0], b - LEVEL_OFFSET_DEG[1]
    ok = ok + 1 if abs(da) < 0.3 and abs(db) < 0.3 and env._pose_ok else 0
    if int((time.time() - t0) * 29) % 29 == 0:
        print(f"{time.time() - t0:4.1f}s alpha {da:+.2f} beta {db:+.2f} motors {env._cmd_ticks}")
    if ok >= 15: break
A, B = [], []
for _ in range(58):
    env._write_action(zero); time.sleep(env.dt); env._read_state()
    if env._pose_ok and env._meas_tilt is not None:
        A.append(env._meas_tilt[0] - LEVEL_OFFSET_DEG[0]); B.append(env._meas_tilt[1] - LEVEL_OFFSET_DEG[1])
print(f"RESULT: alpha {np.median(A):+.2f} beta {np.median(B):+.2f} deg over 2 s, motors {dict(env._cmd_ticks)}")
os._exit(0 if (abs(np.median(A)) < 0.3 and abs(np.median(B)) < 0.3) else 3)
