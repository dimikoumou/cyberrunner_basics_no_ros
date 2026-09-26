#!/usr/bin/env python3
"""Same probe as probe_m3_range.py, but for motor1(id 1). Leveling now pegs
m1 at 2824 (the current calibrated max) simultaneously with m3 pegging at 4050,
with alpha still +4.8deg residual -- check whether there's real, non-stalling
travel beyond 2824 the same way there wasn't (beyond ~4050-4059) for m3."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_env import HardwarePlateEnv  # noqa: E402
from state_est_control import ADDR_PRESENT_POSITION  # noqa: E402
import state_est_control  # noqa: E402
import numpy as np  # noqa: E402


def main():
    env = HardwarePlateEnv(max_episode_steps=1)
    try:
        m1_id, m3_id = sorted(env.calibration.keys())
        _, m3p, _ = env.calibration[m3_id]
        state_est_control.set_position(env.port_handler, env.packet_handler, m3_id, m3p)
        pos = 3200
        step = 30
        max_pos = 4200
        while pos <= max_pos:
            state_est_control.set_position(env.port_handler, env.packet_handler, m1_id, pos)
            time.sleep(1.3)
            present, _, _ = env.packet_handler.read4ByteTxRx(
                env.port_handler, m1_id, ADDR_PRESENT_POSITION
            )
            # Only a SHORTFALL (present < target for an increasing target) is a real
            # stall -- overshoot past target is healthy settling, not a problem
            # (this exact abs() vs one-sided distinction was a real bug in the
            # analogous m3 probe: it flagged normal overshoot as a false stall twice).
            stall_gap = max(0, pos - present)
            _, _, alpha, beta, found = env._read_state()
            alpha_deg = None if alpha is None else float(np.degrees(alpha))
            print(f"m1_target={pos:5d} present={present:5d} gap={stall_gap:3d} "
                  f"alpha={alpha_deg} ball_found={found}")
            if stall_gap > 15:
                print(f"*** STALL detected at m1_target={pos} (present only reached {present}) -- stopping")
                break
            if alpha_deg is not None and abs(alpha_deg) < 0.5:
                print(f"*** alpha crossed into tolerance at m1_target={pos} (alpha={alpha_deg:+.2f}deg)")
                break
            pos += step
    finally:
        env.close()


if __name__ == "__main__":
    main()
