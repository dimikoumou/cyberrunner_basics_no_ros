#!/usr/bin/env python3
"""One-off: leveling's hill-climb pegs motor3(id 3) at the configured ceiling of
4050 ticks and STAYS there for 30+ iterations without beta ever reaching zero
(beta plateaus around -8 to -9 degrees at that ceiling) -- meaning the true level
point for this axis is now beyond 4050, so the "safe max" in plate_env.py's
_level_plate() is stale and needs to move. Probe upward in small increments past
4050, checking for a REAL stall (present position readback vs commanded) at each
step, to find where beta actually crosses zero (or where the motor genuinely
stalls, whichever comes first) rather than assuming either outcome.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_env import HardwarePlateEnv  # noqa: E402
from state_est_control import ADDR_PRESENT_POSITION  # noqa: E402


def main():
    env = HardwarePlateEnv(max_episode_steps=1)
    try:
        m1_id, m3_id = sorted(env.calibration.keys())
        _, m1p, _ = env.calibration[m1_id]
        env_set = lambda mid, pos: __import__("state_est_control").set_position(
            env.port_handler, env.packet_handler, mid, pos
        )
        env_set(m1_id, m1p)
        pos = 3246
        step = 30
        max_pos = 4200
        while pos <= max_pos:
            env_set(m3_id, pos)
            time.sleep(1.3)
            present, _, _ = env.packet_handler.read4ByteTxRx(
                env.port_handler, m3_id, ADDR_PRESENT_POSITION
            )
            # SDK returns unsigned 32-bit; Dynamixel position range is small and
            # positive here, so a direct compare against the commanded tick is fine.
            # A real stall means present FELL SHORT of the (increasing) target --
            # present overshooting past target is healthy eager motion/residual
            # settling, not a stall, and abs() was flagging that as one by mistake.
            stall_gap = max(0, pos - present)
            _, _, alpha, beta, found = env._read_state()
            beta_deg = None if beta is None else float(__import__("numpy").degrees(beta))
            print(f"m3_target={pos:5d} present={present:5d} gap={stall_gap:3d} "
                  f"beta={beta_deg:+.2f}deg ball_found={found}" if beta_deg is not None
                  else f"m3_target={pos:5d} present={present:5d} gap={stall_gap:3d} beta=None")
            if stall_gap > 15:
                print(f"*** STALL detected at m3_target={pos} (present only reached {present}) -- stopping")
                break
            if beta_deg is not None and abs(beta_deg) < 0.5:
                print(f"*** beta crossed into tolerance at m3_target={pos} (beta={beta_deg:+.2f}deg)")
                break
            pos += step
    finally:
        env.close()


if __name__ == "__main__":
    main()
