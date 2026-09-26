#!/usr/bin/env python3
"""
Minimal keyboard-driven motor jog tool -- no camera, no vision, just direct motor
control. For physically inspecting the mechanism (e.g. checking whether a motor
shaft actually turns, or whether a linkage is connected) without needing the
detection pipeline running.

Controls:
    w/s   motor 1 (axis 1) up/down by the current step size
    a/d   motor 2 (axis 2) up/down by the current step size
    1/2/3 set step size to 20/60/150 ticks
    c     recenter both motors
    q     quit (recenters and disables torque before exiting)

Usage:
    python3 motor_jog.py
"""
import os
import sys
import termios
import tty

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_env import _load_motor_calibration, _resolve_dynamixel_port, CALIBRATION_PATH  # noqa: E402
import state_est_control as sec  # noqa: E402


def read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    cal = _load_motor_calibration(CALIBRATION_PATH)
    # Derive the real motor IDs from the calibration itself rather than hardcoding
    # them -- motor_calibration.json is the single source of truth for which DXL IDs
    # are actually the tilt axes (corrected 2026-08-26: axis 2 is ID 3, not 2 -- ID 2
    # is the unrelated ball-reload elevator motor).
    axis1_id, axis2_id = sorted(cal.keys())

    sec.DXL_PORT = _resolve_dynamixel_port()
    port_handler, packet_handler = sec.init_dynamixel()
    if port_handler is None:
        raise SystemExit("could not open the Dynamixel port")
    for dxl_id in (axis1_id, axis2_id):
        packet_handler.write1ByteTxRx(port_handler, dxl_id, sec.ADDR_TORQUE_ENABLE, 1)

    (a1_lo, a1_c, a1_hi) = cal[axis1_id]
    (a2_lo, a2_c, a2_hi) = cal[axis2_id]
    pos = {1: a1_c, 2: a2_c}
    bounds = {1: (a1_lo, a1_hi), 2: (a2_lo, a2_hi)}
    step = 60

    def send():
        sec.set_position(port_handler, packet_handler, axis1_id, pos[1])
        sec.set_position(port_handler, packet_handler, axis2_id, pos[2])
        print(f"\raxis1(id={axis1_id})={pos[1]} (range {bounds[1][0]}-{bounds[1][1]})   "
              f"axis2(id={axis2_id})={pos[2]} (range {bounds[2][0]}-{bounds[2][1]})   step={step}   ",
              end="", flush=True)

    print(__doc__)
    print(f"axis1 -> motor ID {axis1_id}, axis2 -> motor ID {axis2_id}")
    send()
    try:
        while True:
            key = read_key()
            if key == "q":
                break
            elif key == "w":
                pos[1] = clamp(pos[1] + step, *bounds[1])
            elif key == "s":
                pos[1] = clamp(pos[1] - step, *bounds[1])
            elif key == "d":
                pos[2] = clamp(pos[2] + step, *bounds[2])
            elif key == "a":
                pos[2] = clamp(pos[2] - step, *bounds[2])
            elif key == "c":
                pos[1], pos[2] = a1_c, a2_c
            elif key in ("1", "2", "3"):
                step = {"1": 20, "2": 60, "3": 150}[key]
            else:
                continue
            send()
    finally:
        print("\nrecentering and disabling torque...")
        sec.set_position(port_handler, packet_handler, axis1_id, a1_c)
        sec.set_position(port_handler, packet_handler, axis2_id, a2_c)
        import time
        time.sleep(0.3)
        for dxl_id in (axis1_id, axis2_id):
            packet_handler.write1ByteTxRx(port_handler, dxl_id, sec.ADDR_TORQUE_ENABLE, 0)
        port_handler.closePort()


if __name__ == "__main__":
    main()
