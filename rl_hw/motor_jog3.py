#!/usr/bin/env python3
"""
Keyboard jog tool for ALL THREE motors (IDs 1, 2, 3) -- no camera, no calibration
file, no per-motor range restrictions beyond the raw hardware limits. For physically
inspecting the mechanism by hand while directly watching position/current feedback.

Current limit is set to 1193 (the real verified max for these X-series motors) on all
three at startup, so you have full available torque while testing.

Controls:
    q/a   motor 1 up/down
    w/s   motor 2 up/down
    e/d   motor 3 up/down
    z/x   decrease/increase step size (starts at 40 ticks)
    p     print position + present current + hardware error status for all 3
    c     recenter motors 1 and 3 to the camera-verified level point in
          motor_calibration.json (alpha=beta=0); motor 2 (the unrelated,
          non-functional elevator motor -- no calibration exists for it)
          recenters to wherever it was when this script started
    ESC   quit (same recenter as 'c', then disables torque)

Usage:
    python3 motor_jog3.py
"""
import os
import sys
import time
import termios
import tty

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plate_env import _resolve_dynamixel_port, _load_motor_calibration, CALIBRATION_PATH  # noqa: E402
import state_est_control as sec  # noqa: E402

ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_CURRENT = 126
ADDR_CURRENT_LIMIT = 38
ADDR_HARDWARE_ERROR_STATUS = 70
CURRENT_LIMIT = 1193  # verified real max for this motor model -- see session notes
POSITION_MIN, POSITION_MAX = 0, 4095  # raw hardware range, no per-motor restriction
MOTOR_IDS = [1, 2, 3]


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


def signed_current(raw):
    return raw - 65536 if raw > 32767 else raw


def main():
    sec.DXL_PORT = _resolve_dynamixel_port()
    port_handler, packet_handler = sec.init_dynamixel()
    if port_handler is None:
        raise SystemExit("could not open the Dynamixel port")

    print(__doc__)

    starting_pos = {}
    for motor_id in MOTOR_IDS:
        packet_handler.write1ByteTxRx(port_handler, motor_id, sec.ADDR_TORQUE_ENABLE, 0)
        time.sleep(0.05)
        packet_handler.write2ByteTxRx(port_handler, motor_id, ADDR_CURRENT_LIMIT, CURRENT_LIMIT)
        time.sleep(0.05)
        packet_handler.write1ByteTxRx(port_handler, motor_id, sec.ADDR_TORQUE_ENABLE, 1)
        pos, comm, err = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
        if comm != 0:
            print(f"warning: could not read motor {motor_id} present position at startup "
                  f"({packet_handler.getTxRxResult(comm)}) -- is it connected?")
            pos = 2048
        starting_pos[motor_id] = pos
        print(f"motor {motor_id}: current_limit set to {CURRENT_LIMIT}, starting position {pos}")

    calibration = _load_motor_calibration(CALIBRATION_PATH)
    recenter_targets = dict(starting_pos)
    for motor_id, (_, center, _) in calibration.items():
        recenter_targets[motor_id] = center
    print(f"recenter ('c' / ESC) targets: {recenter_targets} "
          f"(motor 2 has no calibration -- recenters to its startup position)")

    pos = dict(starting_pos)
    step = 40
    key_map = {"q": (1, +1), "a": (1, -1), "w": (2, +1), "s": (2, -1), "e": (3, +1), "d": (3, -1)}

    def send(motor_id):
        sec.set_position(port_handler, packet_handler, motor_id, pos[motor_id])

    def print_status():
        for motor_id in MOTOR_IDS:
            p, _, _ = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
            c_raw, _, _ = packet_handler.read2ByteTxRx(port_handler, motor_id, ADDR_PRESENT_CURRENT)
            err, _, _ = packet_handler.read1ByteTxRx(port_handler, motor_id, ADDR_HARDWARE_ERROR_STATUS)
            print(f"  motor {motor_id}: commanded={pos[motor_id]:5d}  present={p:5d}  "
                  f"current={signed_current(c_raw):+5d}  hw_error=0x{err:02X}")

    print("\nready -- press any mapped key to move, 'p' to check status, ESC to quit\n")
    try:
        while True:
            key = read_key()
            if key == "\x1b":  # ESC
                break
            elif key == "p":
                print_status()
            elif key == "c":
                pos = dict(recenter_targets)
                for motor_id in MOTOR_IDS:
                    send(motor_id)
                print(f"recentered to: {pos}")
            elif key == "z":
                step = max(5, step - 10)
                print(f"step size now {step}")
            elif key == "x":
                step = min(200, step + 10)
                print(f"step size now {step}")
            elif key in key_map:
                motor_id, direction = key_map[key]
                pos[motor_id] = clamp(pos[motor_id] + direction * step, POSITION_MIN, POSITION_MAX)
                send(motor_id)
                p, _, _ = packet_handler.read4ByteTxRx(port_handler, motor_id, ADDR_PRESENT_POSITION)
                print(f"motor {motor_id}: commanded={pos[motor_id]:5d}  present={p:5d}  step={step}")
            else:
                continue
    finally:
        print("\nrecentering and disabling torque...")
        for motor_id in MOTOR_IDS:
            sec.set_position(port_handler, packet_handler, motor_id, recenter_targets[motor_id])
        time.sleep(0.5)
        for motor_id in MOTOR_IDS:
            packet_handler.write1ByteTxRx(port_handler, motor_id, sec.ADDR_TORQUE_ENABLE, 0)
        port_handler.closePort()


if __name__ == "__main__":
    main()
