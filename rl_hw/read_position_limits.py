#!/usr/bin/env python3
"""Check whether motor3's apparent 'stall' just past 4050 ticks is a real
mechanical bind or just the servo's own firmware-enforced Max Position Limit
(X-series Dynamixel EEPROM register 52, 4 bytes) silently refusing to go
further -- a pure software/config setting, not a hardware fault."""
import sys
import os

STATE_EST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "state_est"))
sys.path.insert(0, STATE_EST_DIR)
os.chdir(STATE_EST_DIR)
import glob  # noqa: E402
import state_est_control  # noqa: E402
from state_est_control import init_dynamixel  # noqa: E402

ADDR_MAX_POSITION_LIMIT = 52  # 4 bytes
ADDR_MIN_POSITION_LIMIT = 56  # 4 bytes

candidates = sorted(glob.glob("/dev/tty.usbserial-*"))
if candidates:
    state_est_control.DXL_PORT = candidates[0]

port_handler, packet_handler = init_dynamixel()
try:
    for dxl_id in (1, 3):
        max_lim, _, _ = packet_handler.read4ByteTxRx(port_handler, dxl_id, ADDR_MAX_POSITION_LIMIT)
        min_lim, _, _ = packet_handler.read4ByteTxRx(port_handler, dxl_id, ADDR_MIN_POSITION_LIMIT)
        print(f"motor id={dxl_id}: firmware Min Position Limit={min_lim} Max Position Limit={max_lim}")
finally:
    port_handler.closePort()
