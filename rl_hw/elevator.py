"""
Ball-reload elevator (Dynamixel ID 2) control for the web UI (2026-09-27).

Motor 2 is an XL330-family servo in Velocity mode (operating mode 1). It is only ever
touched when the user presses On in the UI; Off and controller shutdown always set the
goal velocity to 0 and switch torque off. Safety notes from the first test run:
  - its Goal Velocity register was left at 1620 (= full speed): always write 0 BEFORE
    enabling torque, then ramp;
  - at 12 units (~2.7 rpm) it drew ~15 mA (max 29 mA): a much higher current while not
    turning means a stall -> switch off.
Only RAM registers are written (torque, goal velocity, profile acceleration).
"""
import json
import os
import time

ELEV_ID = 2
ADDR_TORQUE, ADDR_GOAL_VEL, ADDR_PROFILE_ACC = 64, 104, 108
ADDR_PRESENT_CURRENT, ADDR_PRESENT_VEL, ADDR_HW_ERR = 126, 128, 70
RPM_PER_UNIT = 0.229
ADDR_VEL_LIMIT = 44
MAX_UNITS_FALLBACK = 1620        # the motor's own Velocity Limit (read at startup) = UI maximum
STALL_MA, STALL_S = 250, 1.0
SETTINGS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "elevator_settings.json"))


def _s32(v):
    return v - 2 ** 32 if v >= 2 ** 31 else v


def _s16(v):
    return v - 2 ** 16 if v >= 2 ** 15 else v


class Elevator:
    def __init__(self, port, ph):
        self.port, self.ph = port, ph
        self.on, self.units, self.direction = False, 12, 1
        self.meas = {"vel_rpm": None, "current_ma": None, "error": None}
        self._last_read, self._stall_since, self.msg = 0.0, None, ""
        lim, res, _ = ph.read4ByteTxRx(port, ELEV_ID, ADDR_VEL_LIMIT)
        self.max_units = int(lim) if res == 0 and 0 < lim <= 2047 else MAX_UNITS_FALLBACK
        try:   # the last speed/direction set in the UI survives a controller restart
            with open(SETTINGS_PATH) as f:
                st = json.load(f)
            self.units = int(max(0, min(self.max_units, st.get("units", self.units))))
            self.direction = 1 if st.get("dir", 1) >= 0 else -1
        except (OSError, ValueError):
            pass

    def _w1(self, addr, v):
        return self.ph.write1ByteTxRx(self.port, ELEV_ID, addr, v)

    def _w4(self, addr, v):
        return self.ph.write4ByteTxRx(self.port, ELEV_ID, addr, int(v) & 0xFFFFFFFF)

    def set_speed(self, units=None, direction=None):
        if units is not None:
            self.units = int(max(0, min(self.max_units, units)))
        if direction is not None:
            self.direction = 1 if direction >= 0 else -1
        if self.on:
            self._w4(ADDR_GOAL_VEL, self.direction * self.units)
        try:
            with open(SETTINGS_PATH, "w") as f:
                json.dump({"units": self.units, "dir": self.direction}, f)
        except OSError:
            pass

    def start(self):
        self._w4(ADDR_GOAL_VEL, 0)               # never enable torque with a stale goal
        self._w4(ADDR_PROFILE_ACC, 500)          # gentle ramp (time-based profile: 500 ms)
        self._w1(ADDR_TORQUE, 1)
        self.on = True
        self._stall_since = None
        self._w4(ADDR_GOAL_VEL, self.direction * self.units)
        self.msg = "running"

    def stop(self, why="stopped"):
        try:
            self._w4(ADDR_GOAL_VEL, 0)
            self._w1(ADDR_TORQUE, 0)
        finally:
            self.on = False
            self.msg = why

    def poll(self):
        """read speed/current ~1x per second; switch off on stall or hardware error"""
        now = time.time()
        if now - self._last_read < 1.0:
            return
        self._last_read = now
        vel, _, _ = self.ph.read4ByteTxRx(self.port, ELEV_ID, ADDR_PRESENT_VEL)
        cur, _, _ = self.ph.read2ByteTxRx(self.port, ELEV_ID, ADDR_PRESENT_CURRENT)
        err, _, _ = self.ph.read1ByteTxRx(self.port, ELEV_ID, ADDR_HW_ERR)
        vel_rpm, cur_ma = _s32(vel) * RPM_PER_UNIT, abs(_s16(cur))
        self.meas = {"vel_rpm": round(vel_rpm, 2), "current_ma": cur_ma, "error": err}
        if self.on:
            if err:
                self.stop(f"stopped: hardware error {err}")
            elif cur_ma > STALL_MA and abs(vel_rpm) < 0.3:
                self._stall_since = self._stall_since or now
                if now - self._stall_since > STALL_S:
                    self.stop(f"stopped: stall ({cur_ma} mA, not turning)")
            else:
                self._stall_since = None

    def state(self):
        return {"elev_on": self.on, "elev_units": self.units, "elev_dir": self.direction,
                "elev_max_units": self.max_units,
                "elev_rpm_set": round(self.units * RPM_PER_UNIT * self.direction, 2),
                "elev_msg": self.msg, **{f"elev_{k}": v for k, v in self.meas.items()}}
