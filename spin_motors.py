import argparse, time
from dynamixel_sdk import *

# Protocol 2.0 typical XL/XM addresses (adjust if your model differs)
ADDR_OPERATING_MODE   = 11  # 1 byte
ADDR_TORQUE_ENABLE    = 64  # 1 byte
ADDR_GOAL_VELOCITY    = 104 # 4 bytes
ADDR_PRESENT_VELOCITY = 128 # 4 bytes
ADDR_HARDWARE_ERROR   = 70  # 1 byte (RO)

PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000

VELOCITY_MODE = 1


def main():
    ap = argparse.ArgumentParser(description="Spin motors 1,2,3 at slow velocities or disable torque")
    ap.add_argument("--port", type=str, default='/dev/tty.usbserial-FTA2U13I', help="Serial port, e.g., /dev/tty.usbserial-XXXXX")
    ap.add_argument("--v1", type=int, default=30, help="Goal velocity for motor 1 (ticks)")
    ap.add_argument("--v2", type=int, default=20, help="Goal velocity for motor 2 (ticks)")
    ap.add_argument("--v3", type=int, default=30, help="Goal velocity for motor 3 (ticks)")
    ap.add_argument("--duration", type=float, default=10.0, help="Seconds to run (Ctrl-C to stop earlier)")
    ap.add_argument("--disable", action="store_true", help="Only disable torque on motors 1,2,3 and exit")
    args = ap.parse_args()

    ph = PortHandler(args.port)
    pk = PacketHandler(PROTOCOL_VERSION)

    if not ph.openPort():
        print(f"Failed to open port /dev/tty.usbserial-FTA2U13I")
        return
    if not ph.setBaudRate(BAUDRATE):
        print(f"Failed to set baud {BAUDRATE}")
        return

    def write_goal_vel(motor_id, vel):
        # Goal Velocity is 4 bytes signed (2's complement). Accept small ticks.
        vel = int(vel) & 0xFFFFFFFF
        pk.write4ByteTxRx(ph, motor_id, ADDR_GOAL_VELOCITY, vel)

    if args.disable:
        # Stop and disable torque only
        for mid in (1,2,3):
            write_goal_vel(mid, 0)
            pk.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, 0)
        print("Torque disabled on motors 1,2,3.")
        ph.closePort()
        return

    for mid in (1,2,3):
        # Disable torque, set velocity mode, enable torque
        pk.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, 0)
        pk.write1ByteTxRx(ph, mid, ADDR_OPERATING_MODE, VELOCITY_MODE)
        pk.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, 1)
        herr,_,_ = pk.read1ByteTxRx(ph, mid, ADDR_HARDWARE_ERROR)
        if herr:
            print(f"Motor {mid} hardware error={herr}")

    print("Spinning: 1->v1, 2->v2, 3->v3. Ctrl-C to stop.")
    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            write_goal_vel(1, args.v1)
            write_goal_vel(2, args.v2)
            write_goal_vel(3, args.v3)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop and disable torque
        for mid in (1,2,3):
            write_goal_vel(mid, 0)
            pk.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, 0)
        ph.closePort()

if __name__ == "__main__":
    main()
