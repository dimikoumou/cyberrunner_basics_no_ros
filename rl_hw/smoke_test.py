"""
Manual hardware smoke test for HardwarePlateEnv -- run this once before starting real
training, to confirm the camera, motors, and safety behavior all work end to end.

Usage:
    python3 smoke_test.py
"""
from plate_env import HardwarePlateEnv


def main():
    env = HardwarePlateEnv()
    try:
        print("\n--- reset() ---")
        obs, info = env.reset()
        print("observation:", obs)

        print("\n--- step([0.3, 0.0]) -- plate should visibly tilt ---")
        obs, reward, terminated, truncated, info = env.step([0.3, 0.0])
        print("observation:", obs, "reward:", reward, "info:", info)

        print("\n--- step([-0.3, 0.0]) -- should tilt the other way ---")
        obs, reward, terminated, truncated, info = env.step([-0.3, 0.0])
        print("observation:", obs, "reward:", reward, "info:", info)

        print("\n--- step([0.0, 0.0]) -- back to level ---")
        obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
        print("observation:", obs, "reward:", reward, "info:", info)

        print("\nNow try covering the camera or removing the ball, then press Enter...")
        input()
        obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
        print("observation:", obs, "reward:", reward, "terminated:", terminated, "info:", info)
    finally:
        print("\n--- close() -- motors should de-torque/settle ---")
        env.close()


if __name__ == "__main__":
    main()
