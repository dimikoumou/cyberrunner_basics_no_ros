#!/usr/bin/env python3

import subprocess

def main():
    print("Launching marker calibration tool...")
    subprocess.run(["python", "marker_calibration.py"])
    print("Marker calibration finished.")

if __name__ == "__main__":
    main()
