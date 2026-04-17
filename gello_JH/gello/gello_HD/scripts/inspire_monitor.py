"""
Inspire RH56F 6-channel angle monitor.
Prints all 6 channel positions in real-time.
Press each finger to identify which channel it corresponds to.
"""

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from gello.robots.SRBL_Inspire import SRBL_Inspire_gripper, INSPIRE_regdict


def read_all_positions(gripper):
    val = gripper.readRegister(1, INSPIRE_regdict["angleAct"], 12, True)
    if len(val) < 12:
        return None
    positions = []
    for i in range(6):
        raw = val[i * 2] + (val[i * 2 + 1] << 8)
        if raw > 32767:
            raw -= 65536
        positions.append(raw)
    return positions


def main():
    print("Connecting to Inspire gripper...")
    gripper = SRBL_Inspire_gripper()
    print("Connected. Monitoring 6 channels. Press Ctrl+C to stop.\n")

    try:
        while True:
            pos = read_all_positions(gripper)
            if pos is None:
                print("Read failed")
            else:
                line = "  ".join(f"Ch{i+1}: {pos[i]:5d}" for i in range(6))
                print(f"\r{line}", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
