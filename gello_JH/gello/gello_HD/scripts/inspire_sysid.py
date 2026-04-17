"""
Inspire RH56F system identification script.
Moves index, thumb bending, thumb rotation one joint at a time.
Records commanded and actual positions.
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, "/home/seunghoon/Documents/gello/gello_HD")
from gello.robots.SRBL_Inspire_copy import (
    SRBL_Inspire_gripper,
    INSPIRE_regdict,
    SRBL_INSPIRE_FINGER_LOWER_LIMIT,
    SRBL_INSPIRE_FINGER_UPPER_LIMIT,
)

# Joint indices: little=0, ring=1, middle=2, index=3, thumb_bend=4, thumb_rot=5
ACTIVE_JOINTS = [3, 4, 5]
JOINT_NAMES = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]

LOWER = list(SRBL_INSPIRE_FINGER_LOWER_LIMIT)  # [900, 900, 900, 900, 1100, 600]
UPPER = list(SRBL_INSPIRE_FINGER_UPPER_LIMIT)  # [1740, 1740, 1740, 1740, 1350, 1800]
LOWER[5] = 900  # thumb rotation: limit to 900 instead of 600

SPEED = 2000
NUM_CYCLES = 3
RECORD_HZ = 60


def set_speed(gripper, speed):
    speeds = [speed] * 6
    val_reg = []
    for s in speeds:
        val_reg.append(s & 0xFF)
        val_reg.append((s >> 8) & 0xFF)
    gripper._writeRegister(1, INSPIRE_regdict["speedSet"], 12, val_reg)


def estimate_move_time(joint_idx):
    t = (UPPER[joint_idx] - LOWER[joint_idx]) / SPEED
    return t + 0.5


def main():
    print("Connecting to Inspire gripper...")
    gripper = SRBL_Inspire_gripper(device_name="/dev/ttyUSB0")
    print("Connected.")

    set_speed(gripper, SPEED)
    print(f"Speed set to {SPEED}")

    timestamps = []
    angle_cmds = []
    angle_actuals = []
    joint_log = []

    t_start = time.time()
    record_interval = 1.0 / RECORD_HZ

    def record(cmd, joint_idx):
        t_now = time.time() - t_start
        pos = gripper.get_position_values()
        timestamps.append(t_now)
        angle_cmds.append(cmd[:])
        angle_actuals.append(pos)
        joint_log.append(joint_idx)

    try:
        for j in ACTIVE_JOINTS:
            wait_time = estimate_move_time(j)
            print(f"\n{'='*50}")
            print(f"Joint: {JOINT_NAMES[j]} (ch{j+1})  |  range: {LOWER[j]}-{UPPER[j]}  |  wait: {wait_time:.2f}s")
            print(f"{'='*50}")

            for cycle in range(NUM_CYCLES):
                # Move to UPPER (unfold)
                cmd = [-1] * 6
                cmd[j] = UPPER[j]
                if j in (4, 5):  # thumb joints: keep index extended
                    cmd[3] = UPPER[3]
                gripper.move_fingers(cmd[:])
                print(f"  [Cycle {cycle+1}] {JOINT_NAMES[j]} -> {UPPER[j]}")

                t_end = time.time() + wait_time
                while time.time() < t_end:
                    t_loop = time.time()
                    record(cmd, j)
                    remain = record_interval - (time.time() - t_loop)
                    if remain > 0:
                        time.sleep(remain)

                # Move to LOWER (fold)
                cmd = [-1] * 6
                cmd[j] = LOWER[j]
                if j in (4, 5):  # thumb joints: keep index extended
                    cmd[3] = UPPER[3]
                gripper.move_fingers(cmd[:])
                print(f"  [Cycle {cycle+1}] {JOINT_NAMES[j]} -> {LOWER[j]}")

                t_end = time.time() + wait_time
                while time.time() < t_end:
                    t_loop = time.time()
                    record(cmd, j)
                    remain = record_interval - (time.time() - t_loop)
                    if remain > 0:
                        time.sleep(remain)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # Return to home pose: index folded, thumb extended
    print("\nReturning to home pose...")
    home_cmd = [-1] * 6
    home_cmd[3] = LOWER[3]   # index folded (900)
    home_cmd[4] = UPPER[4]   # thumb bending extended (1350)
    home_cmd[5] = UPPER[5]   # thumb rotation extended (1800)
    gripper.move_fingers(home_cmd[:])
    time.sleep(2.0)

    if len(timestamps) == 0:
        print("No data recorded.")
        return

    data = {
        "timestamp": np.array(timestamps),
        "angle_cmd": np.array(angle_cmds),
        "angle_actual": np.array(angle_actuals),
        "active_joint": np.array(joint_log),
        "joint_names": np.array(JOINT_NAMES),
    }

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"inspire_sysid_{ts_str}.npz")
    np.savez(out_path, **data)

    N = len(timestamps)
    print(f"\n{'='*50}")
    print(f"Saved: {out_path}")
    print(f"  Samples: {N}  |  Total time: {timestamps[-1]:.1f}s")
    for k, v in data.items():
        print(f"  {k:15s}: {v.shape}")
    print(f"{'='*50}")

    gripper.close()


if __name__ == "__main__":
    main()
