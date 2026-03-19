import time
import SRBL_Inspire_copy

def main():
    gripper = SRBL_Inspire_copy.SRBL_Inspire_gripper(device_name="/dev/ttyUSB0")
    finger_lower_limits = SRBL_Inspire_copy.SRBL_INSPIRE_FINGER_LOWER_LIMIT
    finger_upper_limits = SRBL_Inspire_copy.SRBL_INSPIRE_FINGER_UPPER_LIMIT
    SJ_flag = True
    SJ_cnt = 0
    while True:
        if SJ_flag:
            target = [int(finger_upper_limits[i]) for i in range(5)]
        else:
            target = [int(finger_lower_limits[i]) for i in range(5)]
        target.append(-1) # thumb rotation joint is not changed
        gripper.move_fingers(target)
        print(f"Position values: {gripper.get_position_values()}")
        print(f"Current values: {gripper.get_current_values()}")
        # print(f"Sensor values: {gripper.get_sensor_values()}")
        time.sleep(1.0)
        SJ_cnt += 1
        if SJ_cnt >= 5:
            SJ_flag = not SJ_flag
            SJ_cnt = 0

if __name__ == "__main__":
    main()