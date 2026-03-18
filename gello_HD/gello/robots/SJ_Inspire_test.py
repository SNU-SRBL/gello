import time
import SRBL_Inspire_copy

def main():
    gripper = SRBL_Inspire_copy.SRBL_Inspire_gripper()
    finger_lower_limits = SRBL_Inspire_copy.SRBL_INSPIRE_FINGER_LOWER_LIMIT
    finger_upper_limits = SRBL_Inspire_copy.SRBL_INSPIRE_FINGER_UPPER_LIMIT
    SJ_flag = True
    while True:
        if SJ_flag:
            target = [int(finger_upper_limits[i]) for i in range(5)]
        else:
            target = [int(finger_lower_limits[i]) for i in range(5)]
        target.append(-1) # thumb rotation joint is not changed
        gripper.move_fingers(target)
        SJ_flag = not SJ_flag
        print(gripper.get_sensor_values())
        time.sleep(1.0)

if __name__ == "__main__":
    main()