from SRBL_Tesollo import SRBL_Tesollo_gripper
import time

def main():
    gripper = SRBL_Tesollo_gripper()
    SJ_flag = True
    while True:
        if SJ_flag:
            gripper.move(0.0)
        else:
            gripper.move(1.0)
        SJ_flag = not SJ_flag
        print(gripper.get_sensor_values())
        time.sleep(1)

if __name__ == "__main__":
    main()