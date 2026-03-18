import time
import SRBL_Tesollo

def main():
    gripper = SRBL_Tesollo.SRBL_Tesollo_gripper()
    SJ_flag = True
    while True:
        if SJ_flag:
            gripper.move(1.0)
        else:
            gripper.move(0.0)
        SJ_flag = not SJ_flag
        print(gripper.get_sensor_values())
        time.sleep(1.0)

if __name__ == "__main__":
    main()