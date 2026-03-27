import time
import SRBL_Inspire_copy

def main():
    gripper = SRBL_Inspire_copy.SRBL_Inspire_gripper(device_name="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG027RUV-if00-port0")
    finger_lower_limits = SRBL_Inspire_copy.SRBL_INSPIRE_FINGER_LOWER_LIMIT
    finger_upper_limits = SRBL_Inspire_copy.SRBL_INSPIRE_FINGER_UPPER_LIMIT
    SJ_flag = True
    SJ_cnt = 0
    prev_time = time.perf_counter()
    while True:
        if SJ_flag:
            target = [int(finger_upper_limits[i]) for i in range(5)]
        else:
            target = [int(finger_lower_limits[i]) for i in range(5)]
        target.append(-1) # thumb rotation joint is not changed
        gripper.move_fingers(target)
        # print("===============")
        # print(f"Position values: {gripper.get_position_values()}")
        # print(f"Current values: {gripper.get_current_values()}")
        # print(f"Sensor values: {gripper.get_sensor_values(all=False)}")
        # pos = gripper.get_position_values()
        # curr = gripper.get_current_values()
        pos, curr = gripper.get_all_once()
        sens = gripper.get_sensor_values(all=True)
        # time.sleep(0.01)
        SJ_cnt += 1
        # print(SJ_cnt)
        if SJ_cnt >= 100:
            curr_time = time.perf_counter()
            freq = SJ_cnt / (curr_time - prev_time)
            print(f"freq: {freq}, dt: {curr_time - prev_time}")
            prev_time = curr_time
            SJ_flag = not SJ_flag
            SJ_cnt = 0

if __name__ == "__main__":
    main()