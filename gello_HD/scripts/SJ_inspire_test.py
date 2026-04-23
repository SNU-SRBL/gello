import argparse
import matplotlib.pyplot as plt
from collections import deque
import time

from gello.robots.SRBL_Inspire import SRBL_Inspire_gripper

parser = argparse.ArgumentParser(description="Test script for the sensor output of one finger of the Inspire gripper.")

parser.add_argument("device", type=str, help="Serial port to which the gripper is connected. E.g. COM3 or /dev/ttyUSB1.")
parser.add_argument("-f", "--finger", type=int, default=4, choices=[2,3,4,5],help="Number of finger to use.")
parser.add_argument("-v", "--visualize", action="store_true", help="Whether to plot. Prints in terminal if not set.")
parser.add_argument("-l", "--length", type=int, default=100, help="Number of data points to show in the plot. Only used if --visualize is set.")

args = parser.parse_args()

def main():
    gripper = SRBL_Inspire_gripper(finger=args.finger, device_name=args.device)

    if args.visualize:
        max_len = args.length
        data_buffer = [deque([0]*max_len, maxlen=max_len) for _ in range(4)]
        plt.ion()
        fig, axs = plt.subplots(2,2)
        axs = axs.flatten()
        lines = []
        for i in range(4):
            line, = axs[i].plot(data_buffer[i])
            axs[i].set_title(f"Data {i}")
            lines.append(line)

    while True:
        data = gripper.get_sensor_values() # list of length 4
        if args.visualize:
            for i in range(4):
                data_buffer[i].append(data[i])
                lines[i].set_ydata(data_buffer[i])
                axs[i].relim()
                axs[i].autoscale_view()
            plt.pause(0.01)
        else:
            print(f"Sensor values: {data}")
        time.sleep(0.01)

if __name__ == "__main__":
    main()