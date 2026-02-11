import serial
import time
import numpy as np

DF = 50.0                             # Force Divider
DT = 1000.0                           # Torque Divider

class SensorFT:
    def __init__(self, port="/dev/ttyUSB2", baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        time.sleep(1)

        self.df = DF  # Force divider
        self.dt = DT  # Torque divider

        # Command packets
        self.cmd_get_data = bytes([0x55, 0x0A, 0x02, 0x00, 0x00, 0x00,
                                   0x00, 0x00, 0x00, 0x0C, 0xAA])

        # Sensor init commands
        self.ser.write(bytes([0x55, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0xAA]))  # Stop Output
        self.ser.write(bytes([0x55, 0x11, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x12, 0xAA]))  # Set Bias
        self.ser.write(bytes([0x55, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0xAA]))  # Output: 200 Hz
        self.ser.write(bytes([0x55, 0x08, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0xAA]))  # LPF: 30 Hz
        time.sleep(0.1)

        self.offset_force = np.zeros(3)
        self.offset_torque = np.zeros(3)
        self._calibrate_offset()

        print("Sensor FT Started.")

    def _calibrate_offset(self):
        force = np.zeros(3)
        torque = np.zeros(3)
        valid = 0
        for _ in range(100):
            self.ser.write(self.cmd_get_data)
            data = self.ser.read(19)
            if len(data) != 19:
                continue
            valid += 1
            force += np.array([
                int.from_bytes(data[2:4], byteorder='big', signed=True),
                int.from_bytes(data[4:6], byteorder='big', signed=True),
                int.from_bytes(data[6:8], byteorder='big', signed=True)
            ])
            torque += np.array([
                int.from_bytes(data[8:10], byteorder='big', signed=True),
                int.from_bytes(data[10:12], byteorder='big', signed=True),
                int.from_bytes(data[12:14], byteorder='big', signed=True)
            ])
        if valid > 0:
            self.offset_force = force / valid
            self.offset_torque = torque / valid
        else:
            print("Warning: No valid frames for offset calibration")

    def read(self):
        self.ser.write(self.cmd_get_data)
        data = self.ser.read(19)
        if len(data) != 19:
            return None  # Or raise exception

        fx = int.from_bytes(data[2:4], byteorder='big', signed=True)
        fy = int.from_bytes(data[4:6], byteorder='big', signed=True)
        fz = int.from_bytes(data[6:8], byteorder='big', signed=True)
        mx = int.from_bytes(data[8:10], byteorder='big', signed=True)
        my = int.from_bytes(data[10:12], byteorder='big', signed=True)
        mz = int.from_bytes(data[12:14], byteorder='big', signed=True)

        force = np.array([fx, fy, fz]) - self.offset_force
        torque = np.array([mx, my, mz]) - self.offset_torque

        return np.concatenate([force / self.df, torque / self.dt])

    def close(self):
        self.ser.close()


if __name__ == "__main__":

    sensor_ft = SensorFT(port="/dev/ttyUSB2")

    try:
        while True:
            data = sensor_ft.read()
            if data is not None:
                Fx, Fy, Fz, Mx, My, Mz = data
                print(f"Fx: {Fx:.2f}, Fy: {Fy:.2f}, Fz: {Fz:.2f} | Mx: {Mx:.3f}, My: {My:.3f}, Mz: {Mz:.3f}")
            time.sleep(0.01)  # 100 Hz
    except KeyboardInterrupt:
        print("Exiting...")
        sensor_ft.close()