from dynamixel_sdk import *  # Uses Dynamixel SDK library


class Gripper:
    def __init__(self, device_name="/dev/ttyUSB1", dxl_id=1, baudrate=3000000):
        self.DXL_ID = dxl_id
        self.DEVICENAME = device_name
        self.BAUDRATE = baudrate
        self.PROTOCOL_VERSION = 2.0
        self.ADDR_TORQUE_ENABLE = 562
        self.ADDR_GOAL_POSITION = 596
        self.ADDR_PRESENT_POSITION = 611
        self.DXL_MOVING_STATUS_THRESHOLD = 20
        self.DXL_MIN_POSITION = 0
        self.DXL_MAX_POSITION = 740

        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        self._connect()

    def _connect(self):
        if not self.portHandler.openPort():
            raise IOError("Failed to open port")
        if not self.portHandler.setBaudRate(self.BAUDRATE):
            raise IOError("Failed to set baudrate")

        # Enable torque
        result, error = self.packetHandler.write1ByteTxRx(
            self.portHandler, self.DXL_ID, self.ADDR_TORQUE_ENABLE, 1
        )
        if result != COMM_SUCCESS:
            raise RuntimeError(f"Torque enable failed: {self.packetHandler.getTxRxResult(result)}")
        if error != 0:
            print(f"Torque enable warning: {self.packetHandler.getRxPacketError(error)}")

    def get_position(self):
        pos, result, error = self.packetHandler.read4ByteTxRx(
            self.portHandler, self.DXL_ID, self.ADDR_PRESENT_POSITION
        )
        if result != COMM_SUCCESS:
            raise RuntimeError(f"Read failed: {self.packetHandler.getTxRxResult(result)}")
        if error != 0:
            print(f"Read warning: {self.packetHandler.getRxPacketError(error)}")
        return pos

    def set_position(self, position: int):
        position = max(self.DXL_MIN_POSITION, min(self.DXL_MAX_POSITION, position))
        result, error = self.packetHandler.write4ByteTxRx(
            self.portHandler, self.DXL_ID, self.ADDR_GOAL_POSITION, position
        )
        if result != COMM_SUCCESS:
            raise RuntimeError(f"Write failed: {self.packetHandler.getTxRxResult(result)}")
        if error != 0:
            print(f"Write warning: {self.packetHandler.getRxPacketError(error)}")

    def wait_until_reached(self, target_pos):
        while True:
            current = self.get_position()
            print(f"current pos is {current}")
            if abs(current - target_pos) < self.DXL_MOVING_STATUS_THRESHOLD:
                break

    def close(self):
        # Disable torque
        self.packetHandler.write1ByteTxRx(self.portHandler, self.DXL_ID, self.ADDR_TORQUE_ENABLE, 0)
        self.portHandler.closePort()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass



class RobotisGripper(Gripper):
    def __init__(self, device_name="/dev/ttyUSB1", dxl_id=1, baudrate=3000000):
        super().__init__(device_name=device_name, dxl_id=dxl_id, baudrate=baudrate)

    def connect(self):
        pass

    def get_current_position(self):
        return self.get_position()

    def move(self, position):
        self.set_position(position)



if __name__ == "__main__":
    from datetime import datetime # TEST TIME

    gripper = Gripper()

    try:
        # Open gripper
        gripper.set_position(0)
        gripper.wait_until_reached(0)

        # Close gripper
        gripper.set_position(740)
        gripper.wait_until_reached(740)

        # Open gripper
        gripper.set_position(0)
        gripper.wait_until_reached(0)

        for angle in range(0,400,10):
            start_time = datetime.now()
            gripper.set_position(angle)
            current = gripper.get_position()
            print(f"current pos is {current}")
            end_time = datetime.now()
            print(f'elapsed time: {end_time - start_time}')

    finally:
        gripper.close()

