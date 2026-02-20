"""
File containing the SRBL Tesollo gripper class for controlling the gripper and obtaining sensor data.
Referenced the document and code samples given by Tesollo.

For use (Python 3.8+):
uv add dgsdk
OR
pip install dgsdk
OR
git clone https://github.com/tesollo/dgsdk-python.git
cd dgsdk-python
uv sync

Written by Seongjun Koh (Soft Robotics and Bionics Lab, Seoul National University)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "delto_py" / "src"))

from dgsdk import (
    DGGripper,
    GripperSystemSetting,
    GripperSetting,
    ControlMode,
    CommunicationMode,
    ReceivedGripperData,
    DGModel,
    DGResult,
)

SRBL_TESOLLO_FINGER_LOWER_LIMIT = 0.0 # Lower limit of the finger joint position
SRBL_TESOLLO_FINGER_UPPER_LIMIT = 30.0 # Upper limit of the finger joint position

SRBL_TESOLLO_FINGER_NUMBER = 2 # Number of the finger to control
'''
1: Thumb / 2: Index / 3: Middle / 4: Ring / 5: Little
'''

class SRBL_Tesollo_gripper:
    def __init__(self, init_pos: bool = True):
        self.init_pos = init_pos
        self.error_flag = False
        self.joint_number = SRBL_TESOLLO_FINGER_NUMBER * 4 # Joint number of the most tip
        self._cached_gripper_data = None  # Cached gripper data to avoid redundant SDK calls

        self.gripper = DGGripper()
        system_setting = GripperSystemSetting.create(
            ip="169.254.186.73", # Set the computer IP to 169.254.286.x where x is any number between 2 and 255 except 73. Used 10
            port=502,
            control_mode=ControlMode.DEVELOPER,  # DEVELOPER 모드 (TCP 데이터 사용 가능)
            communication_mode=CommunicationMode.ETHERNET,
            read_timeout=1000,
            slave_id=1,
            baudrate=115200,
        )
        self.setting = self.gripper.set_gripper_system(system_setting)
        if self.setting != DGResult.NONE:
            print(f"[ERROR] Gripper setting failed: {self.setting.name}")
            self.error_flag = True
        else:
            print("[INFO] Gripper system setting successful.")
        
        connected = False
        def on_connected():
            nonlocal connected
            connected = True
            print("Connection callback!")
        # self.gripper.on_gripper_data(lambda: print(""))
        self.gripper.on_connected(on_connected)
        self.gripper.on_disconnected(lambda: print("Disconnect callback!"))

        self.connection  = self.gripper.connect()
        if self.connection != DGResult.NONE:
            print(f"[ERROR] Gripper connection failed: {self.connection.name}")
            self.error_flag = True
        else:
            print("[INFO] Gripper connection successful.")

        self.gripper_setting = GripperSetting.create(
            model=DGModel.DG_5F_LEFT,  # 실제 그리퍼 모델에 맞게 변경
            joint_count=20,
            finger_count=5,
            moving_inpose=0.5,
            received_data_type=[1, 2, 0, 4, 5, 0], # Check types.py
        )
        self.gripper_option = self.gripper.set_gripper_option(self.gripper_setting)
        if self.gripper_option != DGResult.NONE:
            print(f"[ERROR] Gripper option setting failed: {self.gripper_option.name}")
            self.error_flag = True
        else:
            print("[INFO] Gripper option setting successful.")
        time.sleep(0.5)

        self.start = self.gripper.start()
        if self.start != DGResult.NONE:
            print(f"[ERROR] Gripper start failed: {self.start.name}")
            self.error_flag = True
        else:
            print("[INFO] Gripper start successful.")
        time.sleep(0.5)

        if self.error_flag:
            raise RuntimeError("Failed to initialize the gripper.")
        
        if self.init_pos:
            self.initialize_finger_position()
        
    def __del__(self):
        self.gripper.stop()
        self.gripper.disconnect()

    def initialize_finger_position(self):
        '''
        TODO : Initialize the finger position to a certain state.
        Current target is to close the gripper fully except for the desired finger.
        
        :param self: Description
        '''
        pass

    def get_current_position(self):
        data = self.gripper.get_gripper_data()
        self._cached_gripper_data = data
        pos = float(data.joint[self.joint_number - 1])
        pos = min(SRBL_TESOLLO_FINGER_UPPER_LIMIT, max(SRBL_TESOLLO_FINGER_LOWER_LIMIT, pos))
        pos = (pos - SRBL_TESOLLO_FINGER_LOWER_LIMIT) / (SRBL_TESOLLO_FINGER_UPPER_LIMIT - SRBL_TESOLLO_FINGER_LOWER_LIMIT) # normalize to [0, 1]
        return pos

    def move(self, target):
        joint_target = target * (SRBL_TESOLLO_FINGER_UPPER_LIMIT - SRBL_TESOLLO_FINGER_LOWER_LIMIT) + SRBL_TESOLLO_FINGER_LOWER_LIMIT
        joint_target = min(SRBL_TESOLLO_FINGER_UPPER_LIMIT, max(SRBL_TESOLLO_FINGER_LOWER_LIMIT, joint_target))
        # Use move_joint_finger() for single Modbus packet instead of 3x move_joint()
        # Finger joint array: [base, ..., tip] — keep base joint at current value
        base_idx = self.joint_number - 4  # 0-indexed base joint of this finger
        if self._cached_gripper_data is not None:
            base_joint = float(self._cached_gripper_data.joint[base_idx])
        else:
            base_joint = 0.0
        self.gripper.move_joint_finger(
            [base_joint, joint_target, joint_target, joint_target],
            SRBL_TESOLLO_FINGER_NUMBER,
        )
        return

    def get_sensor_values(self):
        data = self.gripper.get_fingertip_sensor_data() # ReceivedFingertipSensorData
        lower_idx = 6 * (SRBL_TESOLLO_FINGER_NUMBER - 1)
        upper_idx = 6 * SRBL_TESOLLO_FINGER_NUMBER
        sensor = data.forceTorque[lower_idx:upper_idx] # elements are float type
        # force in 0.1N, torque in 1mNm
        return sensor

    def get_fingertip_tcp(self):
        '''
        Get fingertip TCP (position and orientation) from ReceivedGripperData.TCP.

        Returns:
            list: [x, y, z, rx, ry, rz] for the selected finger
                  Position in mm, rotation in degrees (axis-angle)
        '''
        data = self.gripper.get_gripper_data()
        # TCP array: 6 values per finger (x, y, z, rx, ry, rz)
        # Index finger is finger 2 (0-indexed: 1)
        lower_idx = 6 * (SRBL_TESOLLO_FINGER_NUMBER - 1)
        upper_idx = 6 * SRBL_TESOLLO_FINGER_NUMBER
        tcp = [float(data.TCP[i]) for i in range(lower_idx, upper_idx)]
        return tcp
    
    def get_current_values(self):
        data = self.gripper.get_gripper_data()
        current = []
        current.append(float(data.current[self.joint_number - 1]) / 1000.0) # convert mA to A
        current.append(float(data.current[self.joint_number - 2]) / 1000.0) # convert mA to A
        current.append(float(data.current[self.joint_number - 3]) / 1000.0) # convert mA to A
        return current

    def get_velocity_values(self):
        data = self.gripper.get_gripper_data()
        velocity = []
        velocity.append(float(data.velocity[self.joint_number - 1]))
        velocity.append(float(data.velocity[self.joint_number - 2]))
        velocity.append(float(data.velocity[self.joint_number - 3]))
        # 1 RPM
        return velocity
    
    def get_position_values(self):
        data = self.gripper.get_gripper_data()
        position = []
        position.append(float(data.joint[self.joint_number - 1])) # 0.1 degree?
        position.append(float(data.joint[self.joint_number - 2]))
        position.append(float(data.joint[self.joint_number - 3]))
        return position
    
    def get_all_observations(self):
        '''
        Get all observation data with minimal SDK calls (2 calls instead of 5).
        Returns dict with: normalized_position, position, velocity, current, tcp, sensor.
        '''
        # Single get_gripper_data() call for position, velocity, current, TCP
        data = self.gripper.get_gripper_data()
        self._cached_gripper_data = data

        jn = self.joint_number
        fn = SRBL_TESOLLO_FINGER_NUMBER

        # Normalized position [0, 1]
        pos = float(data.joint[jn - 1])
        pos = min(SRBL_TESOLLO_FINGER_UPPER_LIMIT, max(SRBL_TESOLLO_FINGER_LOWER_LIMIT, pos))
        normalized_pos = (pos - SRBL_TESOLLO_FINGER_LOWER_LIMIT) / (SRBL_TESOLLO_FINGER_UPPER_LIMIT - SRBL_TESOLLO_FINGER_LOWER_LIMIT)

        # Raw per-joint data (3 joints: indices jn-3 to jn-1 in 0-indexed)
        position = [float(data.joint[i]) for i in range(jn - 3, jn)]
        velocity = [float(data.velocity[i]) for i in range(jn - 3, jn)]
        current = [float(data.current[i]) / 1000.0 for i in range(jn - 3, jn)]

        # TCP: 6 values per finger (x, y, z, rx, ry, rz)
        tcp_lower = 6 * (fn - 1)
        tcp_upper = 6 * fn
        tcp = [float(data.TCP[i]) for i in range(tcp_lower, tcp_upper)]

        # Fingertip sensor: separate SDK call (different struct)
        sensor_data = self.gripper.get_fingertip_sensor_data()
        ft_lower = 6 * (fn - 1)
        ft_upper = 6 * fn
        sensor = sensor_data.forceTorque[ft_lower:ft_upper]

        return {
            "normalized_position": normalized_pos,
            "position": position,
            "velocity": velocity,
            "current": current,
            "tcp": tcp,
            "sensor": sensor,
        }

    def get_observation_values(self):
        '''
        This method returns all the observation values from the gripper.
        It is intended to ask the gripper for the data only once and get all the necessary information to avoid multiple communication calls and reduce latency.

        :param self: Default parameter for class methods
        '''
        data = self.gripper.get_gripper_data()
        self._cached_gripper_data = data
        observation = {}
        observation["position"] = []
        observation["velocity"] = []
        observation["current"] = []

        # Iterate over the finger joints
        # Position, Velocity, Current
        for i in range(self.joint_number - 3, self.joint_number):
            observation["position"].append(float(data.joint[i]))
            observation["velocity"].append(float(data.velocity[i]))
            observation["current"].append(float(data.current[i]) / 1000.0) # convert mA to A

        # Fingertip Sensor
        observation["sensor"] = self.get_sensor_values()

        return observation