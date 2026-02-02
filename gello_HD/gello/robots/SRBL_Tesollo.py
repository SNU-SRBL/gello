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

        self.gripper = DGGripper()
        system_setting = GripperSystemSetting.create(
            ip="169.254.186.73", # Set the computer IP to 169.254.286.x where x is any number between 2 and 255 except 73. Used 10
            port=502,
            control_mode=ControlMode.OPERATOR,  # OPERATOR 모드
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
            self.SRBL_initialize()
        
    def __del__(self):
        self.gripper.stop()
        self.gripper.disconnect()

    def SRBL_initialize(self):
        '''
        Initialize the Tesollo gripper
        - Move to initial state
        - Set motion time for gello control
        '''
        init_time = 2000
        move_time = 30 # ms

        # Move to initial position
        self.gripper.set_motion_time_all_equal(init_time)
        thumb_init = [0.0, 0.0, 90.0, 0.0]
        index_init = [0.0, 0.0, 0.0, 0.0]
        middle_init = [0.0, 0.0, 90.0, 90.0]
        ring_init = [0.0, 0.0, 90.0, 90.0]
        little_init = [0.0, 0.0, 90.0, 90.0]
        init_pos = thumb_init + index_init + middle_init + ring_init + little_init
        self.gripper.move_joint_all(init_pos)

        # Set shorter time length for gello control
        self.gripper.set_motion_time_all_equal(move_time)

    def get_current_position(self):
        data = self.gripper.get_gripper_data()
        pos = float(data.joint[self.joint_number - 1])
        pos = min(SRBL_TESOLLO_FINGER_UPPER_LIMIT, max(SRBL_TESOLLO_FINGER_LOWER_LIMIT, pos))
        pos = (pos - SRBL_TESOLLO_FINGER_LOWER_LIMIT) / (SRBL_TESOLLO_FINGER_UPPER_LIMIT - SRBL_TESOLLO_FINGER_LOWER_LIMIT) # normalize to [0, 1]
        return pos

    def move(self, target):
        joint_target = target * (SRBL_TESOLLO_FINGER_UPPER_LIMIT - SRBL_TESOLLO_FINGER_LOWER_LIMIT) + SRBL_TESOLLO_FINGER_LOWER_LIMIT
        joint_target = min(SRBL_TESOLLO_FINGER_UPPER_LIMIT, max(SRBL_TESOLLO_FINGER_LOWER_LIMIT, joint_target))

        SRBL_type = 1
        if SRBL_type == 0:
            self.gripper.move_joint(joint_target, self.joint_number)
            self.gripper.move_joint(joint_target, self.joint_number - 1)
            self.gripper.move_joint(joint_target, self.joint_number - 2)
        elif SRBL_type == 1:
            joint_targets = [0.0, joint_target, joint_target, joint_target]
            self.gripper.move_joint_finger(SRBL_TESOLLO_FINGER_NUMBER, joint_targets)
        return

    def get_sensor_values(self):
        data = self.gripper.get_fingertip_sensor_data() # ReceivedFingertipSensorData
        lower_idx = 6 * (SRBL_TESOLLO_FINGER_NUMBER - 1)
        upper_idx = 6 * SRBL_TESOLLO_FINGER_NUMBER
        sensor = data.forceTorque[lower_idx:upper_idx] # elements are float type
        # force in 0.1N, torque in 1mNm
        return sensor
    
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
    
    def get_observation_values(self):
        '''
        This method returns all the observation values from the gripper.
        It is intended to ask the gripper for the data only once and get all the necessary information to avoid multiple communication calls and reduce latency.
        
        :param self: Default parameter for class methods
        '''
        data = self.gripper.get_gripper_data()
        observation = {}
        observation["position"] = []
        observation["velocity"] = []
        observation["current"] = []

        # Iterate over the finger joints - Position, Velocity, Current
        for i in range(self.joint_number - 3, self.joint_number):
            observation["position"].append(float(data.joint[i]))
            observation["velocity"].append(float(data.velocity[i]))
            observation["current"].append(float(data.current[i]) / 1000.0) # convert mA to A
        
        # Fingertip Sensor
        observation["sensor"] = self.get_sensor_values()

        return observation