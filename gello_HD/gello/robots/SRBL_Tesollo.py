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

sys.path.insert(0, str(Path(__file__).parent.parent / "delto_py" / "src"))

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
SRBL_TESOLLO_FINGER_UPPER_LIMIT = 60.0 # Upper limit of the finger joint position

SRBL_TESOLLO_FINGER_NUMBER = 1 # Number of the finger to control

class SRBL_Tesollo_gripper:
    def __init__(self):
        self.error_flag = False
        self.joint_number = SRBL_TESOLLO_FINGER_NUMBER * 4 - 1

        self.gripper = DGGripper()
        system_setting = GripperSystemSetting.create(
            ip="169.254.186.73",
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
        self.connection  = self.gripper.connect()
        if self.connection != DGResult.NONE:
            print(f"[ERROR] Gripper connection failed: {self.connection.name}")
            self.error_flag = True

        self.gripper_setting = GripperSetting.create(
            model=DGModel.DG_5F_LEFT,  # 실제 그리퍼 모델에 맞게 변경
            joint_count=20,
            finger_count=5,
            moving_inpose=0.5,
        )
        self.gripper_option = self.gripper.set_gripper_option(self.gripper_setting)
        if self.gripper_option != DGResult.NONE:
            print(f"[ERROR] Gripper option setting failed: {self.gripper_option.name}")
            self.error_flag = True
        time.sleep(0.5)

        self.start = self.gripper.start()
        if self.start != DGResult.NONE:
            print(f"[ERROR] Gripper start failed: {self.start.name}")
            self.error_flag = True
        time.sleep(0.5)

        if self.error_flag:
            raise RuntimeError("Failed to initialize the gripper.")
        
    def __del__(self):
        self.gripper.stop()
        self.gripper.disconnect()

    def get_current_position(self):
        data = self.gripper.get_gripper_data()
        pos = float(data.joint[self.joint_number])
        pos = min(SRBL_TESOLLO_FINGER_UPPER_LIMIT, max(SRBL_TESOLLO_FINGER_LOWER_LIMIT, pos))
        pos = (pos - SRBL_TESOLLO_FINGER_LOWER_LIMIT) / (SRBL_TESOLLO_FINGER_UPPER_LIMIT - SRBL_TESOLLO_FINGER_LOWER_LIMIT) # normalize to [0, 1]
        return pos

    def move(self, target):
        joint_target = min(SRBL_TESOLLO_FINGER_UPPER_LIMIT, max(SRBL_TESOLLO_FINGER_LOWER_LIMIT, target))
        joint_target = joint_target * (SRBL_TESOLLO_FINGER_UPPER_LIMIT - SRBL_TESOLLO_FINGER_LOWER_LIMIT) + SRBL_TESOLLO_FINGER_LOWER_LIMIT
        self.gripper.move_joint(joint_target, self.joint_number)
        return

    def get_sensor_values(self):
        data = self.gripper.get_fingertip_sensor_data() # [TODO] Need to confirm the data format
        lower_idx = 4 * (SRBL_TESOLLO_FINGER_NUMBER - 1)
        upper_idx = 4 * SRBL_TESOLLO_FINGER_NUMBER
        sensor = data[lower_idx:upper_idx]
        return sensor