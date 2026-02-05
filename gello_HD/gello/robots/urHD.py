"""
Copy of the ur.py file.
Modified to incorporate the Inspire and Tesollo hands.
Modified by Seongjun Koh (Soft Robotics and Bionics Lab, Seoul National University)
"""


from typing import Dict

import numpy as np

from gello.robots.robot import Robot


class URTesollo(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "192.168.1.10", no_gripper: bool = False):
        import rtde_control
        import rtde_receive

        [print("in ur robot") for _ in range(4)]
        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        if not no_gripper:
            from gello.robots.SRBL_Tesollo import SRBL_Tesollo_gripper 
            self.gripper = SRBL_Tesollo_gripper() 

            print("gripper connected")
            # gripper.activate()

        [print("connect") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()
        self._use_gripper = not no_gripper

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float: # CHANGE to corresponding gripper
        import time

        time.sleep(0.01)
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 1, "Gripper position must be between 0 and 1"
        return gripper_pos

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()
        self.robot.servoJ(
            robot_joints, velocity, acceleration, dt, lookahead_time, gain
        )
        if self._use_gripper:
            gripper_pos = joint_state[-1] # Tesollo # CHANGE to corresponding gripper
            self.gripper.move(gripper_pos) # Tesollo # CHANGE to corresponding gripper
        self.robot.waitPeriod(t_start)

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_joint_velocity(self):
        robot_joint_velocity = self.r_inter.getActualQd()
        return robot_joint_velocity

    def get_robot_current(self):
        robot_current = self.r_inter.getActualCurrent()
        return robot_current
    
    def get_robot_ee_pose(self):
        robot_ee_pose = self.r_inter.getActualTCPPose()
        return robot_ee_pose

    def get_finger_sensor_values(self):
        if self._use_gripper:
            return self.gripper.get_sensor_values()
        else:
            return None
        
    def get_finger_current_values(self):
        if self._use_gripper:
            return self.gripper.get_current_values()
        else:
            return None
        
    def get_finger_velocity_values(self):
        if self._use_gripper:
            return self.gripper.get_velocity_values()
        else:
            return None
    
    def get_finger_position_values(self):
        if self._use_gripper:
            return self.gripper.get_position_values()
        else:
            return None
        
    def get_finger_values(self):
        if self._use_gripper:
            return self.gripper.get_observation_values()
        else:
            return None

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        robot_velocity = self.get_joint_velocity()
        robot_current = self.get_robot_current()
        robot_ee = self.get_robot_ee_pose()
        if False:
            finger_pos = self.get_finger_position_values()
            fingertip_sensor = self.get_finger_sensor_values()
            finger_current = self.get_finger_current_values()
            finger_velocity = self.get_finger_velocity_values()
        else:
            # Code for calling the data from the gripper once. Not tested yet.
            # Currently the approach above doesn't seem to cause too much delay.
            finger_data = self. get_finger_values()
            finger_pos = finger_data["position"]
            finger_velocity = finger_data["velocity"]
            finger_current = finger_data["current"]
            fingertip_sensor = finger_data["sensor"]
            finger_tcp = finger_data["tcp"]
        # finger_data = self.get_finger_values() # position, velocity, current, sensor
        return {
            "joint_positions": joints,
            "ee_pose": robot_ee,
            "finger_positions": finger_pos,
            "robot_velocity": robot_velocity,
            "robot_current": robot_current,
            "fingertip_sensor": fingertip_sensor,
            "finger_current": finger_current, # Maybe can combine with the robot values, but not sure of data type compatibility
            "finger_velocity": finger_velocity,
            "finger_tcp": finger_tcp,
        }

class URInspire(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "192.168.1.10", no_gripper: bool = False):
        import rtde_control
        import rtde_receive

        [print("in ur robot") for _ in range(4)]
        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        if not no_gripper:
            from gello.robots.SRBL_Inspire import SRBL_Inspire_gripper 
            self.gripper = SRBL_Inspire_gripper() 

            print("gripper connected")
            # gripper.activate()

        [print("connect") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()
        self._use_gripper = not no_gripper

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 6

    def _get_gripper_pos(self) -> float: # CHANGE to corresponding gripper
        import time

        time.sleep(0.01)
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 1, "Gripper position must be between 0 and 1"
        return gripper_pos

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            gripper_pos = self._get_gripper_pos()
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500  # 2ms
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()
        self.robot.servoJ(
            robot_joints, velocity, acceleration, dt, lookahead_time, gain
        )
        if self._use_gripper:
            gripper_pos = int(joint_state[-1]) # Tesollo # CHANGE to corresponding gripper
            self.gripper.move(gripper_pos) # Tesollo # CHANGE to corresponding gripper
        self.robot.waitPeriod(t_start)

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_finger_sensor(self):
        if self._use_gripper:
            return self.gripper.get_sensor_values()
        else:
            return None

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]])
        fingertip_sensor = self.get_finger_sensor()
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
            "fingertip_sensor": fingertip_sensor,
        }