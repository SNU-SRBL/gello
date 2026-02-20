# Copyright (c) 2025, SRBL
# Phase 0: Robot configuration for system identification

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field


class RobotType(Enum):
    """Robot type for Phase 0 system identification."""
    UR5 = "ur5"
    HAND = "hand"


@dataclass
class UR5Config:
    """UR5 robot configuration."""
    num_joints: int = 6
    num_ee: int = 1  # Tool flange

    joint_names: list[str] = field(default_factory=lambda: [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ])

    ee_body_name: str = "wrist_3_link"

    # Default parameter bounds
    stiffness_bounds: tuple[float, float] = (100.0, 10000.0)
    damping_bounds: tuple[float, float] = (10.0, 1000.0)

    # Default initial values
    initial_stiffness: float = 1000.0
    initial_damping: float = 100.0


@dataclass
class FingerConfig:
    """Single finger configuration."""
    name: str
    joint_indices: list[int]
    ee_index: int  # Index in fingertip arrays
    joint_names: list[str]
    ee_body_name: str


# Tesollo DG5F Hand finger configurations
FINGER_CONFIGS = {
    "thumb": FingerConfig(
        name="thumb",
        joint_indices=[0, 1, 2, 3],
        ee_index=0,
        joint_names=["thumb_j1", "thumb_j2", "thumb_j3", "thumb_j4"],
        ee_body_name="thumb_tip",
    ),
    "index": FingerConfig(
        name="index",
        joint_indices=[4, 5, 6, 7],
        ee_index=1,
        joint_names=["index_j1", "index_j2", "index_j3", "index_j4"],
        ee_body_name="index_tip",
    ),
    "middle": FingerConfig(
        name="middle",
        joint_indices=[8, 9, 10, 11],
        ee_index=2,
        joint_names=["middle_j1", "middle_j2", "middle_j3", "middle_j4"],
        ee_body_name="middle_tip",
    ),
    "ring": FingerConfig(
        name="ring",
        joint_indices=[12, 13, 14, 15],
        ee_index=3,
        joint_names=["ring_j1", "ring_j2", "ring_j3", "ring_j4"],
        ee_body_name="ring_tip",
    ),
    "pinky": FingerConfig(
        name="pinky",
        joint_indices=[16, 17, 18, 19],
        ee_index=4,
        joint_names=["pinky_j1", "pinky_j2", "pinky_j3", "pinky_j4"],
        ee_body_name="pinky_tip",
    ),
}


@dataclass
class HandConfig:
    """Hand robot configuration (Tesollo DG5F)."""
    num_joints: int = 20
    num_ee: int = 5  # 5 fingertips
    num_fingers: int = 5
    joints_per_finger: int = 4

    fingers: dict = field(default_factory=lambda: FINGER_CONFIGS)

    ee_body_names: list[str] = field(default_factory=lambda: [
        "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"
    ])

    # Default parameter bounds
    stiffness_bounds: tuple[float, float] = (1.0, 100.0)
    damping_bounds: tuple[float, float] = (0.1, 10.0)

    # Default initial values
    initial_stiffness: float = 10.0
    initial_damping: float = 1.0

    def get_finger_config(self, finger_name: str) -> FingerConfig:
        """Get configuration for a specific finger."""
        return self.fingers[finger_name]

    def get_finger_joint_indices(self, finger_name: str) -> list[int]:
        """Get joint indices for a specific finger."""
        return self.fingers[finger_name].joint_indices

    def get_finger_ee_index(self, finger_name: str) -> int:
        """Get EE index for a specific finger."""
        return self.fingers[finger_name].ee_index


# Global config instances
UR5_CONFIG = UR5Config()
HAND_CONFIG = HandConfig()


def get_robot_config(robot_type: RobotType) -> UR5Config | HandConfig:
    """Get robot configuration by type."""
    if robot_type == RobotType.UR5:
        return UR5_CONFIG
    elif robot_type == RobotType.HAND:
        return HAND_CONFIG
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")
