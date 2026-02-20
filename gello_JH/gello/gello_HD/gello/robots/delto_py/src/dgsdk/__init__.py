"""
DGSDK - Python wrapper for Delto Gripper SDK
"""

__version__ = "1.7.2"

from .wrapper import DGGripper
from .types import (
    # Constants
    MAX_JOINT_COUNT,
    MAX_FINGER_COUNT,
    MAX_FINGER_JOINT_COUNT,
    CARTESIAN_COORDINATE_POSE_COUNT,
    MAX_GRIPPER_IP_ADDRESS_SIZE,
    MAX_COMPORT_NAME_SIZE,
    MAX_BLEND_COUNT,
    MAX_RECIPE_POSE_COUNT,
    MAX_RECIPE_GAIN_COUNT,
    MAX_RECIPE_GRASP_COUNT,
    MAX_GRIPPER_GPIO_SIZE,
    PI,
    DEGREE_TO_RADIAN,
    RADIAN_TO_DEGREE,

    # Enums
    DGResult,
    DGModel,
    BlendMotionStatus,
    CommunicationMode,
    ControlMode,
    GainMode,
    DeveloperModeCommand,
    ReceivedDataType,
    DGGraspMode,
    DGGraspOption,
    DGDiagnosis,

    # Structures
    ReceivedGripperData,
    RecipeBlendData,
    GripperSystemSetting,
    GripperSetting,
    RecipePoseData,
    RecipeGainData,
    RecipeGraspData,
    ReceivedFingertipSensorData,
    ReceivedGPIOData,
    DiagnosisSystem,

    # Callback Types
    ReceivedGripperDatasCallback,
    ConnectedToGripperCallback,
    DisconnectedToGripperCallback,
    CommunicationPeriodCallback,
    DiagnosisSystemCallback,
    ReceivedSensorCallback,
    ReceivedGPIOCallback,
    DataProcessingCallback,
)

__all__ = [
    # Main class
    "DGGripper",

    # Version
    "__version__",

    # Constants
    "MAX_JOINT_COUNT",
    "MAX_FINGER_COUNT",
    "MAX_FINGER_JOINT_COUNT",
    "CARTESIAN_COORDINATE_POSE_COUNT",
    "MAX_GRIPPER_IP_ADDRESS_SIZE",
    "MAX_COMPORT_NAME_SIZE",
    "MAX_BLEND_COUNT",
    "MAX_RECIPE_POSE_COUNT",
    "MAX_RECIPE_GAIN_COUNT",
    "MAX_RECIPE_GRASP_COUNT",
    "MAX_GRIPPER_GPIO_SIZE",
    "PI",
    "DEGREE_TO_RADIAN",
    "RADIAN_TO_DEGREE",

    # Enums
    "DGResult",
    "DGModel",
    "BlendMotionStatus",
    "CommunicationMode",
    "ControlMode",
    "GainMode",
    "DeveloperModeCommand",
    "ReceivedDataType",
    "DGGraspMode",
    "DGGraspOption",
    "DGDiagnosis",

    # Structures
    "ReceivedGripperData",
    "RecipeBlendData",
    "GripperSystemSetting",
    "GripperSetting",
    "RecipePoseData",
    "RecipeGainData",
    "RecipeGraspData",
    "ReceivedFingertipSensorData",
    "ReceivedGPIOData",
    "DiagnosisSystem",

    # Callback Types
    "ReceivedGripperDatasCallback",
    "ConnectedToGripperCallback",
    "DisconnectedToGripperCallback",
    "CommunicationPeriodCallback",
    "DiagnosisSystemCallback",
    "ReceivedSensorCallback",
    "ReceivedGPIOCallback",
    "DataProcessingCallback",
]
