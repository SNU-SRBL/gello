"""
DGSDK 데이터 타입 정의
DGDataTypes.h 기반으로 완전히 구현
"""

import ctypes
from enum import IntEnum


# =============================================================================
# Constants
# =============================================================================
MAX_JOINT_COUNT = 20
MAX_FINGER_COUNT = 5
MAX_FINGER_JOINT_COUNT = 4
CARTESIAN_COORDINATE_POSE_COUNT = 6
MAX_RECEIVED_DATA_SIZE = 1024
MAX_GRIPPER_IP_ADDRESS_SIZE = 32
MAX_GRIPPER_IP_BYTE_LENGTH = 4
MAX_COMPORT_NAME_SIZE = 32
MAX_BLEND_COUNT = 10
MAX_BLEND_ADD_POSE_COUNT = 50
MAX_RECIPE_POSE_COUNT = 100
MAX_RECIPE_GAIN_COUNT = 20
MAX_RECIPE_GRASP_COUNT = 20
MAX_GRASP_OPTION_COUNT = 2
MAX_GRIPPER_GPIO_SIZE = 4
MAX_RECEIVED_DATA_TYPE_COUNT = 6
MAX_BASE_JOINT_COUNT = 2

PI = 3.1415927
DEGREE_TO_RADIAN = PI / 180.0
RADIAN_TO_DEGREE = 180.0 / PI


# =============================================================================
# Enums
# =============================================================================
class DGResult(IntEnum):
    """SDK 결과 코드"""
    NONE = 0
    SYSTEM_SETTING_NOT_PERFORMED = 1

    VALUE_IS_NEGATIVE = 100
    OVERFLOW_JOINT_COUNT = 101
    OVERFLOW_FINGER_COUNT = 102
    OVERFLOW_TCP_COUNT = 103
    OVERFLOW_ADD_BLEND_SIZE = 105
    OVERFLOW_RECIPE_BLEND_SIZE = 106
    BLEND_SIZE_ZERO = 107
    NOT_SUPPORTED_MODEL = 108
    DATA_IS_ZERO = 109
    DATA_IS_NOT_BOOLEAN = 110
    NOT_FOUND_MODEL = 111
    OVERFLOW_RECIPE_POSE_COUNT = 112
    OVERFLOW_RECIPE_GAIN_COUNT = 113
    OVERFLOW_RECIPE_GRASP_COUNT = 114
    INVALID_CONTROL_MODE = 115
    INVALID_GRASP_MODE_DATA = 116
    OVERFLOW_GRASP_OPTION_DATA = 117
    OVERFLOW_MAX_BYTE_DATA = 118
    OVERFLOW_CURRENT_LIMIT = 119
    IS_NOT_TORQUE_CONTROL_MODE = 120
    NOT_SUPPORTED_DATA_TYPE = 121
    BACKUP_START = 122
    INVALID_PASSWORD = 123
    OVERFLOW_GPIO_COUNT = 124
    OVERFLOW_GRASP_FORCE = 125
    OVERFLOW_BLEND_WAIT_TIME = 126
    NOT_FOUND_RESTORE_DATA = 127
    RESTORE_START = 128

    NOT_ARRIVED = 200
    NOT_START_BLEND_MOVE = 201
    ALREADY_BLEND_MOVE_STATE = 202
    ALREADY_TCP_MOVE = 203
    RECIPE_IS_NOT_JOINT_MODE = 204
    ACTIVATE_CURRENT_CONTROL_MODE = 205
    ACTIVATE_GRASP_MOTION = 206
    ACTIVATE_MANUAL_CONTROL_MODE = 207

    SOCK_EXCEPTION = 500
    SOCK_FAILED_WSA_START_UP = 501

    NO_GRASP_OBJECT = 1002
    _3F_ONLY_SUPPORTED_3FINGER = 1003
    NOT_SUPPORTED_CONTROL_MODE_OPERATOR = 1004
    NOT_SUPPORTED_CONTROL_MODE_DEVELOPER = 1005
    NOT_SUPPORTED_PORT_NUM = 1006

    PORT_EXCEPTION = 2000
    PORT_FAILED_START_UP = 2001
    PORT_SET_CONFIG_ERROR = 2002
    PORT_SET_TIMEOUT_ERROR = 2003
    PORT_SET_COMMASK_ERROR = 2004

    DIAGNOSING_SYSTEM = 2009


class DGModel(IntEnum):
    """그리퍼 모델"""
    NONE = 0x0000
    DG_1F_M = 0x1F02
    DG_2F_M = 0x2F02
    DG_3F_B = 0x3F01
    DG_3F_M = 0x3F02
    DG_4F_M = 0x4F02
    DG_5F_LEFT = 0x5F12
    DG_5F_RIGHT = 0x5F22


class BlendMotionStatus(IntEnum):
    """블렌드 동작 상태"""
    STOP = 0
    RUN = 1
    COMPLETE = 2


class CommunicationMode(IntEnum):
    """통신 모드"""
    ETHERNET = 0
    RS485 = 1


class ControlMode(IntEnum):
    """제어 모드"""
    OPERATOR = 0
    DEVELOPER = 1


class GainMode(IntEnum):
    """PID 제어 모드"""
    PD = 0
    PID = 1


class DeveloperModeCommand(IntEnum):
    """개발자 모드 명령"""
    GET_DATA = 0x01
    SET_RECEIVED_DATA = 0x02
    SET_DUTY = 0x05
    SET_GPIO = 0x06
    SET_IP_ADDRESS = 0x07
    GET_ID_AND_VERSION = 0x08
    GET_JOINT_ID = 0x09
    SET_BOOT_MODE = 0x0A
    SET_FT_OFFSET_ZERO = 0x0B


class ReceivedDataType(IntEnum):
    """수신 데이터 타입"""
    JOINT = 0x01
    CURRENT = 0x02
    TEMPERATURE = 0x03
    VELOCITY = 0x04
    FINGER_FT_SENSOR = 0x05
    GPIO = 0x06


class DGGraspMode(IntEnum):
    """그립 모드"""
    NONE = 0

    # 3F modes
    _3F_3FINGER = 1
    _3F_2FINGER_1_AND_2 = 2
    _3F_2FINGER_1_AND_3 = 3
    _3F_2FINGER_2_AND_3 = 4
    _3F_3FINGER_PARALLEL = 5
    _3F_3FINGER_ENVELOP = 6

    # 5F modes
    _5F_5FINGER = 21
    _5F_3FINGER = 22
    _5F_3FINGER_PARALLEL = 23
    _5F_2FINGER_1_AND_2 = 24
    _5F_2FINGER_1_AND_3 = 25
    _5F_2FINGER_1_AND_4 = 26
    _5F_2FINGER_1_AND_5 = 27
    _5F_5FINGER_PARALLEL = 28
    _5F_5FINGER_ENVELOP = 29

    # 4F modes
    _4F_4FINGER = 31
    _4F_4FINGER_PARALLEL = 32
    _4F_4FINGER_ENVELOP = 33
    _4F_4FINGER_RIGHT_PARALLEL = 34
    _4F_4FINGER_LEFT_PARALLEL = 35
    _4F_2FINGER_1_AND_2 = 36
    _4F_2FINGER_1_AND_4 = 37
    _4F_2FINGER_3_AND_4 = 38

    # 2F modes
    _2F_2FINGER = 51
    _2F_2FINGER_ENVELOP = 52


class DGGraspOption(IntEnum):
    """그립 옵션"""
    NONE = 0
    PARALLEL = 1
    FIX_TILT = 2
    SET_TILT = 3


class DGDiagnosis(IntEnum):
    """진단 상태"""
    STEP_JOINT_ID = 1
    STEP_PERIOD = 2
    STEP_TEMPERATURE = 3
    STEP_JOINT = 4

    RESULT_NONE = 5
    RESULT_STANDBY = 6

    RESULT_OK = 100
    RESULT_FAULT = 200


# =============================================================================
# Structures
# =============================================================================
class ReceivedGripperData(ctypes.Structure):
    """그리퍼에서 수신하는 데이터"""
    _pack_ = 1
    _fields_ = [
        ("joint", ctypes.c_float * MAX_JOINT_COUNT),
        ("current", ctypes.c_int * MAX_JOINT_COUNT),
        ("velocity", ctypes.c_int * MAX_JOINT_COUNT),
        ("temperature", ctypes.c_float * MAX_JOINT_COUNT),
        ("TCP", ctypes.c_float * (6 * MAX_FINGER_COUNT)),
        ("moving", ctypes.c_int),
        ("targetArrived", ctypes.c_int),
        ("blendMoveState", ctypes.c_int),
        ("currentBlendIndex", ctypes.c_int),
        ("productID", ctypes.c_int),
        ("firmwareVersion", ctypes.c_int),
    ]

    def to_dict(self):
        """딕셔너리로 변환"""
        return {
            "joint": list(self.joint),
            "current": list(self.current),
            "velocity": list(self.velocity),
            "temperature": list(self.temperature),
            "TCP": list(self.TCP),
            "moving": self.moving,
            "targetArrived": self.targetArrived,
            "blendMoveState": self.blendMoveState,
            "currentBlendIndex": self.currentBlendIndex,
            "productID": self.productID,
            "firmwareVersion": self.firmwareVersion,
        }


class RecipeBlendData(ctypes.Structure):
    """블렌드 모션용 레시피 설정"""
    _pack_ = 1
    _fields_ = [
        ("recipePoseNumber", ctypes.c_int),
        ("recipeGainNumber", ctypes.c_int),
        ("blendWaitTime", ctypes.c_int),
        ("number", ctypes.c_int),
    ]


class GripperSystemSetting(ctypes.Structure):
    """그리퍼 시스템 설정"""
    _pack_ = 1
    _fields_ = [
        ("comport", ctypes.c_char * MAX_COMPORT_NAME_SIZE),
        ("ip", ctypes.c_char * MAX_GRIPPER_IP_ADDRESS_SIZE),
        ("port", ctypes.c_int),
        ("readTimeout", ctypes.c_int),
        ("controlMode", ctypes.c_int),
        ("communicationMode", ctypes.c_int),
        ("slaveID", ctypes.c_int),
        ("baudrate", ctypes.c_int),
    ]

    @classmethod
    def create(cls, ip="192.168.1.100", port=8000,
               control_mode=ControlMode.OPERATOR,
               communication_mode=CommunicationMode.ETHERNET,
               read_timeout=1000, slave_id=1, baudrate=115200,
               comport="COM1"):
        """설정 생성 헬퍼"""
        setting = cls()
        setting.ip = ip.encode() if isinstance(ip, str) else ip
        setting.port = port
        setting.controlMode = control_mode
        setting.communicationMode = communication_mode
        setting.readTimeout = read_timeout
        setting.slaveID = slave_id
        setting.baudrate = baudrate
        setting.comport = comport.encode() if isinstance(comport, str) else comport
        return setting


class GripperSetting(ctypes.Structure):
    """그리퍼 옵션 설정"""
    _pack_ = 1
    _fields_ = [
        ("jointOffset", ctypes.c_float * MAX_JOINT_COUNT),
        ("jointInpose", ctypes.c_float * MAX_JOINT_COUNT),
        ("tcpInpose", ctypes.c_float * MAX_FINGER_COUNT),
        ("orientationInpose", ctypes.c_float * MAX_FINGER_COUNT),
        ("receivedDataType", ctypes.c_int * MAX_RECEIVED_DATA_TYPE_COUNT),
        ("movingInpose", ctypes.c_float),
        ("jointCount", ctypes.c_int),
        ("fingerCount", ctypes.c_int),
        ("model", ctypes.c_int),
        ("dutyByteLength", ctypes.c_int8),
    ]

    @classmethod
    def create(cls, model, joint_count, finger_count,
               moving_inpose=0.5, received_data_type=None):
        """
        설정 생성 헬퍼

        Args:
            model: DGModel enum (예: DGModel.DG_3F_B)
            joint_count: 조인트 수
            finger_count: 핑거 수
            moving_inpose: 이동 판정 각도 (기본값: 0.5)
            received_data_type: 수신 데이터 타입 리스트 (기본값: [JOINT, CURRENT])
        """
        setting = cls()
        setting.model = int(model)
        setting.jointCount = joint_count
        setting.fingerCount = finger_count
        setting.movingInpose = moving_inpose

        # 기본값 초기화
        for i in range(MAX_JOINT_COUNT):
            setting.jointOffset[i] = 0.0
            setting.jointInpose[i] = 0.0

        for i in range(MAX_FINGER_COUNT):
            setting.tcpInpose[i] = 0.0
            setting.orientationInpose[i] = 0.0

        # 수신 데이터 타입 설정
        if received_data_type is None:
            received_data_type = [1, 2, 0, 0, 0, 0]  # JOINT, CURRENT
        for i, val in enumerate(received_data_type):
            if i < MAX_RECEIVED_DATA_TYPE_COUNT:
                setting.receivedDataType[i] = val

        return setting


class RecipePoseData(ctypes.Structure):
    """레시피 포즈 데이터"""
    _pack_ = 1
    _fields_ = [
        ("targetJoint", ctypes.c_float * MAX_JOINT_COUNT),
        ("jointMotionTime", ctypes.c_int * MAX_JOINT_COUNT),
        ("number", ctypes.c_int),
        ("mode", ctypes.c_int),
    ]


class RecipeGainData(ctypes.Structure):
    """레시피 게인 데이터"""
    _pack_ = 1
    _fields_ = [
        ("gainP", ctypes.c_float * MAX_JOINT_COUNT),
        ("gainD", ctypes.c_float * MAX_JOINT_COUNT),
        ("gainI", ctypes.c_float * MAX_JOINT_COUNT),
        ("iLimit", ctypes.c_float * MAX_JOINT_COUNT),
        ("controlPIDMode", ctypes.c_int),
        ("number", ctypes.c_int),
        ("mode", ctypes.c_int),
    ]


class RecipeGraspData(ctypes.Structure):
    """레시피 그립 데이터"""
    _pack_ = 1
    _fields_ = [
        ("graspForce", ctypes.c_float),
        ("postionMode", ctypes.c_int * MAX_JOINT_COUNT),
        ("graspMode", ctypes.c_int),
        ("graspOption", ctypes.c_int),
        ("smoothGrasping", ctypes.c_int),
        ("number", ctypes.c_int),
    ]


class ReceivedFingertipSensorData(ctypes.Structure):
    """핑거팁 센서 데이터"""
    _pack_ = 1
    _fields_ = [
        ("forceTorque", ctypes.c_float * (6 * MAX_FINGER_COUNT)),
    ]

    def to_dict(self):
        """딕셔너리로 변환"""
        return {"forceTorque": list(self.forceTorque)}


class ReceivedGPIOData(ctypes.Structure):
    """GPIO 데이터"""
    _pack_ = 1
    _fields_ = [
        ("GPIO", ctypes.c_int * MAX_GRIPPER_GPIO_SIZE),
    ]

    def to_dict(self):
        """딕셔너리로 변환"""
        return {"GPIO": list(self.GPIO)}


class DiagnosisSystem(ctypes.Structure):
    """진단 시스템 데이터"""
    _pack_ = 1
    _fields_ = [
        ("process", ctypes.c_int),
        ("step", ctypes.c_int),
        ("jointId", ctypes.c_int),
        ("period", ctypes.c_int),
        ("joint", ctypes.c_int),
        ("temperature", ctypes.c_int),
    ]

    def to_dict(self):
        """딕셔너리로 변환"""
        return {
            "process": self.process,
            "step": self.step,
            "jointId": self.jointId,
            "period": self.period,
            "joint": self.joint,
            "temperature": self.temperature,
        }


# =============================================================================
# Callback Types
# =============================================================================
ReceivedGripperDatasCallback = ctypes.CFUNCTYPE(None, ReceivedGripperData)
ConnectedToGripperCallback = ctypes.CFUNCTYPE(None)
DisconnectedToGripperCallback = ctypes.CFUNCTYPE(None)
CommunicationPeriodCallback = ctypes.CFUNCTYPE(None, ctypes.c_int)
DiagnosisSystemCallback = ctypes.CFUNCTYPE(None, DiagnosisSystem)
ReceivedSensorCallback = ctypes.CFUNCTYPE(None, ReceivedFingertipSensorData)
ReceivedGPIOCallback = ctypes.CFUNCTYPE(None, ReceivedGPIOData)
DataProcessingCallback = ctypes.CFUNCTYPE(None, ctypes.c_int)
