"""
DGSDK Python Wrapper
DGSDK.h 기반 완전한 바인딩 구현 (cffi 사용)
"""

import ctypes
import platform
from pathlib import Path
from typing import List, Optional, Callable, Union

from .types import (
    # Constants
    MAX_JOINT_COUNT, MAX_FINGER_COUNT, MAX_GRIPPER_GPIO_SIZE,
    MAX_RECEIVED_DATA_TYPE_COUNT,
    # Enums
    DGResult, DGModel, DGGraspMode, DGGraspOption, ControlMode, CommunicationMode,
    # Structures
    GripperSystemSetting, GripperSetting, ReceivedGripperData,
    RecipeBlendData, RecipePoseData, RecipeGainData, RecipeGraspData,
    ReceivedFingertipSensorData, ReceivedGPIOData, DiagnosisSystem,
    # Callbacks
    ReceivedGripperDatasCallback, ConnectedToGripperCallback,
    DisconnectedToGripperCallback, CommunicationPeriodCallback,
    DiagnosisSystemCallback, ReceivedSensorCallback,
    ReceivedGPIOCallback, DataProcessingCallback,
)


class DGGripper:
    """DGSDK 그리퍼 래퍼 클래스"""

    def __init__(self, lib_dir: Optional[Union[str, Path]] = None):
        """
        DGSDK 라이브러리 초기화

        Args:
            lib_dir: 라이브러리 디렉토리 경로 (기본값: 패키지 내 libs 폴더)
        """
        if lib_dir is None:
            # 패키지 내 libs 폴더 또는 상위 libs 폴더 탐색
            pkg_dir = Path(__file__).parent
            if (pkg_dir / "libs").exists():
                lib_dir = pkg_dir / "libs"
            elif (pkg_dir.parent.parent / "libs").exists():
                lib_dir = pkg_dir.parent.parent / "libs"
            else:
                lib_dir = pkg_dir / "libs"
        else:
            lib_dir = Path(lib_dir)

        self.lib_dir = lib_dir
        self._lib = None
        self._ffi = None
        self._callbacks = {}  # 콜백 레퍼런스 유지
        self._load_library()

    def _load_library(self):
        """운영체제에 맞는 라이브러리 로드"""
        system = platform.system()

        if system == "Windows":
            lib_path = self.lib_dir / "DGSDK.dll"
        elif system == "Linux":
            lib_path = self.lib_dir / "libDGSDK.so"
        else:
            raise OSError(f"지원되지 않는 운영체제: {system}")

        if not lib_path.exists():
            raise FileNotFoundError(f"라이브러리를 찾을 수 없습니다: {lib_path}")

        try:
            # cffi를 사용하여 라이브러리 로드
            from cffi import FFI
            self._ffi = FFI()

            # C 구조체 및 함수 선언
            self._ffi.cdef("""
                // Structs
                typedef struct {
                    char comport[32];
                    char ip[32];
                    int port;
                    int readTimeout;
                    int controlMode;
                    int communicationMode;
                    int slaveID;
                    int baudrate;
                } GripperSystemSetting;

                typedef struct {
                    float jointOffset[20];
                    float jointInpose[20];
                    float tcpInpose[5];
                    float orientationInpose[5];
                    int receivedDataType[6];
                    float movingInpose;
                    int jointCount;
                    int fingerCount;
                    int model;
                    int8_t dutyByteLength;
                } GripperSetting;

                typedef struct {
                    float joint[20];
                    int current[20];
                    int velocity[20];
                    float temperature[20];
                    float TCP[30];
                    int moving;
                    int targetArrived;
                    int blendMoveState;
                    int currentBlendIndex;
                    int productID;
                    int firmwareVersion;
                } ReceivedGripperData;

                typedef struct {
                    int recipePoseNumber;
                    int recipeGainNumber;
                    int blendWaitTime;
                    int number;
                } RecipeBlendData;

                typedef struct {
                    float forceTorque[30];
                } ReceivedFingertipSensorData;

                typedef struct {
                    int GPIO[4];
                } ReceivedGPIOData;

                typedef struct {
                    int process;
                    int step;
                    int jointId;
                    int period;
                    int joint;
                    int temperature;
                } DiagnosisSystem;

                // Callback types (matching header exactly)
                typedef void (*ReceivedGripperDatasCallback)(const ReceivedGripperData);
                typedef void (*ConnectedToGripperCallback)();
                typedef void (*DisconnectedToGripperCallback)();
                typedef void (*CommunicationPeriodCallback)(const int);
                typedef void (*DiagnosisSystemCallback)(const DiagnosisSystem);
                typedef void (*ReceivedSensorCallback)(const ReceivedFingertipSensorData);
                typedef void (*ReceivedGPIOCallback)(const ReceivedGPIOData);
                typedef void (*DataProcessingCallback)(const int);

                // System Setting Functions
                int SetGripperSystem(GripperSystemSetting setting);
                int SetGripperOption(GripperSetting setting);
                int ConnectToGripper(void);
                int DisconnectToGripper(void);
                int SystemStart(void);
                int SystemStop(void);
                int SetIp(int* ip, int port);

                // Callback Functions
                int CallbackForOnConnected(ConnectedToGripperCallback cb);
                int CallbackForOnDisconnected(DisconnectedToGripperCallback cb);
                int CallbackForOnReceivedGripperData(ReceivedGripperDatasCallback cb);
                int CallbackForOnCommunicationPeriod(CommunicationPeriodCallback cb);
                int CallbackForOnDiagnosisSystem(DiagnosisSystemCallback cb);
                int CallbackForOnReceivedFingertipSensorData(ReceivedSensorCallback cb);
                int CallbackForOnReceivedGPIOData(ReceivedGPIOCallback cb);
                int CallbackForOnDataProcessing(DataProcessingCallback cb);

                // Other System Functions
                int SetLowPassFilterAlpha(int isUsed, float alpha);
                int SetBootMode(char* password);
                int SetGPIOOuput(int gpio, int outputNumber);
                int SetGPIOOuputAll(int* output);
                int SetTorqueLimitMode(int isOn);
                int SetBootRecipe(int recipePoseNumber, int recipeGainNumber, int recipeGraspNumber);
                int EEPROMWrite(void);
                int SystemDiagnosis(void);
                int BackupRecipeData(char* path);
                int RestoreRecipeData(char* path);

                // Grasp Setting Functions
                int SetGraspData(int graspMode, float graspForce, int graspOption, int smoothGrasping);
                int SetGraspForce(float graspForce);

                // Joint Gain P Functions
                int SetJointGainP(float gainP, int jointNumber);
                int SetJointGainPFinger(float* gainP, int fingerNumber);
                int SetJointGainPBase(float* gainP);
                int SetJointGainPAll(float* gainP);

                // Joint Gain D Functions
                int SetJointGainD(float gainD, int jointNumber);
                int SetJointGainDFinger(float* gainD, int fingerNumber);
                int SetJointGainDBase(float* gainD);
                int SetJointGainDAll(float* gainD);

                // Joint Gain I Functions
                int SetJointGainI(float gainI, float iLimit, int jointNumber);
                int SetJointGainIFinger(float* gainI, float* iLimit, int fingerNumber);
                int SetJointGainIBase(float* gainI, float* iLimit);
                int SetJointGainIAll(float* gainI, float* iLimit);

                // Joint Gain PID Functions
                int SetControlPIDMode(int mode);
                int SetJointGainPID(float gainP, float gainD, float gainI, float iLimit, int jointNumber);
                int SetJointGainPIDFinger(float* gainP, float* gainD, float* gainI, float* iLimit, int fingerNumber);
                int SetJointGainPIDBase(float* gainP, float* gainD, float* gainI, float* iLimit);
                int SetJointGainPIDAll(float* gainP, float* gainD, float* gainI, float* iLimit);
                int SetJointGainPIDAllEqual(float gainP, float gainD, float gainI, float iLimit);

                // Motion Time Functions
                int SetMotionTimeJoint(int motionTime, int jointNumber);
                int SetMotionTimeFinger(int* motionTime, int fingerNumber);
                int SetMotionTimeBase(int* motionTime);
                int SetMotionTimeAll(int* motionTime);
                int SetMotionTimeAllEqual(int motionTime);

                // Position Mode Functions
                int SetPositionModeJoint(int positionMode, int jointNumber);
                int SetPositionModeFinger(int* positionMode, int fingerNumber);
                int SetPositionModeBase(int* positionMode);
                int SetPositionModeAll(int* positionMode);

                // Current Control Functions
                int SetCurrentControlMode(int isOn);
                int SetTargetCurrentJoint(int targetCurrent, int jointNumber);
                int SetTargetCurrentFinger(int* targetCurrent, int fingerNumber);
                int SetTargetCurrentBase(int* targetCurrent);
                int SetTargetCurrentAll(int* targetCurrent);

                // Recipe Functions
                int UpdateRecipeCurrentPoseData(int poseNumber);
                int UpdateRecipeTargetPoseData(int poseNumber);
                int LoadRecipePoseData(int poseNumber);
                int UpdateRecipeGainData(int gainNumber);
                int LoadRecipeGainData(int gainNumber);
                int UpdateRecipeGraspData(int graspNumber);
                int LoadRecipeGraspData(int graspNumber);

                // Blend Motion Functions
                int UpdateBlendJoint(RecipeBlendData blendData);
                int ClearMoveBlendJoint(void);
                int AddMoveBlendJoint(void);
                int SetMoveBlendJoint(int blendNumber);
                int StartMoveBlendJoint(int blendNumber);
                int StopMoveBlendJoint(void);

                // Motion Functions
                int ManualTeachMode(int isOn);
                int StartGraspMotion(int isGrasp);
                int MoveJoint(float targetJoint, int jointNumber);
                int MoveJointFinger(float* targetJoint, int fingerNumber);
                int MoveJointBase(float* targetJoint);
                int MoveJointAll(float* targetJoint);
                int MoveServoJoint(float* targetJoint);

                // TCP Functions
                int SetTCPGainPFinger(float* gainP, int fingerNumber);
                int SetTCPGainPAll(float* gainP);
                int SetTCPGainDFinger(float* gainD, int fingerNumber);
                int SetTCPGainDAll(float* gainD);
                int SetTCPGainIFinger(float* gainI, float* iLimit, int fingerNumber);
                int SetTCPGainIAll(float* gainI, float* iLimit);
                int SetTCPGainPIDFinger(float* gainP, float* gainD, float* gainI, float* iLimit, int fingerNumber);
                int SetTCPGainPIDAll(float* gainP, float* gainD, float* gainI, float* iLimit);
                int SetTCPMotionTimeFinger(int motionTime, int fingerNumber);
                int SetTCPMotionTimeAll(int* motionTime);
                int SetTCPMotionTimeAllEqual(int motionTime);
                int MoveTCPFinger(float* targetTCP, int fingerNumber);
                int MoveTCPAll(float* targetTCP);
                int GetCurrentTcpPose(float* currentTCP);
                int SetManipulationGainPIDAll(float* gainP, float* gainD, float* gainI, float* iLimit);
                int InHandManipulation(float* targetOffset, int motionTime);
                int SetFingerTipDataZero(void);

                // Getter Functions
                int GetReceivedGripperData(ReceivedGripperData* recvGripperData);
                int GetCommunicationPeriod(int* countPeriod);
                int GetReceivedFingertipSensorData(ReceivedFingertipSensorData* recvFingerTipData);
                int GetReceivedGPIOData(ReceivedGPIOData* recvGPIOData);
                int GetDataProcessing(int* status);
                int GetDiagnosisSystem(DiagnosisSystem* diagnosisData);
            """)

            self._lib = self._ffi.dlopen(str(lib_path))

        except Exception as e:
            raise OSError(f"라이브러리 로드 실패: {lib_path}\n{e}")

    @property
    def lib(self):
        """원시 cffi 라이브러리 객체"""
        return self._lib

    @property
    def ffi(self):
        """cffi FFI 객체"""
        return self._ffi

    # =========================================================================
    # System Setting Functions
    # =========================================================================

    def set_gripper_system(self, setting: GripperSystemSetting) -> DGResult:
        """
        그리퍼 시스템 설정 (가장 먼저 호출해야 함)

        Args:
            setting: GripperSystemSetting 구조체
        """
        # ctypes 구조체를 cffi 구조체로 변환
        cffi_setting = self._ffi.new("GripperSystemSetting *")
        cffi_setting.comport = setting.comport
        cffi_setting.ip = setting.ip
        cffi_setting.port = setting.port
        cffi_setting.readTimeout = setting.readTimeout
        cffi_setting.controlMode = setting.controlMode
        cffi_setting.communicationMode = setting.communicationMode
        cffi_setting.slaveID = setting.slaveID
        cffi_setting.baudrate = setting.baudrate

        result = self._lib.SetGripperSystem(cffi_setting[0])
        return DGResult(result)

    def set_gripper_option(self, setting: GripperSetting) -> DGResult:
        """
        그리퍼 옵션 설정 (시스템 설정 후 호출)

        Args:
            setting: GripperSetting 구조체
        """
        cffi_setting = self._ffi.new("GripperSetting *")

        for i in range(MAX_JOINT_COUNT):
            cffi_setting.jointOffset[i] = setting.jointOffset[i]
            cffi_setting.jointInpose[i] = setting.jointInpose[i]

        for i in range(MAX_FINGER_COUNT):
            cffi_setting.tcpInpose[i] = setting.tcpInpose[i]
            cffi_setting.orientationInpose[i] = setting.orientationInpose[i]

        for i in range(MAX_RECEIVED_DATA_TYPE_COUNT):
            cffi_setting.receivedDataType[i] = setting.receivedDataType[i]

        cffi_setting.movingInpose = setting.movingInpose
        cffi_setting.jointCount = setting.jointCount
        cffi_setting.fingerCount = setting.fingerCount
        cffi_setting.model = setting.model
        cffi_setting.dutyByteLength = setting.dutyByteLength

        result = self._lib.SetGripperOption(cffi_setting[0])
        return DGResult(result)

    def connect(self) -> DGResult:
        """그리퍼 연결"""
        return DGResult(self._lib.ConnectToGripper())

    def disconnect(self) -> DGResult:
        """그리퍼 연결 해제"""
        return DGResult(self._lib.DisconnectToGripper())

    def start(self) -> DGResult:
        """시스템 시작 (데이터 수신 및 제어 가능)"""
        return DGResult(self._lib.SystemStart())

    def stop(self) -> DGResult:
        """시스템 정지"""
        return DGResult(self._lib.SystemStop())

    def set_ip(self, ip: List[int], port: int) -> DGResult:
        """
        IP 및 포트 설정

        Args:
            ip: IP 주소 [192, 168, 1, 100]
            port: 포트 번호
        """
        ip_array = self._ffi.new("int[4]", ip)
        return DGResult(self._lib.SetIp(ip_array, port))

    # =========================================================================
    # Callback Functions
    # =========================================================================

    def on_connected(self, callback: Callable[[], None]) -> DGResult:
        """연결 시 콜백 등록"""
        cb = self._ffi.callback("void()", callback)
        self._callbacks['connected'] = cb
        return DGResult(self._lib.CallbackForOnConnected(cb))

    def on_disconnected(self, callback: Callable[[], None]) -> DGResult:
        """연결 해제 시 콜백 등록"""
        cb = self._ffi.callback("void()", callback)
        self._callbacks['disconnected'] = cb
        return DGResult(self._lib.CallbackForOnDisconnected(cb))

    def on_gripper_data(self, callback: Callable[[ReceivedGripperData], None]) -> DGResult:
        """그리퍼 데이터 수신 콜백 등록"""
        def wrapper(cffi_data):
            # cffi 구조체를 ctypes 구조체로 변환
            data = ReceivedGripperData()
            for i in range(MAX_JOINT_COUNT):
                data.joint[i] = cffi_data.joint[i]
                data.current[i] = cffi_data.current[i]
                data.velocity[i] = cffi_data.velocity[i]
                data.temperature[i] = cffi_data.temperature[i]
            for i in range(30):
                data.TCP[i] = cffi_data.TCP[i]
            data.moving = cffi_data.moving
            data.targetArrived = cffi_data.targetArrived
            data.blendMoveState = cffi_data.blendMoveState
            data.currentBlendIndex = cffi_data.currentBlendIndex
            data.productID = cffi_data.productID
            data.firmwareVersion = cffi_data.firmwareVersion
            callback(data)

        cb = self._ffi.callback("void(const ReceivedGripperData)", wrapper)
        self._callbacks['gripper_data'] = cb
        return DGResult(self._lib.CallbackForOnReceivedGripperData(cb))

    def on_communication_period(self, callback: Callable[[int], None]) -> DGResult:
        """통신 주기 콜백 등록"""
        cb = self._ffi.callback("void(const int)", callback)
        self._callbacks['comm_period'] = cb
        return DGResult(self._lib.CallbackForOnCommunicationPeriod(cb))

    def on_diagnosis(self, callback: Callable[[DiagnosisSystem], None]) -> DGResult:
        """진단 결과 콜백 등록"""
        def wrapper(cffi_data):
            data = DiagnosisSystem()
            data.process = cffi_data.process
            data.step = cffi_data.step
            data.jointId = cffi_data.jointId
            data.period = cffi_data.period
            data.joint = cffi_data.joint
            data.temperature = cffi_data.temperature
            callback(data)

        cb = self._ffi.callback("void(const DiagnosisSystem)", wrapper)
        self._callbacks['diagnosis'] = cb
        return DGResult(self._lib.CallbackForOnDiagnosisSystem(cb))

    def on_fingertip_sensor(self, callback: Callable[[ReceivedFingertipSensorData], None]) -> DGResult:
        """핑거팁 센서 데이터 콜백 등록"""
        def wrapper(cffi_data):
            data = ReceivedFingertipSensorData()
            for i in range(30):
                data.forceTorque[i] = cffi_data.forceTorque[i]
            callback(data)

        cb = self._ffi.callback("void(const ReceivedFingertipSensorData)", wrapper)
        self._callbacks['fingertip'] = cb
        return DGResult(self._lib.CallbackForOnReceivedFingertipSensorData(cb))

    def on_gpio(self, callback: Callable[[ReceivedGPIOData], None]) -> DGResult:
        """GPIO 데이터 콜백 등록"""
        def wrapper(cffi_data):
            data = ReceivedGPIOData()
            for i in range(4):
                data.GPIO[i] = cffi_data.GPIO[i]
            callback(data)

        cb = self._ffi.callback("void(const ReceivedGPIOData)", wrapper)
        self._callbacks['gpio'] = cb
        return DGResult(self._lib.CallbackForOnReceivedGPIOData(cb))

    def on_data_processing(self, callback: Callable[[int], None]) -> DGResult:
        """데이터 처리 상태 콜백 등록"""
        cb = self._ffi.callback("void(const int)", callback)
        self._callbacks['data_processing'] = cb
        return DGResult(self._lib.CallbackForOnDataProcessing(cb))

    # =========================================================================
    # Other System Functions
    # =========================================================================

    def set_low_pass_filter(self, is_used: int, alpha: float) -> DGResult:
        """로우패스 필터 설정"""
        return DGResult(self._lib.SetLowPassFilterAlpha(is_used, alpha))

    def set_boot_mode(self, password: str) -> DGResult:
        """펌웨어 다운로드용 부트 모드 진입"""
        return DGResult(self._lib.SetBootMode(password.encode()))

    def set_gpio_output(self, gpio: int, output_number: int) -> DGResult:
        """GPIO 출력 설정"""
        return DGResult(self._lib.SetGPIOOuput(gpio, output_number))

    def set_gpio_output_all(self, output: List[int]) -> DGResult:
        """전체 GPIO 출력 설정"""
        output_array = self._ffi.new("int[]", output)
        return DGResult(self._lib.SetGPIOOuputAll(output_array))

    def set_torque_limit_mode(self, is_on: int) -> DGResult:
        """토크 제한 모드 설정"""
        return DGResult(self._lib.SetTorqueLimitMode(is_on))

    def set_boot_recipe(self, pose_number: int, gain_number: int, grasp_number: int) -> DGResult:
        """부팅 시 시작할 레시피 설정"""
        return DGResult(self._lib.SetBootRecipe(pose_number, gain_number, grasp_number))

    def eeprom_write(self) -> DGResult:
        """EEPROM에 설정 저장"""
        return DGResult(self._lib.EEPROMWrite())

    def system_diagnosis(self) -> DGResult:
        """시스템 자가 진단 실행"""
        return DGResult(self._lib.SystemDiagnosis())

    def backup_recipe_data(self, path: str) -> DGResult:
        """레시피 데이터 백업"""
        return DGResult(self._lib.BackupRecipeData(path.encode()))

    def restore_recipe_data(self, path: str) -> DGResult:
        """레시피 데이터 복원"""
        return DGResult(self._lib.RestoreRecipeData(path.encode()))

    # =========================================================================
    # Grasp Setting Functions
    # =========================================================================

    def set_grasp_data(self, grasp_mode: DGGraspMode, grasp_force: float,
                       grasp_option: int, smooth_grasping: int) -> DGResult:
        """그립 데이터 설정"""
        return DGResult(self._lib.SetGraspData(int(grasp_mode), grasp_force, grasp_option, smooth_grasping))

    def set_grasp_force(self, grasp_force: float) -> DGResult:
        """그립 힘 변경"""
        return DGResult(self._lib.SetGraspForce(grasp_force))

    # =========================================================================
    # Joint Gain P Functions
    # =========================================================================

    def set_joint_gain_p(self, gain_p: float, joint_number: int) -> DGResult:
        """단일 조인트 P 게인 설정"""
        return DGResult(self._lib.SetJointGainP(gain_p, joint_number))

    def set_joint_gain_p_finger(self, gain_p: List[float], finger_number: int) -> DGResult:
        """단일 핑거 P 게인 설정"""
        gain_array = self._ffi.new("float[]", gain_p)
        return DGResult(self._lib.SetJointGainPFinger(gain_array, finger_number))

    def set_joint_gain_p_base(self, gain_p: List[float]) -> DGResult:
        """베이스 조인트 P 게인 설정"""
        gain_array = self._ffi.new("float[]", gain_p)
        return DGResult(self._lib.SetJointGainPBase(gain_array))

    def set_joint_gain_p_all(self, gain_p: List[float]) -> DGResult:
        """모든 조인트 P 게인 설정"""
        gain_array = self._ffi.new("float[]", gain_p)
        return DGResult(self._lib.SetJointGainPAll(gain_array))

    # =========================================================================
    # Joint Gain D Functions
    # =========================================================================

    def set_joint_gain_d(self, gain_d: float, joint_number: int) -> DGResult:
        """단일 조인트 D 게인 설정"""
        return DGResult(self._lib.SetJointGainD(gain_d, joint_number))

    def set_joint_gain_d_finger(self, gain_d: List[float], finger_number: int) -> DGResult:
        """단일 핑거 D 게인 설정"""
        gain_array = self._ffi.new("float[]", gain_d)
        return DGResult(self._lib.SetJointGainDFinger(gain_array, finger_number))

    def set_joint_gain_d_base(self, gain_d: List[float]) -> DGResult:
        """베이스 조인트 D 게인 설정"""
        gain_array = self._ffi.new("float[]", gain_d)
        return DGResult(self._lib.SetJointGainDBase(gain_array))

    def set_joint_gain_d_all(self, gain_d: List[float]) -> DGResult:
        """모든 조인트 D 게인 설정"""
        gain_array = self._ffi.new("float[]", gain_d)
        return DGResult(self._lib.SetJointGainDAll(gain_array))

    # =========================================================================
    # Joint Gain I Functions
    # =========================================================================

    def set_joint_gain_i(self, gain_i: float, i_limit: float, joint_number: int) -> DGResult:
        """단일 조인트 I 게인 설정"""
        return DGResult(self._lib.SetJointGainI(gain_i, i_limit, joint_number))

    def set_joint_gain_i_finger(self, gain_i: List[float], i_limit: List[float], finger_number: int) -> DGResult:
        """단일 핑거 I 게인 설정"""
        gain_array = self._ffi.new("float[]", gain_i)
        limit_array = self._ffi.new("float[]", i_limit)
        return DGResult(self._lib.SetJointGainIFinger(gain_array, limit_array, finger_number))

    def set_joint_gain_i_base(self, gain_i: List[float], i_limit: List[float]) -> DGResult:
        """베이스 조인트 I 게인 설정"""
        gain_array = self._ffi.new("float[]", gain_i)
        limit_array = self._ffi.new("float[]", i_limit)
        return DGResult(self._lib.SetJointGainIBase(gain_array, limit_array))

    def set_joint_gain_i_all(self, gain_i: List[float], i_limit: List[float]) -> DGResult:
        """모든 조인트 I 게인 설정"""
        gain_array = self._ffi.new("float[]", gain_i)
        limit_array = self._ffi.new("float[]", i_limit)
        return DGResult(self._lib.SetJointGainIAll(gain_array, limit_array))

    # =========================================================================
    # Joint Gain PID Functions
    # =========================================================================

    def set_control_pid_mode(self, mode: int) -> DGResult:
        """PID 제어 모드 설정"""
        return DGResult(self._lib.SetControlPIDMode(mode))

    def set_joint_gain_pid(self, gain_p: float, gain_d: float, gain_i: float,
                           i_limit: float, joint_number: int) -> DGResult:
        """단일 조인트 PID 게인 설정"""
        return DGResult(self._lib.SetJointGainPID(gain_p, gain_d, gain_i, i_limit, joint_number))

    def set_joint_gain_pid_finger(self, gain_p: List[float], gain_d: List[float],
                                   gain_i: List[float], i_limit: List[float],
                                   finger_number: int) -> DGResult:
        """단일 핑거 PID 게인 설정"""
        p_array = self._ffi.new("float[]", gain_p)
        d_array = self._ffi.new("float[]", gain_d)
        i_array = self._ffi.new("float[]", gain_i)
        limit_array = self._ffi.new("float[]", i_limit)
        return DGResult(self._lib.SetJointGainPIDFinger(p_array, d_array, i_array, limit_array, finger_number))

    def set_joint_gain_pid_base(self, gain_p: List[float], gain_d: List[float],
                                 gain_i: List[float], i_limit: List[float]) -> DGResult:
        """베이스 조인트 PID 게인 설정"""
        p_array = self._ffi.new("float[]", gain_p)
        d_array = self._ffi.new("float[]", gain_d)
        i_array = self._ffi.new("float[]", gain_i)
        limit_array = self._ffi.new("float[]", i_limit)
        return DGResult(self._lib.SetJointGainPIDBase(p_array, d_array, i_array, limit_array))

    def set_joint_gain_pid_all(self, gain_p: List[float], gain_d: List[float],
                                gain_i: List[float], i_limit: List[float]) -> DGResult:
        """모든 조인트 PID 게인 설정"""
        p_array = self._ffi.new("float[]", gain_p)
        d_array = self._ffi.new("float[]", gain_d)
        i_array = self._ffi.new("float[]", gain_i)
        limit_array = self._ffi.new("float[]", i_limit)
        return DGResult(self._lib.SetJointGainPIDAll(p_array, d_array, i_array, limit_array))

    def set_joint_gain_pid_all_equal(self, gain_p: float, gain_d: float,
                                      gain_i: float, i_limit: float) -> DGResult:
        """모든 조인트에 동일한 PID 게인 설정"""
        return DGResult(self._lib.SetJointGainPIDAllEqual(gain_p, gain_d, gain_i, i_limit))

    # =========================================================================
    # Motion Time Functions
    # =========================================================================

    def set_motion_time_joint(self, motion_time: int, joint_number: int) -> DGResult:
        """단일 조인트 동작 시간 설정"""
        return DGResult(self._lib.SetMotionTimeJoint(motion_time, joint_number))

    def set_motion_time_finger(self, motion_time: List[int], finger_number: int) -> DGResult:
        """단일 핑거 동작 시간 설정"""
        time_array = self._ffi.new("int[]", motion_time)
        return DGResult(self._lib.SetMotionTimeFinger(time_array, finger_number))

    def set_motion_time_base(self, motion_time: List[int]) -> DGResult:
        """베이스 조인트 동작 시간 설정"""
        time_array = self._ffi.new("int[]", motion_time)
        return DGResult(self._lib.SetMotionTimeBase(time_array))

    def set_motion_time_all(self, motion_time: List[int]) -> DGResult:
        """모든 조인트 동작 시간 설정"""
        time_array = self._ffi.new("int[]", motion_time)
        return DGResult(self._lib.SetMotionTimeAll(time_array))

    def set_motion_time_all_equal(self, motion_time: int) -> DGResult:
        """모든 조인트에 동일한 동작 시간 설정"""
        return DGResult(self._lib.SetMotionTimeAllEqual(motion_time))

    # =========================================================================
    # Position Mode Functions
    # =========================================================================

    def set_position_mode_joint(self, position_mode: int, joint_number: int) -> DGResult:
        """단일 조인트 위치 제어 모드 설정"""
        return DGResult(self._lib.SetPositionModeJoint(position_mode, joint_number))

    def set_position_mode_finger(self, position_mode: List[int], finger_number: int) -> DGResult:
        """단일 핑거 위치 제어 모드 설정"""
        mode_array = self._ffi.new("int[]", position_mode)
        return DGResult(self._lib.SetPositionModeFinger(mode_array, finger_number))

    def set_position_mode_base(self, position_mode: List[int]) -> DGResult:
        """베이스 조인트 위치 제어 모드 설정"""
        mode_array = self._ffi.new("int[]", position_mode)
        return DGResult(self._lib.SetPositionModeBase(mode_array))

    def set_position_mode_all(self, position_mode: List[int]) -> DGResult:
        """모든 조인트 위치 제어 모드 설정"""
        mode_array = self._ffi.new("int[]", position_mode)
        return DGResult(self._lib.SetPositionModeAll(mode_array))

    # =========================================================================
    # Current Control Functions
    # =========================================================================

    def set_current_control_mode(self, is_on: int) -> DGResult:
        """전류 기반 위치 제어 모드 설정"""
        return DGResult(self._lib.SetCurrentControlMode(is_on))

    def set_target_current_joint(self, target_current: int, joint_number: int) -> DGResult:
        """단일 조인트 목표 전류 설정"""
        return DGResult(self._lib.SetTargetCurrentJoint(target_current, joint_number))

    def set_target_current_finger(self, target_current: List[int], finger_number: int) -> DGResult:
        """단일 핑거 목표 전류 설정"""
        current_array = self._ffi.new("int[]", target_current)
        return DGResult(self._lib.SetTargetCurrentFinger(current_array, finger_number))

    def set_target_current_base(self, target_current: List[int]) -> DGResult:
        """베이스 조인트 목표 전류 설정"""
        current_array = self._ffi.new("int[]", target_current)
        return DGResult(self._lib.SetTargetCurrentBase(current_array))

    def set_target_current_all(self, target_current: List[int]) -> DGResult:
        """모든 조인트 목표 전류 설정"""
        current_array = self._ffi.new("int[]", target_current)
        return DGResult(self._lib.SetTargetCurrentAll(current_array))

    # =========================================================================
    # Recipe Functions
    # =========================================================================

    def update_recipe_current_pose(self, pose_number: int) -> DGResult:
        """현재 포즈를 레시피에 저장"""
        return DGResult(self._lib.UpdateRecipeCurrentPoseData(pose_number))

    def update_recipe_target_pose(self, pose_number: int) -> DGResult:
        """목표 포즈를 레시피에 저장"""
        return DGResult(self._lib.UpdateRecipeTargetPoseData(pose_number))

    def load_recipe_pose(self, pose_number: int) -> DGResult:
        """레시피에서 포즈 로드"""
        return DGResult(self._lib.LoadRecipePoseData(pose_number))

    def update_recipe_gain(self, gain_number: int) -> DGResult:
        """현재 게인을 레시피에 저장"""
        return DGResult(self._lib.UpdateRecipeGainData(gain_number))

    def load_recipe_gain(self, gain_number: int) -> DGResult:
        """레시피에서 게인 로드"""
        return DGResult(self._lib.LoadRecipeGainData(gain_number))

    def update_recipe_grasp(self, grasp_number: int) -> DGResult:
        """현재 그립 설정을 레시피에 저장"""
        return DGResult(self._lib.UpdateRecipeGraspData(grasp_number))

    def load_recipe_grasp(self, grasp_number: int) -> DGResult:
        """레시피에서 그립 설정 로드"""
        return DGResult(self._lib.LoadRecipeGraspData(grasp_number))

    # =========================================================================
    # Blend Motion Functions
    # =========================================================================

    def update_blend_joint(self, blend_data: RecipeBlendData) -> DGResult:
        """블렌드 모션에 데이터 추가"""
        cffi_data = self._ffi.new("RecipeBlendData *")
        cffi_data.recipePoseNumber = blend_data.recipePoseNumber
        cffi_data.recipeGainNumber = blend_data.recipeGainNumber
        cffi_data.blendWaitTime = blend_data.blendWaitTime
        cffi_data.number = blend_data.number
        return DGResult(self._lib.UpdateBlendJoint(cffi_data[0]))

    def clear_blend_joint(self) -> DGResult:
        """블렌드 모션 데이터 초기화"""
        return DGResult(self._lib.ClearMoveBlendJoint())

    def add_blend_joint(self) -> DGResult:
        """현재 블렌드 데이터를 리스트에 추가"""
        return DGResult(self._lib.AddMoveBlendJoint())

    def set_blend_joint(self, blend_number: int) -> DGResult:
        """블렌드 모션 번호 저장"""
        return DGResult(self._lib.SetMoveBlendJoint(blend_number))

    def start_blend_joint(self, blend_number: int) -> DGResult:
        """블렌드 모션 시작"""
        return DGResult(self._lib.StartMoveBlendJoint(blend_number))

    def stop_blend_joint(self) -> DGResult:
        """블렌드 모션 정지"""
        return DGResult(self._lib.StopMoveBlendJoint())

    # =========================================================================
    # Motion Functions
    # =========================================================================

    def manual_teach_mode(self, is_on: int) -> DGResult:
        """수동 티칭 모드 설정"""
        return DGResult(self._lib.ManualTeachMode(is_on))

    def grasp(self, is_grasp: int) -> DGResult:
        """그립 동작 수행"""
        return DGResult(self._lib.StartGraspMotion(is_grasp))

    def move_joint(self, target_joint: float, joint_number: int) -> DGResult:
        """단일 조인트 이동"""
        return DGResult(self._lib.MoveJoint(target_joint, joint_number))

    def move_joint_finger(self, target_joint: List[float], finger_number: int) -> DGResult:
        """단일 핑거 조인트 이동"""
        joint_array = self._ffi.new("float[]", target_joint)
        return DGResult(self._lib.MoveJointFinger(joint_array, finger_number))

    def move_joint_base(self, target_joint: List[float]) -> DGResult:
        """베이스 조인트 이동"""
        joint_array = self._ffi.new("float[]", target_joint)
        return DGResult(self._lib.MoveJointBase(joint_array))

    def move_joint_all(self, target_joint: List[float]) -> DGResult:
        """모든 조인트 이동"""
        joint_array = self._ffi.new("float[]", target_joint)
        return DGResult(self._lib.MoveJointAll(joint_array))

    def move_servo_joint(self, target_joint: List[float]) -> DGResult:
        """실시간 조인트 이동 (Developer Mode 전용)"""
        joint_array = self._ffi.new("float[]", target_joint)
        return DGResult(self._lib.MoveServoJoint(joint_array))

    # =========================================================================
    # TCP (Experimental) Functions
    # =========================================================================

    def set_tcp_gain_p_finger(self, gain_p: List[float], finger_number: int) -> DGResult:
        """TCP P 게인 설정 (핑거별)"""
        gain_array = self._ffi.new("float[]", gain_p)
        return DGResult(self._lib.SetTCPGainPFinger(gain_array, finger_number))

    def set_tcp_gain_p_all(self, gain_p: List[float]) -> DGResult:
        """TCP P 게인 설정 (전체)"""
        gain_array = self._ffi.new("float[]", gain_p)
        return DGResult(self._lib.SetTCPGainPAll(gain_array))

    def set_tcp_gain_d_finger(self, gain_d: List[float], finger_number: int) -> DGResult:
        """TCP D 게인 설정 (핑거별)"""
        gain_array = self._ffi.new("float[]", gain_d)
        return DGResult(self._lib.SetTCPGainDFinger(gain_array, finger_number))

    def set_tcp_gain_d_all(self, gain_d: List[float]) -> DGResult:
        """TCP D 게인 설정 (전체)"""
        gain_array = self._ffi.new("float[]", gain_d)
        return DGResult(self._lib.SetTCPGainDAll(gain_array))

    def set_tcp_gain_i_finger(self, gain_i: List[float], i_limit: List[float], finger_number: int) -> DGResult:
        """TCP I 게인 설정 (핑거별)"""
        gain_array = self._ffi.new("float[]", gain_i)
        limit_array = self._ffi.new("float[]", i_limit)
        return DGResult(self._lib.SetTCPGainIFinger(gain_array, limit_array, finger_number))

    def set_tcp_gain_i_all(self, gain_i: List[float], i_limit: List[float]) -> DGResult:
        """TCP I 게인 설정 (전체)"""
        gain_array = self._ffi.new("float[]", gain_i)
        limit_array = self._ffi.new("float[]", i_limit)
        return DGResult(self._lib.SetTCPGainIAll(gain_array, limit_array))

    def set_tcp_gain_pid_finger(self, gain_p: List[float], gain_d: List[float],
                                 gain_i: List[float], i_limit: List[float],
                                 finger_number: int) -> DGResult:
        """TCP PID 게인 설정 (핑거별)"""
        p_array = self._ffi.new("float[]", gain_p)
        d_array = self._ffi.new("float[]", gain_d)
        i_array = self._ffi.new("float[]", gain_i)
        limit_array = self._ffi.new("float[]", i_limit)
        return DGResult(self._lib.SetTCPGainPIDFinger(p_array, d_array, i_array, limit_array, finger_number))

    def set_tcp_gain_pid_all(self, gain_p: List[float], gain_d: List[float],
                              gain_i: List[float], i_limit: List[float]) -> DGResult:
        """TCP PID 게인 설정 (전체)"""
        p_array = self._ffi.new("float[]", gain_p)
        d_array = self._ffi.new("float[]", gain_d)
        i_array = self._ffi.new("float[]", gain_i)
        limit_array = self._ffi.new("float[]", i_limit)
        return DGResult(self._lib.SetTCPGainPIDAll(p_array, d_array, i_array, limit_array))

    def set_tcp_motion_time_finger(self, motion_time: int, finger_number: int) -> DGResult:
        """TCP 동작 시간 설정 (핑거별)"""
        return DGResult(self._lib.SetTCPMotionTimeFinger(motion_time, finger_number))

    def set_tcp_motion_time_all(self, motion_time: List[int]) -> DGResult:
        """TCP 동작 시간 설정 (전체)"""
        time_array = self._ffi.new("int[]", motion_time)
        return DGResult(self._lib.SetTCPMotionTimeAll(time_array))

    def set_tcp_motion_time_all_equal(self, motion_time: int) -> DGResult:
        """TCP 동작 시간 설정 (전체 동일)"""
        return DGResult(self._lib.SetTCPMotionTimeAllEqual(motion_time))

    def move_tcp_finger(self, target_tcp: List[float], finger_number: int) -> DGResult:
        """TCP 위치 제어 (핑거별)"""
        tcp_array = self._ffi.new("float[]", target_tcp)
        return DGResult(self._lib.MoveTCPFinger(tcp_array, finger_number))

    def move_tcp_all(self, target_tcp: List[float]) -> DGResult:
        """TCP 위치 제어 (전체)"""
        tcp_array = self._ffi.new("float[]", target_tcp)
        return DGResult(self._lib.MoveTCPAll(tcp_array))

    def get_current_tcp_pose(self) -> List[float]:
        """현재 TCP 좌표 가져오기"""
        tcp = self._ffi.new("float[30]")
        result = self._lib.GetCurrentTcpPose(tcp)
        if result != DGResult.NONE:
            raise RuntimeError(f"TCP 좌표 가져오기 실패: {DGResult(result).name}")
        return list(tcp)

    def set_manipulation_gain_pid_all(self, gain_p: List[float], gain_d: List[float],
                                       gain_i: List[float], i_limit: List[float]) -> DGResult:
        """인핸드 조작용 PID 게인 설정"""
        p_array = self._ffi.new("float[]", gain_p)
        d_array = self._ffi.new("float[]", gain_d)
        i_array = self._ffi.new("float[]", gain_i)
        limit_array = self._ffi.new("float[]", i_limit)
        return DGResult(self._lib.SetManipulationGainPIDAll(p_array, d_array, i_array, limit_array))

    def in_hand_manipulation(self, target_offset: List[float], motion_time: int) -> DGResult:
        """인핸드 조작"""
        offset_array = self._ffi.new("float[]", target_offset)
        return DGResult(self._lib.InHandManipulation(offset_array, motion_time))

    def set_fingertip_data_zero(self) -> DGResult:
        """핑거팁 센서 데이터 영점 조정"""
        return DGResult(self._lib.SetFingerTipDataZero())

    # =========================================================================
    # Getter Functions
    # =========================================================================

    def get_gripper_data(self) -> ReceivedGripperData:
        """그리퍼 데이터 가져오기"""
        cffi_data = self._ffi.new("ReceivedGripperData *")
        result = self._lib.GetReceivedGripperData(cffi_data)
        if result != DGResult.NONE:
            raise RuntimeError(f"데이터 가져오기 실패: {DGResult(result).name}")

        data = ReceivedGripperData()
        for i in range(MAX_JOINT_COUNT):
            data.joint[i] = cffi_data.joint[i]
            data.current[i] = cffi_data.current[i]
            data.velocity[i] = cffi_data.velocity[i]
            data.temperature[i] = cffi_data.temperature[i]
        for i in range(30):
            data.TCP[i] = cffi_data.TCP[i]
        data.moving = cffi_data.moving
        data.targetArrived = cffi_data.targetArrived
        data.blendMoveState = cffi_data.blendMoveState
        data.currentBlendIndex = cffi_data.currentBlendIndex
        data.productID = cffi_data.productID
        data.firmwareVersion = cffi_data.firmwareVersion
        return data

    def get_communication_period(self) -> int:
        """통신 주기 가져오기 (Hz)"""
        period = self._ffi.new("int *")
        result = self._lib.GetCommunicationPeriod(period)
        if result != DGResult.NONE:
            raise RuntimeError(f"통신 주기 가져오기 실패: {DGResult(result).name}")
        return period[0]

    def get_fingertip_sensor_data(self) -> ReceivedFingertipSensorData:
        """핑거팁 센서 데이터 가져오기"""
        cffi_data = self._ffi.new("ReceivedFingertipSensorData *")
        result = self._lib.GetReceivedFingertipSensorData(cffi_data)
        if result != DGResult.NONE:
            raise RuntimeError(f"센서 데이터 가져오기 실패: {DGResult(result).name}")

        data = ReceivedFingertipSensorData()
        for i in range(30):
            data.forceTorque[i] = cffi_data.forceTorque[i]
        return data

    def get_gpio_data(self) -> ReceivedGPIOData:
        """GPIO 데이터 가져오기"""
        cffi_data = self._ffi.new("ReceivedGPIOData *")
        result = self._lib.GetReceivedGPIOData(cffi_data)
        if result != DGResult.NONE:
            raise RuntimeError(f"GPIO 데이터 가져오기 실패: {DGResult(result).name}")

        data = ReceivedGPIOData()
        for i in range(4):
            data.GPIO[i] = cffi_data.GPIO[i]
        return data

    def get_data_processing(self) -> int:
        """데이터 처리 상태 가져오기"""
        status = self._ffi.new("int *")
        result = self._lib.GetDataProcessing(status)
        if result != DGResult.NONE:
            raise RuntimeError(f"처리 상태 가져오기 실패: {DGResult(result).name}")
        return status[0]

    def get_diagnosis_system(self) -> DiagnosisSystem:
        """진단 시스템 데이터 가져오기"""
        cffi_data = self._ffi.new("DiagnosisSystem *")
        result = self._lib.GetDiagnosisSystem(cffi_data)
        if result != DGResult.NONE:
            raise RuntimeError(f"진단 데이터 가져오기 실패: {DGResult(result).name}")

        data = DiagnosisSystem()
        data.process = cffi_data.process
        data.step = cffi_data.step
        data.jointId = cffi_data.jointId
        data.period = cffi_data.period
        data.joint = cffi_data.joint
        data.temperature = cffi_data.temperature
        return data

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.disconnect()
        return False
