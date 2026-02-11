"""
DGSDK 테스트
"""
import time
import sys
from pathlib import Path

# 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dgsdk import (
    DGGripper,
    GripperSystemSetting,
    GripperSetting,
    ControlMode,
    CommunicationMode,
    DGModel,
    DGGraspMode,
    DGResult,
    __version__,
)


def test_library_loading():
    """라이브러리 로딩 테스트"""
    print("=" * 60)
    print("DGSDK 라이브러리 로딩 테스트")
    print("=" * 60)

    try:
        gripper = DGGripper()
        print(f"✓ 라이브러리 로드 성공")
        print(f"  버전: {__version__}")
        print(f"  라이브러리: {gripper._lib}")
        return gripper
    except Exception as e:
        print(f"✗ 라이브러리 로드 실패: {e}")
        return None


def test_system_setting():
    """시스템 설정 테스트"""
    print("\n" + "=" * 60)
    print("시스템 설정 구조체 테스트")
    print("=" * 60)

    # 헬퍼 메서드로 생성
    setting = GripperSystemSetting.create(
        # ip="192.168.1.100",
        ip="169.254.186.73",
        port=502,
        control_mode=ControlMode.DEVELOPER,
        communication_mode=CommunicationMode.ETHERNET,
        read_timeout=1000,
    )

    print(f"✓ 시스템 설정 생성 완료")
    print(f"  IP: {setting.ip.decode()}")
    print(f"  Port: {setting.port}")
    print(f"  Control Mode: {ControlMode(setting.controlMode).name}")
    print(f"  Communication Mode: {CommunicationMode(setting.communicationMode).name}")
    print(f"  Read Timeout: {setting.readTimeout}ms")

    return setting


def test_gripper_setting():
    """그리퍼 옵션 설정 테스트"""
    print("\n" + "=" * 60)
    print("그리퍼 옵션 설정 테스트")
    print("=" * 60)

    # 헬퍼 메서드로 생성
    setting = GripperSetting.create(
        model=DGModel.DG_5F_RIGHT,
        joint_count=12,
        finger_count=3,
        moving_inpose=0.5,
    )

    print(f"✓ 그리퍼 옵션 생성 완료")
    print(f"  Model: {hex(setting.model)} ({DGModel(setting.model).name})")
    print(f"  Joint Count: {setting.jointCount}")
    print(f"  Finger Count: {setting.fingerCount}")
    print(f"  Moving Inpose: {setting.movingInpose}")
    print(f"  Received Data Type: {list(setting.receivedDataType)}")

    return setting


def test_enums():
    """Enum 테스트"""
    print("\n" + "=" * 60)
    print("Enum 테스트")
    print("=" * 60)

    print(f"✓ DGResult.NONE = {DGResult.NONE}")
    print(f"✓ DGResult.SYSTEM_SETTING_NOT_PERFORMED = {DGResult.SYSTEM_SETTING_NOT_PERFORMED}")

    print(f"✓ DGGraspMode._3F_3FINGER = {DGGraspMode._3F_3FINGER}")
    print(f"✓ DGGraspMode._5F_5FINGER = {DGGraspMode._5F_5FINGER}")

    print(f"✓ DGModel.DG_3F_B = {hex(DGModel.DG_3F_B)}")
    print(f"✓ DGModel.DG_5F_LEFT = {hex(DGModel.DG_5F_LEFT)}")

    print(f"✓ ControlMode.OPERATOR = {ControlMode.OPERATOR}")
    print(f"✓ ControlMode.DEVELOPER = {ControlMode.DEVELOPER}")


def test_gripper_methods():
    """그리퍼 메서드 존재 확인"""
    print("\n" + "=" * 60)
    print("그리퍼 메서드 확인")
    print("=" * 60)

    gripper = DGGripper()

    methods = [
        # System
        "set_gripper_system", "set_gripper_option", "connect", "disconnect",
        "start", "stop", "set_ip",
        # Callbacks
        "on_connected", "on_disconnected", "on_gripper_data",
        "on_communication_period", "on_diagnosis", "on_fingertip_sensor",
        "on_gpio", "on_data_processing",
        # Motion
        "grasp", "move_joint", "move_joint_all", "move_servo_joint",
        "manual_teach_mode",
        # Gain
        "set_joint_gain_p", "set_joint_gain_p_all",
        "set_joint_gain_d", "set_joint_gain_d_all",
        "set_joint_gain_i", "set_joint_gain_i_all",
        "set_joint_gain_pid", "set_joint_gain_pid_all",
        # TCP
        "move_tcp_finger", "move_tcp_all", "get_current_tcp_pose",
        # Getters
        "get_gripper_data", "get_communication_period",
        "get_fingertip_sensor_data", "get_gpio_data",
    ]

    for method in methods:
        if hasattr(gripper, method):
            print(f"✓ {method}")
        else:
            print(f"✗ {method} (missing)")

def connect_callback():
    isConnected = True
    print("Connected Gripper")

# def receive_gripper_data(ReceivedGripperData):
    

def main():
    """메인 테스트 함수"""
    print(f"\nDGSDK Python Wrapper v{__version__}")
    print("=" * 60)

    test_library_loading()
    test_system_setting()
    test_gripper_setting()
    test_enums()
    test_gripper_methods()
    
if __name__ == "__main__":
    main()
