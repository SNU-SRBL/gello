#!/usr/bin/env python3
"""
DGSDK 기본 연결 예제

연결 시퀀스:
1. set_gripper_system() - 시스템 설정
2. connect()            - 그리퍼 연결
3. set_gripper_option() - 그리퍼 옵션 설정
4. start()              - 시스템 시작
"""

import sys
import time
from pathlib import Path

# 개발 모드: 패키지 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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

def main():
    # 그리퍼 초기화
    gripper = DGGripper()
    print("✓ 라이브러리 로드 완료")

    try:
        # =====================================================
        # 1. 시스템 설정
        # =====================================================
        system_setting = GripperSystemSetting.create(
            ip="169.254.186.72",
            port=502,
            control_mode=ControlMode.OPERATOR,  # OPERATOR 모드
            communication_mode=CommunicationMode.ETHERNET,
            read_timeout=1000,
            slave_id=1,
            baudrate=115200,
        )

        result = gripper.set_gripper_system(system_setting)
        print(f"1. 시스템 설정: {result.name}")

        if result != DGResult.NONE:
            print(f"   오류: {result.name}")
            return

        # =====================================================
        # 1.5 콜백 설정 (연결 전에 설정해야 함)
        # =====================================================
        connected = False
        def on_connected():
            nonlocal connected
            connected = True
            print("   -> 연결 콜백 수신!")

        gripper.on_connected(on_connected)
        gripper.on_disconnected(lambda: print("   -> 연결 해제 콜백 수신!"))

        # =====================================================
        # 2. 그리퍼 연결
        # =====================================================
        print("2. 그리퍼 연결 시도 중...", flush=True)
        result = gripper.connect()
        print(f"2. 그리퍼 연결: {result.name}")

        if result != DGResult.NONE:
            print(f"   오류: {result.name}")
            return

        # =====================================================
        # 3. 그리퍼 옵션 설정
        # =====================================================
        gripper_setting = GripperSetting.create(
            model=DGModel.DG_5F_LEFT,  # 실제 그리퍼 모델에 맞게 변경
            joint_count=20,
            finger_count=5,
            moving_inpose=0.5,
        )

        result = gripper.set_gripper_option(gripper_setting)
        print(f"3. 그리퍼 옵션: {result.name}")

        if result != DGResult.NONE:
            print(f"   오류: {result.name}")
            return

        time.sleep(0.5)  # 설정 적용 대기

        # =====================================================
        # 4. 시스템 시작
        # =====================================================
        result = gripper.start()
        if result != DGResult.NONE:
            print(f"   오류: {result.name}")

        print(f"4. 시스템 시작: {result.name}")

        if result != DGResult.NONE:
            print(f"   오류: {result.name}")
            return

        print("\n✓ 연결 완료! 그리퍼 제어 가능")

        # =====================================================
        # 데이터 읽기 예제
        # =====================================================
        time.sleep(0.5)

        try:
            data = gripper.get_gripper_data()
            print(f"\n그리퍼 데이터:")
            print(f"  Joint: {list(data.joint[:12])}")
            print(f"  Current: {list(data.current[:12])}")
            print(f"  Moving: {data.moving}")
            print(f"  Product ID: {hex(data.productID)}")
        except RuntimeError as e:
            print(f"  데이터 읽기 실패: {e}")

        # =====================================================
        # 조인트 이동 예제
        # =====================================================
        print("\n조인트 이동 중...")
        target = [0.0] * 12
        # target.append(45.0)  # 12번째 조인트를 45도로 이동
        result = gripper.move_joint_all(target)
        print(f"  결과: {result.name}")

        time.sleep(1.0)

    except Exception as e:
        print(f"\n오류 발생: {e}")

    finally:
        # =====================================================
        # 연결 종료
        # =====================================================
        # print("\n연결 종료 중...")
        # gripper.stop()

        while True:
            try:
                data = gripper.get_gripper_data()
                print(f"\n그리퍼 데이터:")
                print(f"  Joint: " + " ".join(f"{x:2g}," for x in data.joint))
                print(f"  Current: {list(data.current[:12])}")
                print(f"  Moving: {data.moving}")
                print(f"  Product ID: {hex(data.productID)}")
            except RuntimeError as e:
                print(f"  데이터 읽기 실패: {e}")
                
            time.sleep(0.1)
            
        gripper.disconnect()
        print("✓ 연결 종료 완료")

if __name__ == "__main__":
    main()
