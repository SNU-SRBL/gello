# DGSDK

Delto Gripper SDK의 공식 Python 래퍼입니다.

운영체제(Linux/Windows)에 맞는 라이브러리를 자동으로 로드하여 Python에서 사용할 수 있습니다.

## 설치

### uv 사용 (권장)

```bash
uv add dgsdk
```

### pip 사용

```bash
pip install dgsdk
```

### 개발 모드 설치

```bash
git clone https://github.com/tesollo/dgsdk-python.git
cd dgsdk-python
uv sync
```

## 빠른 시작

```python
from dgsdk import (
    DGGripper, GripperSystemSetting, GripperSetting,
    ControlMode, CommunicationMode, DGModel, DGGraspMode
)

gripper = DGGripper()

# 1. 시스템 설정
system_setting = GripperSystemSetting.create(
    ip="169.254.5.72",
    port=502,
    control_mode=ControlMode.DEVELOPER,
    communication_mode=CommunicationMode.ETHERNET,
    read_timeout=1000,
    slave_id=1,
    baudrate=115200,
)
gripper.set_gripper_system(system_setting)

# 2. 그리퍼 연결
gripper.connect()

# 3. 그리퍼 옵션 설정
gripper_setting = GripperSetting.create(
    model=DGModel.DG_3F_B,
    joint_count=12,
    finger_count=3,
)
gripper.set_gripper_option(gripper_setting)

# 4. 시스템 시작
gripper.start()

# 조인트 이동
gripper.move_joint_all([0.0, 45.0, 30.0] * 4)

# 그립 동작
gripper.set_grasp_data(DGGraspMode._3F_3FINGER, grasp_force=100.0, grasp_option=0, smooth_grasping=1)
gripper.grasp(1)  # 그립

# 연결 종료
gripper.stop()
gripper.disconnect()
```

### 연결 시퀀스 (중요!)

```
1. set_gripper_system()  → 시스템 설정 (IP, 포트, 통신모드)
2. connect()             → 그리퍼 연결
3. set_gripper_option()  → 그리퍼 옵션 (모델, 조인트 수)
4. start()               → 시스템 시작
```

**반드시 이 순서를 지켜야 합니다!**

### Context Manager 사용

```python
from dgsdk import DGGripper, GripperSystemSetting, GripperSetting, DGModel

with DGGripper() as gripper:
    # 1. 시스템 설정
    system_setting = GripperSystemSetting.create(ip="169.254.5.72", port=502)
    gripper.set_gripper_system(system_setting)

    # 2. 연결
    gripper.connect()

    # 3. 옵션 설정
    gripper_setting = GripperSetting.create(DGModel.DG_3F_B, joint_count=12, finger_count=3)
    gripper.set_gripper_option(gripper_setting)

    # 4. 시작
    gripper.start()

    gripper.move_joint_all([0.0, 45.0, 30.0] * 4)
    # 자동으로 stop() 및 disconnect() 호출
```

## API 개요

### 시스템 함수

| 함수 | 설명 |
|------|------|
| `set_gripper_system(setting)` | 그리퍼 시스템 설정 |
| `set_gripper_option(setting)` | 그리퍼 옵션 설정 |
| `connect()` | 그리퍼 연결 |
| `disconnect()` | 그리퍼 연결 해제 |
| `start()` | 시스템 시작 |
| `stop()` | 시스템 정지 |

### 모션 함수

| 함수 | 설명 |
|------|------|
| `move_joint(target, joint_number)` | 단일 조인트 이동 |
| `move_joint_all(targets)` | 모든 조인트 이동 |
| `move_joint_finger(targets, finger_number)` | 핑거 조인트 이동 |
| `grasp(is_grasp)` | 그립 동작 (1=그립, 0=릴리스) |
| `manual_teach_mode(is_on)` | 수동 티칭 모드 |

### 설정 함수

| 함수 | 설명 |
|------|------|
| `set_grasp_data(mode, force, option, smooth)` | 그립 데이터 설정 |
| `set_grasp_force(force)` | 그립 힘 설정 |
| `set_joint_gain_pid_all(p, d, i, limit)` | PID 게인 설정 |
| `set_motion_time_all(times)` | 동작 시간 설정 |

### 데이터 함수

| 함수 | 설명 |
|------|------|
| `get_gripper_data()` | 그리퍼 데이터 가져오기 |
| `get_current_tcp_pose()` | 현재 TCP 좌표 |
| `get_communication_period()` | 통신 주기 (Hz) |
| `get_fingertip_sensor_data()` | 핑거팁 센서 데이터 |

### 콜백 함수

```python
def on_data(data):
    print(f"Joint: {list(data.joint)}")

gripper.on_gripper_data(on_data)
gripper.on_connected(lambda: print("Connected!"))
gripper.on_disconnected(lambda: print("Disconnected!"))
```

## 지원 모델

| 모델 | 설명 |
|------|------|
| DG-1F-M | 1핑거 |
| DG-2F-M | 2핑거 |
| DG-3F-B | 3핑거 (Basic) |
| DG-3F-M | 3핑거 (Multi) |
| DG-4F-M | 4핑거 |
| DG-5F-LEFT | 5핑거 (왼손) |
| DG-5F-RIGHT | 5핑거 (오른손) |

## 그립 모드

### 3F 모델

- `DGGraspMode._3F_3FINGER` - 3핑거 그립
- `DGGraspMode._3F_2FINGER_1_AND_2` - 2핑거 그립 (1, 2)
- `DGGraspMode._3F_3FINGER_PARALLEL` - 3핑거 평행 그립
- `DGGraspMode._3F_3FINGER_ENVELOP` - 3핑거 감싸기 그립

### 5F 모델

- `DGGraspMode._5F_5FINGER` - 5핑거 그립
- `DGGraspMode._5F_3FINGER` - 3핑거 그립
- `DGGraspMode._5F_5FINGER_PARALLEL` - 5핑거 평행 그립

## 프로젝트 구조

```
dgsdk/
├── pyproject.toml
├── README.md
├── libs/
│   ├── DGSDK.dll          # Windows
│   ├── libDGSDK.so        # Linux
│   ├── DGSDK.h
│   └── DGDataTypes.h
├── src/
│   └── dgsdk/
│       ├── __init__.py
│       ├── wrapper.py     # DGGripper 클래스
│       └── types.py       # 구조체, Enum 정의
└── tests/
    └── test_dgsdk.py
```

## 지원 플랫폼

- ✅ Linux (libDGSDK.so)
- ✅ Windows (DGSDK.dll)
- ❌ macOS (미지원)

## 요구사항

- Python 3.8+
- ctypes (표준 라이브러리)

## 라이센스

MIT License

## 링크

- [GitHub](https://github.com/tesollo/dgsdk-python)
- [Delto Gripper](https://www.tesollo.com)
