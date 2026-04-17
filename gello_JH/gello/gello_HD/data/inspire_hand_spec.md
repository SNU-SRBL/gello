# Inspire RH56F Hand Specification

## Channel-Finger Mapping

| Channel | Index | Finger |
|---------|-------|--------|
| Ch1 | 0 | Little (새끼) |
| Ch2 | 1 | Ring (약지) |
| Ch3 | 2 | Middle (중지) |
| Ch4 | 3 | Index (검지) |
| Ch5 | 4 | Thumb Bending (엄지 굽힘) |
| Ch6 | 5 | Thumb Rotation (엄지 회전) |

## Joint Position Range

단위: 0.1도 (값 / 10 = 도)

| Joint | Lower (접힘) | Upper (펴짐) | Lower (deg) | Upper (deg) |
|-------|-------------|-------------|-------------|-------------|
| Little | 900 | 1740 | 90 | 174 |
| Ring | 900 | 1740 | 90 | 174 |
| Middle | 900 | 1740 | 90 | 174 |
| Index | 900 | 1740 | 90 | 174 |
| Thumb Bending | 1100 | 1350 | 110 | 135 |
| Thumb Rotation | 600 | 1800 | 60 | 180 |

- **숫자가 클수록 펴진 상태 (UPPER = 펴짐)**
- **숫자가 작을수록 접힌 상태 (LOWER = 접힘)**

## Communication

- Serial: `/dev/ttyUSB0` (환경에 따라 다를 수 있음)
- Baudrate: 115200
- Protocol: Custom register read/write (header: 0xEB 0x90)

## Key Registers

| Register | Address | Description |
|----------|---------|-------------|
| angleSet | 1040 | 위치 명령 (0.1도, -1이면 변경 없음) |
| angleAct | 1064 | 현재 위치 읽기 (0.1도) |
| speedSet | 1052 | 속도 설정 |
| forceSet | 1046 | 힘 설정 |
| forceAct | 1070 | 현재 힘 읽기 |
| currAct | 1076 | 전류 읽기 (mA) |
| sensorData | 3000 | 촉각 센서 데이터 (68 bytes) |

## Speed

- 기본값: 2000
- 단위: 미확인 (0.1도/초 추정)

## Sensor Data Layout (register 3000, 68 bytes)

각 손가락 10 bytes (5 fingers x 10 = 50 bytes) + 손바닥 18 bytes (3 x 6)

Per finger: normal(2) + tangential(2) + tangential_dir(2) + proximity(3) + reserved(1)

## Source Code

- 제어 클래스: `SRBL_Inspire_copy.py` (by Seongjun Koh, SNU SRBL)
- API: `get_position_values()`, `move_fingers()`, `get_sensor_values()`, `get_current_values()`, `get_all_once()`
