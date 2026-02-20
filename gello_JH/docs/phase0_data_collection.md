# Phase 0: 데이터 수집 가이드

## 개요

SIMPLER-style System Identification을 위해 **free motion 궤적**을 수집합니다.
접촉 없이 로봇이 자유롭게 움직이는 동안 joint state와 EE pose를 기록합니다.

---

## 1. 필수 데이터 필드

### UR5 캘리브레이션

| 필드명 | Shape | 단위 | 설명 |
|--------|-------|------|------|
| `timestamps` | (T,) | seconds | 시간 스탬프 |
| `ur5_joint_positions` | (T, 6) | radians | 6 joint 위치 |
| `ur5_joint_velocities` | (T, 6) | rad/s | 6 joint 속도 |
| `ur5_ee_position` | (T, 3) | meters | Tool flange 위치 (x, y, z) |
| `ur5_ee_orientation` | (T, 4) | quaternion | Tool flange 회전 (w, x, y, z) |
| `actions` | (T, 6) | - | 로봇에 보낸 명령 (position/torque) |

### Hand 캘리브레이션

| 필드명 | Shape | 단위 | 설명 |
|--------|-------|------|------|
| `timestamps` | (T,) | seconds | 시간 스탬프 |
| `hand_joint_positions` | (T, 20) | radians | 20 joint 위치 |
| `hand_joint_velocities` | (T, 20) | rad/s | 20 joint 속도 |
| `fingertip_positions` | (T, 5, 3) | meters | 5 fingertip 위치 |
| `fingertip_orientations` | (T, 5, 4) | quaternion | 5 fingertip 회전 (w, x, y, z) |
| `actions` | (T, 20) | - | 로봇에 보낸 명령 |

---

## 2. Fingertip 순서 (Hand)

```
Index 0: Thumb tip
Index 1: Index finger tip
Index 2: Middle finger tip
Index 3: Ring finger tip
Index 4: Pinky finger tip
```

### Joint 순서 (20 DOF)

```
[0:4]   Thumb:  j1, j2, j3, j4
[4:8]   Index:  j1, j2, j3, j4
[8:12]  Middle: j1, j2, j3, j4
[12:16] Ring:   j1, j2, j3, j4
[16:20] Pinky:  j1, j2, j3, j4
```

---

## 3. 수집 방법

### 3.1 UR5 데이터 수집

```python
import numpy as np
from ur_rtde import RTDEReceiveInterface

# UR5 연결
rtde_r = RTDEReceiveInterface("192.168.1.100")

# 데이터 버퍼
timestamps = []
joint_positions = []
joint_velocities = []
ee_positions = []
ee_orientations = []
actions = []

# 수집 루프
dt = 0.002  # 500Hz
start_time = time.time()

while collecting:
    t = time.time() - start_time

    # Joint state
    q = rtde_r.getActualQ()           # 6 joint positions
    qd = rtde_r.getActualQd()         # 6 joint velocities

    # EE pose (tool flange)
    tcp = rtde_r.getActualTCPPose()   # [x, y, z, rx, ry, rz]
    pos = tcp[:3]
    rot_vec = tcp[3:6]
    quat = rotation_vector_to_quaternion(rot_vec)  # → (w, x, y, z)

    # 명령 (position control인 경우)
    cmd = rtde_r.getTargetQ()

    # 저장
    timestamps.append(t)
    joint_positions.append(q)
    joint_velocities.append(qd)
    ee_positions.append(pos)
    ee_orientations.append(quat)
    actions.append(cmd)

    time.sleep(dt)

# NumPy 변환
data = {
    "timestamps": np.array(timestamps),
    "ur5_joint_positions": np.array(joint_positions),
    "ur5_joint_velocities": np.array(joint_velocities),
    "ur5_ee_position": np.array(ee_positions),
    "ur5_ee_orientation": np.array(ee_orientations),
    "actions": np.array(actions),
}
```

### 3.2 Hand 데이터 수집

```python
import numpy as np
from tesollo_sdk import TesolloHand

hand = TesolloHand()

# 데이터 버퍼
timestamps = []
joint_positions = []
joint_velocities = []
fingertip_positions = []
fingertip_orientations = []
actions = []

dt = 0.002  # 500Hz
start_time = time.time()

while collecting:
    t = time.time() - start_time

    # Joint state (20 DOF)
    q = hand.get_joint_positions()      # shape: (20,)
    qd = hand.get_joint_velocities()    # shape: (20,)

    # Fingertip poses (5 fingers)
    # 방법 1: Forward Kinematics로 계산
    ft_pos, ft_rot = hand.get_fingertip_poses()  # (5, 3), (5, 4)

    # 방법 2: 외부 모션캡처 사용 (더 정확)
    # ft_pos, ft_rot = mocap.get_fingertip_poses()

    # 명령
    cmd = hand.get_target_positions()   # shape: (20,)

    # 저장
    timestamps.append(t)
    joint_positions.append(q)
    joint_velocities.append(qd)
    fingertip_positions.append(ft_pos)
    fingertip_orientations.append(ft_rot)
    actions.append(cmd)

    time.sleep(dt)

# NumPy 변환
data = {
    "timestamps": np.array(timestamps),
    "hand_joint_positions": np.array(joint_positions),
    "hand_joint_velocities": np.array(joint_velocities),
    "fingertip_positions": np.array(fingertip_positions),      # (T, 5, 3)
    "fingertip_orientations": np.array(fingertip_orientations), # (T, 5, 4)
    "actions": np.array(actions),
}
```

---

## 4. Fingertip Pose 획득 방법

### 방법 1: Forward Kinematics (FK)

```python
def compute_fingertip_fk(joint_positions, urdf_path):
    """Joint positions → Fingertip poses via FK."""
    import pybullet as p

    # Load URDF
    robot_id = p.loadURDF(urdf_path)

    fingertip_links = [
        "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"
    ]

    # Set joint positions
    for i, q in enumerate(joint_positions):
        p.resetJointState(robot_id, i, q)

    # Get link states
    positions = []
    orientations = []
    for link_name in fingertip_links:
        link_idx = get_link_index(robot_id, link_name)
        state = p.getLinkState(robot_id, link_idx)
        positions.append(state[0])      # (x, y, z)
        orientations.append(state[1])   # (x, y, z, w) → 변환 필요

    return np.array(positions), np.array(orientations)
```

**장점**: 추가 장비 불필요
**단점**: URDF 정확도에 의존, 캘리브레이션 목적에는 circular dependency

### 방법 2: Motion Capture (권장)

```python
def get_fingertip_poses_mocap(optitrack_client):
    """OptiTrack/Vicon으로 fingertip marker 위치 획득."""
    markers = optitrack_client.get_rigid_bodies([
        "thumb_marker", "index_marker", "middle_marker",
        "ring_marker", "pinky_marker"
    ])

    positions = []
    orientations = []
    for marker in markers:
        positions.append(marker.position)        # (x, y, z)
        orientations.append(marker.orientation)  # (w, x, y, z)

    return np.array(positions), np.array(orientations)
```

**장점**: Ground truth EE pose
**단점**: 추가 장비 필요, marker 부착 필요

### 방법 3: RGB-D Camera + Hand Pose Estimation

```python
def get_fingertip_poses_vision(camera, hand_pose_model):
    """RGB-D 카메라로 fingertip 위치 추정."""
    rgb, depth = camera.get_frames()

    # MediaPipe, FrankMocap 등 사용
    hand_landmarks = hand_pose_model.predict(rgb, depth)

    fingertip_indices = [4, 8, 12, 16, 20]  # MediaPipe landmark indices
    positions = []
    for idx in fingertip_indices:
        pos_3d = hand_landmarks[idx]  # (x, y, z) in camera frame
        positions.append(pos_3d)

    # Orientation은 인접 joints로 추정하거나 identity 사용
    orientations = np.tile([1, 0, 0, 0], (5, 1))  # (w, x, y, z)

    return np.array(positions), orientations
```

**장점**: 마커 불필요
**단점**: 정확도 낮을 수 있음, orientation 추정 어려움

---

## 5. 수집할 궤적 유형

### Free Motion Trajectories (필수)

접촉 없이 로봇이 자유롭게 움직이는 궤적:

```python
# 예시: 랜덤 joint space 궤적
def generate_free_motion_trajectory(duration=10.0, num_waypoints=20):
    """무작위 waypoint 사이를 부드럽게 이동."""
    waypoints = []
    for _ in range(num_waypoints):
        q = np.random.uniform(joint_limits_low, joint_limits_high)
        waypoints.append(q)

    # Smooth interpolation
    trajectory = interpolate_trajectory(waypoints, duration)
    return trajectory
```

### 권장 궤적 패턴

1. **Slow sweep**: 각 joint를 천천히 min→max 이동
2. **Random exploration**: 랜덤 waypoint 연결
3. **Sinusoidal**: 각 joint에 서로 다른 주파수 sine wave
4. **Chirp**: 점점 빨라지는 움직임 (dynamics 테스트)

```python
# Sinusoidal trajectory example
def sinusoidal_trajectory(t, amplitudes, frequencies, phases):
    """Multi-frequency sinusoidal trajectory."""
    q = []
    for amp, freq, phase in zip(amplitudes, frequencies, phases):
        q.append(amp * np.sin(2 * np.pi * freq * t + phase))
    return np.array(q)
```

---

## 6. 저장 형식

### HDF5 저장

```python
from Real2Sim.data.storage import TrajectoryStorage, TrajectoryData

# 데이터 객체 생성
trajectory = TrajectoryData(
    timestamps=timestamps,
    hand_joint_positions=joint_positions,
    hand_joint_velocities=joint_velocities,
    fingertip_positions=fingertip_positions,
    fingertip_orientations=fingertip_orientations,
    actions=actions,
    metadata={
        "robot": "tesollo_dg5f",
        "date": "2025-01-30",
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "sampling_rate_hz": 500,
        "trajectory_type": "free_motion",
        "ee_source": "mocap",  # or "fk", "vision"
    }
)

# 저장
storage = TrajectoryStorage("data/real_data/phase0_hand")
storage.save(trajectory, "free_motion_001.h5")
```

---

## 7. 체크리스트

### 수집 전
- [ ] 로봇 캘리브레이션 확인 (joint encoder, FK)
- [ ] EE pose 획득 방법 결정 (FK / Mocap / Vision)
- [ ] Quaternion convention 확인 (wxyz vs xyzw)
- [ ] 샘플링 레이트 설정 (권장: 200-500Hz)

### 수집 중
- [ ] 접촉 없이 free motion만 수집
- [ ] 모든 joint가 움직이는 궤적 포함
- [ ] 다양한 속도의 궤적 포함 (slow/fast)
- [ ] 최소 5-10개 궤적 (총 60초 이상)

### 수집 후
- [ ] 데이터 shape 확인
- [ ] NaN/Inf 체크
- [ ] Quaternion 정규화 확인 (||q|| = 1)
- [ ] 시각화로 sanity check

---

## 8. 권장 수집량

| Robot | 궤적 수 | 궤적당 길이 | 총 데이터 |
|-------|---------|------------|----------|
| UR5 | 5-10개 | 10-30초 | 1-5분 |
| Hand | 10-20개 | 5-15초 | 2-5분 |

**참고**: 더 많은 데이터 = 더 robust한 캘리브레이션

---

## 9. 예시 스크립트

완전한 수집 스크립트: `scripts/collect_phase0_data.py` (TODO)

```bash
# UR5 데이터 수집
python scripts/collect_phase0_data.py \
    --robot ur5 \
    --output_dir data/real_data/phase0_ur5 \
    --duration 30 \
    --num_trajectories 10

# Hand 데이터 수집
python scripts/collect_phase0_data.py \
    --robot hand \
    --output_dir data/real_data/phase0_hand \
    --duration 15 \
    --num_trajectories 20 \
    --ee_source mocap
```
