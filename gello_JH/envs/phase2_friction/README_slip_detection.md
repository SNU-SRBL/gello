# Slip Detection - Relative Velocity Method

## Overview

Position 기반 상대속도를 이용한 slip detection 구현. Fingertip과 object의 position 변화량을 비교하여 slip 여부를 판정합니다.

## Core Algorithm

```python
# 1. Position 기반 velocity 계산
fingertip_vel = (fingertip_pos[t] - fingertip_pos[t-1]) / dt
object_vel = (object_pos[t] - object_pos[t-1]) / dt

# 2. 상대속도 계산
relative_vel = object_vel - fingertip_vel

# 3. Contact 판정
is_contact = ||F_contact|| > force_threshold

# 4. Slip 판정
is_slipping = is_contact AND (||relative_vel|| > velocity_threshold)
```

## Key Design Decisions

### Position 기반 Velocity vs PhysX Velocity

| 방법 | 장점 | 단점 |
|------|------|------|
| **Position 기반** (채택) | 실제 변위 측정, noise 적음 | 1 step 지연 |
| PhysX `get_linear_velocity()` | 즉각적 | Numerical jitter (~2-3 cm/s) |

PhysX solver는 contact 상태에서 numerical solving 과정에서 실제로 움직이지 않아도 ~2-3 cm/s의 velocity를 지속적으로 보고합니다. Position 기반 계산은 실제 변위만 측정하므로 이 문제를 해결합니다.

### Contact Detection

F/T sensor의 joint reaction force를 사용하여 contact 판정:
- `SRBL_TesolloFTSensor`로 각 손가락별 resultant force 측정
- Force magnitude > threshold (default: 0.5N) 일 때 contact로 판정

## Files

| File | Description |
|------|-------------|
| `slip_detection.py` | `RelativeVelocitySlipDetector` 클래스 |
| `slip_data_collector.py` | 데이터 수집 및 저장 |
| `scripts/slip_test_env.py` | 테스트 환경 |

## Usage

### Test Environment

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
cd /home/Isaac/workspace/HD/Real2Sim

# Run sweep test (grip gradually tightens)
python scripts/slip_test_env.py --sweep_test --duration 10

# With custom thresholds
python scripts/slip_test_env.py --sweep_test --force_threshold 0.5 --velocity_threshold 0.01

# Save to specific file
python scripts/slip_test_env.py --sweep_test --output my_data.npz
```

### Using RelativeVelocitySlipDetector

```python
from slip_detection import RelativeVelocitySlipDetector

detector = RelativeVelocitySlipDetector(
    contact_force_threshold=0.5,   # N
    slip_velocity_threshold=0.01,  # m/s (1 cm/s)
    device="cuda:0",
    num_envs=1,
    num_fingers=5,
)

# In physics loop
rel_vel, is_contact, is_slipping = detector.update(
    fingertip_vel=fingertip_vel,  # (N, 5, 3)
    object_vel=object_vel,         # (N, 3)
    contact_forces=contact_forces, # (N, 5, 6)
    timestamp=sim_time,
)
```

## Data Format

Saved `.npz` file contains:

```
slip_data.npz
├── timestamps: (T,)              # Simulation time
├── fingertip_velocities: (T, 5, 3)  # Per-finger velocities
├── object_velocities: (T, 3)     # Object velocity
├── relative_velocities: (T, 5, 3)   # Per-finger relative velocities
├── contact_forces: (T, 5, 6)     # Per-finger forces [Fx,Fy,Fz,Tx,Ty,Tz]
├── is_contact: (T, 5)            # Contact status per finger
├── is_slipping: (T, 5)           # Slip status per finger
├── joint_positions: (T, 20)      # Hand joint positions (optional)
└── object_positions: (T, 3)      # Object positions (optional)
```

## Parameters

### RelativeVelocitySlipDetector

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contact_force_threshold` | 0.5 N | Minimum force for contact |
| `slip_velocity_threshold` | 0.01 m/s | Minimum relative velocity for slip |

### Test Environment (slip_test_env.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--force_threshold` | 0.5 | Contact force threshold (N) |
| `--velocity_threshold` | 0.01 | Slip velocity threshold (m/s) |
| `--duration` | 10.0 | Test duration (s) |
| `--sweep_test` | False | Enable automatic grip sweep |
| `--headless` | False | Run without GUI |

## Limitations

1. **Rolling 미구분**: Slip과 roll을 구분하지 않음 (상대속도만 측정)
2. **Single Object**: 현재 단일 object만 지원
3. **1-step Delay**: Position 기반 계산으로 1 physics step 지연

## Example Output

```
t=  5.0s | grip=1.00 | contact=1/5 | slip=0/5 | rel_vel=0.0062m/s
  [ring] tip_vel=0.0003 | obj_vel=0.0060 | rel=0.0062 | F=2.52N
t=  6.0s | grip=1.04 | contact=1/5 | slip=0/5 | rel_vel=0.0195m/s
  [ring] tip_vel=0.0003 | obj_vel=0.0160 | rel=0.0163 | F=0.84N
t=  9.0s | grip=1.16 | contact=2/5 | slip=2/5 | rel_vel=0.0275m/s [SLIP!]
  [ring] tip_vel=0.0002 | obj_vel=0.0250 | rel=0.0253 | F=0.50N
```

## References

- Tesollo F/T Sensor: `callback/tesollo_ft_sensor.py`
- Base Environment Config: `envs/base/real2sim_base_env_cfg.py`
