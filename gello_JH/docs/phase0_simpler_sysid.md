# Phase 0: SIMPLER-style System Identification

## Overview

Phase 0은 [SIMPLER 논문](https://simpler-env.github.io/) 방식을 따라 로봇의 joint dynamics 파라미터(stiffness, damping)를 최적화합니다.

**핵심 아이디어**: Joint 위치 에러만 보는 것이 아니라, **End-Effector (EE) pose tracking**을 통해 실제 작업 공간에서의 정확도를 최적화합니다.

### 현재 상태 (2026-02-11)

| 대상 | 스크립트 | 상태 | 결과 |
|------|---------|------|------|
| UR5e (6 joints) | `run_sysid_ur.py` | **완료** | L_total = 0.0123 |
| Index Finger (3 joints) | `run_sysid_index.py` | **완료** | L_total = 0.0817 |
| Full Hand (20 joints) | - | 미구현 | - |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       Phase 0 Pipeline                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Real Robot Data (LeRobot)     Isaac Sim                         │
│  ┌───────────────────┐        ┌────────────────────┐            │
│  │ HD_0204 dataset   │        │ World0.usd         │            │
│  │ - ur5e_joint_pos  │───────▶│ - UR5e + Tesollo   │            │
│  │ - ur5e_ee_pos/rot │        │ - Open-loop replay │            │
│  │ - actions         │        └─────────┬──────────┘            │
│  │ - fingertip_pos   │                  │                        │
│  └───────────────────┘                  ▼                        │
│                              ┌─────────────────────┐             │
│                              │   SIMPLER Loss      │             │
│                              │   L = L_transl      │             │
│                              │     + L_rot         │             │
│                              └──────────┬──────────┘             │
│                                         │                        │
│                                         ▼                        │
│                              ┌─────────────────────┐             │
│                              │ Simulated Annealing │             │
│                              │ (3 rounds × 100)    │             │
│                              │ Shrinking bounds    │             │
│                              └──────────┬──────────┘             │
│                                         │                        │
│                                         ▼                        │
│                              ┌─────────────────────┐             │
│                              │ Optimized PD Params │             │
│                              │ - stiffness[N]     │             │
│                              │ - damping[N]       │             │
│                              └─────────────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 캘리브레이션 대상

### 1. UR5e Calibration

```
6 joints → 1 EE (wrist_3_link = tool flange)

Loss = L_transl(tool_flange) + L_rot(tool_flange)

Parameters: stiffness[6], damping[6] = 12 params
Joints: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
```

**특이사항**:
- `wrist_3_joint`에 π offset 적용 (USD 모델과 실제 로봇 간 차이)
- World → UR5e base frame 변환 시 Rz(180°) 보정 필요
- PhysX 내부 단위(radian)와 USD 단위(degree) 변환: `PhysX = USD × 57.296`

### 2. Index Finger Calibration

```
3 joints → 1 EE (lk_dg_2_4 = index fingertip DIP link)

Loss = L_transl(fingertip) + L_rot(fingertip)

Parameters: stiffness[3], damping[3] = 6 params
Joints: lj_dg_2_2, lj_dg_2_3, lj_dg_2_4
```

### 3. Full Hand Calibration (계획)

```
20 joints (5 fingers × 4 joints) → 5 EE (fingertips)

L_hand = Σ L_finger(i) for i in {thumb, index, middle, ring, pinky}

Parameters: stiffness[20], damping[20] = 40 params
```

---

## Loss Functions

### Translation Loss (SIMPLER Eq. 3)

```python
L_transl = (1/T) × Σ_t ||x_real[t] - x_sim[t]||²
```

EE position의 MSE. 단위: m².

### Rotation Loss (SIMPLER Eq. 4)

```python
L_rot = (1/T) × Σ_t 2·arccos(|q_real[t] · q_sim[t]|)
```

Quaternion 간 geodesic distance (각도). 단위: radian.

- Quaternion format: **wxyz** (w가 scalar component)
- 정규화 후 dot product → arccos로 각도 계산

### Combined Loss (SIMPLER Eq. 5)

```python
L_sysid = L_transl + L_rot + λ × L_joint

# 현재 구현에서는 L_joint = 0 (EE pose만 사용)
```

> **참고**: 구현체(`calibration/utils/loss_functions.py`)에서는 `joint_weight`를 설정할 수 있으나, 현재 실행 스크립트에서는 EE pose loss만 사용합니다.

---

## Optimizer: 3-Round Simulated Annealing

SIMPLER 논문 방식의 multi-round SA 최적화:

```
Round 1: Full search space → 100 trials
    ↓ Best 파라미터 주변으로 bounds를 0.5× 축소
Round 2: Narrowed space → 100 trials
    ↓ Best 파라미터 주변으로 bounds를 0.5× 축소
Round 3: Fine-tuning → 100 trials
    ↓
Final: 전체 300 evaluations 중 best parameters 반환
```

### SA 핵심 메커니즘

| 요소 | 구현 |
|------|------|
| Neighbor 생성 | Normalized [0,1] 공간에서 Gaussian perturbation |
| Temperature schedule | Exponential decay: T_init → T_final |
| Acceptance | Metropolis criterion: `exp(-ΔL / T)` |
| Bounds shrinking | 각 round 후 best 중심으로 range × 0.5 축소 |

### SA 파라미터

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_rounds` | 3 | SA round 수 |
| `trials_per_round` | 100 | Round 당 trial 수 |
| `temp_initial` | 1.0 | 시작 온도 |
| `temp_final` | 0.01 | 최종 온도 |
| `shrink_factor` | 0.5 | Bounds 축소 비율 |

---

## 실행 방법

### UR5e System Identification

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab

./isaaclab.sh -p scripts/run_sysid_ur.py \
    --trajectory_dir data/lerobot/HD_0204 \
    --output_dir results/sysid_ur \
    --num_rounds 3 \
    --trials_per_round 100 \
    --headless
```

**CLI 옵션**:

| 옵션 | Default | 설명 |
|------|---------|------|
| `--trajectory_dir` | (필수) | LeRobot 데이터셋 경로 |
| `--output_dir` | `results/sysid_simpler` | 결과 저장 경로 |
| `--num_rounds` | 3 | SA round 수 |
| `--trials_per_round` | 100 | Round 당 trial 수 |
| `--stiffness_min/max` | 286.5 / 85944.0 | Stiffness bounds (PhysX 단위) |
| `--damping_min/max` | 0.573 / 572.96 | Damping bounds (PhysX 단위) |
| `--max_time` | 20.0 | 사용할 궤적 최대 시간 (초) |
| `--seed` | None | Random seed |
| `--headless` | - | GUI 없이 실행 |

### Index Finger System Identification

```bash
./isaaclab.sh -p scripts/run_sysid_index.py \
    --trajectory_dir data/lerobot/HD_0204 \
    --output_dir results/phase0_index \
    --num_rounds 3 \
    --trials_per_round 100 \
    --headless
```

### Validation (검증)

```bash
# UR5e 검증 (캘리브레이션 전후 비교)
./isaaclab.sh -p scripts/validate_sysid_ur.py \
    --trajectory_dir data/lerobot/HD_0204

./isaaclab.sh -p scripts/validate_sysid_ur.py \
    --trajectory_dir data/lerobot/HD_0204 \
    --calibration_file results/sysid_ur/sysid_simpler_result.yaml

# Index finger 검증
./isaaclab.sh -p scripts/validate_sysid_index.py \
    --trajectory_dir data/lerobot/HD_0204 \
    --calibration_file results/phase0_index/phase0_index_result.yaml
```

---

## 캘리브레이션 결과

### UR5e 결과 (2026-02-10)

**Dataset**: LeRobot HD_0204 (658 frames, 20초)

| 메트릭 | 값 |
|--------|-----|
| L_transl | 4.08 × 10⁻⁶ m² |
| L_rot | 0.0123 rad |
| L_total | 0.0126 |
| Total evaluations | 300 (3 rounds × 100) |

**최적화된 PD 파라미터 (PhysX 단위)**:

| Joint | Stiffness | Damping |
|-------|-----------|---------|
| shoulder_pan | 332.2 | 36.1 |
| shoulder_lift | 81887.5 | 165.0 |
| elbow | 81585.5 | 246.2 |
| wrist_1 | 10947.6 | 216.3 |
| wrist_2 | 53370.8 | 12.4 |
| wrist_3 | 83602.3 | 391.8 |

### Index Finger 결과 (2026-02-04)

**Dataset**: LeRobot HD_0204

| 메트릭 | 값 |
|--------|-----|
| L_total | 0.0817 |
| Total evaluations | 100 (1 round × 100) |

**최적화된 PD 파라미터 (PhysX 단위)**:

| Joint | Stiffness | Damping |
|-------|-----------|---------|
| lj_dg_2_2 | 16.93 | 1.48 |
| lj_dg_2_3 | 10.15 | 1.06 |
| lj_dg_2_4 | 10.58 | 0.10 |

---

## 파일 구조

```
Real2Sim/
├── calibration/
│   ├── phase0/
│   │   ├── simulated_annealing.py    # 3-round SA optimizer (SAConfig, SimulatedAnnealingOptimizer)
│   │   ├── joint_dynamics_estimator.py  # 초기 설계 (미사용)
│   │   ├── robot_config.py             # UR5/Hand config (초기 설계)
│   │   └── trajectory_replay.py        # Grid search 방식 (미사용)
│   └── utils/
│       └── loss_functions.py           # SIMPLER loss (simpler_sysid_loss, per_finger, hand)
│
├── data/
│   ├── storage/
│   │   ├── lerobot_loader.py          # LeRobot 데이터셋 로더 (LeRobotLoader)
│   │   ├── trajectory_storage.py      # HDF5 기반 궤적 저장
│   │   └── calibration_results.py     # YAML 결과 저장
│   └── lerobot/
│       └── HD_0204/                   # 실제 데이터셋 (UR5e + Index finger)
│
├── scripts/
│   ├── run_sysid_ur.py               # ★ UR5e SysID 실행 스크립트
│   ├── run_sysid_index.py            # ★ Index finger SysID 실행 스크립트
│   ├── validate_sysid_ur.py          # UR5e 검증 스크립트
│   ├── validate_sysid_index.py       # Index finger 검증 스크립트
│   └── test_phase0_io.py             # I/O 테스트
│
├── results/
│   ├── sysid_ur/
│   │   └── sysid_simpler_result.yaml # UR5e 캘리브레이션 결과
│   └── phase0_index/
│       └── phase0_index_result.yaml  # Index finger 캘리브레이션 결과
│
└── assets/
    └── robots/
        └── World0.usd                # UR5e + Tesollo 통합 USD 모델
```

---

## 데이터 형식

### 입력: LeRobot Dataset

`data/storage/lerobot_loader.py`의 `LeRobotLoader`를 사용하여 LeRobot 형식 데이터셋을 로드합니다.

**UR5e SysID에 필요한 필드**:

| 필드 | Shape | 설명 |
|------|-------|------|
| `timestamp` | (T,) | 시간 스탬프 |
| `observation.state.ur5e_joint_pos` | (T, 6) | UR5e joint 위치 |
| `observation.state.ur5e_ee_pos` | (T, 3) | Tool flange 위치 (base frame) |
| `observation.state.ur5e_ee_rot` | (T, 4) | Tool flange 회전 (wxyz) |
| `action` | (T, 9) | 명령 [ur5e(6) + finger(3)] |

**Index Finger SysID 추가 필드**:

| 필드 | Shape | 설명 |
|------|-------|------|
| `observation.state.index_fingertip_pos` | (T, 3) | Index fingertip 위치 |
| `observation.state.index_fingertip_rot` | (T, 4) | Index fingertip 회전 (wxyz) |

### 출력: Calibration Result YAML

```yaml
# results/sysid_ur/sysid_simpler_result.yaml
method: simpler
loss_type: ee_pose                    # or fingertip_tcp
joint_names:
- shoulder_pan_joint
- shoulder_lift_joint
- elbow_joint
- wrist_1_joint
- wrist_2_joint
- wrist_3_joint
parameters:
  stiffness: [332.2, 81887.5, ...]    # per-joint (PhysX 단위)
  damping: [36.1, 165.0, ...]
final_loss:
  L_transl: 4.08e-06
  L_rot: 0.0123
  L_total: 0.0126
optimization:
  num_rounds: 3
  trials_per_round: 100
  stiffness_bounds: [286.5, 85944.0]
  damping_bounds: [0.573, 572.96]
  total_evaluations: 300
trajectory_info:
  dataset: data/lerobot/HD_0204
  max_time_used: 20.0
  frames_used: 658
timestamp: '2026-02-10T22:34:49'
```

---

## 실행 파이프라인 상세

### Step-by-Step 동작

1. **Load Data**: LeRobotLoader로 데이터셋 로드 (joint pos, EE pose, actions)
2. **Create Simulation**: Isaac Sim + World0.usd 로드 (UR5e + Tesollo)
3. **Setup Robot**: `ArticulationCfg`로 로봇 생성, joint/body index 매핑
4. **Initial Analysis**: 초기 자세에서 Real vs Sim EE 위치 비교
5. **Loss Function 정의**:
   - Sim reset → PD 파라미터 적용 → Initial pose settle (50 steps)
   - Open-loop trajectory replay (action → position target → step × N)
   - Real vs Sim EE pose 비교 → SIMPLER loss 계산
6. **SA Optimization**: 3-round SA 실행 (total 300 evaluations)
7. **Final Evaluation**: Best 파라미터로 최종 loss 계산
8. **Save Results**: YAML 파일로 저장

### 좌표 변환 주의사항

```
World Frame → UR5e Base Frame 변환:
1. position: quat_rotate_inverse(base_quat, pos - base_pos) → negate x, y (Rz180)
2. rotation: conj(base_quat) * quat → Rz180 * result * conj(Rz180)
```

USD 모델의 base frame이 UR5e native base frame 대비 Z축으로 180° 회전되어 있어 보정이 필요합니다.

---

## 데이터 수집 가이드

> 상세 가이드: [phase0_data_collection.md](phase0_data_collection.md)

### 요약

- **궤적 유형**: Free motion (접촉 없이 자유롭게 움직이는 궤적)
- **수집 방식**: GELLO teleoperation → LeRobot 형식으로 변환 (`convert_gello_to_lerobot.py`)
- **필수 정보**: Joint positions + EE poses (FK 또는 Mocap으로 획득)
- **Quaternion convention**: wxyz
- **권장 샘플링**: ~30Hz (LeRobot default)

---

## References

- SIMPLER Paper: ["Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition"](https://simpler-env.github.io/)
  - Section 3.2: System Identification
  - Equations 3-5: Translation, Rotation, Combined Loss
