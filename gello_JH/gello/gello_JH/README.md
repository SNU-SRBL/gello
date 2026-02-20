# Real2Sim Calibration Framework

UR5e + Tesollo DG5F Hand의 Real2Sim calibration을 위한 3-Phase 파이프라인입니다.

## 목표

실제 로봇에서 측정한 데이터를 시뮬레이션에서 replay했을 때, 동일한 동작을 재현하도록 시뮬레이션 파라미터를 튜닝합니다.

## 3-Phase 구조

| Phase | 목적 | 파라미터 | 방법 | 상태 |
|-------|------|---------|------|------|
| **Phase 0** | Joint dynamics | stiffness, damping | SIMPLER-style EE tracking + SA | **UR5e 완료, Index 완료** |
| **Phase 1** | I ↔ τ 매핑 | k_t, offset | q_cmd Replay + Paired Sim-Real | **구현됨** |
| Phase 2 | 접촉 마찰 | μ_static, μ_dynamic | Position Control + Slip detection | 구현됨 |

---

## Phase 0: System Identification (SIMPLER style) - 완료

SIMPLER 논문 방식으로 joint dynamics 파라미터를 최적화합니다.

### 핵심 방법

- **Open-loop trajectory replay**: 실제 로봇의 action 명령을 시뮬레이션에서 그대로 재생
- **EE pose tracking loss**: Joint 에러가 아닌 End-Effector (tool flange / fingertip) pose 비교
- **3-round Simulated Annealing**: 매 round마다 best 주변으로 search space 축소

### Loss Function

```
L_sysid = L_transl + L_rot

L_transl = (1/T) × Σ ||x_real - x_sim||²     (EE position MSE)
L_rot    = (1/T) × Σ 2·arccos(|q_real · q_sim|) (quaternion geodesic distance)
```

### 캘리브레이션 결과

**UR5e** (2026-02-10):

| 메트릭 | 값 |
|--------|-----|
| L_transl | 4.08 × 10⁻⁶ m² |
| L_rot | 0.0123 rad |
| Evaluations | 300 (3 rounds × 100) |
| Dataset | LeRobot HD_0204 (658 frames, 20s) |

| Joint | Stiffness | Damping |
|-------|-----------|---------|
| shoulder_pan | 332.2 | 36.1 |
| shoulder_lift | 81887.5 | 165.0 |
| elbow | 81585.5 | 246.2 |
| wrist_1 | 10947.6 | 216.3 |
| wrist_2 | 53370.8 | 12.4 |
| wrist_3 | 83602.3 | 391.8 |

**Index Finger** (2026-02-04):

| 메트릭 | 값 |
|--------|-----|
| L_total | 0.0817 |
| Evaluations | 100 (1 round × 100) |

| Joint | Stiffness | Damping |
|-------|-----------|---------|
| lj_dg_2_2 | 16.93 | 1.48 |
| lj_dg_2_3 | 10.15 | 1.06 |
| lj_dg_2_4 | 10.58 | 0.10 |

### 실행 방법

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab

# UR5e System ID
./isaaclab.sh -p scripts/run_sysid_ur.py \
    --trajectory_dir data/lerobot/HD_0204 \
    --output_dir results/sysid_ur \
    --num_rounds 3 --trials_per_round 100 --headless

# Index Finger System ID
./isaaclab.sh -p scripts/run_sysid_index.py \
    --trajectory_dir data/lerobot/HD_0204 \
    --output_dir results/phase0_index \
    --num_rounds 3 --trials_per_round 100 --headless

# Validation
./isaaclab.sh -p scripts/validate_sysid_ur.py \
    --trajectory_dir data/lerobot/HD_0204 \
    --calibration_file results/sysid_ur/sysid_simpler_result.yaml
```

> 상세 문서: [docs/phase0_simpler_sysid.md](docs/phase0_simpler_sysid.md)

---

## Phase 1: Paired Sim-Real Current-Torque Calibration

Motor current (I)와 joint torque (τ) 사이의 매핑을 추정합니다: `τ_sim = k_t × I_real + b`

### 왜 필요한가?

- Real robot: motor current (I) 만 측정 가능 (joint torque sensor 없음)
- Simulation: joint torque (τ) 에 직접 접근 가능
- **Real2Sim transfer 시 두 도메인을 연결하는 calibration이 필수**

### 이론적 배경

준정적 조건 (q̈ ≈ 0)에서 Euler-Lagrange 방정식:

```
Sim:  τ_sim    = g_sim(q)    + friction_sim(qdot)    + J^T(q) × F_ext
Real: k_t × I  = g_real(q)   + friction_real(qdot)   + J^T(q) × F_ext
```

같은 (q, F_ext) 조건에서 J^T × F_ext는 동일하므로:

```
τ_sim ≈ k_t × I_real + (g_sim - g_real) + (friction_sim - friction_real)
      = k_t × I_real + b   ← offset이 sim-real 차이를 흡수
```

### 핵심 방법: q_cmd Replay + Paired Matching

```
[Real Robot]                              [Simulation]
1. 손가락이 q_cmd에 도달                  1. 초기 상태 = q_cmd 설정 (밀기 전 동일 위치)
2. F_ext 인가 → q_actual로 밀림           2. PD target = q_cmd 설정 (동일 컨트롤러)
3. PD가 q_cmd를 향해 I 출력               3. F_ext를 fingertip에 직접 인가
4. 기록: (q_actual, q_cmd, I, F_ext)      4. Physics settle → q_sim으로 밀림, τ_sim 기록
                ↓                                       ↓
     q_sim ↔ q_actual 비교 (밀린 위치 매칭)
     τ_sim = k_t × I_real + b (current-torque 매핑)
```

**왜 q_cmd로 초기화하는가?**
- Real/Sim 모두 **같은 시작 위치(q_cmd)**에서 같은 힘(F_ext)을 받아 밀림
- q_deflection = |q_sim - q_actual|로 sim-real 밀림 정도 비교 → Phase 0 PD 게인 정확도 검증
- q_actual로 초기화하면 이미 밀린 상태 → PD error 없이 F_ext가 추가로 밀어서 State Mismatch 발생

### 코드 구조

| 파일 | 역할 |
|------|------|
| `scripts/collect_phase1_real.py` | **Real 데이터 수집** (Tesollo DG5F 직접 제어, UR/GELLO 불필요) |
| `envs/phase1_current_torque/current_torque_env.py` | Isaac Lab 환경 + 15-state State Machine |
| `envs/phase1_current_torque/current_torque_env_cfg.py` | 환경 설정 (PD gains, FT fixture 등) |
| `calibration/phase1/current_torque_model.py` | `JacobianCurrentTorqueModel` (per-joint LinearRegression) |
| `calibration/phase1/learned_model.py` | `LearnedCurrentTorqueModel` (MLP + Residual) |
| `scripts/run_phase1.py` | CLI 실행 스크립트 (6개 mode) |

### 데이터 흐름

```
[Real Robot Data]                    [Simulation]
q_actual, q_cmd, I_motor, F_ext
        ↓
  ┌─────────────────────────────────────────────────┐
  │  Step 1: sim_from_real                           │
  │  각 데이터포인트에 대해:                           │
  │    write_joint_state_to_sim(q_cmd)  ← 밀기 전 위치│
  │    set_joint_position_target(q_cmd)               │
  │    set_external_force_and_torque(F_ext)           │
  │    settle 0.5s → q_sim으로 밀림                   │
  │    record applied_torque × 5 (τ_sim 평균)        │
  │    q_deflection = q_sim - q_actual (매칭 검증)   │
  └──────────────────────┬──────────────────────────┘
                         ↓
                sim_matched.npz (τ_sim)
                         ↓
  ┌─────────────────────────────────────────────────┐
  │  Step 2: calibrate                               │
  │  Per-joint LinearRegression:                     │
  │    τ_sim_j = k_t_j × I_real_j + b_j             │
  │  Outlier 제거 (residual > 3σ), R² ≥ 0.9 검증   │
  └──────────────────────┬──────────────────────────┘
                         ↓
          phase1_result.yaml (k_t, offset, R²)
                         ↓ (optional)
  ┌─────────────────────────────────────────────────┐
  │  Step 3: calibrate_learned                       │
  │  MLP: [q+qdot+FT+I = 90D] → [128,64,32] → τ   │
  │  Residual: k_t×I + f_residual(features)         │
  └──────────────────────┬──────────────────────────┘
                         ↓
                learned_model.pt
```

### State Machine (`CalibrationState`)

```
Sim-From-Real (핵심 모드):
  SFR_SETTING_STATE → SFR_APPLYING_FORCE → SFR_SETTLING (0.5s)
  → SFR_RECORDING (5 samples avg) → SFR_NEXT_POINT → ... → COMPLETED

Phase 1A (Gravity/Friction Baseline, sim only):
  GRAVITY_BASELINE_MOVING (1.5s settle) → GRAVITY_BASELINE_RECORDING (30 samples)
  → 반복 20회 → GRAVITY_BASELINE_VELOCITY (sinusoidal sweep 0.5~3Hz, 10회)

Phase 1B (Contact Calibration, sim only):
  MOVING_TO_CONTACT (1s) → PRESSING (5s ramp, 50 steps, max 15N)
  → SETTLING (0.5s) → RECORDING (50 samples) → RETRACTING
  → REPOSITION_FT_SENSOR (3 directions) → ... → COMPLETED
```

### 실행 방법

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab

# Step 0: Real 데이터 수집 (Tesollo DG5F 직접 제어, Isaac Lab 불필요)
#   - Tesollo에 q_cmd 전송 → PID로 위치 유지
#   - 사용자가 수동으로 각 fingertip에 외력 인가
#   - 내장 fingertip FT 센서로 F_ext 측정 (외부 센서 불필요)
#   - 접촉 자동 감지 → (q, q_cmd, I_motor, F_ext, finger_idx) 기록
python scripts/collect_phase1_real.py \
    --num_positions 5 --position_mode grid \
    --output data/real_data/phase1_contact.npz

# Step 0-alt: 센서 검증 (dry run)
python scripts/collect_phase1_real.py --dry_run

# Step 1: Real 데이터 → sim에서 τ_sim 계산 (q_cmd replay)
./isaaclab.sh -p scripts/run_phase1.py --mode sim_from_real \
    --real_contact data/real_data/phase1_contact.npz \
    --phase0_result results/phase0/phase0_hand_result.yaml \
    --output_dir results/phase1 --headless

# Step 2: τ_sim vs I_real per-joint 선형 회귀
./isaaclab.sh -p scripts/run_phase1.py --mode calibrate \
    --sim_matched results/phase1/sim_matched.npz \
    --real_contact data/real_data/phase1_contact.npz \
    --output_dir results/phase1

# Step 3: (Optional) Learning-based calibration (MLP + Residual)
./isaaclab.sh -p scripts/run_phase1.py --mode calibrate_learned \
    --sim_matched results/phase1/sim_matched.npz \
    --real_contact data/real_data/phase1_contact.npz \
    --model_type both --num_epochs 200 --output_dir results/phase1

# Step 4: (Optional) Sim Dry Run — Jacobian/gravity 검증용
./isaaclab.sh -p scripts/run_phase1.py --mode sim_full \
    --phase0_result results/phase0/phase0_hand_result.yaml \
    --fingers thumb,index,middle,ring,pinky \
    --output_dir results/phase1 --headless
```

### Real 데이터 수집 (`collect_phase1_real.py`)

Tesollo DG5F를 직접 제어하여 Phase 1 real 데이터를 수집합니다:

- **UR/GELLO 불필요**: Tesollo SDK (DGSDK)로 직접 20 관절 제어
- **내장 FT 센서 사용**: Tesollo fingertip FT 센서 (5 fingers × 6 DOF) → 외부 센서 불필요
- **접촉 자동 감지**: 5개 손가락 FT 크기 비교 → 활성 손가락 판별 (히스테리시스: onset 0.3N / release 0.15N)
- **Ctrl-C 안전 종료**: 수집 중 interrupt 시 partial data 자동 저장

```
수집 프로토콜:
  1. move_joint_all(q_cmd) → 모든 관절 PID 위치 유지
  2. set_fingertip_data_zero() → 중력 보상
  3. 사용자가 fingertip에 외력 인가
  4. FT > threshold → 접촉 감지, 50Hz 샘플링 시작
  5. FT < threshold 0.2s → 접촉 종료, steady-state 평균 저장
  6. 다음 position 또는 다음 손가락으로 반복
```

q_cmd 생성 모드:

| 모드 | 설명 |
|------|------|
| `grid` (기본) | 5~25도 균일 분할, 모든 관절 동일값 |
| `per_finger` | 한 손가락씩 변경, 나머지 15도 고정 |
| `random` | 3~27도 랜덤 |
| `manual` | 사용자가 직접 값 입력 |

### Real 데이터 형식 (`phase1_contact.npz`)

| 필드 | Shape | 설명 | 필수 |
|------|-------|------|------|
| `q` | (N, 20) | 실제 관절 위치 (접촉 후 밀린 위치) [rad] | O |
| `q_cmd` | (N, 20) | 위치 명령 (PD controller target) [rad] | **O (핵심)** |
| `I_motor` | (N, 20) | 모터 전류 [A] | O |
| `F_ext` | (N, 6) | 활성 손가락 fingertip FT [Fx,Fy,Fz,Tx,Ty,Tz] [N, Nm] | O |
| `finger_idx` | (N,) | 활성 손가락 인덱스 (0=thumb ~ 4=pinky) | O |
| `F_internal` | (N, 30) | 전체 5손가락 FT sensor (5 × 6D) [N, Nm] | (learned model용) |
| `config_idx` | (N,) | q_cmd position 인덱스 | (메타데이터) |

### 출력 파일

| 파일 | 내용 |
|------|------|
| `sim_matched.npz` | τ_sim(N,20), q_actual, q_cmd, q_deflection, F_applied, finger_idx |
| `calibration.npz` | k_t(20,), offset(20,), r_squared(20,), jacobian_consistency(20,) |
| `phase1_result.yaml` | Per-joint k_t, offset, R² + per-finger summary |
| `learned_model.pt` | (Optional) MLP/Residual model weights + normalization stats |

### 검증 기준

| 메트릭 | 기준 | 의미 |
|--------|------|------|
| R² (per-joint) | ≥ 0.90 | τ_sim = k_t × I + b의 선형 적합도 |
| q_deflection | 모니터링 | Phase 0 PD 게인이 정확할수록 작아짐 |
| k_t stability | CV < 10% | 다른 config에서 추정한 k_t의 변동계수 |
| Jacobian consistency | < 0.1 relative error | J^T × F_ext ≈ τ_sim (kinematic model 검증) |

> 상세 문서: [docs/phase1_current_torque.md](docs/phase1_current_torque.md)

---

## Phase 2: Friction & Contact Calibration

Position control grip sweep + slip detection을 사용하여 마찰 파라미터를 튜닝합니다.

**방법**:
1.*Dynamic Friction (μ_d)**: Slip 후 물체 추적
 객체 파지 (Position control)
2. Position Sweep (손가락 벌리기) → 힘 감소 → Slip 발생
3. **Static Friction (μ_s)**: Slip 발생 시점의 F_tangential / F_normal
4. *
```bash
# Simulation slip test
./isaaclab.sh -p scripts/run_phase2.py --mode sim \
    --phase0_result results/sysid_ur/sysid_simpler_result.yaml \
    --phase1_result results/phase1/phase1_result.yaml \
    --output_dir results/phase2
```

> 상세 문서: [docs/phase2_friction_calibration.md](docs/phase2_friction_calibration.md)

---

## 설치

```bash
# Isaac Lab 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab

# 의존성 설치
pip install optuna h5py pyyaml matplotlib scipy scikit-learn torch
```

---

## 디렉토리 구조

```
Real2Sim/
├── assets/                          # USD 모델
│   └── robots/
│       └── World0.usd              # UR5e + Tesollo 통합 모델
│
├── calibration/                     # 캘리브레이션 알고리즘
│   ├── phase0/
│   │   ├── simulated_annealing.py  # ★ 3-round SA optimizer
│   │   ├── joint_dynamics_estimator.py
│   │   ├── robot_config.py
│   │   └── trajectory_replay.py
│   ├── phase1/                      # Paired sim-real calibration + learned models
│   ├── phase2/
│   └── utils/
│       └── loss_functions.py       # ★ SIMPLER loss functions
│
├── data/                            # 데이터
│   ├── storage/
│   │   ├── lerobot_loader.py       # ★ LeRobot 데이터셋 로더
│   │   ├── trajectory_storage.py   # HDF5 저장
│   │   └── calibration_results.py  # YAML 저장
│   └── lerobot/
│       └── HD_0204/                # ★ 실제 데이터셋
│
├── envs/                            # Isaac Lab 환경
│   ├── base/
│   ├── phase0_sysid/
│   ├── phase1_current_torque/
│   └── phase2_friction/
│
├── scripts/                         # 실행 스크립트
│   ├── run_sysid_ur.py             # ★ UR5e SysID
│   ├── run_sysid_index.py          # ★ Index finger SysID
│   ├── validate_sysid_ur.py        # UR5e 검증
│   ├── validate_sysid_index.py     # Index finger 검증
│   ├── collect_phase1_real.py       # ★ Phase 1 Real 데이터 수집 (Tesollo 직접 제어)
│   ├── convert_gello_to_lerobot.py # GELLO → LeRobot 변환
│   ├── replay_gello_data.py        # GELLO 데이터 재생
│   ├── slip_test_env.py            # Slip 테스트 환경
│   ├── run_phase1.py               # Phase 1 Sim + Calibration
│   └── run_phase2.py
│
├── results/                         # 캘리브레이션 결과
│   ├── sysid_ur/
│   │   └── sysid_simpler_result.yaml
│   └── phase0_index/
│       └── phase0_index_result.yaml
│
├── gello/                           # GELLO teleoperation
├── gello_HD/
├── sensors/                         # 센서 인터페이스
├── replay/                          # 궤적 재생
├── analysis/                        # 분석 도구
├── configs/                         # 설정 파일
└── docs/                            # 문서
    ├── phase0_simpler_sysid.md     # ★ Phase 0 상세 문서
    ├── phase0_data_collection.md   # 데이터 수집 가이드
    ├── phase1_current_torque.md
    └── phase2_friction_calibration.md
```

---

## 주요 클래스 및 모듈

### Phase 0 (System ID)

| 모듈 | 클래스/함수 | 역할 |
|------|------------|------|
| `calibration/phase0/simulated_annealing.py` | `SimulatedAnnealingOptimizer` | 3-round SA with bounds shrinking |
| `calibration/phase0/simulated_annealing.py` | `SAConfig` | SA 설정 (rounds, trials, temp) |
| `calibration/utils/loss_functions.py` | `simpler_sysid_loss()` | L_transl + L_rot + λ·L_joint |
| `calibration/utils/loss_functions.py` | `hand_simpler_loss()` | Per-finger 분해 loss |
| `data/storage/lerobot_loader.py` | `LeRobotLoader` | LeRobot 데이터셋 로드 |

### 센서

| 모듈 | 클래스 | 역할 |
|------|--------|------|
| `sensors/tesollo_ft_sensor.py` | `TesolloFTSensorIsaacLab` | 35개 F/T 센서 (7 pads × 5 fingers) |
| `sensors/external_ft_sensor.py` | - | 외부 F/T 센서 |

---

## 검증 기준

| Phase | Robot | 메트릭 | 기준 |
|-------|-------|--------|------|
| Phase 0 | UR5e | Tool flange position error | < 5mm |
| Phase 0 | UR5e | Tool flange rotation error | < 2° |
| Phase 0 | Hand | Fingertip position error | < 3mm |
| Phase 0 | Hand | Fingertip rotation error | < 5° |
| Phase 1 | All | R² (paired sim-real) | > 0.90 |
| Phase 1 | All | q_deflection (sim-real 상태 차이) | 모니터링 |
| Phase 2 | All | μ_s 오차 | < 0.1 |
| Phase 2 | All | μ_d 오차 | < 0.15 |

---

## 의존성

- Isaac Lab >= 2.0.0
- PyTorch >= 2.0.0
- Optuna >= 3.0.0
- h5py >= 3.0.0
- PyYAML >= 6.0.0
- matplotlib >= 3.5.0
- scipy >= 1.9.0

## 참고 문서

- [Phase 0: SIMPLER-style SysID](docs/phase0_simpler_sysid.md)
- [Phase 0: 데이터 수집 가이드](docs/phase0_data_collection.md)
- [Phase 1: Current-Torque](docs/phase1_current_torque.md)
- [Phase 2: Friction Calibration](docs/phase2_friction_calibration.md)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
