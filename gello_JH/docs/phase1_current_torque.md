# Phase 1: Paired Sim-Real Current-Torque Calibration

## 1. 개요

Phase 1은 real robot의 **motor current (I)**와 simulation의 **joint torque (τ)** 사이의
정확한 매핑을 구하는 단계이다.

최종 목표: per-joint **torque constant k_t [Nm/A]** 및 **offset b [Nm]** 추정
```
τ_sim_j = k_t_j × I_real_j + b_j
```

### 왜 필요한가?

- Real robot: motor current (I) 만 측정 가능, joint torque sensor 없음
- Simulation: joint torque (τ) 에 직접 접근 가능
- Sim-to-Real transfer 시 이 두 도메인을 연결하는 **calibration**이 필수

### 핵심 아이디어: Paired Sim-Real Matching

```
[Real Robot]                          [Simulation]
1. 손가락을 config q로 이동          1. 같은 q로 설정
2. FT sensor에 pressing              2. real의 F_ext를 fingertip에 직접 인가
3. 기록: (q, qdot, I_real, F_ext)    3. 기록: (q, τ_sim)
                     ↓                              ↓
              같은 조건에서 매칭: τ_sim = k_t × I_real + b
```

**왜 이 방법인가?**
- g(q)는 real robot에서 직접 측정 불가 (IsaacLab API는 sim에서만 사용 가능)
- 목적은 물리량 분해가 아니라, **실용적인 I ↔ τ 매핑**
- τ_sim은 sim의 모든 physics (gravity, friction, contact response) 포함
- k_t × I_real도 real의 모든 physics 포함
- 같은 (q, F_ext) 조건에서: gravity/friction 차이는 **offset b에 흡수**

---

## 2. 이론적 배경

### 2.1 Euler-Lagrange 방정식

로봇 동역학:
```
M(q)q̈ + C(q,qdot)qdot + g(q) = τ_motor + J^T(q) × F_contact
```

| 항 | 의미 | 차원 |
|---|------|------|
| M(q)q̈ | 관성력 | Nm |
| C(q,qdot)qdot | 코리올리/원심력 | Nm |
| g(q) | 중력 토크 | Nm |
| τ_motor | 모터 토크 = k_t × I | Nm |
| J^T(q) × F | 외력의 관절 토크 기여 | Nm |

### 2.2 Paired Matching 원리

준정적 조건 (q̈ ≈ 0)에서:

**Sim**:
```
τ_sim = g_sim(q) + friction_sim(qdot) + J_sim^T(q) × F_ext
```

**Real**:
```
k_t × I_real = g_real(q) + friction_real(qdot) + J_real^T(q) × F_ext
```

같은 (q, F_ext)를 적용하면, J와 F_ext 기여분은 같으므로:
```
τ_sim ≈ k_t × I_real + (g_sim - g_real) + (friction_sim - friction_real)
      = k_t × I_real + b   (offset이 차이를 흡수)
```

### 2.3 Jacobian의 역할 (Validation Only)

Jacobian J(q)는 calibration에 직접 사용하지 않고, **sim 내부 검증**에 사용:
```
J^T(q) × F_ext ≈ τ_sim - g(q) - friction
```
→ Kinematic model이 정확한지 확인 (k_t 추정과는 별개)

### 2.4 Multi-Configuration의 이점

다양한 q에서 데이터를 수집하면:
- k_t 추정의 overdetermined system → 더 robust
- configuration-dependent 오차 감지 가능
- k_t가 진짜 상수인지 검증 가능 (q 변해도 k_t 일정해야 함)

---

## 3. 캘리브레이션 프로토콜

### Step 1: Real Robot 데이터 수집

```
프로토콜:
1. 각 손가락 i를 config j로 이동
2. 외부 FT sensor에 direction k로 pressing
3. 기록: (q_actual, q_cmd, qdot, I_real, F_ext_6D, F_internal_30D)
   - q_actual: 실제 관절 위치 (접촉으로 밀린 후)
   - q_cmd: 위치 명령 (PD controller target)
4. 다양한 (finger, config, direction)에서 반복
→ 출력: real_contact_data.npz
```

**q_cmd로 초기화하는 이유 (State Mismatch 방지)**:
- Real: 로봇이 q_cmd에 도달 → F_ext 인가 → q_actual로 밀림
- Sim: q_cmd로 초기화 → F_ext 인가 → q_sim으로 밀림 (real과 동일한 물리 시나리오)
- Sim에서 q_actual로 초기화하면 → 이미 밀린 상태에서 시작 → F_ext가 추가로 밀어서 State Mismatch
- **핵심**: Real/Sim 모두 같은 시작점(q_cmd) + 같은 힘(F_ext) → 비교 가능한 결과

**F_internal (Fingertip FT sensor) 기록 이유**:
- Real에서 FT sensor와 6축 외부 센서 값 비교 검증
- Learning-based calibration의 고차원 feature로 활용

### Step 2: Sim에서 τ_sim 계산 (q_cmd Replay 모드)

```
프로토콜:
1. real_contact_data.npz 로드 (q_actual, q_cmd, F_ext 포함)
2. 각 데이터포인트 n에 대해:
   a. sim 초기 상태 = q_cmd_n (write_joint_state_to_sim) ← 밀기 전 위치
   b. PD target = q_cmd_n (set_joint_position_target) ← 동일 컨트롤러
   c. F_ext_n을 fingertip에 직접 인가 (set_external_force_and_torque)
   d. Physics settle → sim이 q_cmd에서 q_sim으로 밀림 (real은 q_actual로 밀림)
   e. τ_sim_n = applied_torque 기록
3. 출력: sim_matched.npz (τ_sim, q_sim, q_deflection 등)
4. 검증: q_deflection = q_sim - q_actual 확인 (sim-real 밀림 차이)
```

**핵심**: Real/Sim 모두 **같은 시작 위치(q_cmd)**에서 같은 힘(F_ext)을 받아 밀림
- `write_joint_state_to_sim(q_cmd)` → 밀기 전 동일 위치로 초기화
- `set_external_force_and_torque(F_ext)` → 동일한 외력 인가
- Real은 q_actual로, Sim은 q_sim으로 밀림 → q_deflection으로 비교

### Step 3: Linear Calibration

```
1. 매칭된 (τ_sim, I_real) 데이터
2. Per-joint linear regression: τ_sim_j = k_t_j × I_j + b_j
3. R² 검증, outlier 제거 (Z-score > 3.0)
4. 출력: k_t per joint, offset per joint
```

### Step 4: Learning-Based Calibration (Optional)

Linear regression을 넘어서, 고차원 feature를 사용한 모델 학습:

**입력 features**: (q, qdot, F_internal_6D, I_real)
**출력**: τ_sim (20D)

#### MLP Model
```
features → [128] → [64] → [32] → τ_sim
```
Direct mapping으로, 비선형 관계까지 포착

#### Residual Model
```
τ_sim = k_t × I_real + f_residual(features)
```
Linear calibration을 base로, MLP가 residual을 보정
- k_t, offset은 learnable parameters (linear calibration 결과로 초기화)
- f_residual이 configuration-dependent 오차를 학습

### Step 5: Validation

```
1. q_deflection 확인: |q_sim_settled - q_real_actual| 확인
   → Phase 0 PD 게인이 정확할수록 deflection 작아짐
2. Jacobian consistency: J^T × F ≈ τ_sim - g(q) in sim
3. R² > 0.9: per-joint regression quality
4. k_t stability: 다른 config에서 추정한 k_t의 CV < 10%
5. Cross-validation: hold-out 데이터에서 예측 정확도
6. Cross-sensor: F_external vs F_internal (Tesollo pads) 일치도
```

---

## 4. 데이터 형식

### Real 데이터 (real_contact.npz)

| 필드 | Shape | 설명 |
|------|-------|------|
| `q` | (N, 20) | 실제 관절 위치 (접촉 후 밀린 위치) |
| `q_cmd` | (N, 20) | 위치 명령 (PD controller target) **필수** |
| `qdot` | (N, 20) | 관절 속도 |
| `I_motor` | (N, 20) | 모터 전류 [A] |
| `F_ext` | (N, 6) | 외부 FT sensor 6D wrench |
| `F_internal` | (N, 30) | 내장 FT sensor (5 fingers × 6D) |
| `finger_idx` | (N,) | 손가락 인덱스 |
| `config_idx` | (N,) | configuration 인덱스 |
| `direction_idx` | (N,) | 힘 방향 인덱스 |

### Sim Matched 데이터 (sim_matched.npz)

| 필드 | Shape | 설명 |
|------|-------|------|
| `tau_sim` | (N, 20) | sim에서 PD controller가 출력한 joint torque [Nm] |
| `q_actual` | (N, 20) | sim settled 관절 위치 |
| `q_cmd` | (N, 20) | PD target (= real q_cmd) |
| `q_deflection` | (N, 20) | q_sim_settled - q_real_actual (매칭 검증) |
| `F_applied` | (N, 6) | sim에서 인가된 외력 |
| `F_internal` | (N, 30) | sim 내장 FT sensor |
| `finger_idx` | (N,) | 손가락 인덱스 |

### Sim Validation 데이터 (sim_data.npz)

| 필드 | Shape | 설명 |
|------|-------|------|
| `q_positions` | (N, 20) | Phase 1A baseline 관절 위치 |
| `tau_applied` | (N, 20) | 적용 토크 |
| `gravity_torques` | (N, 20) | IsaacLab gravity API |
| `F_internal` | (N, 30) | 내장 FT sensor |
| `contact_q` | (M, 20) | Phase 1B 접촉 관절 위치 |
| `contact_tau` | (M, 20) | 접촉 시 토크 |
| `contact_F_ext` | (M, 6) | 접촉 외력 |
| `contact_F_internal` | (M, 30) | 접촉 시 내장 FT |
| `contact_jacobian` | (M, 6, 4) | fingertip Jacobian |

### 최종 결과 (phase1_result.yaml)

```yaml
# Linear Calibration
method: "paired_sim_real"
global:
  mean_r_squared: 0.97
  num_joints: 20

per_joint:
  thumb_j0:
    k_t: 0.152           # Nm/A (torque constant)
    offset: 0.003         # Nm (gravity/friction offset)
    r_squared: 0.98

per_finger:
  thumb:
    mean_r_squared: 0.97
    k_t_values: [0.152, 0.148, 0.155, 0.150]
    all_success: true

# Learning-Based (optional)
learned:
  mlp:
    per_joint_r_squared: [0.99, 0.98, ...]
    mean_r_squared: 0.98
  residual:
    per_joint_r_squared: [0.99, 0.99, ...]
    mean_r_squared: 0.99
```

---

## 5. IsaacLab API Reference

### set_external_force_and_torque()
```python
# sim_from_real 모드의 핵심 API
# fingertip body에 직접 외력 인가
hand.root_physx_view.set_external_force_and_torque(
    forces,   # (num_envs, num_bodies, 3)
    torques,  # (num_envs, num_bodies, 3)
)
```

### write_joint_state_to_sim()
```python
# real robot의 q를 sim에 직접 설정
hand.write_joint_state_to_sim(
    joint_pos,  # (num_envs, num_dofs)
    joint_vel,  # (num_envs, num_dofs)
)
```

### data.applied_torque
```python
# τ_sim 기록
tau_sim = hand.data.applied_torque  # (num_envs, num_dofs)
```

### get_jacobians()
```python
J = hand.root_physx_view.get_jacobians()
# Shape: (num_envs, num_bodies, 6, num_dofs)
# 주의: fixed-base robot은 body_idx = frame_idx - 1
```

### get_gravity_compensation_forces()
```python
g = hand.root_physx_view.get_gravity_compensation_forces()
# Shape: (num_envs, num_dofs) → (1, 20)
# Jacobian validation에서 사용 (sim only)
```

### get_link_incoming_joint_force()
```python
W = hand.root_physx_view.get_link_incoming_joint_force()
# Shape: (num_envs, num_bodies, 6) = [Fx,Fy,Fz,Tx,Ty,Tz]
# 내장 FT sensor 읽기에 사용
```

---

## 6. 사용법

### Step 1: (Optional) Sim Validation Dry Run
```bash
# Phase 1A + 1B simulation (Jacobian/gravity 검증용)
./isaaclab.sh -p scripts/run_phase1.py \
    --mode sim_full \
    --phase0_result results/phase0/phase0_hand_result.yaml \
    --fingers thumb,index,middle,ring,pinky \
    --num_gravity_configs 20 \
    --num_contact_configs 5 \
    --num_directions 3 \
    --output_dir results/phase1
```

### Step 2: Sim-From-Real (τ_sim 계산)
```bash
# Real 데이터의 (q, F_ext)를 sim에서 재현 → τ_sim 기록
./isaaclab.sh -p scripts/run_phase1.py \
    --mode sim_from_real \
    --real_contact data/real_data/phase1_contact.npz \
    --output_dir results/phase1
```

### Step 3: Linear Calibration
```bash
# τ_sim vs I_real per-joint regression
./isaaclab.sh -p scripts/run_phase1.py \
    --mode calibrate \
    --sim_matched results/phase1/sim_matched.npz \
    --real_contact data/real_data/phase1_contact.npz \
    --output_dir results/phase1
```

### Step 4: (Optional) Learning-Based Calibration
```bash
# MLP + Residual model 학습
./isaaclab.sh -p scripts/run_phase1.py \
    --mode calibrate_learned \
    --sim_matched results/phase1/sim_matched.npz \
    --real_contact data/real_data/phase1_contact.npz \
    --model_type both \
    --num_epochs 200 \
    --output_dir results/phase1
```

### Real Robot 데이터 수집 형식

```python
import numpy as np

np.savez("data/real_data/phase1_contact.npz",
    q=q_actual,            # (N, 20) 실제 관절 위치 (접촉 후)
    q_cmd=q_cmd,           # (N, 20) 위치 명령 (PD target) ← 필수!
    qdot=qdot_data,        # (N, 20) 관절 속도
    I_motor=I_data,        # (N, 20) 모터 전류
    F_ext=ft_data,         # (N, 6) 외부 FT sensor wrench
    F_internal=ft_int,     # (N, 30) 내장 FT sensor (optional)
    finger_idx=finger_idx, # (N,) 손가락 인덱스
    config_idx=config_idx, # (N,) configuration 인덱스
    direction_idx=dir_idx, # (N,) 힘 방향 인덱스
)
```

---

## 7. 파일 구조

```
Real2Sim/
├── envs/phase1_current_torque/
│   ├── current_torque_env.py      # Sim 환경 (state machine + sim_from_real)
│   └── current_torque_env_cfg.py  # 환경 설정
│
├── calibration/phase1/
│   ├── current_torque_model.py    # JacobianCurrentTorqueModel (paired regression)
│   └── learned_model.py           # LearnedCurrentTorqueModel (MLP + Residual)
│
├── scripts/
│   └── run_phase1.py              # 실행 스크립트 (sim_full/sim_from_real/calibrate/calibrate_learned)
│
├── data/
│   ├── real_data/                  # Real robot 데이터
│   └── storage/calibration_results.py  # 결과 저장 클래스
│
├── results/phase1/
│   ├── sim_data.npz               # Sim validation 데이터
│   ├── sim_matched.npz            # Sim-from-real matched τ_sim
│   ├── calibration.npz            # Linear k_t 파라미터
│   ├── learned_model.pt           # Learning-based model weights
│   └── phase1_result.yaml         # 최종 결과
│
└── docs/
    └── phase1_current_torque.md   # 이 문서
```
