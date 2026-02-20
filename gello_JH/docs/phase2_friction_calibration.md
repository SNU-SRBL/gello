# Phase 2: Friction & Contact Calibration

## Overview

Phase 2 calibrates friction and contact parameters using **position control** to perform slip tests. The method:
1. Grips an object with position-controlled fingers
2. Gradually loosens the grip (position sweep) until slip occurs
3. Measures **static friction (μ_s)** at slip onset
4. Tracks post-slip object motion using **ArUco markers** to calculate **dynamic friction (μ_d)**

### Comparison with Previous Method

| Aspect | Previous (Current Control) | New (Position Control) |
|--------|---------------------------|------------------------|
| Grip Control | Ramp motor current 0.5A → 0A | Position sweep (tight → loose) |
| Slip Detection | Current at slip (I_slip) | Position/Force at slip |
| Dynamic Friction | Not measured | ArUco marker tracking |
| Real Robot | Requires current control | Standard position control |

## Method

### Static Friction Measurement (μ_s)

At slip onset:
```
μ_s = F_tangential / F_normal
```

Where:
- `F_tangential`: Friction force parallel to contact surface
- `F_normal`: Force perpendicular to contact surface

### Dynamic Friction Measurement (μ_d)

After slip, the object falls/slides. Using ArUco marker tracking:

```
Motion equation: m × a = m × g - μ_d × N
Therefore:       μ_d = (g - a) × m / N
```

Where:
- `m`: Object mass (kg)
- `g`: Gravitational acceleration (9.81 m/s²)
- `a`: Measured vertical acceleration (m/s²)
- `N`: Normal force (N)

## Usage

### 1. Simulation Slip Test

Run a slip test in simulation to measure friction coefficients:

```bash
./isaaclab.sh -p scripts/run_phase2.py --mode sim \
    --output_dir results/phase2 \
    --object_mass 0.1 \
    --sweep_duration 10.0 \
    --phase0_result results/phase0/phase0_hand_result.yaml \
    --phase1_result results/phase1/phase1_result.yaml
```

**Arguments:**
- `--mode sim`: Run simulation slip test only
- `--object_mass`: Test object mass in kg (default: 0.1)
- `--sweep_duration`: Time to sweep grip from tight to loose (default: 10s)
- `--phase0_result`: Apply Phase 0 stiffness/damping calibration
- `--phase1_result`: Apply Phase 1 I→τ calibration

### 2. Calibration with Real Robot Targets

Optimize friction parameters to match real robot measurements:

```bash
./isaaclab.sh -p scripts/run_phase2.py --mode calibrate \
    --real_mu_s 0.6 \
    --real_mu_d 0.4 \
    --phase0_result results/phase0/phase0_hand_result.yaml \
    --phase1_result results/phase1/phase1_result.yaml \
    --output_dir results/phase2 \
    --n_trials 100
```

**Arguments:**
- `--real_mu_s`: Target static friction coefficient from real robot (required)
- `--real_mu_d`: Target dynamic friction coefficient (optional)
- `--real_q_slip`: Target grip position at slip (optional)
- `--n_trials`: Number of optimization trials (default: 100)
- `--use_bayesian`: Use Bayesian optimization (default: grid search)

### 3. Full Pipeline

Run both simulation and calibration:

```bash
./isaaclab.sh -p scripts/run_phase2.py --mode both \
    --real_mu_s 0.6 --real_mu_d 0.4 \
    --output_dir results/phase2
```

## ArUco Marker Tracking

### Simulation

In simulation, the object's ground truth pose is used directly. The `ArucoObjectTracker` class:
1. Records position history
2. Computes velocity/acceleration via finite differences
3. Calculates μ_d from post-slip motion

### Real Robot

For real experiments, attach an ArUco marker to the test object:

```python
# Marker configuration (in friction_env_cfg.py)
aruco_marker_id: int = 0          # Marker ID
aruco_marker_size: float = 0.02   # 2cm marker
tracking_fps: float = 30.0        # Camera FPS
tracking_history_length: int = 100
```

## Calibration Pipeline Chain

Phase 2 uses results from Phase 0 and Phase 1:

```
Phase 0 (System ID)
├── joint_stiffness: [20 values]
└── joint_damping: [20 values]
        ↓
Phase 1 (Current-Torque)
├── k_gain: [20 values]
└── k_offset: [20 values]
        ↓
Phase 2 (Friction)
├── Applies Phase 0 stiffness/damping to actuators
├── Uses Phase 1 I→τ mapping (if needed)
└── Optimizes μ_s and μ_d
```

## Data Format

### Input (Real Robot Measurements)

Required from real robot slip tests:
```python
real_data = {
    "static_friction": 0.6,        # μ_s at slip
    "dynamic_friction": 0.4,       # μ_d from post-slip tracking (optional)
    "q_slip": 0.45,                # Grip position at slip (optional)
    "F_tangential": 1.2,           # Tangential force at slip (N)
    "F_normal": 2.0,               # Normal force at slip (N)
}
```

### Output (sim_slip_data.npz)

```python
{
    "timestamps": np.ndarray,           # (T,) - Time stamps
    "grip_positions": np.ndarray,       # (T, num_envs) - Grip position
    "joint_positions": np.ndarray,      # (T, num_envs, 20) - Joint angles
    "joint_torques": np.ndarray,        # (T, num_envs, 20) - Applied torques
    "object_positions": np.ndarray,     # (T, num_envs, 3) - Object position
    "object_velocities": np.ndarray,    # (T, num_envs, 3) - Object velocity
    "slip_detected": np.ndarray,        # (num_envs,) - Slip detection flag
    "static_friction": np.ndarray,      # (num_envs,) - μ_s values
    "dynamic_friction": np.ndarray,     # (num_envs,) - μ_d values
    "dynamic_friction_valid": np.ndarray,  # (num_envs,) - Valid μ_d flag
    "q_slip": np.ndarray,               # (num_envs, 20) - Joint pos at slip
    "tau_slip": np.ndarray,             # (num_envs, 20) - Torque at slip
    "F_tangential_at_slip": np.ndarray, # (num_envs,) - F_t at slip
    "F_normal_at_slip": np.ndarray,     # (num_envs,) - F_n at slip
}
```

### Output (phase2_result.yaml)

```yaml
phase: 2
type: friction_calibration
parameters:
  static_friction: 0.5832
  dynamic_friction: 0.4021
simulation_results:
  static_friction: 0.5845
  dynamic_friction: 0.4156
targets:
  static_friction: 0.6
  dynamic_friction: 0.4
loss: 0.000234
config:
  n_trials: 100
  use_bayesian: false
  object_mass: 0.1
```

## Configuration Options

### Environment Configuration (FrictionEnvCfg)

```python
@configclass
class FrictionEnvCfg(Real2SimBaseEnvCfg):
    # Control mode
    control_mode: str = "position"

    # Position sweep settings
    initial_grip_position: float = 0.8    # Tight grip (normalized)
    final_grip_position: float = 0.3      # Loose grip
    sweep_duration: float = 10.0          # Seconds
    settle_time_before_sweep: float = 1.0 # Settle time

    # Object configuration
    object_mass: float = 0.1              # kg

    # Slip detection thresholds
    slip_velocity_threshold: float = 0.001      # m/s
    slip_displacement_threshold: float = 0.002  # m
    slip_acceleration_threshold: float = 0.5    # m/s²

    # ArUco tracking
    use_aruco_tracking: bool = True
    aruco_marker_id: int = 0
    aruco_marker_size: float = 0.02       # meters
    tracking_history_length: int = 100
    tracking_fps: float = 30.0

    # Dynamic friction measurement
    post_slip_tracking_duration: float = 2.0  # seconds
    min_slip_distance: float = 0.01           # meters
    gravity: float = 9.81                     # m/s²

    # Phase 0/1 calibration
    phase0_result_file: str | None = None
    phase1_result_file: str | None = None
```

## Verification Criteria

| Metric | Target |
|--------|--------|
| μ_s error | < 0.1 |
| μ_d error | < 0.15 |
| q_slip error | < 0.05 rad |
| μ_d measurement validity | > 80% of trials |

## Algorithm Flow

```
1. Initialize Environment
   ├── Load Phase 0 calibration (stiffness/damping)
   └── Load Phase 1 calibration (I→τ mapping)

2. Grip Object
   ├── Position control to initial_grip_position (0.8)
   └── Wait for settle_time (1.0s)

3. Position Sweep
   ├── Linear interpolation: 0.8 → 0.3 over 10s
   └── Monitor for slip

4. Slip Detection
   ├── Object velocity > threshold
   ├── Object displacement > threshold
   └── Object acceleration > threshold

5. Record Slip Event
   ├── q_slip: Joint positions at slip
   ├── τ_slip: Joint torques at slip
   ├── F_tangential, F_normal: Forces at slip
   └── μ_s = F_tangential / F_normal

6. Post-Slip Tracking (2s)
   ├── Track object position (ArUco)
   ├── Compute velocity/acceleration
   └── μ_d = (g - a) × m / N

7. Optimization
   ├── Grid search or Bayesian optimization
   ├── Match μ_s, μ_d with real robot
   └── Output optimal friction parameters
```

## File Structure

```
Real2Sim/
├── envs/phase2_friction/
│   ├── __init__.py
│   ├── friction_env_cfg.py      # Configuration
│   ├── friction_env.py          # Environment implementation
│   └── slip_detection.py        # Slip detectors + ArUco tracker
├── calibration/phase2/
│   ├── __init__.py
│   ├── friction_estimator.py    # Grid search estimator
│   ├── bayesian_optimizer.py    # Bayesian optimization
│   └── slip_matcher.py          # Position/Force matching
├── scripts/
│   └── run_phase2.py            # Main script
└── docs/
    └── phase2_friction_calibration.md
```

## Troubleshooting

### No Slip Detected

If no slip occurs during the test:
1. Decrease `initial_grip_position` (weaker initial grip)
2. Increase `sweep_duration` (slower sweep)
3. Check object friction material in simulation
4. Verify finger contact with object

### Invalid Dynamic Friction

If μ_d measurements are invalid:
1. Increase `post_slip_tracking_duration`
2. Lower `min_slip_distance` threshold
3. Ensure sufficient object motion after slip
4. Check ArUco marker visibility

### Poor Calibration Convergence

If optimization doesn't converge:
1. Increase `n_trials`
2. Adjust friction parameter bounds in config
3. Use Bayesian optimization (`--use_bayesian`)
4. Check Phase 0/1 calibration quality
