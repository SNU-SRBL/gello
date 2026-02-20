# Copyright (c) 2025, SRBL
# Phase 2: Friction & Contact Calibration Environment Configuration

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from ..base.real2sim_base_env_cfg import Real2SimBaseEnvCfg


@configclass
class FrictionEnvCfg(Real2SimBaseEnvCfg):
    """Configuration for Phase 2: Friction & Contact Calibration environment.

    This environment calibrates friction and contact parameters using position
    control to perform slip tests.

    Method:
    - UR5 arm is fixed with high stiffness position control
    - Hand grips object with position control
    - Gradually loosen grip (position sweep) until slip occurs
    - Measure μ_s (static friction) at slip onset
    - Track object post-slip motion (ArUco) to calculate μ_d (dynamic friction)

    Calibration target:
    - Static friction coefficient (μ_s): F_tangential / F_normal at slip onset
    - Dynamic friction coefficient (μ_d): Derived from post-slip object acceleration
    """

    # ==========================================================================
    # Control Mode Settings
    # ==========================================================================

    control_mode: str = "position"
    """Control mode for calibration (position control)."""

    # UR5 arm - fixed in place
    ur5_stiffness: float = 10000.0
    """High stiffness to keep UR5 fixed in position."""

    ur5_damping: float = 1000.0
    """High damping for UR5 position control."""

    # Hand position control (defaults - overridden by Phase 0 results if provided)
    hand_stiffness: float | list[float] = 100.0
    """Position control stiffness for hand joints. Can be per-joint list from Phase 0."""

    hand_damping: float | list[float] = 10.0
    """Position control damping for hand joints. Can be per-joint list from Phase 0."""

    # ==========================================================================
    # Phase 0/1 Calibration Results (for parameter chaining)
    # ==========================================================================

    phase0_result_file: str | None = None
    """Path to Phase 0 calibration result YAML file. Uses calibrated stiffness/damping."""

    phase1_result_file: str | None = None
    """Path to Phase 1 calibration result YAML file. Uses calibrated I→τ mapping."""

    # ==========================================================================
    # Position Sweep Settings (Grip Loosening)
    # ==========================================================================

    initial_grip_position: float = 0.8
    """Initial grip position (normalized 0~1). Higher = tighter grip."""

    final_grip_position: float = 0.3
    """Final grip position (normalized). Lower = looser grip."""

    sweep_duration: float = 10.0
    """Duration to sweep from initial to final position (seconds)."""

    settle_time_before_sweep: float = 1.0
    """Time to settle after grasping before starting sweep (seconds)."""

    # ==========================================================================
    # Observation Settings
    # ==========================================================================

    obs_order: list = [
        "hand_joint_pos",
        "hand_joint_vel",
        "hand_joint_torque",
        "fingertip_forces",
        "object_position",
        "object_velocity",
    ]
    """Observation components for Phase 2."""

    # ==========================================================================
    # Test Object Configuration
    # ==========================================================================

    test_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TestObject",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),  # 4cm cube
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),  # 100g
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3),  # In front of hand (grasp pose)
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    """Test object configuration for slip test."""

    object_mass: float = 0.1
    """Object mass in kg (for dynamic friction calculation)."""

    # ==========================================================================
    # Slip Detection Settings
    # ==========================================================================

    slip_velocity_threshold: float = 0.001
    """Object velocity threshold for slip detection (m/s)."""

    slip_displacement_threshold: float = 0.002
    """Object displacement threshold for slip detection (m)."""

    slip_acceleration_threshold: float = 0.5
    """Object acceleration threshold for slip confirmation (m/s²)."""

    # ==========================================================================
    # ArUco Marker Tracking Settings
    # ==========================================================================

    use_aruco_tracking: bool = True
    """Whether to use ArUco marker tracking for object motion."""

    aruco_marker_id: int = 0
    """ArUco marker ID attached to test object."""

    aruco_marker_size: float = 0.02
    """ArUco marker size in meters (2cm default)."""

    tracking_history_length: int = 100
    """Number of position samples to keep for velocity/acceleration calculation."""

    tracking_fps: float = 30.0
    """Tracking update frequency (Hz) for derivative calculation."""

    # ==========================================================================
    # Static Friction Measurement Settings
    # ==========================================================================

    min_contact_force: float = 0.5
    """Minimum force (N) to consider contact established."""

    force_measurement_window: float = 0.1
    """Time window (seconds) for averaging force measurement at slip point."""

    # ==========================================================================
    # Dynamic Friction Measurement Settings
    # ==========================================================================

    post_slip_tracking_duration: float = 2.0
    """Duration to track object after slip for μ_d calculation (seconds)."""

    min_slip_distance: float = 0.01
    """Minimum distance object must fall/slide for valid μ_d measurement (m)."""

    gravity: float = 9.81
    """Gravitational acceleration (m/s²)."""

    # ==========================================================================
    # Friction Parameter Bounds for Optimization
    # ==========================================================================

    friction_bounds: dict = {
        "static_friction": (0.1, 2.0),
        "dynamic_friction": (0.05, 1.5),
        "contact_stiffness": (1e4, 1e7),
        "contact_damping": (1e2, 1e5),
        "contact_offset": (0.0001, 0.01),
    }
    """Parameter bounds for friction optimization."""

    # ==========================================================================
    # Bayesian Optimization Settings
    # ==========================================================================

    n_optimization_trials: int = 100
    """Number of optimization trials."""

    n_initial_samples: int = 10
    """Number of initial random samples."""

    # ==========================================================================
    # Data Recording Settings
    # ==========================================================================

    record_frequency: int = 10
    """Record data every N physics steps."""

    num_trials_per_object: int = 5
    """Number of slip test trials per object configuration."""

    # ==========================================================================
    # Gripper Configuration
    # ==========================================================================

    grip_fingers: list[str] = ["thumb", "index"]
    """Fingers used for gripping."""

    finger_joint_indices: dict = {
        "thumb": [0, 1, 2, 3],
        "index": [4, 5, 6, 7],
        "middle": [8, 9, 10, 11],
        "ring": [12, 13, 14, 15],
        "pinky": [16, 17, 18, 19],
    }
    """Joint indices for each finger."""
