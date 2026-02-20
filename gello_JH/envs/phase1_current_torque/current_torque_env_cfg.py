# Copyright (c) 2025, SRBL
# Phase 1: Jacobian-Based Current-Torque Calibration Environment Configuration

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from ..base.real2sim_base_env_cfg import Real2SimBaseEnvCfg, ASSET_DIR


@configclass
class CurrentTorqueEnvCfg(Real2SimBaseEnvCfg):
    """Configuration for Phase 1: Jacobian-Based Current-Torque Calibration.

    Calibrates the relationship between motor current (I) and joint torque (τ)
    using the Jacobian-based method:
        k_t × I = g(q) + friction(qdot) + J^T(q) × F_ext

    Phase 1A: Gravity/Friction Baseline (non-contact)
    Phase 1B: Contact Calibration (pressing against FT sensor)
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
    """Position control stiffness for hand joints."""

    hand_damping: float | list[float] = 10.0
    """Position control damping for hand joints."""

    # Phase 0 calibration result file (optional)
    phase0_result_file: str | None = None
    """Path to Phase 0 calibration result YAML file."""

    # ==========================================================================
    # Phase 1A: Gravity/Friction Baseline Settings
    # ==========================================================================

    num_gravity_configs: int = 20
    """Number of random joint configurations for gravity baseline."""

    gravity_settle_time: float = 1.5
    """Time to settle at each gravity config before recording (seconds)."""

    gravity_record_samples: int = 30
    """Number of samples to record at each gravity config."""

    num_velocity_sweeps: int = 10
    """Number of velocity sweep passes for friction measurement."""

    velocity_sweep_duration: float = 3.0
    """Duration of each velocity sweep (seconds)."""

    # ==========================================================================
    # Phase 1B: Contact Calibration Settings
    # ==========================================================================

    num_contact_configs: int = 5
    """Number of different joint configurations for contact calibration per finger."""

    num_force_directions: int = 3
    """Number of FT sensor directions per configuration."""

    contact_record_samples: int = 50
    """Number of samples to record at each contact condition."""

    ramp_duration: float = 5.0
    """Duration for position ramp (seconds)."""

    position_steps: int = 50
    """Number of discrete position steps during ramp."""

    settle_time: float = 0.5
    """Time to settle at each position step (seconds)."""

    # Joint position range for pressing (normalized 0~1 relative to joint limits)
    press_start_position: float = 0.3
    """Starting position for press motion (normalized)."""

    press_end_position: float = 0.8
    """End position for press motion (normalized)."""

    # Force thresholds
    min_contact_force: float = 0.5
    """Minimum force (N) to consider contact established."""

    max_contact_force: float = 15.0
    """Maximum force (N) before stopping press motion."""

    # ==========================================================================
    # Fingers to Calibrate
    # ==========================================================================

    fingers_to_calibrate: list[str] = ["thumb", "index", "middle", "ring", "pinky"]
    """Fingers to calibrate."""

    # Joint indices per finger (4 joints each)
    finger_joint_indices: dict = {
        "thumb": [0, 1, 2, 3],
        "index": [4, 5, 6, 7],
        "middle": [8, 9, 10, 11],
        "ring": [12, 13, 14, 15],
        "pinky": [16, 17, 18, 19],
    }
    """Joint indices for each finger."""

    # ==========================================================================
    # External FT Sensor Fixture
    # ==========================================================================

    ft_fixture: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/FTFixture",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.01),  # Small plate for fingertip contact
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Fixed in space
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.6, 0.2),  # Green color
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.15, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    """External FT sensor fixture configuration."""

    # FT sensor base positions per finger (relative to hand base)
    ft_sensor_positions: dict = {
        "thumb": (0.05, 0.08, 0.02),
        "index": (0.0, 0.15, 0.05),
        "middle": (-0.01, 0.16, 0.05),
        "ring": (-0.02, 0.15, 0.04),
        "pinky": (-0.03, 0.13, 0.03),
    }
    """FT sensor base positions for each finger."""

    # FT sensor directions for multi-direction calibration
    ft_directions: list[dict] = [
        {
            "name": "front",
            "offset": (0.0, 0.0, 0.0),
            "rot": (1.0, 0.0, 0.0, 0.0),
        },
        {
            "name": "side",
            "offset": (0.02, -0.01, 0.01),
            "rot": (0.92, 0.38, 0.0, 0.0),
        },
        {
            "name": "below",
            "offset": (-0.01, 0.0, -0.01),
            "rot": (0.96, 0.0, 0.26, 0.0),
        },
    ]
    """FT sensor positions/orientations for multi-direction calibration."""

    # ==========================================================================
    # Data Recording Settings
    # ==========================================================================

    record_frequency: int = 10
    """Record data every N physics steps (for contact phase)."""

    num_trials_per_finger: int = 3
    """Legacy: Number of press trials per finger. Use num_contact_configs instead."""

    # ==========================================================================
    # Observation Settings
    # ==========================================================================

    obs_order: list = [
        "hand_joint_pos",
        "hand_joint_vel",
        "hand_joint_torque",
        "fingertip_forces",
        "external_ft_force",
    ]
    """Observation components for Phase 1."""

    # ==========================================================================
    # Internal FT Sensor Usage
    # ==========================================================================

    use_internal_ft_sensors: bool = True
    """Whether to record internal Tesollo FT sensors (cross-validation)."""

    use_external_ft_sensor: bool = True
    """Whether to use external FT sensor for force measurement."""
