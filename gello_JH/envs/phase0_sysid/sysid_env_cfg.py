# Copyright (c) 2025, SRBL
# Phase 0: System Identification Environment Configuration

from __future__ import annotations

from isaaclab.utils import configclass

from ..base.real2sim_base_env_cfg import Real2SimBaseEnvCfg
from ...calibration.phase0.robot_config import RobotType


@configclass
class SysIDEnvCfg(Real2SimBaseEnvCfg):
    """Configuration for Phase 0: System Identification environment.

    This environment is used to identify joint dynamics parameters
    (stiffness, damping, friction) by replaying real robot trajectories.

    Supports both UR5 and Hand calibration using SIMPLER-style
    EE pose tracking loss.
    """

    # Override for SysID
    episode_length_s: float = 60.0  # Longer episodes for trajectory replay

    # Robot type for calibration
    robot_type: RobotType = RobotType.HAND
    """Robot type to calibrate: UR5 or HAND."""

    # Observation order for SysID
    obs_order: list = [
        "hand_joint_pos",
        "hand_joint_vel",
    ]

    # State includes privileged information
    state_order: list = [
        "hand_joint_pos",
        "hand_joint_vel",
        "hand_joint_torque",
        "joint_stiffness",
        "joint_damping",
        "joint_friction",
    ]

    # SysID specific settings
    trajectory_dir: str = ""
    """Directory containing real robot trajectories."""

    loss_weights: dict = {
        "position": 1.0,
        "velocity": 0.1,
        "joint": 0.1,
    }
    """Weights for position, velocity, and joint losses (SIMPLER style)."""

    # Control mode
    control_mode: str = "torque"
    """Control mode for replaying: 'torque' or 'position'."""

    # Settle time before recording
    settle_steps: int = 100
    """Number of steps to settle before comparing."""

    # SIMPLER optimization settings
    use_simpler_loss: bool = True
    """Use SIMPLER-style EE tracking loss instead of joint-only loss."""
