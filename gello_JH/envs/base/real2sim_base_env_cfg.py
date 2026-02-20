# Copyright (c) 2025, SRBL
# Real2Sim Calibration Framework - Base Environment Configuration

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

import os
import sys

# Add Real2Sim root to path for imports
REAL2SIM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REAL2SIM_DIR)

from sensors.tesollo_ft_sensor import TesolloFTSensorCfg

# Asset directories
ASSET_DIR = os.path.join(REAL2SIM_DIR, "assets")
TESOLLO_ASSET_DIR = "/home/Isaac/workspace/HD/Tesollo/assets"

# Observation and State dimensions
OBS_DIM_CFG = {
    # Hand proprioception
    "hand_joint_pos": 20,        # Tesollo DG5F joint positions
    "hand_joint_vel": 20,        # Tesollo DG5F joint velocities
    "hand_joint_torque": 20,     # Tesollo DG5F joint torques
    # Arm proprioception
    "arm_joint_pos": 6,          # UR5e joint positions
    "arm_joint_vel": 6,          # UR5e joint velocities
    # F/T sensor
    "fingertip_forces": 30,      # 5 fingers × 6 DOF (resultant per finger)
    "fingertip_forces_all": 210, # 35 pads × 6 DOF (all pads)
    # End-effector
    "ee_pos": 3,
    "ee_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    **OBS_DIM_CFG,
    # Privileged information (calibration parameters)
    "joint_stiffness": 20,
    "joint_damping": 20,
    "joint_friction": 20,
    "contact_friction": 2,       # static, dynamic
    "contact_params": 3,         # stiffness, damping, offset
}


@configclass
class CalibrationParamsCfg:
    """Initial calibration parameters (to be tuned)."""
    # Phase 0: Joint dynamics
    initial_joint_stiffness: float = 100.0
    initial_joint_damping: float = 10.0
    initial_joint_friction: float = 0.1

    # Phase 1: Current-Torque mapping (per joint)
    initial_k_gain: float = 0.15
    initial_k_offset: float = 0.02

    # Phase 2: Friction and contact
    initial_static_friction: float = 1.0
    initial_dynamic_friction: float = 0.8
    initial_contact_stiffness: float = 1e5
    initial_contact_damping: float = 1e3
    initial_contact_offset: float = 0.001


@configclass
class Real2SimBaseEnvCfg(DirectRLEnvCfg):
    """Base configuration for Real2Sim calibration environments."""

    # Environment settings
    decimation: int = 4  # 30Hz control at 120Hz physics
    episode_length_s: float = 30.0

    # Observation/action spaces (will be computed based on obs_order)
    observation_space: int = 0
    state_space: int = 0
    action_space: int = 20  # Tesollo hand joints

    # Observation ordering
    obs_order: list = [
        "hand_joint_pos",
        "hand_joint_vel",
        "fingertip_forces",
    ]
    state_order: list = [
        "hand_joint_pos",
        "hand_joint_vel",
        "hand_joint_torque",
        "fingertip_forces",
        "joint_stiffness",
        "joint_damping",
        "joint_friction",
    ]

    # Simulation configuration
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,  # TGS solver for better stability
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            enable_ccd=True,  # Continuous collision detection
            enable_stabilization=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,  # Single env for calibration
        env_spacing=2.0,
        clone_in_fabric=True,
    )

    # Tesollo Hand articulation configuration
    tesollo_hand: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot/dg5f_left",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{TESOLLO_ASSET_DIR}/dg5f_left_flattened.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                # Default hand pose (slightly open)
                "lj_dg_.*": 0.0,
            },
        ),
        actuators={
            "tesollo_fingers": ImplicitActuatorCfg(
                joint_names_expr=["lj_dg_.*"],
                stiffness=100.0,  # Initial value, will be calibrated
                damping=10.0,     # Initial value, will be calibrated
                friction=0.1,    # Initial value, will be calibrated
                effort_limit_sim=10.0,
                velocity_limit_sim=10.0,
            ),
        },
    )

    # UR5e arm configuration (optional, for Phase 1)
    ur5e_arm: ArticulationCfg | None = None  # Will be configured if needed

    # F/T sensor configuration
    tesollo_ft_sensor: TesolloFTSensorCfg = TesolloFTSensorCfg()

    # Calibration parameters
    calibration_params: CalibrationParamsCfg = CalibrationParamsCfg()

    # Data logging
    enable_logging: bool = True
    log_dir: str = os.path.join(REAL2SIM_DIR, "data", "sim_data")

    def __post_init__(self):
        """Compute observation and state spaces based on ordering."""
        self.observation_space = sum(OBS_DIM_CFG[obs] for obs in self.obs_order)
        self.state_space = sum(STATE_DIM_CFG[state] for state in self.state_order)


@configclass
class Real2SimWithArmEnvCfg(Real2SimBaseEnvCfg):
    """Configuration with UR5e arm for Phase 1 calibration."""

    ur5e_arm: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot/ur5e",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{REAL2SIM_DIR}/isaacsim_assets_for_export/ur5e/ur5e.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.57,
                "elbow_joint": 1.57,
                "wrist_1_joint": -1.57,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
            },
        ),
        actuators={
            "ur5e_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*_joint"],
                stiffness=0.0,  # Torque control
                damping=0.0,
                effort_limit_sim=150.0,
                velocity_limit_sim=3.14,
            ),
        },
    )
