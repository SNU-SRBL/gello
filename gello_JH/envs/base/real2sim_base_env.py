# Copyright (c) 2025, SRBL
# Real2Sim Calibration Framework - Base Environment

from __future__ import annotations

import torch
import numpy as np
from typing import Any

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation

from .real2sim_base_env_cfg import Real2SimBaseEnvCfg, OBS_DIM_CFG, STATE_DIM_CFG


class Real2SimBaseEnv(DirectRLEnv):
    """Base environment for Real2Sim calibration.

    Provides:
    - Tesollo DG5F hand articulation management
    - F/T sensor integration (35 sensors)
    - Trajectory replay capability
    - Data logging infrastructure

    This class is meant to be subclassed for specific calibration phases.
    """

    cfg: Real2SimBaseEnvCfg

    def __init__(self, cfg: Real2SimBaseEnvCfg, render_mode: str | None = None, **kwargs):
        # Compute observation/state dimensions from config
        cfg.observation_space = sum(OBS_DIM_CFG[obs] for obs in cfg.obs_order)
        cfg.state_space = sum(STATE_DIM_CFG[state] for state in cfg.state_order)

        super().__init__(cfg, render_mode, **kwargs)

        # Tesollo F/T sensor indices
        self._ft_sensor_indices: dict[str, list[int]] = {}

        # Data buffers
        self._init_data_buffers()

        # Calibration parameters (modifiable)
        self._calibration_params = {
            "joint_stiffness": torch.ones(self.num_envs, 20, device=self.device)
            * cfg.calibration_params.initial_joint_stiffness,
            "joint_damping": torch.ones(self.num_envs, 20, device=self.device)
            * cfg.calibration_params.initial_joint_damping,
            "joint_friction": torch.ones(self.num_envs, 20, device=self.device)
            * cfg.calibration_params.initial_joint_friction,
            "k_gain": torch.ones(self.num_envs, 20, device=self.device)
            * cfg.calibration_params.initial_k_gain,
            "k_offset": torch.ones(self.num_envs, 20, device=self.device)
            * cfg.calibration_params.initial_k_offset,
            "static_friction": torch.ones(self.num_envs, 1, device=self.device)
            * cfg.calibration_params.initial_static_friction,
            "dynamic_friction": torch.ones(self.num_envs, 1, device=self.device)
            * cfg.calibration_params.initial_dynamic_friction,
        }

    def _setup_scene(self):
        """Set up the simulation scene with Tesollo hand."""
        # Spawn Tesollo hand
        self._tesollo_hand = Articulation(self.cfg.tesollo_hand)
        self.scene.articulations["tesollo_hand"] = self._tesollo_hand

        # Spawn UR5e arm if configured
        if self.cfg.ur5e_arm is not None:
            self._ur5e_arm = Articulation(self.cfg.ur5e_arm)
            self.scene.articulations["ur5e_arm"] = self._ur5e_arm
        else:
            self._ur5e_arm = None

        # Add ground plane
        self._add_ground_plane()

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Set up F/T sensor indices after scene is ready
        self._setup_ft_sensor_indices()

    def _add_ground_plane(self):
        """Add ground plane to the scene."""
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners.materials import physics_materials_cfg

        ground_cfg = sim_utils.GroundPlaneCfg(
            physics_material=physics_materials_cfg.RigidBodyMaterialCfg(
                static_friction=self.cfg.calibration_params.initial_static_friction,
                dynamic_friction=self.cfg.calibration_params.initial_dynamic_friction,
                restitution=0.0,
            )
        )
        ground_cfg.func("/World/ground", ground_cfg)

    def _setup_ft_sensor_indices(self):
        """Set up indices for F/T sensor readings from joint forces."""
        fingers = self.cfg.tesollo_ft_sensor.fingers
        pads_per_finger = self.cfg.tesollo_ft_sensor.pads_per_finger

        # Get joint names from articulation
        joint_names = self._tesollo_hand.joint_names

        for finger in fingers:
            self._ft_sensor_indices[finger] = []
            for i in range(1, pads_per_finger + 1):
                joint_name = f"pad_{i}_{finger}_joint"
                if joint_name in joint_names:
                    idx = joint_names.index(joint_name)
                    self._ft_sensor_indices[finger].append(idx)

    def _init_data_buffers(self):
        """Initialize data buffers for logging and analysis."""
        self._data_buffer = {
            "timestamps": [],
            "joint_pos": [],
            "joint_vel": [],
            "joint_torques": [],
            "ft_sensor_data": [],
        }

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions before physics simulation step."""
        self._actions = actions.clone()

    def _apply_action(self):
        """Apply actions to the robot.

        Default: direct torque control on Tesollo hand joints.
        Override in subclasses for different control modes.
        """
        if hasattr(self, "_actions"):
            self._tesollo_hand.set_joint_effort_target(self._actions)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute observations based on cfg.obs_order."""
        obs_dict = {}

        for obs_name in self.cfg.obs_order:
            if obs_name == "hand_joint_pos":
                obs_dict[obs_name] = self._tesollo_hand.data.joint_pos
            elif obs_name == "hand_joint_vel":
                obs_dict[obs_name] = self._tesollo_hand.data.joint_vel
            elif obs_name == "hand_joint_torque":
                obs_dict[obs_name] = self._tesollo_hand.data.applied_torque
            elif obs_name == "fingertip_forces":
                obs_dict[obs_name] = self._get_fingertip_forces_resultant()
            elif obs_name == "fingertip_forces_all":
                obs_dict[obs_name] = self._get_fingertip_forces_all()
            elif obs_name == "ee_pos":
                obs_dict[obs_name] = self._get_ee_position()
            elif obs_name == "ee_quat":
                obs_dict[obs_name] = self._get_ee_quaternion()
            elif obs_name == "ee_linvel":
                obs_dict[obs_name] = self._get_ee_linear_velocity()
            elif obs_name == "ee_angvel":
                obs_dict[obs_name] = self._get_ee_angular_velocity()

        # Concatenate observations
        policy_obs = torch.cat([obs_dict[k] for k in self.cfg.obs_order], dim=-1)

        return {"policy": policy_obs}

    def _get_privileged_observations(self) -> dict[str, torch.Tensor]:
        """Compute privileged observations (state) for asymmetric training."""
        state_dict = {}

        for state_name in self.cfg.state_order:
            if state_name in ["hand_joint_pos", "hand_joint_vel", "hand_joint_torque",
                              "fingertip_forces", "ee_pos", "ee_quat", "ee_linvel", "ee_angvel"]:
                # Same as policy observations
                state_dict[state_name] = self._get_observations()["policy"]
            elif state_name == "joint_stiffness":
                state_dict[state_name] = self._calibration_params["joint_stiffness"]
            elif state_name == "joint_damping":
                state_dict[state_name] = self._calibration_params["joint_damping"]
            elif state_name == "joint_friction":
                state_dict[state_name] = self._calibration_params["joint_friction"]
            elif state_name == "contact_friction":
                state_dict[state_name] = torch.cat([
                    self._calibration_params["static_friction"],
                    self._calibration_params["dynamic_friction"],
                ], dim=-1)

        return state_dict

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards. Override in subclasses for specific calibration objectives."""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions."""
        # Default: no early termination
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)

        super()._reset_idx(env_ids)

        # Reset hand to default pose
        default_joint_pos = self._tesollo_hand.data.default_joint_pos[env_ids]
        default_joint_vel = torch.zeros_like(default_joint_pos)

        self._tesollo_hand.write_joint_state_to_sim(
            default_joint_pos,
            default_joint_vel,
            env_ids=env_ids,
        )

        # Clear data buffers
        if self.cfg.enable_logging:
            self._clear_data_buffers()

    # ===== F/T Sensor Methods =====

    def _get_fingertip_forces_all(self) -> torch.Tensor:
        """Get all F/T sensor readings (35 pads × 6 DOF = 210 dims)."""
        # Get measured joint forces from articulation
        joint_forces = self._tesollo_hand.root_physx_view.get_link_incoming_joint_force()

        # Extract forces for all pads
        all_forces = []
        for finger in self.cfg.tesollo_ft_sensor.fingers:
            indices = self._ft_sensor_indices.get(finger, [])
            if indices:
                finger_forces = joint_forces[:, indices, :]  # (num_envs, 7, 6)
                all_forces.append(finger_forces.reshape(self.num_envs, -1))

        if all_forces:
            return torch.cat(all_forces, dim=-1)
        return torch.zeros(self.num_envs, 210, device=self.device)

    def _get_fingertip_forces_resultant(self) -> torch.Tensor:
        """Get resultant F/T per finger (5 fingers × 6 DOF = 30 dims)."""
        joint_forces = self._tesollo_hand.root_physx_view.get_link_incoming_joint_force()

        resultants = []
        for finger in self.cfg.tesollo_ft_sensor.fingers:
            indices = self._ft_sensor_indices.get(finger, [])
            if indices:
                finger_forces = joint_forces[:, indices, :]  # (num_envs, 7, 6)
                # Sum across pads
                resultant = finger_forces.sum(dim=1)  # (num_envs, 6)
                # Apply axis transformation
                resultant = self._transform_ft_axes(resultant, finger)
                resultants.append(resultant)
            else:
                resultants.append(torch.zeros(self.num_envs, 6, device=self.device))

        return torch.cat(resultants, dim=-1)

    def _transform_ft_axes(self, force_torque: torch.Tensor, finger: str) -> torch.Tensor:
        """Transform F/T sensor axes to standard frame."""
        if finger == "thumb":
            # Thumb: negate X and Y
            transformed = torch.stack([
                -force_torque[:, 0],  # Fx
                -force_torque[:, 1],  # Fy
                force_torque[:, 2],   # Fz
                -force_torque[:, 3],  # Tx
                -force_torque[:, 4],  # Ty
                force_torque[:, 5],   # Tz
            ], dim=-1)
        else:
            # Other fingers: permute axes
            transformed = torch.stack([
                force_torque[:, 1],   # Fx = old Fy
                force_torque[:, 2],   # Fy = old Fz
                force_torque[:, 0],   # Fz = old Fx
                force_torque[:, 4],   # Tx = old Ty
                force_torque[:, 5],   # Ty = old Tz
                force_torque[:, 3],   # Tz = old Tx
            ], dim=-1)

        return transformed

    # ===== End-Effector Methods =====

    def _get_ee_position(self) -> torch.Tensor:
        """Get end-effector position."""
        # Use wrist link as end-effector for now
        return self._tesollo_hand.data.root_pos_w

    def _get_ee_quaternion(self) -> torch.Tensor:
        """Get end-effector quaternion."""
        return self._tesollo_hand.data.root_quat_w

    def _get_ee_linear_velocity(self) -> torch.Tensor:
        """Get end-effector linear velocity."""
        return self._tesollo_hand.data.root_lin_vel_w

    def _get_ee_angular_velocity(self) -> torch.Tensor:
        """Get end-effector angular velocity."""
        return self._tesollo_hand.data.root_ang_vel_w

    # ===== Calibration Parameter Methods =====

    def set_calibration_params(self, params: dict[str, torch.Tensor]):
        """Set calibration parameters and apply to simulation."""
        for key, value in params.items():
            if key in self._calibration_params:
                self._calibration_params[key] = value.to(self.device)

        # Apply joint dynamics parameters to actuators
        if "joint_stiffness" in params:
            self._apply_joint_stiffness(params["joint_stiffness"])
        if "joint_damping" in params:
            self._apply_joint_damping(params["joint_damping"])

    def _apply_joint_stiffness(self, stiffness: torch.Tensor):
        """Apply joint stiffness to implicit actuators.

        Args:
            stiffness: Stiffness values, shape (num_joints,) or (num_envs, num_joints).
        """
        actuator_name = "tesollo_fingers"
        if actuator_name not in self._tesollo_hand.actuators:
            return

        actuator = self._tesollo_hand.actuators[actuator_name]

        # Ensure correct shape: (num_envs, num_joints)
        if stiffness.dim() == 1:
            stiffness = stiffness.unsqueeze(0).expand(self.num_envs, -1)

        # Update the stiffness tensor in the actuator
        if hasattr(actuator, 'stiffness'):
            actuator.stiffness[:] = stiffness

    def _apply_joint_damping(self, damping: torch.Tensor):
        """Apply joint damping to implicit actuators.

        Args:
            damping: Damping values, shape (num_joints,) or (num_envs, num_joints).
        """
        actuator_name = "tesollo_fingers"
        if actuator_name not in self._tesollo_hand.actuators:
            return

        actuator = self._tesollo_hand.actuators[actuator_name]

        # Ensure correct shape: (num_envs, num_joints)
        if damping.dim() == 1:
            damping = damping.unsqueeze(0).expand(self.num_envs, -1)

        # Update the damping tensor in the actuator
        if hasattr(actuator, 'damping'):
            actuator.damping[:] = damping

    def get_calibration_params(self) -> dict[str, torch.Tensor]:
        """Get current calibration parameters."""
        return self._calibration_params.copy()

    # ===== Data Logging Methods =====

    def _log_step_data(self):
        """Log data for current step."""
        if not self.cfg.enable_logging:
            return

        self._data_buffer["timestamps"].append(self.sim.current_time)
        self._data_buffer["joint_pos"].append(
            self._tesollo_hand.data.joint_pos.cpu().numpy().copy()
        )
        self._data_buffer["joint_vel"].append(
            self._tesollo_hand.data.joint_vel.cpu().numpy().copy()
        )
        self._data_buffer["joint_torques"].append(
            self._tesollo_hand.data.applied_torque.cpu().numpy().copy()
        )
        self._data_buffer["ft_sensor_data"].append(
            self._get_fingertip_forces_resultant().cpu().numpy().copy()
        )

    def _clear_data_buffers(self):
        """Clear data buffers."""
        for key in self._data_buffer:
            self._data_buffer[key] = []

    def get_logged_data(self) -> dict:
        """Get logged data as numpy arrays."""
        return {
            key: np.array(value) if value else np.array([])
            for key, value in self._data_buffer.items()
        }

    # ===== Utility Methods =====

    def current_to_torque(self, currents: torch.Tensor) -> torch.Tensor:
        """Convert motor currents to joint torques using Phase 1 calibration.

        τ = k_gain × I + k_offset
        """
        k_gain = self._calibration_params["k_gain"]
        k_offset = self._calibration_params["k_offset"]

        return k_gain * currents + k_offset

    def apply_external_force(self, body_idx: int, force: torch.Tensor):
        """Apply external force to a body (for Phase 1 contact-free calibration).

        Uses PhysX API to apply force directly without contact/friction.
        """
        forces = torch.zeros(self.num_envs, self._tesollo_hand.num_bodies, 3, device=self.device)
        forces[:, body_idx, :] = force

        torques = torch.zeros_like(forces)

        self._tesollo_hand.set_external_force_and_torque(forces, torques)
