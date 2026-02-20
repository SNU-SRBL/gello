# Copyright (c) 2025, SRBL
# Phase 0: System Identification Environment

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path

from ..base.real2sim_base_env import Real2SimBaseEnv
from .sysid_env_cfg import SysIDEnvCfg
from ...replay.trajectory_replayer import TrajectoryReplayer
from ...data.storage import TrajectoryStorage, TrajectoryData
from ...calibration.phase0.robot_config import RobotType, FINGER_CONFIGS, get_robot_config
from ...calibration.utils.loss_functions import simpler_sysid_loss, hand_simpler_loss


class SysIDEnv(Real2SimBaseEnv):
    """System Identification Environment for Phase 0.

    This environment replays real robot trajectories and compares
    the simulated states with the real ones. Supports SIMPLER-style
    system identification with EE pose tracking.

    Supports:
    - UR5: Tool flange EE tracking
    - Hand: Per-finger fingertip tracking

    Workflow:
    1. Load real robot trajectory
    2. Apply same actions to simulation
    3. Compare EE poses and joint positions
    4. Compute SIMPLER loss for parameter optimization
    """

    cfg: SysIDEnvCfg
    """System Identification Environment for Phase 0.

    This environment replays real robot trajectories and compares
    the simulated states with the real ones. Supports SIMPLER-style
    system identification with EE pose tracking.

    Supports:
    - UR5: Tool flange EE tracking
    - Hand: Per-finger fingertip tracking

    Workflow:
    1. Load real robot trajectory
    2. Apply same actions to simulation
    3. Compare EE poses and joint positions
    4. Compute SIMPLER loss for parameter optimization
    """

    cfg: SysIDEnvCfg

    def __init__(self, cfg: SysIDEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Robot type
        self.robot_type = cfg.robot_type
        self.robot_config = get_robot_config(cfg.robot_type)

        # Trajectory replayer
        storage = TrajectoryStorage(cfg.trajectory_dir) if cfg.trajectory_dir else None
        self._trajectory_replayer = TrajectoryReplayer(storage, device=self.device)

        # Current trajectory loaded
        self._trajectory_loaded = False
        self._current_trajectory: TrajectoryData | None = None

        # Tracking data for loss computation (joint states)
        self._sim_positions: list[torch.Tensor] = []
        self._sim_velocities: list[torch.Tensor] = []
        self._real_positions: list[torch.Tensor] = []
        self._real_velocities: list[torch.Tensor] = []

        # SIMPLER EE tracking data
        self._sim_ee_positions: list[torch.Tensor] = []
        self._sim_ee_orientations: list[torch.Tensor] = []
        self._real_ee_positions: list[torch.Tensor] = []
        self._real_ee_orientations: list[torch.Tensor] = []

        # Step counter
        self._replay_step = 0
        self._settle_counter = 0

    def load_trajectory(self, filename: str):
        """Load a real robot trajectory for replay.

        Args:
            filename: Trajectory filename.
        """
        self._trajectory_replayer.load(filename)
        self._trajectory_replayer.prepare_for_sim(self.physics_dt)
        self._trajectory_loaded = True
        self._replay_step = 0

    def load_trajectory_data(self, trajectory: TrajectoryData):
        """Load trajectory from data object.

        Args:
            trajectory: Trajectory data.
        """
        self._trajectory_replayer.load_from_data(trajectory)
        self._trajectory_replayer.prepare_for_sim(self.physics_dt)
        self._trajectory_loaded = True
        self._current_trajectory = trajectory
        self._replay_step = 0

    def _apply_action(self):
        """Apply torque commands from real trajectory."""
        if not self._trajectory_loaded or self._trajectory_replayer.is_done:
            return

        if self.cfg.control_mode == "torque":
            # Get torques from real trajectory
            real_torques = self._trajectory_replayer.get_joint_torques_batch(self.num_envs)
            self._tesollo_hand.set_joint_effort_target(real_torques)
        else:
            # Position control mode
            real_positions = self._trajectory_replayer.get_joint_positions_batch(self.num_envs)
            self._tesollo_hand.set_joint_position_target(real_positions)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute observations and record tracking data."""
        obs = super()._get_observations()

        # Record data for loss computation (after settle period)
        if self._trajectory_loaded and self._settle_counter >= self.cfg.settle_steps:
            self._record_tracking_data()

        return obs

    def _record_tracking_data(self):
        """Record sim and real data for loss computation."""
        if self._trajectory_replayer.is_done:
            return

        # Simulated joint states
        sim_pos = self._tesollo_hand.data.joint_pos.clone()
        sim_vel = self._tesollo_hand.data.joint_vel.clone()

        # Real joint states from trajectory
        real_pos = self._trajectory_replayer.get_joint_positions_batch(self.num_envs)
        real_vel = self._trajectory_replayer.get_joint_velocities_batch(self.num_envs)

        self._sim_positions.append(sim_pos)
        self._sim_velocities.append(sim_vel)
        self._real_positions.append(real_pos)
        self._real_velocities.append(real_vel)

        # Record EE data based on robot type
        if self.robot_type == RobotType.UR5:
            self._record_ur5_ee_data()
        elif self.robot_type == RobotType.HAND:
            self._record_hand_ee_data()

        # Advance trajectory
        self._trajectory_replayer.step()
        self._replay_step += 1

    def _record_ur5_ee_data(self):
        """Record UR5 tool flange EE pose."""
        # Get simulated EE pose from robot body
        sim_ee_pos, sim_ee_rot = self._get_ur5_ee_pose()
        self._sim_ee_positions.append(sim_ee_pos)
        self._sim_ee_orientations.append(sim_ee_rot)

        # Get real EE pose from trajectory
        real_ee_pos = self._trajectory_replayer.get_ur5_ee_position_batch(self.num_envs)
        real_ee_rot = self._trajectory_replayer.get_ur5_ee_orientation_batch(self.num_envs)
        self._real_ee_positions.append(real_ee_pos)
        self._real_ee_orientations.append(real_ee_rot)

    def _record_hand_ee_data(self):
        """Record Hand fingertip poses (5 fingers)."""
        # Get simulated fingertip poses
        sim_fingertip_pos, sim_fingertip_rot = self._get_fingertip_poses()
        self._sim_ee_positions.append(sim_fingertip_pos)
        self._sim_ee_orientations.append(sim_fingertip_rot)

        # Get real fingertip poses from trajectory
        real_fingertip_pos = self._trajectory_replayer.get_fingertip_positions_batch(self.num_envs)
        real_fingertip_rot = self._trajectory_replayer.get_fingertip_orientations_batch(self.num_envs)
        self._real_ee_positions.append(real_fingertip_pos)
        self._real_ee_orientations.append(real_fingertip_rot)

    def _get_ur5_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get UR5 tool flange pose from simulation.

        Returns:
            Tuple of (position, orientation) tensors.
            Position: (num_envs, 3)
            Orientation: (num_envs, 4) quaternion wxyz
        """
        # Get body state for tool flange
        body_idx = self._tesollo_hand.find_bodies(self.robot_config.ee_body_name)[0][0]
        ee_pos = self._tesollo_hand.data.body_pos_w[:, body_idx, :].clone()
        ee_rot = self._tesollo_hand.data.body_quat_w[:, body_idx, :].clone()
        return ee_pos, ee_rot

    def _get_fingertip_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get all fingertip poses from simulation.

        Returns:
            Tuple of (positions, orientations) tensors.
            Positions: (num_envs, 5, 3)
            Orientations: (num_envs, 5, 4) quaternion wxyz
        """
        fingertip_pos = []
        fingertip_rot = []

        for finger_name in ["thumb", "index", "middle", "ring", "pinky"]:
            finger_cfg = FINGER_CONFIGS[finger_name]
            body_idx = self._tesollo_hand.find_bodies(finger_cfg.ee_body_name)[0][0]

            pos = self._tesollo_hand.data.body_pos_w[:, body_idx, :]
            rot = self._tesollo_hand.data.body_quat_w[:, body_idx, :]

            fingertip_pos.append(pos)
            fingertip_rot.append(rot)

        # Stack: (num_envs, 5, 3) and (num_envs, 5, 4)
        return torch.stack(fingertip_pos, dim=1), torch.stack(fingertip_rot, dim=1)

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward as negative tracking error."""
        if not self._trajectory_loaded:
            return torch.zeros(self.num_envs, device=self.device)

        # Compute instantaneous error
        real_pos = self._trajectory_replayer.get_joint_positions_batch(self.num_envs)
        real_vel = self._trajectory_replayer.get_joint_velocities_batch(self.num_envs)

        sim_pos = self._tesollo_hand.data.joint_pos
        sim_vel = self._tesollo_hand.data.joint_vel

        pos_error = torch.mean((sim_pos - real_pos) ** 2, dim=-1)
        vel_error = torch.mean((sim_vel - real_vel) ** 2, dim=-1)

        # Weighted loss as negative reward
        loss = (
            self.cfg.loss_weights["position"] * pos_error +
            self.cfg.loss_weights["velocity"] * vel_error
        )

        return -loss

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if trajectory replay is complete."""
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Truncate when trajectory is done or episode length reached
        trajectory_done = self._trajectory_replayer.is_done if self._trajectory_loaded else False
        time_done = self.episode_length_buf >= self.max_episode_length

        truncated = torch.full(
            (self.num_envs,),
            trajectory_done,
            dtype=torch.bool,
            device=self.device
        ) | time_done

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments and trajectory replayer."""
        super()._reset_idx(env_ids)

        # Reset trajectory replayer
        if self._trajectory_loaded:
            self._trajectory_replayer.reset()

        # Clear tracking data
        self._sim_positions = []
        self._sim_velocities = []
        self._real_positions = []
        self._real_velocities = []

        # Clear EE tracking data
        self._sim_ee_positions = []
        self._sim_ee_orientations = []
        self._real_ee_positions = []
        self._real_ee_orientations = []

        self._replay_step = 0
        self._settle_counter = 0

        # Set initial joint state from trajectory
        if self._trajectory_loaded:
            init_pos = self._trajectory_replayer.get_joint_positions_batch(self.num_envs)
            init_vel = torch.zeros_like(init_pos)
            self._tesollo_hand.write_joint_state_to_sim(init_pos, init_vel, env_ids=env_ids)

    def step(self, action: torch.Tensor | None = None):
        """Override step to handle trajectory replay."""
        # Increment settle counter
        self._settle_counter += 1

        # For SysID, we ignore external actions and use trajectory replay
        return super().step(torch.zeros(self.num_envs, self.cfg.action_space, device=self.device))

    # ===== Calibration Methods =====

    def compute_sysid_loss(self) -> torch.Tensor:
        """Compute system identification loss over entire trajectory.

        Returns:
            Total loss tensor (for gradient computation).
        """
        if not self._sim_positions:
            return torch.tensor(0.0, device=self.device)

        # Stack recorded data
        sim_pos = torch.stack(self._sim_positions, dim=0)  # (T, num_envs, num_joints)
        sim_vel = torch.stack(self._sim_velocities, dim=0)
        real_pos = torch.stack(self._real_positions, dim=0)
        real_vel = torch.stack(self._real_velocities, dim=0)

        # Compute MSE losses
        pos_mse = torch.mean((sim_pos - real_pos) ** 2)
        vel_mse = torch.mean((sim_vel - real_vel) ** 2)

        # Weighted total loss
        total_loss = (
            self.cfg.loss_weights["position"] * pos_mse +
            self.cfg.loss_weights["velocity"] * vel_mse
        )

        return total_loss

    def compute_simpler_loss(self) -> tuple[float, dict[str, float]]:
        """Compute SIMPLER-style system identification loss.

        Uses EE pose tracking (translation + rotation) plus optional joint loss.

        Returns:
            Tuple of (total_loss, loss_components dict).
        """
        if not self._sim_ee_positions:
            return 0.0, {}

        # Stack recorded EE data: (T, num_envs, ...) -> (T, ...)
        # Take mean over environments for now
        sim_ee_pos = torch.stack(self._sim_ee_positions, dim=0).mean(dim=1).cpu().numpy()
        sim_ee_rot = torch.stack(self._sim_ee_orientations, dim=0).mean(dim=1).cpu().numpy()
        real_ee_pos = torch.stack(self._real_ee_positions, dim=0).mean(dim=1).cpu().numpy()
        real_ee_rot = torch.stack(self._real_ee_orientations, dim=0).mean(dim=1).cpu().numpy()

        # Joint data
        sim_joint_pos = torch.stack(self._sim_positions, dim=0).mean(dim=1).cpu().numpy()
        real_joint_pos = torch.stack(self._real_positions, dim=0).mean(dim=1).cpu().numpy()

        if self.robot_type == RobotType.UR5:
            # UR5: Single EE (tool flange)
            return simpler_sysid_loss(
                real_ee_pos=real_ee_pos,
                sim_ee_pos=sim_ee_pos,
                real_ee_rot=real_ee_rot,
                sim_ee_rot=sim_ee_rot,
                real_joint_pos=real_joint_pos,
                sim_joint_pos=sim_joint_pos,
                joint_weight=self.cfg.loss_weights.get("joint", 0.1),
            )
        elif self.robot_type == RobotType.HAND:
            # Hand: Per-finger loss decomposition
            return hand_simpler_loss(
                real_fingertip_pos=real_ee_pos,  # (T, 5, 3)
                sim_fingertip_pos=sim_ee_pos,
                real_fingertip_rot=real_ee_rot,  # (T, 5, 4)
                sim_fingertip_rot=sim_ee_rot,
                real_joint_pos=real_joint_pos,  # (T, 20)
                sim_joint_pos=sim_joint_pos,
                finger_config=FINGER_CONFIGS,
                joint_weight=self.cfg.loss_weights.get("joint", 0.1),
            )
        else:
            return 0.0, {}

    def get_tracking_metrics(self) -> dict[str, float]:
        """Get tracking error metrics.

        Returns:
            Dictionary of metrics.
        """
        if not self._sim_positions:
            return {}

        sim_pos = torch.stack(self._sim_positions, dim=0)
        sim_vel = torch.stack(self._sim_velocities, dim=0)
        real_pos = torch.stack(self._real_positions, dim=0)
        real_vel = torch.stack(self._real_velocities, dim=0)

        pos_error = sim_pos - real_pos
        vel_error = sim_vel - real_vel

        return {
            "position_mse": torch.mean(pos_error ** 2).item(),
            "position_rmse": torch.sqrt(torch.mean(pos_error ** 2)).item(),
            "position_mae": torch.mean(torch.abs(pos_error)).item(),
            "position_max_error": torch.max(torch.abs(pos_error)).item(),
            "velocity_mse": torch.mean(vel_error ** 2).item(),
            "velocity_rmse": torch.sqrt(torch.mean(vel_error ** 2)).item(),
            "total_loss": self.compute_sysid_loss().item(),
        }

    def set_joint_dynamics(
        self,
        stiffness: torch.Tensor | None = None,
        damping: torch.Tensor | None = None,
        friction: torch.Tensor | None = None,
    ):
        """Set joint dynamics parameters for optimization.

        Args:
            stiffness: Joint stiffness values, shape (num_joints,) or scalar.
            damping: Joint damping values, shape (num_joints,) or scalar.
            friction: Joint friction values, shape (num_joints,) or scalar.
        """
        if stiffness is not None:
            stiffness_tensor = stiffness.to(self.device) if isinstance(stiffness, torch.Tensor) else torch.tensor(stiffness, device=self.device)
            self._calibration_params["joint_stiffness"] = stiffness_tensor.expand(self.num_envs, -1) if stiffness_tensor.dim() == 1 else stiffness_tensor
            self._apply_stiffness_to_actuators(stiffness_tensor)

        if damping is not None:
            damping_tensor = damping.to(self.device) if isinstance(damping, torch.Tensor) else torch.tensor(damping, device=self.device)
            self._calibration_params["joint_damping"] = damping_tensor.expand(self.num_envs, -1) if damping_tensor.dim() == 1 else damping_tensor
            self._apply_damping_to_actuators(damping_tensor)

        if friction is not None:
            friction_tensor = friction.to(self.device) if isinstance(friction, torch.Tensor) else torch.tensor(friction, device=self.device)
            self._calibration_params["joint_friction"] = friction_tensor.expand(self.num_envs, -1) if friction_tensor.dim() == 1 else friction_tensor
            self._apply_friction_to_actuators(friction_tensor)

    def _apply_stiffness_to_actuators(self, stiffness: torch.Tensor):
        """Apply stiffness values to implicit actuators.

        Args:
            stiffness: Stiffness values, shape (num_joints,).
        """
        # Access the implicit actuator model
        actuator_name = "tesollo_fingers"
        if actuator_name not in self._tesollo_hand.actuators:
            return

        actuator = self._tesollo_hand.actuators[actuator_name]

        # Ensure correct shape: (num_envs, num_joints)
        if stiffness.dim() == 1:
            stiffness = stiffness.unsqueeze(0).expand(self.num_envs, -1)

        # Update the stiffness tensor in the actuator
        # For ImplicitActuator, stiffness is stored as a parameter
        if hasattr(actuator, 'stiffness'):
            actuator.stiffness[:] = stiffness

    def _apply_damping_to_actuators(self, damping: torch.Tensor):
        """Apply damping values to implicit actuators.

        Args:
            damping: Damping values, shape (num_joints,).
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

    def _apply_friction_to_actuators(self, friction: torch.Tensor):
        """Apply friction values to implicit actuators.

        Args:
            friction: Friction values, shape (num_joints,).
        """
        actuator_name = "tesollo_fingers"
        if actuator_name not in self._tesollo_hand.actuators:
            return

        actuator = self._tesollo_hand.actuators[actuator_name]

        # Ensure correct shape: (num_envs, num_joints)
        if friction.dim() == 1:
            friction = friction.unsqueeze(0).expand(self.num_envs, -1)

        # Update the friction tensor in the actuator
        if hasattr(actuator, 'friction'):
            actuator.friction[:] = friction

    @property
    def trajectory_progress(self) -> float:
        """Get trajectory replay progress (0.0 to 1.0)."""
        return self._trajectory_replayer.progress

    @property
    def replay_step(self) -> int:
        """Get current replay step."""
        return self._replay_step
