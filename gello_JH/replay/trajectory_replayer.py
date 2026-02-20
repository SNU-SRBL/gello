# Copyright (c) 2025, SRBL
# Trajectory replayer for Real2Sim calibration

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

from ..data.storage import TrajectoryStorage, TrajectoryData
from .sync_utils import interpolate_trajectory

if TYPE_CHECKING:
    from ..envs.base import Real2SimBaseEnv


class TrajectoryReplayer:
    """Replay real robot trajectories in simulation.

    Used for:
    - Phase 0: System identification by matching joint trajectories
    - Phase 1: Force profile replay for current-torque calibration
    - Phase 2: Slip test replay for friction calibration
    """

    def __init__(
        self,
        storage: TrajectoryStorage | None = None,
        device: str = "cuda:0",
    ):
        """Initialize trajectory replayer.

        Args:
            storage: Trajectory storage instance.
            device: Device for tensor operations.
        """
        self.storage = storage or TrajectoryStorage()
        self.device = device

        # Current trajectory
        self._trajectory: TrajectoryData | None = None
        self._playback_idx: int = 0
        self._is_playing: bool = False

        # Interpolated trajectory (sim dt)
        self._interp_trajectory: TrajectoryData | None = None

    def load(self, filename: str) -> TrajectoryData:
        """Load trajectory from file.

        Args:
            filename: Trajectory filename.

        Returns:
            Loaded trajectory data.
        """
        self._trajectory = self.storage.load(filename)
        self._playback_idx = 0
        self._is_playing = False
        return self._trajectory

    def load_from_data(self, trajectory: TrajectoryData):
        """Load trajectory from data object.

        Args:
            trajectory: Trajectory data object.
        """
        self._trajectory = trajectory
        self._playback_idx = 0
        self._is_playing = False

    def prepare_for_sim(self, sim_dt: float):
        """Prepare trajectory for simulation by interpolating to sim dt.

        Args:
            sim_dt: Simulation time step.
        """
        if self._trajectory is None:
            raise ValueError("No trajectory loaded")

        self._interp_trajectory = interpolate_trajectory(
            self._trajectory, sim_dt
        )
        self._playback_idx = 0

    def reset(self):
        """Reset playback to beginning."""
        self._playback_idx = 0
        self._is_playing = True

    def step(self) -> bool:
        """Advance to next time step.

        Returns:
            True if more data available, False if trajectory complete.
        """
        traj = self._interp_trajectory or self._trajectory
        if traj is None:
            return False

        self._playback_idx += 1
        return self._playback_idx < traj.num_steps

    @property
    def is_done(self) -> bool:
        """Check if trajectory playback is complete."""
        traj = self._interp_trajectory or self._trajectory
        if traj is None:
            return True
        return self._playback_idx >= traj.num_steps

    @property
    def current_time(self) -> float:
        """Get current playback time."""
        traj = self._interp_trajectory or self._trajectory
        if traj is None or len(traj.timestamps) == 0:
            return 0.0
        idx = min(self._playback_idx, len(traj.timestamps) - 1)
        return traj.timestamps[idx]

    @property
    def progress(self) -> float:
        """Get playback progress (0.0 to 1.0)."""
        traj = self._interp_trajectory or self._trajectory
        if traj is None or traj.num_steps <= 1:
            return 1.0
        return self._playback_idx / (traj.num_steps - 1)

    # ===== Data Access Methods =====

    def get_joint_positions(self) -> torch.Tensor:
        """Get joint positions at current time step.

        Returns:
            Tensor of shape (num_joints,) with joint positions.
        """
        traj = self._interp_trajectory or self._trajectory
        if traj is None or len(traj.joint_positions) == 0:
            return torch.zeros(20, device=self.device)

        idx = min(self._playback_idx, len(traj.joint_positions) - 1)
        return torch.from_numpy(traj.joint_positions[idx]).to(self.device).float()

    def get_joint_velocities(self) -> torch.Tensor:
        """Get joint velocities at current time step."""
        traj = self._interp_trajectory or self._trajectory
        if traj is None or len(traj.joint_velocities) == 0:
            return torch.zeros(20, device=self.device)

        idx = min(self._playback_idx, len(traj.joint_velocities) - 1)
        return torch.from_numpy(traj.joint_velocities[idx]).to(self.device).float()

    def get_joint_torques(self) -> torch.Tensor:
        """Get joint torques at current time step."""
        traj = self._interp_trajectory or self._trajectory
        if traj is None or len(traj.joint_torques) == 0:
            return torch.zeros(20, device=self.device)

        idx = min(self._playback_idx, len(traj.joint_torques) - 1)
        return torch.from_numpy(traj.joint_torques[idx]).to(self.device).float()

    def get_motor_currents(self) -> torch.Tensor:
        """Get motor currents at current time step."""
        traj = self._interp_trajectory or self._trajectory
        if traj is None or len(traj.motor_currents) == 0:
            return torch.zeros(20, device=self.device)

        idx = min(self._playback_idx, len(traj.motor_currents) - 1)
        return torch.from_numpy(traj.motor_currents[idx]).to(self.device).float()

    def get_ft_sensor_data(self, finger: str | None = None) -> torch.Tensor | dict:
        """Get F/T sensor data at current time step.

        Args:
            finger: Specific finger name, or None for all fingers.

        Returns:
            Tensor or dict of F/T data.
        """
        traj = self._interp_trajectory or self._trajectory
        if traj is None or not traj.ft_sensor_data:
            if finger:
                return torch.zeros(6, device=self.device)
            return {}

        idx = min(self._playback_idx, len(traj.timestamps) - 1)

        if finger:
            if finger in traj.ft_sensor_data:
                data = traj.ft_sensor_data[finger]
                if idx < len(data):
                    return torch.from_numpy(data[idx]).to(self.device).float()
            return torch.zeros(6, device=self.device)

        result = {}
        for f, data in traj.ft_sensor_data.items():
            if idx < len(data):
                result[f] = torch.from_numpy(data[idx]).to(self.device).float()
        return result

    def get_object_position(self) -> torch.Tensor:
        """Get object position at current time step (for Phase 2)."""
        traj = self._interp_trajectory or self._trajectory
        if traj is None or len(traj.object_positions) == 0:
            return torch.zeros(3, device=self.device)

        idx = min(self._playback_idx, len(traj.object_positions) - 1)
        return torch.from_numpy(traj.object_positions[idx]).to(self.device).float()

    # ===== Time-Based Access =====

    def get_positions_at_time(self, t: float) -> torch.Tensor:
        """Get joint positions at specified time.

        Args:
            t: Time in seconds.

        Returns:
            Interpolated joint positions.
        """
        traj = self._trajectory
        if traj is None or len(traj.joint_positions) == 0:
            return torch.zeros(20, device=self.device)

        idx = np.searchsorted(traj.timestamps, t)
        idx = min(idx, len(traj.joint_positions) - 1)

        # Linear interpolation
        if idx > 0 and idx < len(traj.timestamps):
            t0, t1 = traj.timestamps[idx - 1], traj.timestamps[idx]
            alpha = (t - t0) / (t1 - t0 + 1e-8)
            pos = (1 - alpha) * traj.joint_positions[idx - 1] + alpha * traj.joint_positions[idx]
        else:
            pos = traj.joint_positions[idx]

        return torch.from_numpy(pos).to(self.device).float()

    def get_velocities_at_time(self, t: float) -> torch.Tensor:
        """Get joint velocities at specified time."""
        traj = self._trajectory
        if traj is None or len(traj.joint_velocities) == 0:
            return torch.zeros(20, device=self.device)

        idx = np.searchsorted(traj.timestamps, t)
        idx = min(idx, len(traj.joint_velocities) - 1)

        if idx > 0 and idx < len(traj.timestamps):
            t0, t1 = traj.timestamps[idx - 1], traj.timestamps[idx]
            alpha = (t - t0) / (t1 - t0 + 1e-8)
            vel = (1 - alpha) * traj.joint_velocities[idx - 1] + alpha * traj.joint_velocities[idx]
        else:
            vel = traj.joint_velocities[idx]

        return torch.from_numpy(vel).to(self.device).float()

    def get_torques_at_time(self, t: float) -> torch.Tensor:
        """Get joint torques at specified time."""
        traj = self._trajectory
        if traj is None or len(traj.joint_torques) == 0:
            return torch.zeros(20, device=self.device)

        idx = np.searchsorted(traj.timestamps, t)
        idx = min(idx, len(traj.joint_torques) - 1)

        if idx > 0 and idx < len(traj.timestamps):
            t0, t1 = traj.timestamps[idx - 1], traj.timestamps[idx]
            alpha = (t - t0) / (t1 - t0 + 1e-8)
            torque = (1 - alpha) * traj.joint_torques[idx - 1] + alpha * traj.joint_torques[idx]
        else:
            torque = traj.joint_torques[idx]

        return torch.from_numpy(torque).to(self.device).float()

    def get_currents_at_time(self, t: float) -> torch.Tensor:
        """Get motor currents at specified time."""
        traj = self._trajectory
        if traj is None or len(traj.motor_currents) == 0:
            return torch.zeros(20, device=self.device)

        idx = np.searchsorted(traj.timestamps, t)
        idx = min(idx, len(traj.motor_currents) - 1)

        if idx > 0 and idx < len(traj.timestamps):
            t0, t1 = traj.timestamps[idx - 1], traj.timestamps[idx]
            alpha = (t - t0) / (t1 - t0 + 1e-8)
            current = (1 - alpha) * traj.motor_currents[idx - 1] + alpha * traj.motor_currents[idx]
        else:
            current = traj.motor_currents[idx]

        return torch.from_numpy(current).to(self.device).float()

    # ===== Batch Access for Multi-Env =====

    def get_joint_positions_batch(self, num_envs: int) -> torch.Tensor:
        """Get joint positions expanded for multiple environments.

        Args:
            num_envs: Number of environments.

        Returns:
            Tensor of shape (num_envs, num_joints).
        """
        pos = self.get_joint_positions()
        return pos.unsqueeze(0).expand(num_envs, -1)

    def get_joint_torques_batch(self, num_envs: int) -> torch.Tensor:
        """Get joint torques expanded for multiple environments."""
        torques = self.get_joint_torques()
        return torques.unsqueeze(0).expand(num_envs, -1)

    def get_motor_currents_batch(self, num_envs: int) -> torch.Tensor:
        """Get motor currents expanded for multiple environments."""
        currents = self.get_motor_currents()
        return currents.unsqueeze(0).expand(num_envs, -1)

    def get_ur5_ee_position_batch(self, num_envs: int) -> torch.Tensor:
        """Get UR5 tool flange position for multiple environments.

        Args:
            num_envs: Number of environments.

        Returns:
            Tensor of shape (num_envs, 3) with EE positions.
        """
        traj = self._interp_trajectory or self._trajectory
        if traj is None or len(traj.ur5_ee_position) == 0:
            return torch.zeros(num_envs, 3, device=self.device)

        idx = min(self._playback_idx, len(traj.ur5_ee_position) - 1)
        pos = torch.from_numpy(traj.ur5_ee_position[idx]).to(self.device).float()
        return pos.unsqueeze(0).expand(num_envs, -1)

    def get_ur5_ee_orientation_batch(self, num_envs: int) -> torch.Tensor:
        """Get UR5 tool flange orientation for multiple environments.

        Args:
            num_envs: Number of environments.

        Returns:
            Tensor of shape (num_envs, 4) with EE orientations (quaternion wxyz).
        """
        traj = self._interp_trajectory or self._trajectory
        if traj is None or len(traj.ur5_ee_orientation) == 0:
            return torch.zeros(num_envs, 4, device=self.device)

        idx = min(self._playback_idx, len(traj.ur5_ee_orientation) - 1)
        rot = torch.from_numpy(traj.ur5_ee_orientation[idx]).to(self.device).float()
        return rot.unsqueeze(0).expand(num_envs, -1)

    def get_fingertip_positions_batch(self, num_envs: int) -> torch.Tensor:
        """Get fingertip positions for multiple environments.

        Args:
            num_envs: Number of environments.

        Returns:
            Tensor of shape (num_envs, 5, 3) with fingertip positions.
        """
        traj = self._interp_trajectory or self._trajectory
        if traj is None or len(traj.fingertip_positions) == 0:
            return torch.zeros(num_envs, 5, 3, device=self.device)

        idx = min(self._playback_idx, len(traj.fingertip_positions) - 1)
        pos = torch.from_numpy(traj.fingertip_positions[idx]).to(self.device).float()
        return pos.unsqueeze(0).expand(num_envs, -1, -1)

    def get_fingertip_orientations_batch(self, num_envs: int) -> torch.Tensor:
        """Get fingertip orientations for multiple environments.

        Args:
            num_envs: Number of environments.

        Returns:
            Tensor of shape (num_envs, 5, 4) with fingertip orientations (quaternion wxyz).
        """
        traj = self._interp_trajectory or self._trajectory
        if traj is None or len(traj.fingertip_orientations) == 0:
            return torch.zeros(num_envs, 5, 4, device=self.device)

        idx = min(self._playback_idx, len(traj.fingertip_orientations) - 1)
        rot = torch.from_numpy(traj.fingertip_orientations[idx]).to(self.device).float()
        return rot.unsqueeze(0).expand(num_envs, -1, -1)
