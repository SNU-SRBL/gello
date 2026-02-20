# Copyright (c) 2025, SRBL
# Time synchronization utilities for Real2Sim

from __future__ import annotations

import numpy as np
from scipy import interpolate
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.storage import TrajectoryData


@dataclass
class TimeSync:
    """Time synchronization between real and simulation."""

    # Time offset (sim_time = real_time + offset)
    offset: float = 0.0

    # Time scale (sim_time = real_time * scale)
    scale: float = 1.0

    def real_to_sim(self, real_time: float) -> float:
        """Convert real time to simulation time."""
        return real_time * self.scale + self.offset

    def sim_to_real(self, sim_time: float) -> float:
        """Convert simulation time to real time."""
        return (sim_time - self.offset) / self.scale

    @classmethod
    def from_alignment(
        cls,
        real_timestamps: np.ndarray,
        sim_timestamps: np.ndarray,
    ) -> TimeSync:
        """Create sync from aligned timestamps.

        Uses linear regression to find best offset and scale.
        """
        # Simple linear regression: sim = scale * real + offset
        A = np.vstack([real_timestamps, np.ones_like(real_timestamps)]).T
        scale, offset = np.linalg.lstsq(A, sim_timestamps, rcond=None)[0]

        return cls(offset=offset, scale=scale)


def interpolate_trajectory(
    trajectory: TrajectoryData,
    target_dt: float,
    method: str = "linear",
) -> TrajectoryData:
    """Interpolate trajectory to a different time step.

    Args:
        trajectory: Original trajectory data.
        target_dt: Target time step.
        method: Interpolation method ('linear', 'cubic', 'nearest').

    Returns:
        Interpolated trajectory data.
    """
    from ..data.storage import TrajectoryData

    if len(trajectory.timestamps) < 2:
        return trajectory

    # Create new time array
    t_start = trajectory.timestamps[0]
    t_end = trajectory.timestamps[-1]
    new_timestamps = np.arange(t_start, t_end, target_dt)

    # Interpolate each array
    interp_data = TrajectoryData(
        timestamps=new_timestamps,
        metadata=trajectory.metadata.copy(),
    )

    # Joint positions
    if len(trajectory.joint_positions) > 0:
        interp_data.joint_positions = _interpolate_array(
            trajectory.timestamps,
            trajectory.joint_positions,
            new_timestamps,
            method,
        )

    # Joint velocities
    if len(trajectory.joint_velocities) > 0:
        interp_data.joint_velocities = _interpolate_array(
            trajectory.timestamps,
            trajectory.joint_velocities,
            new_timestamps,
            method,
        )

    # Joint torques
    if len(trajectory.joint_torques) > 0:
        interp_data.joint_torques = _interpolate_array(
            trajectory.timestamps,
            trajectory.joint_torques,
            new_timestamps,
            method,
        )

    # Motor currents
    if len(trajectory.motor_currents) > 0:
        interp_data.motor_currents = _interpolate_array(
            trajectory.timestamps,
            trajectory.motor_currents,
            new_timestamps,
            method,
        )

    # Object positions
    if len(trajectory.object_positions) > 0:
        interp_data.object_positions = _interpolate_array(
            trajectory.timestamps,
            trajectory.object_positions,
            new_timestamps,
            method,
        )

    # Object velocities
    if len(trajectory.object_velocities) > 0:
        interp_data.object_velocities = _interpolate_array(
            trajectory.timestamps,
            trajectory.object_velocities,
            new_timestamps,
            method,
        )

    # F/T sensor data
    if trajectory.ft_sensor_data:
        interp_data.ft_sensor_data = {}
        for finger, data in trajectory.ft_sensor_data.items():
            if len(data) > 0:
                # Reshape for interpolation if needed
                orig_shape = data.shape
                if len(orig_shape) == 3:
                    # (T, pads, 6) -> (T, pads*6)
                    data_flat = data.reshape(orig_shape[0], -1)
                    interp_flat = _interpolate_array(
                        trajectory.timestamps,
                        data_flat,
                        new_timestamps,
                        method,
                    )
                    interp_data.ft_sensor_data[finger] = interp_flat.reshape(
                        len(new_timestamps), orig_shape[1], orig_shape[2]
                    )
                else:
                    interp_data.ft_sensor_data[finger] = _interpolate_array(
                        trajectory.timestamps,
                        data,
                        new_timestamps,
                        method,
                    )

    return interp_data


def _interpolate_array(
    old_times: np.ndarray,
    old_data: np.ndarray,
    new_times: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate a multi-dimensional array.

    Args:
        old_times: Original timestamps, shape (T_old,).
        old_data: Original data, shape (T_old, ...).
        new_times: New timestamps, shape (T_new,).
        method: Interpolation method.

    Returns:
        Interpolated data, shape (T_new, ...).
    """
    if len(old_data) == 0:
        return np.array([])

    # Handle different dimensions
    if old_data.ndim == 1:
        f = interpolate.interp1d(
            old_times, old_data, kind=method, bounds_error=False, fill_value="extrapolate"
        )
        return f(new_times)

    # For multi-dimensional data, interpolate each column
    new_shape = (len(new_times),) + old_data.shape[1:]
    new_data = np.zeros(new_shape)

    # Flatten to 2D for interpolation
    flat_old = old_data.reshape(len(old_data), -1)
    flat_new = new_data.reshape(len(new_data), -1)

    for i in range(flat_old.shape[1]):
        f = interpolate.interp1d(
            old_times, flat_old[:, i], kind=method, bounds_error=False, fill_value="extrapolate"
        )
        flat_new[:, i] = f(new_times)

    return flat_new.reshape(new_shape)


def resample_to_common_times(
    trajectory1: TrajectoryData,
    trajectory2: TrajectoryData,
    target_dt: float | None = None,
) -> tuple[TrajectoryData, TrajectoryData]:
    """Resample two trajectories to common timestamps.

    Args:
        trajectory1: First trajectory.
        trajectory2: Second trajectory.
        target_dt: Target time step. If None, uses the smaller dt.

    Returns:
        Tuple of resampled trajectories.
    """
    # Determine target dt
    if target_dt is None:
        dt1 = trajectory1.dt if trajectory1.dt > 0 else 0.01
        dt2 = trajectory2.dt if trajectory2.dt > 0 else 0.01
        target_dt = min(dt1, dt2)

    # Find common time range
    t_start = max(trajectory1.timestamps[0], trajectory2.timestamps[0])
    t_end = min(trajectory1.timestamps[-1], trajectory2.timestamps[-1])

    # Create common timestamps
    common_times = np.arange(t_start, t_end, target_dt)

    # Interpolate both trajectories
    traj1_resampled = interpolate_trajectory(trajectory1, target_dt)
    traj2_resampled = interpolate_trajectory(trajectory2, target_dt)

    # Trim to common range
    mask1 = (traj1_resampled.timestamps >= t_start) & (traj1_resampled.timestamps <= t_end)
    mask2 = (traj2_resampled.timestamps >= t_start) & (traj2_resampled.timestamps <= t_end)

    # Apply mask to all arrays
    traj1_resampled.timestamps = traj1_resampled.timestamps[mask1]
    traj2_resampled.timestamps = traj2_resampled.timestamps[mask2]

    if len(traj1_resampled.joint_positions) > 0:
        traj1_resampled.joint_positions = traj1_resampled.joint_positions[mask1]
    if len(traj2_resampled.joint_positions) > 0:
        traj2_resampled.joint_positions = traj2_resampled.joint_positions[mask2]

    # Joint velocities
    if len(traj1_resampled.joint_velocities) > 0:
        traj1_resampled.joint_velocities = traj1_resampled.joint_velocities[mask1]
    if len(traj2_resampled.joint_velocities) > 0:
        traj2_resampled.joint_velocities = traj2_resampled.joint_velocities[mask2]

    # Joint torques
    if len(traj1_resampled.joint_torques) > 0:
        traj1_resampled.joint_torques = traj1_resampled.joint_torques[mask1]
    if len(traj2_resampled.joint_torques) > 0:
        traj2_resampled.joint_torques = traj2_resampled.joint_torques[mask2]

    # Motor currents
    if len(traj1_resampled.motor_currents) > 0:
        traj1_resampled.motor_currents = traj1_resampled.motor_currents[mask1]
    if len(traj2_resampled.motor_currents) > 0:
        traj2_resampled.motor_currents = traj2_resampled.motor_currents[mask2]

    # Object positions
    if len(traj1_resampled.object_positions) > 0:
        traj1_resampled.object_positions = traj1_resampled.object_positions[mask1]
    if len(traj2_resampled.object_positions) > 0:
        traj2_resampled.object_positions = traj2_resampled.object_positions[mask2]

    # Object velocities
    if len(traj1_resampled.object_velocities) > 0:
        traj1_resampled.object_velocities = traj1_resampled.object_velocities[mask1]
    if len(traj2_resampled.object_velocities) > 0:
        traj2_resampled.object_velocities = traj2_resampled.object_velocities[mask2]

    # F/T sensor data
    if traj1_resampled.ft_sensor_data:
        for finger, data in traj1_resampled.ft_sensor_data.items():
            if len(data) > 0:
                traj1_resampled.ft_sensor_data[finger] = data[mask1]
    if traj2_resampled.ft_sensor_data:
        for finger, data in traj2_resampled.ft_sensor_data.items():
            if len(data) > 0:
                traj2_resampled.ft_sensor_data[finger] = data[mask2]

    return traj1_resampled, traj2_resampled


def compute_trajectory_error(
    real_traj: TrajectoryData,
    sim_traj: TrajectoryData,
) -> dict[str, float]:
    """Compute error metrics between real and simulated trajectories.

    Args:
        real_traj: Real robot trajectory.
        sim_traj: Simulated trajectory.

    Returns:
        Dictionary of error metrics.
    """
    errors = {}

    # Joint position error
    if len(real_traj.joint_positions) > 0 and len(sim_traj.joint_positions) > 0:
        # Ensure same length
        min_len = min(len(real_traj.joint_positions), len(sim_traj.joint_positions))
        pos_error = real_traj.joint_positions[:min_len] - sim_traj.joint_positions[:min_len]

        errors["position_mse"] = np.mean(pos_error ** 2)
        errors["position_rmse"] = np.sqrt(errors["position_mse"])
        errors["position_mae"] = np.mean(np.abs(pos_error))
        errors["position_max_error"] = np.max(np.abs(pos_error))

    # Joint velocity error
    if len(real_traj.joint_velocities) > 0 and len(sim_traj.joint_velocities) > 0:
        min_len = min(len(real_traj.joint_velocities), len(sim_traj.joint_velocities))
        vel_error = real_traj.joint_velocities[:min_len] - sim_traj.joint_velocities[:min_len]

        errors["velocity_mse"] = np.mean(vel_error ** 2)
        errors["velocity_rmse"] = np.sqrt(errors["velocity_mse"])
        errors["velocity_mae"] = np.mean(np.abs(vel_error))

    # F/T sensor error
    if real_traj.ft_sensor_data and sim_traj.ft_sensor_data:
        ft_errors = []
        for finger in real_traj.ft_sensor_data:
            if finger in sim_traj.ft_sensor_data:
                real_ft = real_traj.ft_sensor_data[finger]
                sim_ft = sim_traj.ft_sensor_data[finger]
                min_len = min(len(real_ft), len(sim_ft))
                ft_error = real_ft[:min_len] - sim_ft[:min_len]
                ft_errors.append(np.mean(ft_error ** 2))

        if ft_errors:
            errors["ft_sensor_mse"] = np.mean(ft_errors)

    return errors
