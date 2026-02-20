# Copyright (c) 2025, SRBL
# Slip detection and object tracking for Phase 2 calibration

from __future__ import annotations

import torch
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Tuple


@dataclass
class SlipEvent:
    """Data recorded at slip onset."""

    timestamp: float
    """Time when slip occurred."""

    q_slip: np.ndarray
    """Joint positions at slip (20,)."""

    tau_slip: np.ndarray
    """Joint torques at slip (20,)."""

    F_tangential: float
    """Tangential (friction) force at slip (N)."""

    F_normal: float
    """Normal force at slip (N)."""

    static_friction: float
    """Computed static friction coefficient (μ_s = F_tangential / F_normal)."""

    grip_position: float
    """Normalized grip position at slip."""

    object_position: np.ndarray
    """Object position at slip (3,)."""


@dataclass
class DynamicFrictionData:
    """Data from post-slip object tracking for dynamic friction calculation."""

    timestamps: np.ndarray
    """Time stamps (T,)."""

    positions: np.ndarray
    """Object positions (T, 3)."""

    velocities: np.ndarray
    """Computed velocities (T, 3)."""

    accelerations: np.ndarray
    """Computed accelerations (T, 3)."""

    dynamic_friction: float
    """Computed dynamic friction coefficient (μ_d)."""

    measurement_valid: bool
    """Whether the measurement is valid (enough motion, stable tracking)."""


class SlipDetector:
    """Slip detector for friction calibration.

    Detects object slip based on:
    1. Object velocity exceeding threshold
    2. Object displacement exceeding threshold
    3. Object acceleration (sudden motion)

    In real experiments, this would use vision-based tracking
    (ArUco markers) or tactile sensing.
    """

    def __init__(
        self,
        velocity_threshold: float = 0.001,
        displacement_threshold: float = 0.002,
        acceleration_threshold: float = 0.5,
        device: str = "cuda:0",
        num_envs: int = 1,
    ):
        """Initialize slip detector.

        Args:
            velocity_threshold: Velocity threshold for slip (m/s).
            displacement_threshold: Displacement threshold for slip (m).
            acceleration_threshold: Acceleration threshold for slip (m/s²).
            device: Device for tensor operations.
            num_envs: Number of environments.
        """
        self.velocity_threshold = velocity_threshold
        self.displacement_threshold = displacement_threshold
        self.acceleration_threshold = acceleration_threshold
        self.device = device
        self._num_envs = num_envs

        # Initial object position (set on first update or reset)
        self._initial_position: torch.Tensor | None = None

        # Current state
        self._current_position: torch.Tensor | None = None
        self._current_velocity: torch.Tensor | None = None
        self._prev_velocity: torch.Tensor | None = None

        # Slip flags
        self._slip_detected = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._slip_timestamp = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset slip detector for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self._initial_position = None
            self._current_position = None
            self._current_velocity = None
            self._prev_velocity = None
            self._slip_detected[:] = False
            self._slip_timestamp[:] = 0.0
        else:
            if self._initial_position is not None:
                self._initial_position[env_ids] = self._current_position[env_ids].clone() \
                    if self._current_position is not None else torch.zeros(len(env_ids), 3, device=self.device)
            self._slip_detected[env_ids] = False
            self._slip_timestamp[env_ids] = 0.0

    def update(self, position: torch.Tensor, velocity: torch.Tensor, timestamp: float = 0.0):
        """Update detector with current object state.

        Args:
            position: Object position, shape (num_envs, 3).
            velocity: Object velocity, shape (num_envs, 3).
            timestamp: Current simulation time.
        """
        self._prev_velocity = self._current_velocity.clone() if self._current_velocity is not None else None
        self._current_position = position.clone()
        self._current_velocity = velocity.clone()

        # Set initial position on first update
        if self._initial_position is None:
            self._initial_position = position.clone()

        # Update num_envs if needed
        if position.shape[0] != self._num_envs:
            self._num_envs = position.shape[0]
            self._slip_detected = torch.zeros(
                self._num_envs, dtype=torch.bool, device=self.device
            )
            self._slip_timestamp = torch.zeros(self._num_envs, device=self.device)

    def check_slip(self, timestamp: float = 0.0) -> torch.Tensor:
        """Check if slip has occurred.

        Args:
            timestamp: Current simulation time.

        Returns:
            Boolean tensor of shape (num_envs,) indicating slip.
        """
        if self._current_position is None or self._initial_position is None:
            return self._slip_detected

        # Velocity-based detection
        velocity_magnitude = torch.norm(self._current_velocity, dim=-1)
        velocity_slip = velocity_magnitude > self.velocity_threshold

        # Displacement-based detection (Z direction for dropping)
        displacement = self._current_position - self._initial_position
        displacement_z = torch.abs(displacement[:, 2])
        displacement_slip = displacement_z > self.displacement_threshold

        # Acceleration-based detection
        if self._prev_velocity is not None:
            acceleration = self._current_velocity - self._prev_velocity
            accel_magnitude = torch.norm(acceleration, dim=-1)
            accel_slip = accel_magnitude > self.acceleration_threshold
        else:
            accel_slip = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)

        # Combined detection
        slip = velocity_slip | displacement_slip | accel_slip

        # Record slip timestamp for newly detected slips
        newly_slipped = slip & ~self._slip_detected
        self._slip_timestamp[newly_slipped] = timestamp

        # Update internal flag
        self._slip_detected |= slip

        return slip

    def get_displacement(self) -> torch.Tensor:
        """Get object displacement from initial position.

        Returns:
            Displacement magnitude, shape (num_envs,).
        """
        if self._current_position is None or self._initial_position is None:
            return torch.zeros(self._num_envs, device=self.device)

        displacement = self._current_position - self._initial_position
        return torch.norm(displacement, dim=-1)

    def get_velocity(self) -> torch.Tensor:
        """Get current object velocity magnitude.

        Returns:
            Velocity magnitude, shape (num_envs,).
        """
        if self._current_velocity is None:
            return torch.zeros(self._num_envs, device=self.device)

        return torch.norm(self._current_velocity, dim=-1)

    def is_slipped(self) -> torch.Tensor:
        """Get slip status.

        Returns:
            Boolean tensor indicating if slip has been detected.
        """
        return self._slip_detected

    def get_slip_timestamp(self) -> torch.Tensor:
        """Get timestamp when slip was detected.

        Returns:
            Tensor of slip timestamps.
        """
        return self._slip_timestamp

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._num_envs


class ArucoObjectTracker:
    """ArUco marker-based object tracker for friction measurement.

    Tracks object position over time using ArUco markers and computes
    velocity and acceleration for dynamic friction calculation.

    In simulation, this uses ground truth object pose.
    In real experiments, this would use camera + ArUco detection.
    """

    def __init__(
        self,
        marker_id: int = 0,
        marker_size: float = 0.02,
        history_length: int = 100,
        tracking_fps: float = 30.0,
        device: str = "cuda:0",
        num_envs: int = 1,
    ):
        """Initialize ArUco object tracker.

        Args:
            marker_id: ArUco marker ID to track.
            marker_size: Marker size in meters.
            history_length: Number of position samples to keep.
            tracking_fps: Expected tracking frequency (Hz).
            device: Device for tensor operations.
            num_envs: Number of environments.
        """
        self.marker_id = marker_id
        self.marker_size = marker_size
        self.history_length = history_length
        self.tracking_fps = tracking_fps
        self.expected_dt = 1.0 / tracking_fps
        self.device = device
        self._num_envs = num_envs

        # Position history: list of (timestamp, positions) tuples
        # positions shape: (num_envs, 3)
        self._history: deque = deque(maxlen=history_length)

        # Computed derivatives
        self._velocity: torch.Tensor | None = None
        self._acceleration: torch.Tensor | None = None

        # Tracking state
        self._is_tracking = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._slip_start_index = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset tracker.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self._history.clear()
            self._velocity = None
            self._acceleration = None
            self._is_tracking[:] = False
            self._slip_start_index[:] = 0
        else:
            # For partial reset, mark these envs as not tracking
            # Full history reset would require more complex bookkeeping
            self._is_tracking[env_ids] = False
            self._slip_start_index[env_ids] = len(self._history)

    def update(self, timestamp: float, position: torch.Tensor):
        """Update tracker with new position measurement.

        Args:
            timestamp: Measurement timestamp.
            position: Object position, shape (num_envs, 3).
        """
        self._history.append((timestamp, position.clone()))
        self._is_tracking[:] = True

        # Compute derivatives if we have enough samples
        self._compute_derivatives()

    def _compute_derivatives(self):
        """Compute velocity and acceleration from position history."""
        if len(self._history) < 2:
            self._velocity = None
            self._acceleration = None
            return

        # Get last two samples for velocity
        t1, p1 = self._history[-2]
        t2, p2 = self._history[-1]
        dt = t2 - t1

        if dt > 0:
            self._velocity = (p2 - p1) / dt
        else:
            self._velocity = torch.zeros_like(p2)

        # Compute acceleration if we have 3+ samples
        if len(self._history) >= 3:
            t0, p0 = self._history[-3]
            dt1 = t1 - t0
            dt2 = t2 - t1

            if dt1 > 0 and dt2 > 0:
                v1 = (p1 - p0) / dt1
                v2 = (p2 - p1) / dt2
                dt_avg = (dt1 + dt2) / 2
                self._acceleration = (v2 - v1) / dt_avg
            else:
                self._acceleration = torch.zeros_like(p2)
        else:
            self._acceleration = torch.zeros_like(p2)

    def get_velocity(self) -> torch.Tensor:
        """Get current velocity estimate.

        Returns:
            Velocity tensor, shape (num_envs, 3).
        """
        if self._velocity is None:
            return torch.zeros(self._num_envs, 3, device=self.device)
        return self._velocity

    def get_acceleration(self) -> torch.Tensor:
        """Get current acceleration estimate.

        Returns:
            Acceleration tensor, shape (num_envs, 3).
        """
        if self._acceleration is None:
            return torch.zeros(self._num_envs, 3, device=self.device)
        return self._acceleration

    def get_velocity_magnitude(self) -> torch.Tensor:
        """Get velocity magnitude.

        Returns:
            Velocity magnitude, shape (num_envs,).
        """
        vel = self.get_velocity()
        return torch.norm(vel, dim=-1)

    def get_acceleration_magnitude(self) -> torch.Tensor:
        """Get acceleration magnitude.

        Returns:
            Acceleration magnitude, shape (num_envs,).
        """
        accel = self.get_acceleration()
        return torch.norm(accel, dim=-1)

    def mark_slip_start(self, env_ids: torch.Tensor):
        """Mark the start of slip for specified environments.

        Args:
            env_ids: Environment indices where slip started.
        """
        self._slip_start_index[env_ids] = len(self._history) - 1

    def get_post_slip_trajectory(
        self,
        env_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get trajectory data since slip started for a single environment.

        Args:
            env_idx: Environment index.

        Returns:
            Tuple of (timestamps, positions, velocities, accelerations).
        """
        start_idx = int(self._slip_start_index[env_idx].item())

        if start_idx >= len(self._history):
            return (
                np.array([]),
                np.array([]).reshape(0, 3),
                np.array([]).reshape(0, 3),
                np.array([]).reshape(0, 3),
            )

        # Extract data from history
        timestamps = []
        positions = []

        for i in range(start_idx, len(self._history)):
            t, p = self._history[i]
            timestamps.append(t)
            positions.append(p[env_idx].cpu().numpy())

        timestamps = np.array(timestamps)
        positions = np.array(positions)

        # Compute velocities and accelerations
        velocities = np.zeros_like(positions)
        accelerations = np.zeros_like(positions)

        if len(timestamps) >= 2:
            for i in range(1, len(timestamps)):
                dt = timestamps[i] - timestamps[i - 1]
                if dt > 0:
                    velocities[i] = (positions[i] - positions[i - 1]) / dt

        if len(timestamps) >= 3:
            for i in range(2, len(timestamps)):
                dt = timestamps[i] - timestamps[i - 2]
                if dt > 0:
                    accelerations[i] = (velocities[i] - velocities[i - 1]) / (timestamps[i] - timestamps[i - 1])

        return timestamps, positions, velocities, accelerations

    def compute_dynamic_friction(
        self,
        env_idx: int,
        object_mass: float,
        normal_force: float,
        gravity: float = 9.81,
        min_samples: int = 5,
    ) -> Tuple[float, bool]:
        """Compute dynamic friction coefficient from post-slip motion.

        Uses the equation: m × a = m × g - μ_d × N
        Therefore: μ_d = (g - a) × m / N

        Args:
            env_idx: Environment index.
            object_mass: Object mass in kg.
            normal_force: Normal force at contact (N).
            gravity: Gravitational acceleration (m/s²).
            min_samples: Minimum samples required for valid measurement.

        Returns:
            Tuple of (dynamic_friction, measurement_valid).
        """
        timestamps, positions, velocities, accelerations = self.get_post_slip_trajectory(env_idx)

        if len(timestamps) < min_samples:
            return 0.0, False

        # Use vertical (Z) acceleration for falling object
        # Average acceleration magnitude during slip
        a_vertical = np.abs(accelerations[:, 2])

        # Filter out initial zero values and outliers
        valid_mask = a_vertical > 0.1  # Minimum detectable acceleration
        if not valid_mask.any():
            return 0.0, False

        a_avg = np.mean(a_vertical[valid_mask])

        # Compute μ_d
        if normal_force > 0.01:  # Minimum force threshold
            mu_d = (gravity - a_avg) * object_mass / normal_force
            # Clamp to reasonable range
            mu_d = max(0.0, min(mu_d, 2.0))
            return mu_d, True

        return 0.0, False

    def get_tracking_data(self) -> DynamicFrictionData | None:
        """Get complete tracking data as a DynamicFrictionData object.

        Returns:
            DynamicFrictionData object or None if insufficient data.
        """
        if len(self._history) < 3:
            return None

        timestamps = []
        positions = []

        for t, p in self._history:
            timestamps.append(t)
            positions.append(p[0].cpu().numpy())  # Assuming single env for now

        timestamps = np.array(timestamps)
        positions = np.array(positions)

        # Compute full velocity/acceleration arrays
        velocities = np.gradient(positions, timestamps, axis=0)
        accelerations = np.gradient(velocities, timestamps, axis=0)

        return DynamicFrictionData(
            timestamps=timestamps,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            dynamic_friction=0.0,  # To be computed separately
            measurement_valid=True,
        )

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._num_envs

    @property
    def history_size(self) -> int:
        """Current history size."""
        return len(self._history)


class VisionSlipDetector(SlipDetector):
    """Vision-based slip detector using ArUco marker tracking.

    Combines SlipDetector with ArucoObjectTracker for complete
    slip detection and post-slip motion analysis.
    """

    def __init__(
        self,
        marker_id: int = 0,
        marker_size: float = 0.02,
        history_length: int = 100,
        tracking_fps: float = 30.0,
        **kwargs,
    ):
        """Initialize vision slip detector.

        Args:
            marker_id: ArUco marker ID to track.
            marker_size: Marker size in meters.
            history_length: Number of samples for tracking history.
            tracking_fps: Expected tracking frequency.
            **kwargs: Additional arguments for base class.
        """
        super().__init__(**kwargs)

        self.marker_id = marker_id
        self.marker_size = marker_size

        # Create object tracker
        self._tracker = ArucoObjectTracker(
            marker_id=marker_id,
            marker_size=marker_size,
            history_length=history_length,
            tracking_fps=tracking_fps,
            device=self.device,
            num_envs=self._num_envs,
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset detector and tracker."""
        super().reset(env_ids)
        self._tracker.reset(env_ids)

    def update(self, position: torch.Tensor, velocity: torch.Tensor, timestamp: float = 0.0):
        """Update detector and tracker with current state.

        Args:
            position: Object position, shape (num_envs, 3).
            velocity: Object velocity, shape (num_envs, 3).
            timestamp: Current simulation time.
        """
        super().update(position, velocity, timestamp)
        self._tracker.update(timestamp, position)

    def check_slip(self, timestamp: float = 0.0) -> torch.Tensor:
        """Check slip and mark slip start in tracker.

        Args:
            timestamp: Current simulation time.

        Returns:
            Boolean tensor indicating newly detected slips.
        """
        pre_slip = self._slip_detected.clone()
        slip = super().check_slip(timestamp)
        newly_slipped = slip & ~pre_slip

        # Mark slip start in tracker for newly slipped environments
        if newly_slipped.any():
            slip_env_ids = torch.where(newly_slipped)[0]
            self._tracker.mark_slip_start(slip_env_ids)

        return slip

    def get_post_slip_trajectory(self, env_idx: int):
        """Get post-slip trajectory from tracker."""
        return self._tracker.get_post_slip_trajectory(env_idx)

    def compute_dynamic_friction(
        self,
        env_idx: int,
        object_mass: float,
        normal_force: float,
        gravity: float = 9.81,
    ) -> Tuple[float, bool]:
        """Compute dynamic friction from post-slip motion."""
        return self._tracker.compute_dynamic_friction(
            env_idx=env_idx,
            object_mass=object_mass,
            normal_force=normal_force,
            gravity=gravity,
        )

    @property
    def tracker(self) -> ArucoObjectTracker:
        """Get the underlying object tracker."""
        return self._tracker


class TactileSlipDetector:
    """Tactile-based slip detector using F/T sensor data.

    Detects slip by monitoring changes in tangential force
    relative to normal force on fingertip sensors.
    """

    def __init__(
        self,
        force_ratio_threshold: float = 0.8,
        force_derivative_threshold: float = 0.5,
        device: str = "cuda:0",
        num_envs: int = 1,
    ):
        """Initialize tactile slip detector.

        Args:
            force_ratio_threshold: Tangential/normal force ratio threshold.
            force_derivative_threshold: Force change rate threshold.
            device: Device for tensor operations.
            num_envs: Number of environments.
        """
        self.force_ratio_threshold = force_ratio_threshold
        self.force_derivative_threshold = force_derivative_threshold
        self.device = device
        self._num_envs = num_envs

        # Previous force reading for derivative
        self._prev_forces: torch.Tensor | None = None

        # Slip flags
        self._slip_detected = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Force measurements at slip
        self._F_tangential_at_slip = torch.zeros(num_envs, device=device)
        self._F_normal_at_slip = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset detector."""
        if env_ids is None:
            self._prev_forces = None
            self._slip_detected[:] = False
            self._F_tangential_at_slip[:] = 0.0
            self._F_normal_at_slip[:] = 0.0
        else:
            self._slip_detected[env_ids] = False
            self._F_tangential_at_slip[env_ids] = 0.0
            self._F_normal_at_slip[env_ids] = 0.0

    def update(self, fingertip_forces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update with fingertip F/T sensor data.

        Args:
            fingertip_forces: Force tensor, shape (num_envs, num_fingers, 6).
                Format: [Fx, Fy, Fz, Tx, Ty, Tz] per finger.

        Returns:
            Tuple of (tangential_force, normal_force) tensors.
        """
        # Sum forces across fingers used for gripping
        # Assume fingers 0 (thumb) and 1 (index) for pinch grip
        grip_forces = fingertip_forces[:, :2, :].sum(dim=1)  # (num_envs, 6)

        # Extract force components
        tangential = torch.norm(grip_forces[:, :2], dim=-1)  # XY plane
        normal = torch.abs(grip_forces[:, 2])  # Z direction

        # Force ratio detection
        ratio = tangential / (normal + 1e-6)
        ratio_slip = ratio > self.force_ratio_threshold

        # Force derivative detection
        if self._prev_forces is not None:
            force_change = torch.norm(grip_forces - self._prev_forces, dim=-1)
            derivative_slip = force_change > self.force_derivative_threshold
        else:
            derivative_slip = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)

        self._prev_forces = grip_forces.clone()

        # Combined detection
        slip = ratio_slip | derivative_slip

        # Record forces at slip for newly slipped envs
        newly_slipped = slip & ~self._slip_detected
        self._F_tangential_at_slip[newly_slipped] = tangential[newly_slipped]
        self._F_normal_at_slip[newly_slipped] = normal[newly_slipped]

        self._slip_detected |= slip

        return tangential, normal

    def check_slip(self) -> torch.Tensor:
        """Check if slip has been detected."""
        return self._slip_detected

    def is_slipped(self) -> torch.Tensor:
        """Get slip status."""
        return self._slip_detected

    def get_static_friction(self) -> torch.Tensor:
        """Compute static friction coefficient from forces at slip.

        Returns:
            Static friction coefficient (μ_s) for each environment.
        """
        # μ_s = F_tangential / F_normal
        mu_s = self._F_tangential_at_slip / (self._F_normal_at_slip + 1e-6)
        return mu_s

    def get_forces_at_slip(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get forces recorded at slip onset.

        Returns:
            Tuple of (F_tangential, F_normal) tensors.
        """
        return self._F_tangential_at_slip, self._F_normal_at_slip


class RelativeVelocitySlipDetector:
    """Fingertip-Object relative velocity based slip detector.

    Detects slip by comparing fingertip and object velocities when in contact.
    Slip is detected when:
    1. Contact force > threshold (contact exists)
    2. Relative velocity > threshold (slip is occurring)

    This method does not require contact normal direction, making it simpler
    to implement for manipulation tasks where contact geometry is complex.
    """

    def __init__(
        self,
        contact_force_threshold: float = 1.0,
        slip_velocity_threshold: float = 0.01,
        device: str = "cuda:0",
        num_envs: int = 1,
        num_fingers: int = 5,
    ):
        """Initialize relative velocity slip detector.

        Args:
            contact_force_threshold: Force magnitude threshold for contact (N).
            slip_velocity_threshold: Velocity threshold for slip (m/s).
            device: Device for tensor operations.
            num_envs: Number of environments.
            num_fingers: Number of fingers to track.
        """
        self.contact_force_threshold = contact_force_threshold
        self.slip_velocity_threshold = slip_velocity_threshold
        self.device = device
        self._num_envs = num_envs
        self._num_fingers = num_fingers

        # Slip state buffers
        self._is_slipping = torch.zeros(
            num_envs, num_fingers, dtype=torch.bool, device=device
        )
        self._slip_start_time = torch.zeros(
            num_envs, num_fingers, device=device
        )

        # History for analysis
        self._slip_count = torch.zeros(num_envs, num_fingers, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset detector for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self._is_slipping[:] = False
            self._slip_start_time[:] = 0.0
            self._slip_count[:] = 0
        else:
            self._is_slipping[env_ids] = False
            self._slip_start_time[env_ids] = 0.0
            self._slip_count[env_ids] = 0

    def update(
        self,
        fingertip_vel: torch.Tensor,
        object_vel: torch.Tensor,
        contact_forces: torch.Tensor,
        timestamp: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update detector with current state and compute slip.

        Args:
            fingertip_vel: Fingertip linear velocities, shape (num_envs, num_fingers, 3).
            object_vel: Object linear velocity, shape (num_envs, 3).
            contact_forces: Contact forces per finger, shape (num_envs, num_fingers, 3)
                or (num_envs, num_fingers, 6) if including torques.
            timestamp: Current simulation time.

        Returns:
            Tuple of:
                - relative_vel: Relative velocity (num_envs, num_fingers, 3)
                - is_contact: Contact status (num_envs, num_fingers)
                - is_slipping: Slip status (num_envs, num_fingers)
        """
        # Compute relative velocity: object_vel - fingertip_vel
        # Positive relative velocity means object is moving faster than fingertip
        relative_vel = object_vel.unsqueeze(1) - fingertip_vel  # (N, F, 3)

        # Contact detection: force magnitude > threshold
        force_components = contact_forces[..., :3]  # Take only force, not torque
        force_magnitude = torch.norm(force_components, dim=-1)  # (N, F)
        is_contact = force_magnitude > self.contact_force_threshold

        # Slip detection: contact exists AND relative velocity > threshold
        relative_vel_magnitude = torch.norm(relative_vel, dim=-1)  # (N, F)
        is_slipping = is_contact & (relative_vel_magnitude > self.slip_velocity_threshold)

        # Track slip transitions (for counting slip events)
        newly_slipping = is_slipping & ~self._is_slipping
        self._slip_count += newly_slipping.float()
        self._slip_start_time[newly_slipping] = timestamp

        # Update state
        self._is_slipping = is_slipping.clone()

        return relative_vel, is_contact, is_slipping

    def get_slip_status(self) -> torch.Tensor:
        """Get current slip status.

        Returns:
            Boolean tensor of shape (num_envs, num_fingers).
        """
        return self._is_slipping

    def get_slip_count(self) -> torch.Tensor:
        """Get cumulative slip event count.

        Returns:
            Tensor of shape (num_envs, num_fingers).
        """
        return self._slip_count

    def is_any_slipping(self) -> torch.Tensor:
        """Check if any finger is slipping in each environment.

        Returns:
            Boolean tensor of shape (num_envs,).
        """
        return self._is_slipping.any(dim=-1)

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._num_envs

    @property
    def num_fingers(self) -> int:
        """Number of fingers."""
        return self._num_fingers
