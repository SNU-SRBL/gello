# Copyright (c) 2025, SRBL
# Slip data collection for analysis and calibration

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SlipDataPoint:
    """Single timestep of slip detection data."""

    timestamp: float
    """Simulation time (s)."""

    fingertip_velocities: np.ndarray
    """Fingertip linear velocities (num_fingers, 3)."""

    object_velocity: np.ndarray
    """Object linear velocity (3,)."""

    relative_velocities: np.ndarray
    """Relative velocities per finger (num_fingers, 3)."""

    contact_forces: np.ndarray
    """Contact forces per finger (num_fingers, 6) - [Fx, Fy, Fz, Tx, Ty, Tz]."""

    is_contact: np.ndarray
    """Contact status per finger (num_fingers,)."""

    is_slipping: np.ndarray
    """Slip status per finger (num_fingers,)."""

    # Optional additional data
    joint_positions: np.ndarray | None = None
    """Hand joint positions (num_joints,)."""

    object_position: np.ndarray | None = None
    """Object position (3,)."""

    grip_force: float | None = None
    """Total grip force magnitude (N)."""


@dataclass
class SlipEvent:
    """Data for a single slip event (onset to end)."""

    start_time: float
    """Time when slip started."""

    end_time: float | None = None
    """Time when slip ended (None if ongoing)."""

    finger_idx: int = 0
    """Finger index where slip occurred."""

    peak_relative_velocity: float = 0.0
    """Maximum relative velocity during slip."""

    contact_force_at_onset: float = 0.0
    """Contact force when slip started."""


class SlipDataCollector:
    """Collect and save slip detection data for analysis.

    Collects timestamped data including:
    - Fingertip and object velocities
    - Relative velocities between fingertips and object
    - Contact forces
    - Slip detection flags

    Data can be saved to .npz files for offline analysis.
    """

    def __init__(
        self,
        num_fingers: int = 5,
        max_samples: int = 100000,
        record_joint_state: bool = False,
    ):
        """Initialize slip data collector.

        Args:
            num_fingers: Number of fingers to track.
            max_samples: Maximum number of samples to store (for memory management).
            record_joint_state: Whether to record joint positions.
        """
        self.num_fingers = num_fingers
        self.max_samples = max_samples
        self.record_joint_state = record_joint_state

        self.data: list[SlipDataPoint] = []
        self.slip_events: list[SlipEvent] = []

        # Track ongoing slip events
        self._active_slip_start: dict[int, float] = {}  # finger_idx -> start_time

    def log_step(
        self,
        timestamp: float,
        fingertip_vel: torch.Tensor | np.ndarray,
        object_vel: torch.Tensor | np.ndarray,
        relative_vel: torch.Tensor | np.ndarray,
        contact_forces: torch.Tensor | np.ndarray,
        is_contact: torch.Tensor | np.ndarray,
        is_slipping: torch.Tensor | np.ndarray,
        joint_positions: torch.Tensor | np.ndarray | None = None,
        object_position: torch.Tensor | np.ndarray | None = None,
        grip_force: float | None = None,
    ):
        """Log data for a single simulation step.

        Args:
            timestamp: Current simulation time.
            fingertip_vel: Fingertip velocities (num_fingers, 3).
            object_vel: Object velocity (3,).
            relative_vel: Relative velocities (num_fingers, 3).
            contact_forces: Contact forces (num_fingers, 6).
            is_contact: Contact flags (num_fingers,).
            is_slipping: Slip flags (num_fingers,).
            joint_positions: Optional joint positions.
            object_position: Optional object position.
            grip_force: Optional total grip force.
        """
        # Check memory limit
        if len(self.data) >= self.max_samples:
            return

        # Convert tensors to numpy
        def to_numpy(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        fingertip_vel_np = to_numpy(fingertip_vel)
        object_vel_np = to_numpy(object_vel)
        relative_vel_np = to_numpy(relative_vel)
        contact_forces_np = to_numpy(contact_forces)
        is_contact_np = to_numpy(is_contact).astype(bool)
        is_slipping_np = to_numpy(is_slipping).astype(bool)

        # Handle single env case (squeeze batch dimension)
        if fingertip_vel_np.ndim == 3:
            fingertip_vel_np = fingertip_vel_np[0]
        if object_vel_np.ndim == 2:
            object_vel_np = object_vel_np[0]
        if relative_vel_np.ndim == 3:
            relative_vel_np = relative_vel_np[0]
        if contact_forces_np.ndim == 3:
            contact_forces_np = contact_forces_np[0]
        if is_contact_np.ndim == 2:
            is_contact_np = is_contact_np[0]
        if is_slipping_np.ndim == 2:
            is_slipping_np = is_slipping_np[0]

        data_point = SlipDataPoint(
            timestamp=timestamp,
            fingertip_velocities=fingertip_vel_np.copy(),
            object_velocity=object_vel_np.copy(),
            relative_velocities=relative_vel_np.copy(),
            contact_forces=contact_forces_np.copy(),
            is_contact=is_contact_np.copy(),
            is_slipping=is_slipping_np.copy(),
            joint_positions=to_numpy(joint_positions),
            object_position=to_numpy(object_position),
            grip_force=grip_force,
        )

        self.data.append(data_point)

        # Track slip events
        self._update_slip_events(timestamp, is_slipping_np, relative_vel_np, contact_forces_np)

    def _update_slip_events(
        self,
        timestamp: float,
        is_slipping: np.ndarray,
        relative_vel: np.ndarray,
        contact_forces: np.ndarray,
    ):
        """Track slip event start/end times."""
        for finger_idx in range(self.num_fingers):
            slipping = is_slipping[finger_idx]

            if slipping and finger_idx not in self._active_slip_start:
                # Slip started
                self._active_slip_start[finger_idx] = timestamp
                force_mag = np.linalg.norm(contact_forces[finger_idx, :3])
                self.slip_events.append(SlipEvent(
                    start_time=timestamp,
                    finger_idx=finger_idx,
                    contact_force_at_onset=force_mag,
                ))

            elif not slipping and finger_idx in self._active_slip_start:
                # Slip ended
                del self._active_slip_start[finger_idx]
                if self.slip_events:
                    # Update the last event for this finger
                    for event in reversed(self.slip_events):
                        if event.finger_idx == finger_idx and event.end_time is None:
                            event.end_time = timestamp
                            break

            elif slipping and finger_idx in self._active_slip_start:
                # Update peak velocity during ongoing slip
                vel_mag = np.linalg.norm(relative_vel[finger_idx])
                for event in reversed(self.slip_events):
                    if event.finger_idx == finger_idx and event.end_time is None:
                        event.peak_relative_velocity = max(
                            event.peak_relative_velocity, vel_mag
                        )
                        break

    def save(self, filepath: str | Path):
        """Save collected data to .npz file.

        Args:
            filepath: Output file path (should end with .npz).
        """
        if not self.data:
            print("Warning: No data to save")
            return

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert to arrays
        n_samples = len(self.data)

        timestamps = np.array([d.timestamp for d in self.data])
        fingertip_velocities = np.stack([d.fingertip_velocities for d in self.data])
        object_velocities = np.stack([d.object_velocity for d in self.data])
        relative_velocities = np.stack([d.relative_velocities for d in self.data])
        contact_forces = np.stack([d.contact_forces for d in self.data])
        is_contact = np.stack([d.is_contact for d in self.data])
        is_slipping = np.stack([d.is_slipping for d in self.data])

        save_dict = {
            "timestamps": timestamps,
            "fingertip_velocities": fingertip_velocities,
            "object_velocities": object_velocities,
            "relative_velocities": relative_velocities,
            "contact_forces": contact_forces,
            "is_contact": is_contact,
            "is_slipping": is_slipping,
            "num_fingers": self.num_fingers,
            "num_samples": n_samples,
        }

        # Optional data
        if self.data[0].joint_positions is not None:
            save_dict["joint_positions"] = np.stack([
                d.joint_positions for d in self.data
            ])

        if self.data[0].object_position is not None:
            save_dict["object_positions"] = np.stack([
                d.object_position for d in self.data
            ])

        if self.data[0].grip_force is not None:
            save_dict["grip_forces"] = np.array([
                d.grip_force for d in self.data
            ])

        # Save slip events summary
        if self.slip_events:
            save_dict["slip_event_count"] = len(self.slip_events)
            save_dict["slip_event_start_times"] = np.array([
                e.start_time for e in self.slip_events
            ])
            save_dict["slip_event_fingers"] = np.array([
                e.finger_idx for e in self.slip_events
            ])

        np.savez(filepath, **save_dict)
        print(f"Saved {n_samples} samples to {filepath}")

    def clear(self):
        """Clear all collected data."""
        self.data.clear()
        self.slip_events.clear()
        self._active_slip_start.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of collected data.

        Returns:
            Dictionary with summary statistics.
        """
        if not self.data:
            return {"num_samples": 0}

        is_slipping = np.stack([d.is_slipping for d in self.data])
        is_contact = np.stack([d.is_contact for d in self.data])
        relative_vel = np.stack([d.relative_velocities for d in self.data])

        # Slip statistics
        slip_ratio = is_slipping.mean()
        contact_ratio = is_contact.mean()

        # Per-finger statistics
        per_finger_slip_ratio = is_slipping.mean(axis=0)

        # Velocity statistics during slip
        slip_mask = is_slipping
        if slip_mask.any():
            rel_vel_mag = np.linalg.norm(relative_vel, axis=-1)
            slip_velocities = rel_vel_mag[slip_mask]
            mean_slip_vel = slip_velocities.mean()
            max_slip_vel = slip_velocities.max()
        else:
            mean_slip_vel = 0.0
            max_slip_vel = 0.0

        return {
            "num_samples": len(self.data),
            "duration_s": self.data[-1].timestamp - self.data[0].timestamp,
            "slip_ratio": float(slip_ratio),
            "contact_ratio": float(contact_ratio),
            "per_finger_slip_ratio": per_finger_slip_ratio.tolist(),
            "num_slip_events": len(self.slip_events),
            "mean_slip_velocity": float(mean_slip_vel),
            "max_slip_velocity": float(max_slip_vel),
        }

    @property
    def num_samples(self) -> int:
        """Number of samples collected."""
        return len(self.data)

    @property
    def is_empty(self) -> bool:
        """Check if collector has no data."""
        return len(self.data) == 0
