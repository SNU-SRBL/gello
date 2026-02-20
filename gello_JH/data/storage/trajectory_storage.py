# Copyright (c) 2025, SRBL
# Trajectory data storage for Real2Sim calibration

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from datetime import datetime

import h5py


@dataclass
class TrajectoryData:
    """Container for trajectory data.

    Stores time-series data from real robot or simulation for
    calibration and replay purposes. Supports separate UR5 and Hand data
    for Phase 0 SIMPLER-style system identification.
    """

    # Time stamps
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    """Time stamps in seconds, shape (T,)."""

    # ===== Legacy joint states (for backward compatibility) =====
    joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    """Joint positions in radians, shape (T, num_joints)."""

    joint_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    """Joint velocities in rad/s, shape (T, num_joints)."""

    joint_torques: np.ndarray = field(default_factory=lambda: np.array([]))
    """Joint torques in Nm, shape (T, num_joints)."""

    # ===== UR5 specific data =====
    ur5_joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    """UR5 joint positions, shape (T, 6)."""

    ur5_joint_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    """UR5 joint velocities, shape (T, 6)."""

    ur5_ee_position: np.ndarray = field(default_factory=lambda: np.array([]))
    """UR5 tool flange position, shape (T, 3)."""

    ur5_ee_orientation: np.ndarray = field(default_factory=lambda: np.array([]))
    """UR5 tool flange orientation (quaternion wxyz), shape (T, 4)."""

    # ===== Hand specific data =====
    hand_joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    """Hand joint positions, shape (T, 20)."""

    hand_joint_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    """Hand joint velocities, shape (T, 20)."""

    fingertip_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    """5 fingertip positions, shape (T, 5, 3)."""

    fingertip_orientations: np.ndarray = field(default_factory=lambda: np.array([]))
    """5 fingertip orientations (quaternion wxyz), shape (T, 5, 4)."""

    # ===== Actions (for SIMPLER replay) =====
    actions: np.ndarray = field(default_factory=lambda: np.array([]))
    """Actions sent to robot, shape (T, num_actions)."""

    # Motor currents (real robot only)
    motor_currents: np.ndarray = field(default_factory=lambda: np.array([]))
    """Motor currents in Amperes, shape (T, num_joints)."""

    # F/T sensor data
    ft_sensor_data: dict[str, np.ndarray] = field(default_factory=dict)
    """F/T sensor data per finger, {finger: (T, 7, 6)} or {finger: (T, 6)}."""

    # Object state (for Phase 2 slip test)
    object_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    """Object positions, shape (T, 3)."""

    object_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    """Object velocities, shape (T, 6) [lin_vel, ang_vel]."""

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata about the recording (date, robot, etc.)."""

    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        array_fields = [
            "timestamps", "joint_positions", "joint_velocities", "joint_torques",
            "ur5_joint_positions", "ur5_joint_velocities", "ur5_ee_position", "ur5_ee_orientation",
            "hand_joint_positions", "hand_joint_velocities", "fingertip_positions", "fingertip_orientations",
            "actions", "motor_currents", "object_positions", "object_velocities",
        ]
        for field_name in array_fields:
            value = getattr(self, field_name)
            if isinstance(value, list):
                setattr(self, field_name, np.array(value))

    @property
    def duration(self) -> float:
        """Total duration of the trajectory in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def num_steps(self) -> int:
        """Number of time steps."""
        return len(self.timestamps)

    @property
    def dt(self) -> float:
        """Average time step in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.duration / (self.num_steps - 1)

    @property
    def num_joints(self) -> int:
        """Number of joints."""
        if len(self.joint_positions) == 0:
            return 0
        return self.joint_positions.shape[-1]


class TrajectoryStorage:
    """HDF5-based trajectory storage.

    Provides efficient storage and retrieval of trajectory data
    for calibration experiments.
    """

    def __init__(self, base_dir: str | Path | None = None):
        """Initialize trajectory storage.

        Args:
            base_dir: Base directory for storage. If None, uses current directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, data: TrajectoryData, filename: str, overwrite: bool = False):
        """Save trajectory data to HDF5 file.

        Args:
            data: Trajectory data to save.
            filename: Filename (with or without .h5 extension).
            overwrite: Whether to overwrite existing file.
        """
        if not filename.endswith(".h5"):
            filename += ".h5"

        filepath = self.base_dir / filename

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {filepath}")

        with h5py.File(filepath, "w") as f:
            # Save main arrays (legacy)
            if len(data.timestamps) > 0:
                f.create_dataset("timestamps", data=data.timestamps, compression="gzip")
            if len(data.joint_positions) > 0:
                f.create_dataset("joint_positions", data=data.joint_positions, compression="gzip")
            if len(data.joint_velocities) > 0:
                f.create_dataset("joint_velocities", data=data.joint_velocities, compression="gzip")
            if len(data.joint_torques) > 0:
                f.create_dataset("joint_torques", data=data.joint_torques, compression="gzip")

            # Save UR5 data
            if len(data.ur5_joint_positions) > 0:
                f.create_dataset("ur5_joint_positions", data=data.ur5_joint_positions, compression="gzip")
            if len(data.ur5_joint_velocities) > 0:
                f.create_dataset("ur5_joint_velocities", data=data.ur5_joint_velocities, compression="gzip")
            if len(data.ur5_ee_position) > 0:
                f.create_dataset("ur5_ee_position", data=data.ur5_ee_position, compression="gzip")
            if len(data.ur5_ee_orientation) > 0:
                f.create_dataset("ur5_ee_orientation", data=data.ur5_ee_orientation, compression="gzip")

            # Save Hand data
            if len(data.hand_joint_positions) > 0:
                f.create_dataset("hand_joint_positions", data=data.hand_joint_positions, compression="gzip")
            if len(data.hand_joint_velocities) > 0:
                f.create_dataset("hand_joint_velocities", data=data.hand_joint_velocities, compression="gzip")
            if len(data.fingertip_positions) > 0:
                f.create_dataset("fingertip_positions", data=data.fingertip_positions, compression="gzip")
            if len(data.fingertip_orientations) > 0:
                f.create_dataset("fingertip_orientations", data=data.fingertip_orientations, compression="gzip")

            # Save actions
            if len(data.actions) > 0:
                f.create_dataset("actions", data=data.actions, compression="gzip")

            # Save motor currents and object state
            if len(data.motor_currents) > 0:
                f.create_dataset("motor_currents", data=data.motor_currents, compression="gzip")
            if len(data.object_positions) > 0:
                f.create_dataset("object_positions", data=data.object_positions, compression="gzip")
            if len(data.object_velocities) > 0:
                f.create_dataset("object_velocities", data=data.object_velocities, compression="gzip")

            # Save F/T sensor data
            if data.ft_sensor_data:
                ft_group = f.create_group("ft_sensor_data")
                for finger, forces in data.ft_sensor_data.items():
                    ft_group.create_dataset(finger, data=forces, compression="gzip")

            # Save metadata
            f.attrs["metadata"] = json.dumps(data.metadata)
            f.attrs["save_time"] = datetime.now().isoformat()

    def load(self, filename: str) -> TrajectoryData:
        """Load trajectory data from HDF5 file.

        Args:
            filename: Filename to load.

        Returns:
            Loaded trajectory data.
        """
        if not filename.endswith(".h5"):
            filename += ".h5"

        filepath = self.base_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with h5py.File(filepath, "r") as f:
            # Helper function to load array if exists
            def load_array(key):
                return f[key][:] if key in f else np.array([])

            # Load main arrays (legacy)
            timestamps = load_array("timestamps")
            joint_positions = load_array("joint_positions")
            joint_velocities = load_array("joint_velocities")
            joint_torques = load_array("joint_torques")

            # Load UR5 data
            ur5_joint_positions = load_array("ur5_joint_positions")
            ur5_joint_velocities = load_array("ur5_joint_velocities")
            ur5_ee_position = load_array("ur5_ee_position")
            ur5_ee_orientation = load_array("ur5_ee_orientation")

            # Load Hand data
            hand_joint_positions = load_array("hand_joint_positions")
            hand_joint_velocities = load_array("hand_joint_velocities")
            fingertip_positions = load_array("fingertip_positions")
            fingertip_orientations = load_array("fingertip_orientations")

            # Load actions and other
            actions = load_array("actions")
            motor_currents = load_array("motor_currents")
            object_positions = load_array("object_positions")
            object_velocities = load_array("object_velocities")

            # Load F/T sensor data
            ft_sensor_data = {}
            if "ft_sensor_data" in f:
                for finger in f["ft_sensor_data"].keys():
                    ft_sensor_data[finger] = f["ft_sensor_data"][finger][:]

            # Load metadata
            metadata = json.loads(f.attrs.get("metadata", "{}"))

        return TrajectoryData(
            timestamps=timestamps,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_torques=joint_torques,
            ur5_joint_positions=ur5_joint_positions,
            ur5_joint_velocities=ur5_joint_velocities,
            ur5_ee_position=ur5_ee_position,
            ur5_ee_orientation=ur5_ee_orientation,
            hand_joint_positions=hand_joint_positions,
            hand_joint_velocities=hand_joint_velocities,
            fingertip_positions=fingertip_positions,
            fingertip_orientations=fingertip_orientations,
            actions=actions,
            motor_currents=motor_currents,
            ft_sensor_data=ft_sensor_data,
            object_positions=object_positions,
            object_velocities=object_velocities,
            metadata=metadata,
        )

    def list_trajectories(self) -> list[str]:
        """List all trajectory files in the storage directory.

        Returns:
            List of trajectory filenames.
        """
        return [f.name for f in self.base_dir.glob("*.h5")]

    def delete(self, filename: str):
        """Delete a trajectory file.

        Args:
            filename: Filename to delete.
        """
        if not filename.endswith(".h5"):
            filename += ".h5"

        filepath = self.base_dir / filename
        if filepath.exists():
            filepath.unlink()


@dataclass
class SlipTrialData:
    """Container for slip test trial data (Phase 2)."""

    # Trial info
    object_name: str = ""
    object_weight: float = 0.0
    trial_idx: int = 0

    # Time series data
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    motor_currents: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    ft_thumb: np.ndarray = field(default_factory=lambda: np.array([]))
    ft_index: np.ndarray = field(default_factory=lambda: np.array([]))
    object_positions: np.ndarray = field(default_factory=lambda: np.array([]))

    # Slip detection
    slip_detected: bool = False
    slip_idx: int | None = None
    I_slip: np.ndarray | float | None = None  # Motor current at slip

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class SlipTrialStorage(TrajectoryStorage):
    """Storage for slip test trials."""

    def save_trial(self, trial: SlipTrialData, filename: str, overwrite: bool = False):
        """Save slip trial data."""
        if not filename.endswith(".h5"):
            filename += ".h5"

        filepath = self.base_dir / filename

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {filepath}")

        with h5py.File(filepath, "w") as f:
            # Trial info
            f.attrs["object_name"] = trial.object_name
            f.attrs["object_weight"] = trial.object_weight
            f.attrs["trial_idx"] = trial.trial_idx
            f.attrs["slip_detected"] = trial.slip_detected

            if trial.slip_idx is not None:
                f.attrs["slip_idx"] = trial.slip_idx
            if trial.I_slip is not None:
                if isinstance(trial.I_slip, np.ndarray):
                    f.create_dataset("I_slip", data=trial.I_slip)
                else:
                    f.attrs["I_slip"] = trial.I_slip

            # Time series
            if len(trial.timestamps) > 0:
                f.create_dataset("timestamps", data=trial.timestamps, compression="gzip")
            if len(trial.motor_currents) > 0:
                f.create_dataset("motor_currents", data=trial.motor_currents, compression="gzip")
            if len(trial.joint_positions) > 0:
                f.create_dataset("joint_positions", data=trial.joint_positions, compression="gzip")
            if len(trial.ft_thumb) > 0:
                f.create_dataset("ft_thumb", data=trial.ft_thumb, compression="gzip")
            if len(trial.ft_index) > 0:
                f.create_dataset("ft_index", data=trial.ft_index, compression="gzip")
            if len(trial.object_positions) > 0:
                f.create_dataset("object_positions", data=trial.object_positions, compression="gzip")

            # Metadata
            f.attrs["metadata"] = json.dumps(trial.metadata)

    def load_trial(self, filename: str) -> SlipTrialData:
        """Load slip trial data."""
        if not filename.endswith(".h5"):
            filename += ".h5"

        filepath = self.base_dir / filename

        with h5py.File(filepath, "r") as f:
            trial = SlipTrialData(
                object_name=f.attrs.get("object_name", ""),
                object_weight=f.attrs.get("object_weight", 0.0),
                trial_idx=f.attrs.get("trial_idx", 0),
                slip_detected=f.attrs.get("slip_detected", False),
                slip_idx=f.attrs.get("slip_idx", None),
            )

            # Load I_slip
            if "I_slip" in f:
                trial.I_slip = f["I_slip"][:]
            elif "I_slip" in f.attrs:
                trial.I_slip = f.attrs["I_slip"]

            # Load time series
            trial.timestamps = f["timestamps"][:] if "timestamps" in f else np.array([])
            trial.motor_currents = f["motor_currents"][:] if "motor_currents" in f else np.array([])
            trial.joint_positions = f["joint_positions"][:] if "joint_positions" in f else np.array([])
            trial.ft_thumb = f["ft_thumb"][:] if "ft_thumb" in f else np.array([])
            trial.ft_index = f["ft_index"][:] if "ft_index" in f else np.array([])
            trial.object_positions = f["object_positions"][:] if "object_positions" in f else np.array([])

            # Load metadata
            trial.metadata = json.loads(f.attrs.get("metadata", "{}"))

        return trial
