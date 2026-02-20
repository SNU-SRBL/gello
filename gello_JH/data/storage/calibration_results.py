# Copyright (c) 2025, SRBL
# Calibration results storage for Real2Sim

from __future__ import annotations

import json
import yaml
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from datetime import datetime


@dataclass
class CalibrationResult:
    """Container for calibration results."""

    # Calibration phase (0, 1, or 2)
    phase: int = 0

    # Calibrated parameters
    parameters: dict[str, np.ndarray | float] = field(default_factory=dict)

    # Optimization history
    loss_history: list[float] = field(default_factory=list)

    # Validation metrics
    validation_metrics: dict[str, float] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "phase": self.phase,
            "parameters": {},
            "loss_history": self.loss_history,
            "validation_metrics": self.validation_metrics,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

        # Convert numpy arrays to lists
        for key, value in self.parameters.items():
            if isinstance(value, np.ndarray):
                result["parameters"][key] = value.tolist()
            else:
                result["parameters"][key] = value

        return result

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationResult:
        """Create from dictionary."""
        parameters = {}
        for key, value in data.get("parameters", {}).items():
            if isinstance(value, list):
                parameters[key] = np.array(value)
            else:
                parameters[key] = value

        return cls(
            phase=data.get("phase", 0),
            parameters=parameters,
            loss_history=data.get("loss_history", []),
            validation_metrics=data.get("validation_metrics", {}),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


@dataclass
class Phase0Result(CalibrationResult):
    """Phase 0: Robot System Identification result."""

    def __post_init__(self):
        self.phase = 0

    @property
    def joint_stiffness(self) -> np.ndarray | None:
        return self.parameters.get("joint_stiffness")

    @property
    def joint_damping(self) -> np.ndarray | None:
        return self.parameters.get("joint_damping")

    @property
    def joint_friction(self) -> np.ndarray | None:
        return self.parameters.get("joint_friction")


@dataclass
class Phase1Result(CalibrationResult):
    """Phase 1: Paired Sim-Real Current-Torque Calibration result.

    Stores per-joint torque constant k_t [Nm/A] and regression offset.
    τ_sim = k_t × I_real + offset
    """

    def __post_init__(self):
        self.phase = 1

    @property
    def k_t(self) -> dict[str, float] | None:
        """Per-joint torque constant [Nm/A]."""
        return self.parameters.get("k_t")

    @property
    def offset(self) -> dict[str, float] | None:
        """Per-joint regression offset (absorbs gravity/friction diff)."""
        return self.parameters.get("offset")

    @property
    def r_squared(self) -> dict[str, float] | None:
        """Per-joint R² score."""
        return self.validation_metrics.get("r_squared")

    @property
    def jacobian_consistency(self) -> dict[str, float] | None:
        """Per-joint Jacobian consistency score."""
        return self.validation_metrics.get("jacobian_consistency")

    # Legacy properties for backward compatibility
    @property
    def k_gain(self) -> dict[str, float] | None:
        return self.k_t

    @property
    def k_offset(self) -> dict[str, float] | None:
        return self.offset


@dataclass
class Phase1BaselineData:
    """Phase 1A baseline data (gravity/friction, non-contact).

    Attributes:
        q_positions: Joint positions (N, 20).
        qdot: Joint velocities (N, 20).
        tau_applied: Applied torques [sim] (N, 20).
        I_motor: Motor currents [real] (N, 20).
        gravity_torques: Gravity compensation torques [sim] (N, 20).
        finger_idx: Finger index per sample (N,).
        metadata: Additional metadata.
    """

    q_positions: np.ndarray  # (N, 20)
    qdot: np.ndarray  # (N, 20)
    tau_applied: np.ndarray | None = None  # (N, 20) - sim
    I_motor: np.ndarray | None = None  # (N, 20) - real
    gravity_torques: np.ndarray | None = None  # (N, 20) - IsaacLab API
    finger_idx: np.ndarray | None = None  # (N,)
    metadata: dict = field(default_factory=dict)

    def save(self, filepath: str | Path):
        data = {"q_positions": self.q_positions, "qdot": self.qdot}
        if self.tau_applied is not None:
            data["tau_applied"] = self.tau_applied
        if self.I_motor is not None:
            data["I_motor"] = self.I_motor
        if self.gravity_torques is not None:
            data["gravity_torques"] = self.gravity_torques
        if self.finger_idx is not None:
            data["finger_idx"] = self.finger_idx
        np.savez(filepath, **data)

    @classmethod
    def load(cls, filepath: str | Path) -> Phase1BaselineData:
        data = np.load(filepath, allow_pickle=True)
        return cls(
            q_positions=data["q_positions"],
            qdot=data["qdot"],
            tau_applied=data.get("tau_applied"),
            I_motor=data.get("I_motor"),
            gravity_torques=data.get("gravity_torques"),
            finger_idx=data.get("finger_idx"),
        )

    @property
    def num_measurements(self) -> int:
        return len(self.q_positions)


@dataclass
class Phase1ContactData:
    """Phase 1B contact calibration data.

    Attributes:
        q: Joint positions (M, 20).
        qdot: Joint velocities (M, 20).
        tau_applied: Applied torques [sim] (M, 20).
        I_motor: Motor currents [real] (M, 20).
        F_ext: External FT sensor 6D wrench (M, 6).
        F_internal: Internal Tesollo FT sensors (M, 30).
        jacobian: Fingertip Jacobian (M, 6, 4) per finger.
        finger_idx: Finger index per sample (M,).
        config_idx: Configuration index (M,).
        direction_idx: Force direction index (M,).
        metadata: Additional metadata.
    """

    q: np.ndarray  # (M, 20)
    qdot: np.ndarray  # (M, 20)
    tau_applied: np.ndarray | None = None  # (M, 20) - sim
    I_motor: np.ndarray | None = None  # (M, 20) - real
    F_ext: np.ndarray | None = None  # (M, 6) - external FT 6D wrench
    F_internal: np.ndarray | None = None  # (M, 30) - 5 fingers × 6D
    jacobian: np.ndarray | None = None  # (M, 6, 4) - fingertip Jacobian
    finger_idx: np.ndarray | None = None  # (M,)
    config_idx: np.ndarray | None = None  # (M,)
    direction_idx: np.ndarray | None = None  # (M,)
    metadata: dict = field(default_factory=dict)

    def save(self, filepath: str | Path):
        data = {"q": self.q, "qdot": self.qdot}
        for attr in ["tau_applied", "I_motor", "F_ext", "F_internal",
                      "jacobian", "finger_idx", "config_idx", "direction_idx"]:
            val = getattr(self, attr)
            if val is not None:
                data[attr] = val
        np.savez(filepath, **data)

    @classmethod
    def load(cls, filepath: str | Path) -> Phase1ContactData:
        data = np.load(filepath, allow_pickle=True)
        return cls(
            q=data["q"],
            qdot=data["qdot"],
            tau_applied=data.get("tau_applied"),
            I_motor=data.get("I_motor"),
            F_ext=data.get("F_ext"),
            F_internal=data.get("F_internal"),
            jacobian=data.get("jacobian"),
            finger_idx=data.get("finger_idx"),
            config_idx=data.get("config_idx"),
            direction_idx=data.get("direction_idx"),
        )

    def get_data_for_finger(self, finger_idx: int) -> Phase1ContactData:
        if self.finger_idx is None:
            raise ValueError("finger_idx not available in data")
        mask = self.finger_idx == finger_idx
        return Phase1ContactData(
            q=self.q[mask],
            qdot=self.qdot[mask],
            tau_applied=self.tau_applied[mask] if self.tau_applied is not None else None,
            I_motor=self.I_motor[mask] if self.I_motor is not None else None,
            F_ext=self.F_ext[mask] if self.F_ext is not None else None,
            F_internal=self.F_internal[mask] if self.F_internal is not None else None,
            jacobian=self.jacobian[mask] if self.jacobian is not None else None,
            finger_idx=self.finger_idx[mask],
            config_idx=self.config_idx[mask] if self.config_idx is not None else None,
            direction_idx=self.direction_idx[mask] if self.direction_idx is not None else None,
            metadata=self.metadata,
        )

    @property
    def num_measurements(self) -> int:
        return len(self.q)

    def summary(self) -> str:
        lines = [
            "Phase 1 Contact Data Summary",
            f"  Measurements: {self.num_measurements}",
        ]
        if self.finger_idx is not None:
            lines.append(f"  Fingers: {list(np.unique(self.finger_idx))}")
        if self.config_idx is not None:
            lines.append(f"  Configs: {list(np.unique(self.config_idx))}")
        if self.direction_idx is not None:
            lines.append(f"  Directions: {list(np.unique(self.direction_idx))}")
        lines.append(f"  Has motor current: {self.I_motor is not None}")
        lines.append(f"  Has Jacobian: {self.jacobian is not None}")
        lines.append(f"  Has internal FT: {self.F_internal is not None}")
        return "\n".join(lines)


# Legacy alias for backward compatibility
Phase1CalibrationData = Phase1ContactData


@dataclass
class Phase2Result(CalibrationResult):
    """Phase 2: Friction & Contact Calibration result."""

    def __post_init__(self):
        self.phase = 2

    @property
    def static_friction(self) -> float | None:
        return self.parameters.get("static_friction")

    @property
    def dynamic_friction(self) -> float | None:
        return self.parameters.get("dynamic_friction")

    @property
    def contact_stiffness(self) -> float | None:
        return self.parameters.get("contact_stiffness")

    @property
    def contact_damping(self) -> float | None:
        return self.parameters.get("contact_damping")

    @property
    def contact_offset(self) -> float | None:
        return self.parameters.get("contact_offset")


@dataclass
class FullCalibrationResult:
    """Complete calibration result from all phases."""

    # Phase results
    phase0: Phase0Result | None = None
    phase1: Phase1Result | None = None
    phase2: Phase2Result | None = None

    # Overall metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

        if self.phase0:
            result["phase0_sysid"] = self.phase0.to_dict()
        if self.phase1:
            result["phase1_calibration"] = self.phase1.to_dict()
        if self.phase2:
            result["phase2_contact"] = self.phase2.to_dict()

        return result

    @classmethod
    def from_dict(cls, data: dict) -> FullCalibrationResult:
        """Create from dictionary."""
        result = cls(
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )

        if "phase0_sysid" in data:
            result.phase0 = Phase0Result.from_dict(data["phase0_sysid"])
        if "phase1_calibration" in data:
            result.phase1 = Phase1Result.from_dict(data["phase1_calibration"])
        if "phase2_contact" in data:
            result.phase2 = Phase2Result.from_dict(data["phase2_contact"])

        return result


class CalibrationResultStorage:
    """YAML-based calibration result storage."""

    def __init__(self, base_dir: str | Path | None = None):
        """Initialize calibration result storage.

        Args:
            base_dir: Base directory for storage.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        result: CalibrationResult | FullCalibrationResult,
        filename: str,
        overwrite: bool = False,
    ):
        """Save calibration result to YAML file.

        Args:
            result: Calibration result to save.
            filename: Filename (with or without .yaml extension).
            overwrite: Whether to overwrite existing file.
        """
        if not filename.endswith(".yaml") and not filename.endswith(".yml"):
            filename += ".yaml"

        filepath = self.base_dir / filename

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {filepath}")

        data = result.to_dict()

        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def load(self, filename: str) -> CalibrationResult | FullCalibrationResult:
        """Load calibration result from YAML file.

        Args:
            filename: Filename to load.

        Returns:
            Loaded calibration result.
        """
        if not filename.endswith(".yaml") and not filename.endswith(".yml"):
            filename += ".yaml"

        filepath = self.base_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        # Determine result type
        if "phase0_sysid" in data or "phase1_calibration" in data or "phase2_contact" in data:
            return FullCalibrationResult.from_dict(data)
        else:
            return CalibrationResult.from_dict(data)

    def save_json(
        self,
        result: CalibrationResult | FullCalibrationResult,
        filename: str,
        overwrite: bool = False,
    ):
        """Save calibration result to JSON file.

        Args:
            result: Calibration result to save.
            filename: Filename (with or without .json extension).
            overwrite: Whether to overwrite existing file.
        """
        if not filename.endswith(".json"):
            filename += ".json"

        filepath = self.base_dir / filename

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {filepath}")

        data = result.to_dict()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_json(self, filename: str) -> CalibrationResult | FullCalibrationResult:
        """Load calibration result from JSON file."""
        if not filename.endswith(".json"):
            filename += ".json"

        filepath = self.base_dir / filename

        with open(filepath, "r") as f:
            data = json.load(f)

        if "phase0_sysid" in data or "phase1_calibration" in data or "phase2_contact" in data:
            return FullCalibrationResult.from_dict(data)
        else:
            return CalibrationResult.from_dict(data)

    def list_results(self) -> list[str]:
        """List all calibration result files."""
        yaml_files = list(self.base_dir.glob("*.yaml")) + list(self.base_dir.glob("*.yml"))
        json_files = list(self.base_dir.glob("*.json"))
        return [f.name for f in yaml_files + json_files]


def create_calibration_summary(result: FullCalibrationResult) -> str:
    """Create a human-readable summary of calibration results.

    Args:
        result: Full calibration result.

    Returns:
        Formatted summary string.
    """
    lines = [
        "=" * 60,
        "Real2Sim Calibration Summary",
        "=" * 60,
        f"Timestamp: {result.timestamp}",
        "",
    ]

    if result.phase0:
        lines.extend([
            "Phase 0: Robot System Identification",
            "-" * 40,
            f"  Joint Stiffness: {result.phase0.joint_stiffness}",
            f"  Joint Damping: {result.phase0.joint_damping}",
            f"  Joint Friction: {result.phase0.joint_friction}",
            "",
        ])

    if result.phase1:
        lines.extend([
            "Phase 1: Jacobian-Based Current-Torque Calibration",
            "-" * 40,
        ])
        if result.phase1.k_t:
            for joint, kt in result.phase1.k_t.items():
                r2 = result.phase1.r_squared.get(joint, 0) if result.phase1.r_squared else 0
                jc = result.phase1.jacobian_consistency.get(joint, 0) if result.phase1.jacobian_consistency else 0
                lines.append(f"  {joint}: k_t={kt:.4f} Nm/A, R²={r2:.4f}, J_consistency={jc:.3f}")
        lines.append("")

    if result.phase2:
        lines.extend([
            "Phase 2: Friction & Contact Calibration",
            "-" * 40,
            f"  Static Friction: {result.phase2.static_friction}",
            f"  Dynamic Friction: {result.phase2.dynamic_friction}",
            f"  Contact Stiffness: {result.phase2.contact_stiffness}",
            f"  Contact Damping: {result.phase2.contact_damping}",
            f"  Contact Offset: {result.phase2.contact_offset}",
            "",
        ])

    lines.append("=" * 60)

    return "\n".join(lines)
