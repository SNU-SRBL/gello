# Copyright (c) 2025, SRBL
# Parameter bounds for calibration

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ParameterBounds:
    """Bounds for a calibration parameter."""

    min_value: float
    max_value: float
    log_scale: bool = False
    name: str = ""

    def clamp(self, value: float) -> float:
        """Clamp value to bounds."""
        return max(self.min_value, min(self.max_value, value))

    def sample_uniform(self) -> float:
        """Sample uniformly from bounds."""
        if self.log_scale:
            log_min = np.log(self.min_value)
            log_max = np.log(self.max_value)
            return float(np.exp(np.random.uniform(log_min, log_max)))
        return float(np.random.uniform(self.min_value, self.max_value))

    def linspace(self, num: int) -> np.ndarray:
        """Generate linearly spaced values."""
        if self.log_scale:
            return np.logspace(
                np.log10(self.min_value),
                np.log10(self.max_value),
                num,
            )
        return np.linspace(self.min_value, self.max_value, num)


# Phase 0: Joint Dynamics
PHASE0_BOUNDS = {
    "joint_stiffness": ParameterBounds(
        min_value=1.0,
        max_value=1000.0,
        log_scale=True,
        name="Joint Stiffness [N·m/rad]",
    ),
    "joint_damping": ParameterBounds(
        min_value=0.1,
        max_value=100.0,
        log_scale=True,
        name="Joint Damping [N·m·s/rad]",
    ),
    "joint_friction": ParameterBounds(
        min_value=0.0,
        max_value=1.0,
        log_scale=False,
        name="Joint Friction [N·m]",
    ),
}

# Phase 1: Current-Torque
PHASE1_BOUNDS = {
    "k_gain": ParameterBounds(
        min_value=0.01,
        max_value=1.0,
        log_scale=False,
        name="Current-Torque Gain [N·m/A]",
    ),
    "k_offset": ParameterBounds(
        min_value=-0.1,
        max_value=0.1,
        log_scale=False,
        name="Current-Torque Offset [N·m]",
    ),
}

# Phase 2: Friction & Contact
PHASE2_BOUNDS = {
    "static_friction": ParameterBounds(
        min_value=0.1,
        max_value=2.0,
        log_scale=False,
        name="Static Friction Coefficient",
    ),
    "dynamic_friction": ParameterBounds(
        min_value=0.05,
        max_value=1.5,
        log_scale=False,
        name="Dynamic Friction Coefficient",
    ),
    "contact_stiffness": ParameterBounds(
        min_value=1e4,
        max_value=1e7,
        log_scale=True,
        name="Contact Stiffness [N/m]",
    ),
    "contact_damping": ParameterBounds(
        min_value=1e2,
        max_value=1e5,
        log_scale=True,
        name="Contact Damping [N·s/m]",
    ),
    "contact_offset": ParameterBounds(
        min_value=0.0001,
        max_value=0.01,
        log_scale=True,
        name="Contact Offset [m]",
    ),
}


def get_all_bounds() -> dict[str, ParameterBounds]:
    """Get all parameter bounds."""
    all_bounds = {}
    all_bounds.update(PHASE0_BOUNDS)
    all_bounds.update(PHASE1_BOUNDS)
    all_bounds.update(PHASE2_BOUNDS)
    return all_bounds


def validate_parameters(params: dict, phase: int) -> tuple[bool, list[str]]:
    """Validate parameters against bounds.

    Args:
        params: Parameter dictionary.
        phase: Calibration phase (0, 1, or 2).

    Returns:
        Tuple of (is_valid, error_messages).
    """
    if phase == 0:
        bounds = PHASE0_BOUNDS
    elif phase == 1:
        bounds = PHASE1_BOUNDS
    elif phase == 2:
        bounds = PHASE2_BOUNDS
    else:
        return False, [f"Invalid phase: {phase}"]

    errors = []

    for name, bound in bounds.items():
        if name in params:
            value = params[name]
            if isinstance(value, np.ndarray):
                if np.any(value < bound.min_value) or np.any(value > bound.max_value):
                    errors.append(
                        f"{name}: values outside bounds [{bound.min_value}, {bound.max_value}]"
                    )
            else:
                if value < bound.min_value or value > bound.max_value:
                    errors.append(
                        f"{name}: {value} outside bounds [{bound.min_value}, {bound.max_value}]"
                    )

    # Phase 2: dynamic friction must be <= static friction
    if phase == 2:
        if "static_friction" in params and "dynamic_friction" in params:
            if params["dynamic_friction"] > params["static_friction"]:
                errors.append(
                    f"dynamic_friction ({params['dynamic_friction']}) must be <= "
                    f"static_friction ({params['static_friction']})"
                )

    return len(errors) == 0, errors
