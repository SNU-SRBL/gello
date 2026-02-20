# Phase 1: Current-Torque Calibration

from .current_torque_model import (
    JacobianCurrentTorqueModel,
    Phase1Config,
    JointCalibrationResult,
    FingerCalibrationResult,
)
from .learned_model import (
    LearnedCurrentTorqueModel,
    LearnedModelConfig,
    TorqueEstimatorMLP,
    ResidualTorqueEstimator,
)

# Legacy alias
CurrentTorqueModel = JacobianCurrentTorqueModel

__all__ = [
    "JacobianCurrentTorqueModel",
    "CurrentTorqueModel",
    "Phase1Config",
    "JointCalibrationResult",
    "FingerCalibrationResult",
    "LearnedCurrentTorqueModel",
    "LearnedModelConfig",
    "TorqueEstimatorMLP",
    "ResidualTorqueEstimator",
]
