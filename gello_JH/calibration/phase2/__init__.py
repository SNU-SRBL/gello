# Phase 2: Friction & Contact Calibration

from .friction_estimator import (
    FrictionEstimator,
    FrictionEstimatorConfig,
    compute_dynamic_friction_from_motion,
    compute_static_friction_from_forces,
)
from .bayesian_optimizer import BayesianOptimizer, BayesianOptimizerConfig
from .slip_matcher import SlipMatcher, SlipMatchingConfig

# Legacy compatibility
from .slip_matcher import SlipCurrentMatcher

__all__ = [
    "FrictionEstimator",
    "FrictionEstimatorConfig",
    "compute_dynamic_friction_from_motion",
    "compute_static_friction_from_forces",
    "BayesianOptimizer",
    "BayesianOptimizerConfig",
    "SlipMatcher",
    "SlipMatchingConfig",
    "SlipCurrentMatcher",  # Legacy compatibility
]
