# Copyright (c) 2025, SRBL
# Phase 2: Bayesian Optimizer for Friction Calibration

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Callable
from dataclasses import dataclass

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

if TYPE_CHECKING:
    from ...envs.phase2_friction import FrictionEnv


@dataclass
class BayesianOptimizerConfig:
    """Configuration for Bayesian optimization."""

    # Number of trials
    n_trials: int = 100
    n_initial_samples: int = 10

    # Parameter bounds
    static_friction_bounds: tuple[float, float] = (0.1, 2.0)
    dynamic_friction_bounds: tuple[float, float] = (0.05, 1.5)
    contact_stiffness_bounds: tuple[float, float] = (1e4, 1e7)
    contact_damping_bounds: tuple[float, float] = (1e2, 1e5)
    contact_offset_bounds: tuple[float, float] = (0.0001, 0.01)

    # Optimization
    direction: str = "minimize"
    sampler: str = "tpe"  # "tpe", "cmaes", "random"


class BayesianOptimizer:
    """Bayesian optimization for friction and contact parameters.

    Uses Optuna for efficient parameter search to match I_slip
    between real and simulated experiments.
    """

    def __init__(
        self,
        env: FrictionEnv,
        cfg: BayesianOptimizerConfig | None = None,
    ):
        """Initialize optimizer.

        Args:
            env: Friction calibration environment.
            cfg: Optimization configuration.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization. "
                              "Install with: pip install optuna")

        self.env = env
        self.cfg = cfg or BayesianOptimizerConfig()
        self.device = env.device

        # Target I_slip (from real experiment)
        self._real_I_slip: float | None = None

        # Best result
        self.best_params: dict | None = None
        self.best_loss: float = float("inf")

        # History
        self.trial_history: list[dict] = []

    def set_target_I_slip(self, I_slip: float):
        """Set target slip current from real experiment.

        Args:
            I_slip: Real robot's slip current.
        """
        self._real_I_slip = I_slip

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial.

        Returns:
            Loss value (I_slip error).
        """
        # Sample parameters
        static_friction = trial.suggest_float(
            "static_friction",
            *self.cfg.static_friction_bounds,
        )
        dynamic_friction = trial.suggest_float(
            "dynamic_friction",
            self.cfg.dynamic_friction_bounds[0],
            min(static_friction, self.cfg.dynamic_friction_bounds[1]),
        )
        contact_stiffness = trial.suggest_float(
            "contact_stiffness",
            *self.cfg.contact_stiffness_bounds,
            log=True,
        )
        contact_damping = trial.suggest_float(
            "contact_damping",
            *self.cfg.contact_damping_bounds,
            log=True,
        )
        contact_offset = trial.suggest_float(
            "contact_offset",
            *self.cfg.contact_offset_bounds,
            log=True,
        )

        # Apply parameters
        self.env.set_friction_params(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            contact_stiffness=contact_stiffness,
            contact_damping=contact_damping,
            contact_offset=contact_offset,
        )

        # Run slip test
        results = self.env.run_slip_test()

        # Compute loss
        if results["slip_detected"].any():
            sim_I_slip = results["I_slip"].mean().item()
            loss = abs(sim_I_slip - self._real_I_slip)
        else:
            # Large penalty for no slip
            loss = 1e6

        # Store in history
        self.trial_history.append({
            "trial": len(self.trial_history),
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "contact_stiffness": contact_stiffness,
            "contact_damping": contact_damping,
            "contact_offset": contact_offset,
            "sim_I_slip": results["I_slip"].mean().item() if results["slip_detected"].any() else None,
            "loss": loss,
        })

        return loss

    def optimize(
        self,
        real_I_slip: float | None = None,
        callback: Callable[[int, float, dict], None] | None = None,
    ) -> dict:
        """Run Bayesian optimization.

        Args:
            real_I_slip: Target slip current from real experiment.
            callback: Optional callback(trial_num, loss, params).

        Returns:
            Best parameters found.
        """
        if real_I_slip is not None:
            self._real_I_slip = real_I_slip

        if self._real_I_slip is None:
            raise ValueError("Target I_slip not set. Call set_target_I_slip() first.")

        # Create Optuna study
        if self.cfg.sampler == "tpe":
            sampler = optuna.samplers.TPESampler(n_startup_trials=self.cfg.n_initial_samples)
        elif self.cfg.sampler == "cmaes":
            sampler = optuna.samplers.CmaEsSampler(n_startup_trials=self.cfg.n_initial_samples)
        else:
            sampler = optuna.samplers.RandomSampler()

        study = optuna.create_study(
            direction=self.cfg.direction,
            sampler=sampler,
        )

        # Custom callback wrapper
        def optuna_callback(study: optuna.Study, trial: optuna.FrozenTrial):
            if callback:
                callback(
                    trial.number,
                    trial.value,
                    trial.params,
                )

        # Run optimization
        study.optimize(
            self._objective,
            n_trials=self.cfg.n_trials,
            callbacks=[optuna_callback] if callback else None,
            show_progress_bar=True,
        )

        # Get best parameters
        self.best_params = study.best_params
        self.best_loss = study.best_value

        return self.best_params

    def get_result(self) -> dict:
        """Get optimization result.

        Returns:
            Dictionary with best parameters and metadata.
        """
        from ...data.storage import Phase2Result

        if self.best_params is None:
            return {}

        return Phase2Result(
            phase=2,
            parameters={
                "static_friction": self.best_params["static_friction"],
                "dynamic_friction": self.best_params["dynamic_friction"],
                "contact_stiffness": self.best_params["contact_stiffness"],
                "contact_damping": self.best_params["contact_damping"],
                "contact_offset": self.best_params["contact_offset"],
            },
            loss_history=[t["loss"] for t in self.trial_history],
            validation_metrics={
                "best_loss": self.best_loss,
                "real_I_slip": self._real_I_slip,
            },
            metadata={
                "n_trials": len(self.trial_history),
                "config": {
                    "n_trials": self.cfg.n_trials,
                    "sampler": self.cfg.sampler,
                },
            },
        )
