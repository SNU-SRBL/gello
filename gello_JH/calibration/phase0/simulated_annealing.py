# Copyright (c) 2025, SRBL
# Phase 0: Simulated Annealing Optimizer (SIMPLER style)

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class SAConfig:
    """Configuration for Simulated Annealing optimizer."""

    # SIMPLER style: 3 rounds with shrinking bounds
    num_rounds: int = 3
    trials_per_round: int = 100

    # Temperature schedule
    temp_initial: float = 1.0
    temp_final: float = 0.01

    # Bounds shrinking factor per round (SIMPLER style)
    shrink_factor: float = 0.5

    # Random seed
    seed: int | None = None


class SimulatedAnnealingOptimizer:
    """Simulated Annealing optimizer for system identification.

    Implements SIMPLER paper style optimization:
    - 3 rounds of simulated annealing
    - Each round shrinks the search bounds around the best solution
    - Normalized parameter space [0, 1] for equal treatment of all params
    - Metropolis acceptance criterion for escaping local minima
    """

    def __init__(
        self,
        param_names: list[str],
        param_bounds: dict[str, tuple[float, float]],
        cfg: SAConfig | None = None,
    ):
        """Initialize optimizer.

        Args:
            param_names: List of parameter names to optimize.
            param_bounds: Dictionary mapping parameter names to (low, high) bounds.
            cfg: Configuration.
        """
        self.param_names = param_names
        self.initial_bounds = param_bounds.copy()
        self.current_bounds = param_bounds.copy()
        self.cfg = cfg or SAConfig()

        self.rng = np.random.default_rng(self.cfg.seed)

        # History
        self.loss_history: list[float] = []
        self.param_history: list[dict[str, np.ndarray]] = []
        self.best_loss: float = float("inf")
        self.best_params: dict[str, np.ndarray] | None = None

    def optimize(
        self,
        loss_fn: Callable[[dict[str, np.ndarray]], float],
        initial_params: dict[str, np.ndarray] | None = None,
        callback: Callable[[int, int, float, dict], None] | None = None,
        round_callback: Callable[[int, dict, dict, float], None] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run multi-round simulated annealing optimization.

        Args:
            loss_fn: Function that takes params dict and returns loss value.
            initial_params: Optional initial parameter values.
            callback: Optional callback(round_idx, trial_idx, loss, params).
            round_callback: Optional callback(round_idx, current_bounds, best_params, best_loss)
                called after each round completes.

        Returns:
            Best parameters found.
        """
        # Initialize parameters
        if initial_params is not None:
            current_params = {k: v.copy() for k, v in initial_params.items()}
        else:
            current_params = self._sample_random_params()

        # Evaluate initial
        current_loss = loss_fn(current_params)
        self.best_loss = current_loss
        self.best_params = {k: v.copy() for k, v in current_params.items()}

        # Run rounds
        for round_idx in range(self.cfg.num_rounds):
            print(f"[SA] Round {round_idx + 1}/{self.cfg.num_rounds}")

            # Run single round of SA
            current_params, current_loss = self._run_sa_round(
                loss_fn,
                current_params,
                current_loss,
                round_idx,
                callback,
            )

            # Round callback
            if round_callback:
                round_callback(round_idx, self.current_bounds, self.best_params, self.best_loss)

            # Shrink bounds around best solution for next round
            if round_idx < self.cfg.num_rounds - 1:
                self._shrink_bounds_around(self.best_params)

        return self.best_params

    def _run_sa_round(
        self,
        loss_fn: Callable,
        initial_params: dict[str, np.ndarray],
        initial_loss: float,
        round_idx: int,
        callback: Callable | None,
    ) -> tuple[dict[str, np.ndarray], float]:
        """Run a single round of simulated annealing."""

        current_params = {k: v.copy() for k, v in initial_params.items()}
        current_loss = initial_loss

        for trial_idx in range(self.cfg.trials_per_round):
            # Compute temperature (exponential decay)
            progress = trial_idx / max(1, self.cfg.trials_per_round - 1)
            temp = self.cfg.temp_initial * (self.cfg.temp_final / self.cfg.temp_initial) ** progress

            # Generate neighbor
            neighbor_params = self._generate_neighbor(current_params, temp)

            # Evaluate
            neighbor_loss = loss_fn(neighbor_params)

            # Accept or reject
            if self._accept(current_loss, neighbor_loss, temp):
                current_params = neighbor_params
                current_loss = neighbor_loss

            # Update best
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.best_params = {k: v.copy() for k, v in current_params.items()}

            # Record history
            self.loss_history.append(current_loss)
            self.param_history.append({k: v.copy() for k, v in current_params.items()})

            # Callback
            if callback:
                callback(round_idx, trial_idx, current_loss, current_params)

            # Progress print every 20 trials
            if (trial_idx + 1) % 20 == 0:
                print(f"  Trial {trial_idx + 1}/{self.cfg.trials_per_round}: "
                      f"loss={current_loss:.6f}, best={self.best_loss:.6f}, temp={temp:.4f}")

        return current_params, current_loss

    def _sample_random_params(self) -> dict[str, np.ndarray]:
        """Sample random parameters within current bounds."""
        params = {}
        for name in self.param_names:
            low, high = self.current_bounds[name]
            if isinstance(low, np.ndarray):
                params[name] = self.rng.uniform(low, high)
            else:
                params[name] = self.rng.uniform(low, high)
        return params

    def _to_normalized(self, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convert parameters to normalized [0,1] space (SIMPLER style)."""
        normalized = {}
        for name, value in params.items():
            low, high = self.current_bounds[name]
            if isinstance(low, np.ndarray):
                normalized[name] = (value - low) / (high - low + 1e-10)
            else:
                normalized[name] = (value - low) / (high - low + 1e-10)
        return normalized

    def _from_normalized(self, normalized: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convert parameters from [0,1] space to original space."""
        params = {}
        for name, value in normalized.items():
            low, high = self.current_bounds[name]
            if isinstance(low, np.ndarray):
                params[name] = value * (high - low) + low
            else:
                params[name] = value * (high - low) + low
        return params

    def _generate_neighbor(
        self,
        params: dict[str, np.ndarray],
        temperature: float,
    ) -> dict[str, np.ndarray]:
        """Generate neighbor solution in normalized [0,1] space (SIMPLER style).

        All parameters are normalized to [0,1] before perturbation,
        ensuring equal treatment regardless of original scale.
        """
        # Convert to normalized [0,1] space
        normalized = self._to_normalized(params)
        neighbor_norm = {}

        for name in self.param_names:
            current = normalized[name]

            # In [0,1] space, perturbation scale is uniform for all params
            if isinstance(current, np.ndarray):
                scale = temperature * 0.3
                perturbation = self.rng.normal(0, scale, size=current.shape)
                new_value = current + perturbation
                neighbor_norm[name] = np.clip(new_value, 0.0, 1.0)
            else:
                scale = temperature * 0.3
                perturbation = self.rng.normal(0, scale)
                new_value = current + perturbation
                neighbor_norm[name] = np.clip(new_value, 0.0, 1.0)

        # Convert back to original space
        return self._from_normalized(neighbor_norm)

    def _accept(self, current_loss: float, new_loss: float, temperature: float) -> bool:
        """Decide whether to accept new solution (Metropolis criterion)."""
        if new_loss < current_loss:
            return True

        # Accept worse solution with probability exp(-(new - current) / temp)
        delta = new_loss - current_loss
        probability = np.exp(-delta / (temperature + 1e-10))
        return self.rng.random() < probability

    def _shrink_bounds_around(self, center_params: dict[str, np.ndarray]):
        """Shrink bounds around center parameters for next round."""
        for name in self.param_names:
            init_low, init_high = self.initial_bounds[name]
            center = center_params[name]

            # New range is shrink_factor * original range, centered on best
            if isinstance(init_low, np.ndarray):
                init_range = init_high - init_low
                new_range = init_range * self.cfg.shrink_factor

                new_low = np.maximum(init_low, center - new_range / 2)
                new_high = np.minimum(init_high, center + new_range / 2)
            else:
                init_range = init_high - init_low
                new_range = init_range * self.cfg.shrink_factor

                new_low = max(init_low, center - new_range / 2)
                new_high = min(init_high, center + new_range / 2)

            self.current_bounds[name] = (new_low, new_high)

        print(f"[SA] Bounds shrunk by factor {self.cfg.shrink_factor}")

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get summary of optimization results."""
        return {
            "best_loss": self.best_loss,
            "best_params": self.best_params,
            "num_evaluations": len(self.loss_history),
            "loss_history": self.loss_history,
            "config": {
                "num_rounds": self.cfg.num_rounds,
                "trials_per_round": self.cfg.trials_per_round,
                "shrink_factor": self.cfg.shrink_factor,
            },
        }
