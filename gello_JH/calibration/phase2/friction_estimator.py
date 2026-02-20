# Copyright (c) 2025, SRBL
# Phase 2: Friction Estimator

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ...envs.phase2_friction import FrictionEnv


@dataclass
class FrictionEstimatorConfig:
    """Configuration for friction estimation."""

    # Grid search parameters
    static_friction_range: tuple[float, float, int] = (0.2, 1.5, 10)
    """Range for static friction coefficient (min, max, num_points)."""

    dynamic_friction_range: tuple[float, float, int] = (0.1, 1.0, 10)
    """Range for dynamic friction coefficient (min, max, num_points)."""

    # Matching tolerances
    mu_s_tolerance: float = 0.1
    """Tolerance for static friction coefficient matching."""

    mu_d_tolerance: float = 0.15
    """Tolerance for dynamic friction coefficient matching."""

    q_slip_tolerance: float = 0.05
    """Tolerance for grip position at slip matching (rad)."""

    # Loss weights
    static_friction_weight: float = 1.0
    """Weight for static friction matching loss."""

    dynamic_friction_weight: float = 0.5
    """Weight for dynamic friction matching loss."""

    grip_position_weight: float = 0.3
    """Weight for grip position matching loss."""


class FrictionEstimator:
    """Friction coefficient estimator using grid search.

    Estimates both static friction (μ_s) and dynamic friction (μ_d)
    by matching simulation slip tests with real robot measurements.

    Matching criteria:
    1. Static friction: μ_s = F_tangential / F_normal at slip onset
    2. Dynamic friction: μ_d from post-slip object motion (ArUco tracking)
    3. Grip position at slip: q_slip matching between sim and real
    """

    def __init__(
        self,
        env: FrictionEnv,
        cfg: FrictionEstimatorConfig | None = None,
    ):
        """Initialize estimator.

        Args:
            env: Friction calibration environment.
            cfg: Estimator configuration.
        """
        self.env = env
        self.cfg = cfg or FrictionEstimatorConfig()
        self.device = env.device

        # Results
        self.grid_results: list[dict] = []
        self.best_params: dict | None = None

    def grid_search(
        self,
        real_static_friction: float,
        real_dynamic_friction: float | None = None,
        real_q_slip: float | None = None,
    ) -> dict:
        """Run grid search over friction parameters.

        Args:
            real_static_friction: Target static friction coefficient from real experiment.
            real_dynamic_friction: Target dynamic friction coefficient (optional).
            real_q_slip: Target grip position at slip (optional).

        Returns:
            Best friction parameters found.
        """
        static_vals = np.linspace(*self.cfg.static_friction_range)
        dynamic_vals = np.linspace(*self.cfg.dynamic_friction_range)

        best_loss = float("inf")
        best_params = None

        total = len(static_vals) * len(dynamic_vals)
        count = 0

        print(f"\nGrid search: {total} combinations")
        print(f"  Target μ_s: {real_static_friction:.3f}")
        if real_dynamic_friction is not None:
            print(f"  Target μ_d: {real_dynamic_friction:.3f}")
        if real_q_slip is not None:
            print(f"  Target q_slip: {real_q_slip:.3f}")

        for mu_s in static_vals:
            for mu_d in dynamic_vals:
                # Skip invalid combinations (dynamic > static)
                if mu_d > mu_s:
                    continue

                count += 1

                # Set friction parameters
                self.env.set_friction_params(
                    static_friction=mu_s,
                    dynamic_friction=mu_d,
                )

                # Run slip test
                results = self.env.run_slip_test()

                # Compute loss
                loss = self._compute_loss(
                    results,
                    real_static_friction,
                    real_dynamic_friction,
                    real_q_slip,
                )

                # Extract simulation results
                if results["slip_detected"].any():
                    sim_mu_s = float(np.mean(results["static_friction"][results["slip_detected"]]))
                    sim_mu_d = float(np.mean(results["dynamic_friction"][results["dynamic_friction_valid"]]))
                    sim_q_slip = float(np.mean(results["grip_position_at_slip"][results["slip_detected"]]))
                else:
                    sim_mu_s = None
                    sim_mu_d = None
                    sim_q_slip = None

                # Store result
                result_entry = {
                    "static_friction": mu_s,
                    "dynamic_friction": mu_d,
                    "sim_mu_s": sim_mu_s,
                    "sim_mu_d": sim_mu_d,
                    "sim_q_slip": sim_q_slip,
                    "loss": loss,
                    "slip_detected": results["slip_detected"].any(),
                }
                self.grid_results.append(result_entry)

                # Update best
                if loss < best_loss:
                    best_loss = loss
                    best_params = {
                        "static_friction": mu_s,
                        "dynamic_friction": mu_d,
                        "sim_mu_s": sim_mu_s,
                        "sim_mu_d": sim_mu_d,
                        "sim_q_slip": sim_q_slip,
                        "loss": loss,
                    }

                if count % 10 == 0:
                    print(f"Progress: {count}/{total}, Best loss: {best_loss:.6f}")

        self.best_params = best_params
        return best_params

    def _compute_loss(
        self,
        sim_results: dict,
        real_static_friction: float,
        real_dynamic_friction: float | None,
        real_q_slip: float | None,
    ) -> float:
        """Compute matching loss.

        Args:
            sim_results: Simulation slip test results.
            real_static_friction: Target static friction.
            real_dynamic_friction: Target dynamic friction (optional).
            real_q_slip: Target grip position at slip (optional).

        Returns:
            Loss value.
        """
        if not sim_results["slip_detected"].any():
            return 1e6  # Large penalty for no slip

        loss = 0.0

        # Static friction loss
        sim_mu_s = np.mean(sim_results["static_friction"][sim_results["slip_detected"]])
        static_loss = (sim_mu_s - real_static_friction) ** 2
        loss += self.cfg.static_friction_weight * static_loss

        # Dynamic friction loss (if target provided)
        if real_dynamic_friction is not None and sim_results["dynamic_friction_valid"].any():
            sim_mu_d = np.mean(sim_results["dynamic_friction"][sim_results["dynamic_friction_valid"]])
            dynamic_loss = (sim_mu_d - real_dynamic_friction) ** 2
            loss += self.cfg.dynamic_friction_weight * dynamic_loss

        # Grip position loss (if target provided)
        if real_q_slip is not None:
            sim_q_slip = np.mean(sim_results["grip_position_at_slip"][sim_results["slip_detected"]])
            q_slip_loss = (sim_q_slip - real_q_slip) ** 2
            loss += self.cfg.grip_position_weight * q_slip_loss

        return float(loss)

    def refine_search(
        self,
        real_static_friction: float,
        real_dynamic_friction: float | None,
        center_params: dict,
        search_radius: float = 0.1,
        num_points: int = 5,
    ) -> dict:
        """Refine search around found parameters.

        Args:
            real_static_friction: Target static friction.
            real_dynamic_friction: Target dynamic friction.
            center_params: Center parameters for refinement.
            search_radius: Search radius around center.
            num_points: Number of points in each dimension.

        Returns:
            Refined parameters.
        """
        mu_s_center = center_params["static_friction"]
        mu_d_center = center_params["dynamic_friction"]

        static_vals = np.linspace(
            max(0.1, mu_s_center - search_radius),
            min(2.0, mu_s_center + search_radius),
            num_points,
        )
        dynamic_vals = np.linspace(
            max(0.05, mu_d_center - search_radius),
            min(mu_s_center, mu_d_center + search_radius),
            num_points,
        )

        best_loss = float("inf")
        best_params = center_params

        for mu_s in static_vals:
            for mu_d in dynamic_vals:
                if mu_d > mu_s:
                    continue

                self.env.set_friction_params(
                    static_friction=mu_s,
                    dynamic_friction=mu_d,
                )

                results = self.env.run_slip_test()
                loss = self._compute_loss(
                    results,
                    real_static_friction,
                    real_dynamic_friction,
                    None,  # No q_slip target for refinement
                )

                if loss < best_loss:
                    best_loss = loss
                    sim_mu_s = np.mean(results["static_friction"][results["slip_detected"]]) \
                        if results["slip_detected"].any() else None
                    sim_mu_d = np.mean(results["dynamic_friction"][results["dynamic_friction_valid"]]) \
                        if results["dynamic_friction_valid"].any() else None

                    best_params = {
                        "static_friction": mu_s,
                        "dynamic_friction": mu_d,
                        "sim_mu_s": sim_mu_s,
                        "sim_mu_d": sim_mu_d,
                        "loss": loss,
                    }

        self.best_params = best_params
        return best_params

    def get_friction_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get loss surface as 2D array for visualization.

        Returns:
            Tuple of (static_vals, dynamic_vals, loss_map).
        """
        if not self.grid_results:
            return np.array([]), np.array([]), np.array([])

        static_vals = sorted(set(r["static_friction"] for r in self.grid_results))
        dynamic_vals = sorted(set(r["dynamic_friction"] for r in self.grid_results))

        loss_map = np.full((len(static_vals), len(dynamic_vals)), np.nan)

        for r in self.grid_results:
            i = static_vals.index(r["static_friction"])
            j = dynamic_vals.index(r["dynamic_friction"])
            loss_map[i, j] = r["loss"]

        return np.array(static_vals), np.array(dynamic_vals), loss_map

    def summary(self) -> str:
        """Generate summary of estimation results.

        Returns:
            Summary string.
        """
        if self.best_params is None:
            return "No estimation results available."

        lines = [
            "Friction Estimation Results",
            "=" * 40,
            f"Best Static Friction (μ_s):  {self.best_params['static_friction']:.4f}",
            f"Best Dynamic Friction (μ_d): {self.best_params['dynamic_friction']:.4f}",
            f"Simulated μ_s:               {self.best_params.get('sim_mu_s', 'N/A')}",
            f"Simulated μ_d:               {self.best_params.get('sim_mu_d', 'N/A')}",
            f"Best Loss:                   {self.best_params['loss']:.6f}",
            f"Total trials:                {len(self.grid_results)}",
        ]

        return "\n".join(lines)


def compute_dynamic_friction_from_motion(
    object_mass: float,
    normal_force: float,
    measured_acceleration: float | np.ndarray,
    gravity: float = 9.81,
) -> float:
    """Compute dynamic friction coefficient from object motion.

    Uses the equation: m × a = m × g - μ_d × N
    Therefore: μ_d = (g - a) × m / N

    Args:
        object_mass: Object mass in kg.
        normal_force: Normal force at contact (N).
        measured_acceleration: Measured vertical acceleration (m/s²).
        gravity: Gravitational acceleration (m/s²).

    Returns:
        Dynamic friction coefficient (μ_d).
    """
    if isinstance(measured_acceleration, np.ndarray):
        # Use vertical component or magnitude
        a = np.abs(measured_acceleration[-1]) if len(measured_acceleration) == 3 else np.mean(np.abs(measured_acceleration))
    else:
        a = abs(measured_acceleration)

    if normal_force < 0.01:  # Minimum threshold
        return 0.0

    mu_d = (gravity - a) * object_mass / normal_force
    return max(0.0, min(mu_d, 2.0))  # Clamp to reasonable range


def compute_static_friction_from_forces(
    tangential_force: float,
    normal_force: float,
) -> float:
    """Compute static friction coefficient from force measurements at slip.

    Uses the equation: μ_s = F_tangential / F_normal

    Args:
        tangential_force: Tangential (friction) force at slip (N).
        normal_force: Normal force at slip (N).

    Returns:
        Static friction coefficient (μ_s).
    """
    if normal_force < 0.01:  # Minimum threshold
        return 0.0

    mu_s = tangential_force / normal_force
    return max(0.0, min(mu_s, 2.0))  # Clamp to reasonable range
