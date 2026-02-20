# Copyright (c) 2025, SRBL
# Phase 2: Slip Matcher (Position/Force based)

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ...envs.phase2_friction import FrictionEnv
    from ...data.storage import Phase2SlipData


@dataclass
class SlipMatchingConfig:
    """Configuration for slip matching."""

    # Grip joints to compare
    grip_joint_indices: list[int] | None = None
    """If None, use all joints."""

    # Matching tolerances
    q_slip_tolerance: float = 0.05
    """Joint position tolerance at slip (rad)."""

    mu_s_tolerance: float = 0.1
    """Static friction coefficient tolerance."""

    mu_d_tolerance: float = 0.15
    """Dynamic friction coefficient tolerance."""

    F_slip_tolerance: float = 0.5
    """Force tolerance at slip (N)."""

    # Weighting
    use_joint_weights: bool = False
    joint_weights: np.ndarray | None = None

    # Loss weights
    static_friction_weight: float = 1.0
    dynamic_friction_weight: float = 0.5
    position_weight: float = 0.3
    force_weight: float = 0.2


class SlipMatcher:
    """Match slip characteristics between real and simulated experiments.

    Computes loss functions for friction optimization based on:
    1. Static friction coefficient (μ_s) matching
    2. Dynamic friction coefficient (μ_d) matching
    3. Grip position at slip (q_slip) matching
    4. Force at slip (F_slip) matching

    Replaces the previous SlipCurrentMatcher which used current (I) matching.
    """

    def __init__(self, cfg: SlipMatchingConfig | None = None):
        """Initialize matcher.

        Args:
            cfg: Matching configuration.
        """
        self.cfg = cfg or SlipMatchingConfig()

        if self.cfg.grip_joint_indices is None:
            # Default: thumb and index joints (0-3, 4-7)
            self.cfg.grip_joint_indices = list(range(8))

    def compute_friction_loss(
        self,
        real_mu_s: float,
        sim_mu_s: float,
        real_mu_d: float | None = None,
        sim_mu_d: float | None = None,
    ) -> float:
        """Compute friction coefficient matching loss.

        Args:
            real_mu_s: Real robot's static friction coefficient.
            sim_mu_s: Simulated static friction coefficient.
            real_mu_d: Real robot's dynamic friction coefficient (optional).
            sim_mu_d: Simulated dynamic friction coefficient (optional).

        Returns:
            Loss value.
        """
        loss = 0.0

        # Static friction loss
        static_loss = (sim_mu_s - real_mu_s) ** 2
        loss += self.cfg.static_friction_weight * static_loss

        # Dynamic friction loss (if both provided)
        if real_mu_d is not None and sim_mu_d is not None:
            dynamic_loss = (sim_mu_d - real_mu_d) ** 2
            loss += self.cfg.dynamic_friction_weight * dynamic_loss

        return float(loss)

    def compute_position_loss(
        self,
        real_q_slip: np.ndarray | float,
        sim_q_slip: np.ndarray | float,
    ) -> float:
        """Compute grip position at slip matching loss.

        Args:
            real_q_slip: Real robot's joint positions at slip.
            sim_q_slip: Simulated joint positions at slip.

        Returns:
            Loss value.
        """
        if isinstance(real_q_slip, (int, float)):
            real_q_slip = np.array([real_q_slip])
        if isinstance(sim_q_slip, (int, float)):
            sim_q_slip = np.array([sim_q_slip])

        # Extract grip joints
        indices = self.cfg.grip_joint_indices

        if len(real_q_slip) > len(indices):
            real_grip = real_q_slip[indices]
        else:
            real_grip = real_q_slip

        if len(sim_q_slip) > len(indices):
            sim_grip = sim_q_slip[indices]
        else:
            sim_grip = sim_q_slip

        # Compute loss
        if self.cfg.use_joint_weights and self.cfg.joint_weights is not None:
            weights = self.cfg.joint_weights[:len(sim_grip)]
            loss = np.sum(weights * (sim_grip - real_grip) ** 2)
        else:
            loss = np.mean((sim_grip - real_grip) ** 2)

        return float(loss) * self.cfg.position_weight

    def compute_force_loss(
        self,
        real_F_tangential: float,
        real_F_normal: float,
        sim_F_tangential: float,
        sim_F_normal: float,
    ) -> float:
        """Compute force at slip matching loss.

        Args:
            real_F_tangential: Real tangential force at slip (N).
            real_F_normal: Real normal force at slip (N).
            sim_F_tangential: Simulated tangential force at slip (N).
            sim_F_normal: Simulated normal force at slip (N).

        Returns:
            Loss value.
        """
        tangential_loss = (sim_F_tangential - real_F_tangential) ** 2
        normal_loss = (sim_F_normal - real_F_normal) ** 2

        loss = tangential_loss + normal_loss
        return float(loss) * self.cfg.force_weight

    def compute_total_loss(
        self,
        real_data: dict,
        sim_results: dict,
    ) -> float:
        """Compute total matching loss.

        Args:
            real_data: Real robot slip test data with keys:
                - static_friction (float)
                - dynamic_friction (float, optional)
                - q_slip (array or float)
                - F_tangential (float)
                - F_normal (float)
            sim_results: Simulation slip test results.

        Returns:
            Total loss value.
        """
        if not sim_results.get("slip_detected", np.array([False])).any():
            return 1e6  # Large penalty for no slip

        total_loss = 0.0

        # Get simulation values (average over environments)
        slip_mask = sim_results["slip_detected"]

        sim_mu_s = np.mean(sim_results["static_friction"][slip_mask])
        sim_q_slip = np.mean(sim_results["grip_position_at_slip"][slip_mask])
        sim_F_tangential = np.mean(sim_results["F_tangential_at_slip"][slip_mask])
        sim_F_normal = np.mean(sim_results["F_normal_at_slip"][slip_mask])

        # Friction loss
        total_loss += self.compute_friction_loss(
            real_mu_s=real_data["static_friction"],
            sim_mu_s=sim_mu_s,
            real_mu_d=real_data.get("dynamic_friction"),
            sim_mu_d=np.mean(sim_results["dynamic_friction"][sim_results["dynamic_friction_valid"]])
                if sim_results["dynamic_friction_valid"].any() else None,
        )

        # Position loss
        total_loss += self.compute_position_loss(
            real_q_slip=real_data.get("q_slip", sim_q_slip),
            sim_q_slip=sim_q_slip,
        )

        # Force loss
        total_loss += self.compute_force_loss(
            real_F_tangential=real_data.get("F_tangential", sim_F_tangential),
            real_F_normal=real_data.get("F_normal", sim_F_normal),
            sim_F_tangential=sim_F_tangential,
            sim_F_normal=sim_F_normal,
        )

        return total_loss

    def compute_batch_loss(
        self,
        real_trials: list[dict],
        sim_results: list[dict],
    ) -> float:
        """Compute average loss over multiple trials.

        Args:
            real_trials: List of real robot trial data.
            sim_results: List of simulation results.

        Returns:
            Average loss.
        """
        if len(real_trials) != len(sim_results):
            raise ValueError("Number of real and sim trials must match")

        total_loss = 0.0
        valid_count = 0

        for real, sim in zip(real_trials, sim_results):
            loss = self.compute_total_loss(real, sim)
            if loss < 1e5:  # Valid comparison
                total_loss += loss
                valid_count += 1

        return total_loss / max(valid_count, 1)

    def check_convergence(
        self,
        real_mu_s: float,
        sim_mu_s: float,
        real_mu_d: float | None = None,
        sim_mu_d: float | None = None,
    ) -> bool:
        """Check if calibration has converged.

        Args:
            real_mu_s: Real static friction.
            sim_mu_s: Simulated static friction.
            real_mu_d: Real dynamic friction (optional).
            sim_mu_d: Simulated dynamic friction (optional).

        Returns:
            True if within tolerance.
        """
        static_converged = abs(real_mu_s - sim_mu_s) < self.cfg.mu_s_tolerance

        if real_mu_d is not None and sim_mu_d is not None:
            dynamic_converged = abs(real_mu_d - sim_mu_d) < self.cfg.mu_d_tolerance
            return static_converged and dynamic_converged

        return static_converged

    def analyze_matching(
        self,
        real_trials: list[dict],
        sim_results: list[dict],
    ) -> dict:
        """Analyze slip matching quality.

        Args:
            real_trials: List of real robot trials.
            sim_results: List of simulation results.

        Returns:
            Dictionary of analysis metrics.
        """
        mu_s_errors = []
        mu_d_errors = []
        q_slip_errors = []

        for real, sim in zip(real_trials, sim_results):
            if not sim.get("slip_detected", np.array([False])).any():
                continue

            slip_mask = sim["slip_detected"]

            # Static friction error
            sim_mu_s = np.mean(sim["static_friction"][slip_mask])
            mu_s_errors.append(abs(real["static_friction"] - sim_mu_s))

            # Dynamic friction error (if available)
            if "dynamic_friction" in real and sim["dynamic_friction_valid"].any():
                sim_mu_d = np.mean(sim["dynamic_friction"][sim["dynamic_friction_valid"]])
                mu_d_errors.append(abs(real["dynamic_friction"] - sim_mu_d))

            # Grip position error
            if "q_slip" in real:
                sim_q_slip = np.mean(sim["grip_position_at_slip"][slip_mask])
                q_slip_errors.append(abs(real["q_slip"] - sim_q_slip))

        result = {
            "num_comparisons": len(mu_s_errors),
        }

        if mu_s_errors:
            result["mu_s_mean_error"] = float(np.mean(mu_s_errors))
            result["mu_s_std_error"] = float(np.std(mu_s_errors))
            result["mu_s_within_tolerance"] = int(np.sum(np.array(mu_s_errors) < self.cfg.mu_s_tolerance))

        if mu_d_errors:
            result["mu_d_mean_error"] = float(np.mean(mu_d_errors))
            result["mu_d_std_error"] = float(np.std(mu_d_errors))
            result["mu_d_within_tolerance"] = int(np.sum(np.array(mu_d_errors) < self.cfg.mu_d_tolerance))

        if q_slip_errors:
            result["q_slip_mean_error"] = float(np.mean(q_slip_errors))
            result["q_slip_std_error"] = float(np.std(q_slip_errors))
            result["q_slip_within_tolerance"] = int(np.sum(np.array(q_slip_errors) < self.cfg.q_slip_tolerance))

        return result


# Legacy compatibility alias
SlipCurrentMatcher = SlipMatcher
