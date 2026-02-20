# Copyright (c) 2025, SRBL
# Phase 2: Slip Current Matcher

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ...envs.phase2_friction import FrictionEnv
    from ...data.storage import SlipTrialData


@dataclass
class SlipMatchingConfig:
    """Configuration for slip current matching."""

    # Grip joints to compare
    grip_joint_indices: list[int] = None  # If None, use all joints

    # Matching tolerance
    I_slip_tolerance: float = 0.05  # Amperes

    # Weighting
    use_joint_weights: bool = False
    joint_weights: np.ndarray | None = None


class SlipCurrentMatcher:
    """Match slip currents between real and simulated experiments.

    Computes the loss function for friction optimization based on
    I_slip matching.
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

    def compute_loss(
        self,
        real_trial: SlipTrialData,
        sim_I_slip: np.ndarray | float,
    ) -> float:
        """Compute I_slip matching loss.

        Args:
            real_trial: Real robot slip trial data.
            sim_I_slip: Simulated slip current (scalar or per-joint array).

        Returns:
            Loss value.
        """
        real_I_slip = real_trial.I_slip

        if real_I_slip is None:
            return 1e6  # No slip in real data

        # Handle different formats
        if isinstance(sim_I_slip, (int, float)):
            sim_I_slip = np.array([sim_I_slip])
        if isinstance(real_I_slip, (int, float)):
            real_I_slip = np.array([real_I_slip])

        # Extract grip joints
        indices = self.cfg.grip_joint_indices

        if len(sim_I_slip) > 1 and len(indices) <= len(sim_I_slip):
            sim_grip = sim_I_slip[indices] if len(sim_I_slip) > len(indices) else sim_I_slip
        else:
            sim_grip = sim_I_slip

        if len(real_I_slip) > 1 and len(indices) <= len(real_I_slip):
            real_grip = real_I_slip[indices] if len(real_I_slip) > len(indices) else real_I_slip
        else:
            real_grip = real_I_slip

        # Compute loss
        if self.cfg.use_joint_weights and self.cfg.joint_weights is not None:
            weights = self.cfg.joint_weights[:len(sim_grip)]
            loss = np.sum(weights * (sim_grip - real_grip) ** 2)
        else:
            loss = np.mean((sim_grip - real_grip) ** 2)

        return float(loss)

    def compute_batch_loss(
        self,
        real_trials: list[SlipTrialData],
        sim_results: list[dict],
    ) -> float:
        """Compute average loss over multiple trials.

        Args:
            real_trials: List of real robot trials.
            sim_results: List of simulation results.

        Returns:
            Average loss.
        """
        if len(real_trials) != len(sim_results):
            raise ValueError("Number of real and sim trials must match")

        total_loss = 0.0
        valid_count = 0

        for real, sim in zip(real_trials, sim_results):
            if not real.slip_detected:
                continue

            sim_I_slip = sim.get("I_slip", sim.get("sim_I_slip"))
            if sim_I_slip is None:
                total_loss += 1e6
            else:
                total_loss += self.compute_loss(real, sim_I_slip)

            valid_count += 1

        return total_loss / max(valid_count, 1)

    def check_convergence(
        self,
        real_I_slip: float | np.ndarray,
        sim_I_slip: float | np.ndarray,
    ) -> bool:
        """Check if calibration has converged.

        Args:
            real_I_slip: Real slip current.
            sim_I_slip: Simulated slip current.

        Returns:
            True if within tolerance.
        """
        if isinstance(real_I_slip, np.ndarray):
            real_I_slip = np.mean(real_I_slip[self.cfg.grip_joint_indices])
        if isinstance(sim_I_slip, np.ndarray):
            sim_I_slip = np.mean(sim_I_slip[self.cfg.grip_joint_indices])

        return abs(real_I_slip - sim_I_slip) < self.cfg.I_slip_tolerance

    def analyze_slip_matching(
        self,
        real_trials: list[SlipTrialData],
        sim_results: list[dict],
    ) -> dict:
        """Analyze slip matching quality.

        Returns:
            Dictionary of analysis metrics.
        """
        errors = []
        for real, sim in zip(real_trials, sim_results):
            if not real.slip_detected:
                continue

            sim_I_slip = sim.get("I_slip", sim.get("sim_I_slip"))
            if sim_I_slip is None:
                continue

            real_I_slip = real.I_slip
            if isinstance(real_I_slip, np.ndarray):
                real_I_slip = np.mean(real_I_slip)
            if isinstance(sim_I_slip, np.ndarray):
                sim_I_slip = np.mean(sim_I_slip)

            errors.append(abs(real_I_slip - sim_I_slip))

        if not errors:
            return {"error": "No valid comparisons"}

        errors = np.array(errors)

        return {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "max_error": float(np.max(errors)),
            "min_error": float(np.min(errors)),
            "num_comparisons": len(errors),
            "within_tolerance": int(np.sum(errors < self.cfg.I_slip_tolerance)),
        }
