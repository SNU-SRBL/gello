# Copyright (c) 2025, SRBL
# Phase 0: Trajectory Replay Calibrator

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...envs.phase0_sysid import SysIDEnv
    from ...data.storage import TrajectoryData


class TrajectoryReplayCalibrator:
    """Calibrator using trajectory replay for system identification.

    Alternative to gradient-based optimization using grid search
    or Bayesian optimization over parameter space.
    """

    def __init__(
        self,
        env: SysIDEnv,
        param_ranges: dict[str, tuple[float, float, int]] | None = None,
    ):
        """Initialize calibrator.

        Args:
            env: System identification environment.
            param_ranges: Parameter ranges as {name: (min, max, num_points)}.
        """
        self.env = env
        self.device = env.device

        self.param_ranges = param_ranges or {
            "stiffness": (10.0, 500.0, 10),
            "damping": (1.0, 50.0, 10),
            "friction": (0.01, 0.5, 5),
        }

        self.results: list[dict] = []

    def grid_search(
        self,
        trajectory: str | TrajectoryData,
        joint_idx: int | None = None,
    ) -> dict:
        """Run grid search over parameter space.

        Args:
            trajectory: Trajectory to replay.
            joint_idx: Specific joint to calibrate (None for all).

        Returns:
            Best parameters found.
        """
        # Load trajectory
        if isinstance(trajectory, str):
            self.env.load_trajectory(trajectory)
        else:
            self.env.load_trajectory_data(trajectory)

        # Generate parameter grid
        stiffness_vals = np.linspace(*self.param_ranges["stiffness"])
        damping_vals = np.linspace(*self.param_ranges["damping"])
        friction_vals = np.linspace(*self.param_ranges["friction"])

        best_loss = float("inf")
        best_params = {}

        total = len(stiffness_vals) * len(damping_vals) * len(friction_vals)
        count = 0

        for stiffness in stiffness_vals:
            for damping in damping_vals:
                for friction in friction_vals:
                    count += 1

                    # Set parameters
                    params = self._create_param_tensors(
                        stiffness, damping, friction, joint_idx
                    )
                    self.env.set_joint_dynamics(**params)

                    # Run simulation
                    self.env.reset()
                    while not self.env._trajectory_replayer.is_done:
                        self.env.step()

                    # Get loss
                    loss = self.env.compute_sysid_loss().item()

                    # Store result
                    self.results.append({
                        "stiffness": stiffness,
                        "damping": damping,
                        "friction": friction,
                        "loss": loss,
                    })

                    # Update best
                    if loss < best_loss:
                        best_loss = loss
                        best_params = {
                            "stiffness": stiffness,
                            "damping": damping,
                            "friction": friction,
                            "loss": loss,
                        }

                    if count % 10 == 0:
                        print(f"Progress: {count}/{total}, Best loss: {best_loss:.6f}")

        return best_params

    def _create_param_tensors(
        self,
        stiffness: float,
        damping: float,
        friction: float,
        joint_idx: int | None,
    ) -> dict:
        """Create parameter tensors."""
        num_joints = 20

        if joint_idx is not None:
            # Single joint
            stiffness_t = torch.ones(num_joints, device=self.device) * self.env.cfg.calibration_params.initial_joint_stiffness
            damping_t = torch.ones(num_joints, device=self.device) * self.env.cfg.calibration_params.initial_joint_damping
            friction_t = torch.ones(num_joints, device=self.device) * self.env.cfg.calibration_params.initial_joint_friction

            stiffness_t[joint_idx] = stiffness
            damping_t[joint_idx] = damping
            friction_t[joint_idx] = friction
        else:
            # All joints
            stiffness_t = torch.ones(num_joints, device=self.device) * stiffness
            damping_t = torch.ones(num_joints, device=self.device) * damping
            friction_t = torch.ones(num_joints, device=self.device) * friction

        return {
            "stiffness": stiffness_t,
            "damping": damping_t,
            "friction": friction_t,
        }

    def get_results_as_array(self) -> np.ndarray:
        """Get results as numpy array for analysis."""
        if not self.results:
            return np.array([])

        return np.array([
            [r["stiffness"], r["damping"], r["friction"], r["loss"]]
            for r in self.results
        ])
