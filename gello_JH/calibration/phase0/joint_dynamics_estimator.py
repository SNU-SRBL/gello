# Copyright (c) 2025, SRBL
# Phase 0: Joint Dynamics Estimator (SIMPLER style with SA)

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Any
from dataclasses import dataclass, field

from .simulated_annealing import SimulatedAnnealingOptimizer, SAConfig
from .robot_config import RobotType, get_robot_config, UR5_CONFIG, HAND_CONFIG

if TYPE_CHECKING:
    from ...envs.phase0_sysid import SysIDEnv
    from ...data.storage import TrajectoryData


@dataclass
class Phase0Config:
    """Configuration for Phase 0 calibration."""

    # Robot type
    robot_type: RobotType = RobotType.HAND

    # Simulated Annealing settings (SIMPLER style)
    num_rounds: int = 3
    trials_per_round: int = 100
    temp_initial: float = 1.0
    temp_final: float = 0.01
    shrink_factor: float = 0.5

    # Parameter bounds (will be overridden by robot config)
    stiffness_bounds: tuple[float, float] | None = None
    damping_bounds: tuple[float, float] | None = None
    friction_bounds: tuple[float, float] | None = None

    # Loss weights
    joint_weight: float = 0.1
    """Weight for joint position loss in SIMPLER loss."""

    # Use SIMPLER EE tracking loss
    use_simpler_loss: bool = True

    # Random seed
    seed: int | None = None

    def __post_init__(self):
        """Set bounds from robot config if not specified."""
        robot_cfg = get_robot_config(self.robot_type)

        if self.stiffness_bounds is None:
            self.stiffness_bounds = robot_cfg.stiffness_bounds
        if self.damping_bounds is None:
            self.damping_bounds = robot_cfg.damping_bounds
        if self.friction_bounds is None:
            self.friction_bounds = (0.0, 1.0)


class JointDynamicsEstimator:
    """Estimate joint dynamics parameters using Simulated Annealing.

    Implements SIMPLER paper style optimization:
    - 3 rounds of simulated annealing with shrinking bounds
    - EE pose tracking loss (translation + rotation + joint)
    - Supports both UR5 and Hand calibration

    For Hand: Computes per-finger losses, then sums them.
    """

    def __init__(
        self,
        env: SysIDEnv,
        cfg: Phase0Config | None = None,
    ):
        """Initialize estimator.

        Args:
            env: System identification environment.
            cfg: Configuration.
        """
        self.env = env
        self.cfg = cfg or Phase0Config()
        self.device = env.device

        # Get robot config
        self.robot_config = get_robot_config(self.cfg.robot_type)
        self.num_joints = self.robot_config.num_joints

        # Initialize parameters
        self._init_parameters()

        # Setup SA optimizer
        self._setup_optimizer()

        # History
        self.loss_history: list[float] = []
        self.loss_components_history: list[dict] = []

    def _init_parameters(self):
        """Initialize calibration parameters."""
        self.params = {
            "stiffness": np.ones(self.num_joints) * self.robot_config.initial_stiffness,
            "damping": np.ones(self.num_joints) * self.robot_config.initial_damping,
        }

        self.param_bounds = {
            "stiffness": (
                np.ones(self.num_joints) * self.cfg.stiffness_bounds[0],
                np.ones(self.num_joints) * self.cfg.stiffness_bounds[1],
            ),
            "damping": (
                np.ones(self.num_joints) * self.cfg.damping_bounds[0],
                np.ones(self.num_joints) * self.cfg.damping_bounds[1],
            ),
        }

    def _setup_optimizer(self):
        """Setup Simulated Annealing optimizer."""
        sa_cfg = SAConfig(
            num_rounds=self.cfg.num_rounds,
            trials_per_round=self.cfg.trials_per_round,
            temp_initial=self.cfg.temp_initial,
            temp_final=self.cfg.temp_final,
            shrink_factor=self.cfg.shrink_factor,
            seed=self.cfg.seed,
        )

        self.optimizer = SimulatedAnnealingOptimizer(
            param_names=list(self.params.keys()),
            param_bounds=self.param_bounds,
            cfg=sa_cfg,
        )

    def calibrate(
        self,
        trajectories: list[str | TrajectoryData],
        callback: Callable[[int, int, float, dict], None] | None = None,
    ) -> dict:
        """Run SIMPLER-style calibration optimization.

        Args:
            trajectories: List of trajectory filenames or data objects.
            callback: Optional callback(round_idx, trial_idx, loss, params).

        Returns:
            Dictionary with calibration results.
        """
        # Store trajectories for loss function
        self._trajectories = trajectories

        print(f"[Phase0] Starting calibration for {self.cfg.robot_type.value}")
        print(f"[Phase0] Number of trajectories: {len(trajectories)}")
        print(f"[Phase0] Using {'SIMPLER' if self.cfg.use_simpler_loss else 'joint-only'} loss")

        # Run SA optimization
        best_params = self.optimizer.optimize(
            loss_fn=self._compute_loss,
            initial_params=self.params,
            callback=callback,
        )

        # Store history
        self.loss_history = self.optimizer.loss_history

        print(f"[Phase0] Optimization complete. Best loss: {self.optimizer.best_loss:.6f}")

        return self._create_result(best_params)

    def _compute_loss(self, params: dict[str, np.ndarray]) -> float:
        """Compute loss for given parameters.

        Args:
            params: Dictionary of parameter arrays.

        Returns:
            Total loss value.
        """
        # Apply parameters to simulation
        self._apply_parameters(params)

        total_loss = 0.0
        all_components = {}
        num_valid = 0

        for traj in self._trajectories:
            # Load trajectory
            if isinstance(traj, str):
                self.env.load_trajectory(traj)
            else:
                self.env.load_trajectory_data(traj)

            # Run simulation
            self.env.reset()
            while not self.env._trajectory_replayer.is_done:
                self.env.step()

            # Compute loss
            if self.cfg.use_simpler_loss:
                loss, components = self.env.compute_simpler_loss()
            else:
                loss = self.env.compute_sysid_loss().item()
                components = {"joint_loss": loss}

            if not np.isnan(loss) and not np.isinf(loss):
                total_loss += loss
                for k, v in components.items():
                    all_components[k] = all_components.get(k, 0.0) + v
                num_valid += 1

        if num_valid > 0:
            avg_loss = total_loss / num_valid
            avg_components = {k: v / num_valid for k, v in all_components.items()}
            self.loss_components_history.append(avg_components)
            return avg_loss
        else:
            return float("inf")

    def _apply_parameters(self, params: dict[str, np.ndarray]):
        """Apply parameters to simulation environment."""
        stiffness = torch.from_numpy(params["stiffness"]).float().to(self.device)
        damping = torch.from_numpy(params["damping"]).float().to(self.device)

        self.env.set_joint_dynamics(
            stiffness=stiffness,
            damping=damping,
        )

    def _create_result(self, best_params: dict[str, np.ndarray]) -> dict:
        """Create calibration result."""
        from ...data.storage import Phase0Result

        return Phase0Result(
            phase=0,
            parameters={
                "joint_stiffness": best_params["stiffness"],
                "joint_damping": best_params["damping"],
            },
            loss_history=self.loss_history,
            validation_metrics={
                "final_loss": self.optimizer.best_loss,
                "num_evaluations": len(self.loss_history),
            },
            metadata={
                "robot_type": self.cfg.robot_type.value,
                "num_joints": self.num_joints,
                "optimizer": "simulated_annealing",
                "config": {
                    "num_rounds": self.cfg.num_rounds,
                    "trials_per_round": self.cfg.trials_per_round,
                    "shrink_factor": self.cfg.shrink_factor,
                    "use_simpler_loss": self.cfg.use_simpler_loss,
                    "joint_weight": self.cfg.joint_weight,
                },
                "loss_components": self.loss_components_history[-1] if self.loss_components_history else {},
            },
        )

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get summary of optimization."""
        return self.optimizer.get_optimization_summary()
