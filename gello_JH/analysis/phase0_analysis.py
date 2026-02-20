# Phase 0 Analysis: System Identification Results

import numpy as np
import yaml
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


class Phase0Analyzer:
    """Analyzer for Phase 0 (System Identification) calibration results."""

    def __init__(self, result_path: Optional[str] = None):
        """Initialize analyzer.

        Args:
            result_path: Path to Phase 0 calibration result YAML file
        """
        self.result_path = result_path
        self.results = None
        if result_path:
            self.load_results(result_path)

    def load_results(self, result_path: str) -> dict:
        """Load calibration results from YAML file.

        Args:
            result_path: Path to result file

        Returns:
            Loaded results dictionary
        """
        with open(result_path, "r") as f:
            self.results = yaml.safe_load(f)
        return self.results

    def get_joint_parameters(self) -> dict:
        """Get calibrated joint parameters.

        Returns:
            Dictionary with stiffness, damping, friction per joint
        """
        if self.results is None:
            raise ValueError("No results loaded")

        params = self.results.get("parameters", {})
        return {
            "joint_stiffness": params.get("joint_stiffness", []),
            "joint_damping": params.get("joint_damping", []),
            "joint_friction": params.get("joint_friction", []),
        }

    def get_loss_history(self) -> np.ndarray:
        """Get optimization loss history.

        Returns:
            Array of loss values per epoch
        """
        if self.results is None:
            raise ValueError("No results loaded")

        return np.array(self.results.get("loss_history", []))

    def compute_metrics(self, real_trajectory: dict, sim_trajectory: dict) -> dict:
        """Compute validation metrics between real and simulated trajectories.

        Args:
            real_trajectory: Real robot trajectory data
            sim_trajectory: Simulated trajectory data

        Returns:
            Dictionary with RMSE, MAE, correlation metrics
        """
        real_pos = np.array(real_trajectory["joint_positions"])
        sim_pos = np.array(sim_trajectory["joint_positions"])

        # Position metrics
        pos_error = real_pos - sim_pos
        pos_rmse = np.sqrt(np.mean(pos_error**2, axis=0))
        pos_mae = np.mean(np.abs(pos_error), axis=0)

        # Velocity metrics (if available)
        vel_rmse = None
        vel_mae = None
        if "joint_velocities" in real_trajectory and "joint_velocities" in sim_trajectory:
            real_vel = np.array(real_trajectory["joint_velocities"])
            sim_vel = np.array(sim_trajectory["joint_velocities"])
            vel_error = real_vel - sim_vel
            vel_rmse = np.sqrt(np.mean(vel_error**2, axis=0))
            vel_mae = np.mean(np.abs(vel_error), axis=0)

        return {
            "position_rmse": pos_rmse,
            "position_mae": pos_mae,
            "position_rmse_mean": float(np.mean(pos_rmse)),
            "velocity_rmse": vel_rmse,
            "velocity_mae": vel_mae,
            "velocity_rmse_mean": float(np.mean(vel_rmse)) if vel_rmse is not None else None,
        }

    def plot_convergence(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot optimization convergence curve.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        loss_history = self.get_loss_history()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(loss_history, "b-", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Phase 0: System Identification Convergence")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_parameter_comparison(
        self, initial_params: dict, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot initial vs calibrated parameters.

        Args:
            initial_params: Initial parameter values
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        calibrated = self.get_joint_parameters()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        param_names = ["joint_stiffness", "joint_damping", "joint_friction"]
        titles = ["Joint Stiffness", "Joint Damping", "Joint Friction"]

        for ax, param_name, title in zip(axes, param_names, titles):
            init_vals = np.atleast_1d(initial_params.get(param_name, []))
            cal_vals = np.atleast_1d(calibrated.get(param_name, []))

            if len(init_vals) > 0 and len(cal_vals) > 0:
                x = np.arange(len(cal_vals))
                width = 0.35

                ax.bar(x - width / 2, init_vals[: len(cal_vals)], width, label="Initial")
                ax.bar(x + width / 2, cal_vals, width, label="Calibrated")

                ax.set_xlabel("Joint Index")
                ax.set_ylabel("Value")
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def summary(self) -> str:
        """Generate text summary of calibration results.

        Returns:
            Summary string
        """
        if self.results is None:
            return "No results loaded"

        params = self.get_joint_parameters()
        loss_history = self.get_loss_history()

        lines = [
            "=" * 50,
            "Phase 0: System Identification Summary",
            "=" * 50,
            f"Final Loss: {loss_history[-1]:.6f}" if len(loss_history) > 0 else "No loss history",
            f"Epochs: {len(loss_history)}",
            "",
            "Calibrated Parameters:",
            f"  Joint Stiffness: {params['joint_stiffness']}",
            f"  Joint Damping: {params['joint_damping']}",
            f"  Joint Friction: {params['joint_friction']}",
            "=" * 50,
        ]

        return "\n".join(lines)
