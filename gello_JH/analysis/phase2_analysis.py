# Phase 2 Analysis: Friction & Contact Calibration Results

import numpy as np
import yaml
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt


class Phase2Analyzer:
    """Analyzer for Phase 2 (Friction & Contact) calibration results."""

    def __init__(self, result_path: Optional[str] = None):
        """Initialize analyzer.

        Args:
            result_path: Path to Phase 2 calibration result YAML file
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

    def get_friction_params(self) -> dict:
        """Get calibrated friction parameters.

        Returns:
            Dictionary with static_friction, dynamic_friction, contact params
        """
        if self.results is None:
            raise ValueError("No results loaded")

        params = self.results.get("parameters", {})
        return {
            "static_friction": params.get("static_friction", 0.0),
            "dynamic_friction": params.get("dynamic_friction", 0.0),
            "contact_stiffness": params.get("contact_stiffness", 0.0),
            "contact_damping": params.get("contact_damping", 0.0),
            "contact_offset": params.get("contact_offset", 0.0),
        }

    def get_trial_history(self) -> List[dict]:
        """Get optimization trial history.

        Returns:
            List of trial dictionaries with params and loss
        """
        if self.results is None:
            raise ValueError("No results loaded")

        return self.results.get("trial_history", [])

    def get_i_slip_comparison(self) -> dict:
        """Get I_slip comparison between real and simulated.

        Returns:
            Dictionary with real_i_slip, sim_i_slip, error
        """
        if self.results is None:
            raise ValueError("No results loaded")

        validation = self.results.get("validation", {})
        return {
            "real_i_slip": validation.get("real_i_slip", 0.0),
            "sim_i_slip": validation.get("sim_i_slip", 0.0),
            "error": validation.get("i_slip_error", 0.0),
        }

    def plot_optimization_progress(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot Bayesian optimization progress.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        trials = self.get_trial_history()

        if len(trials) == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No trial history available", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Extract data
        trial_nums = list(range(len(trials)))
        losses = [t.get("loss", float("inf")) for t in trials]
        static_frictions = [t.get("params", {}).get("static_friction", 0) for t in trials]
        dynamic_frictions = [t.get("params", {}).get("dynamic_friction", 0) for t in trials]

        # Loss over trials
        ax = axes[0, 0]
        ax.plot(trial_nums, losses, "b-", alpha=0.5)
        ax.scatter(trial_nums, losses, c="blue", s=20, alpha=0.7)

        # Best so far
        best_so_far = np.minimum.accumulate(losses)
        ax.plot(trial_nums, best_so_far, "r-", linewidth=2, label="Best so far")

        ax.set_xlabel("Trial")
        ax.set_ylabel("Loss (|I_slip_sim - I_slip_real|)")
        ax.set_title("Optimization Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Static friction exploration
        ax = axes[0, 1]
        scatter = ax.scatter(trial_nums, static_frictions, c=losses, cmap="viridis_r", s=30)
        plt.colorbar(scatter, ax=ax, label="Loss")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Static Friction (μ_s)")
        ax.set_title("Static Friction Exploration")
        ax.grid(True, alpha=0.3)

        # Dynamic friction exploration
        ax = axes[1, 0]
        scatter = ax.scatter(trial_nums, dynamic_frictions, c=losses, cmap="viridis_r", s=30)
        plt.colorbar(scatter, ax=ax, label="Loss")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Dynamic Friction (μ_d)")
        ax.set_title("Dynamic Friction Exploration")
        ax.grid(True, alpha=0.3)

        # Friction parameter space
        ax = axes[1, 1]
        scatter = ax.scatter(
            static_frictions, dynamic_frictions, c=losses, cmap="viridis_r", s=30
        )
        plt.colorbar(scatter, ax=ax, label="Loss")

        # Mark best point
        best_idx = np.argmin(losses)
        ax.scatter(
            static_frictions[best_idx],
            dynamic_frictions[best_idx],
            c="red",
            s=200,
            marker="*",
            edgecolors="white",
            linewidths=2,
            label="Best",
            zorder=5,
        )

        ax.set_xlabel("Static Friction (μ_s)")
        ax.set_ylabel("Dynamic Friction (μ_d)")
        ax.set_title("Friction Parameter Space")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_i_slip_comparison(
        self,
        real_i_slip: float,
        sim_i_slip: float,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot I_slip comparison between real and simulated.

        Args:
            real_i_slip: Real robot slip current
            sim_i_slip: Simulated slip current
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        categories = ["Real Robot", "Simulation"]
        values = [real_i_slip, sim_i_slip]
        colors = ["steelblue", "coral"]

        bars = ax.bar(categories, values, color=colors, width=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.4f} A",
                ha="center",
                va="bottom",
                fontsize=12,
            )

        # Error annotation
        error = abs(sim_i_slip - real_i_slip)
        ax.annotate(
            f"Error: {error:.4f} A\n({error/real_i_slip*100:.1f}%)",
            xy=(0.5, max(values) * 0.8),
            ha="center",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax.set_ylabel("Slip Current I_slip (A)")
        ax.set_title("Phase 2: Slip Current Comparison")
        ax.set_ylim(0, max(values) * 1.3)
        ax.grid(True, alpha=0.3, axis="y")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_slip_profile(
        self,
        time: np.ndarray,
        current: np.ndarray,
        object_velocity: np.ndarray,
        i_slip: float,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot slip test profile showing current ramp and slip detection.

        Args:
            time: Time array
            current: Current array during ramp
            object_velocity: Object velocity array
            i_slip: Detected slip current
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Current profile
        ax = axes[0]
        ax.plot(time, current, "b-", linewidth=2, label="Motor Current")
        ax.axhline(y=i_slip, color="r", linestyle="--", linewidth=2, label=f"I_slip = {i_slip:.4f} A")

        # Find slip time
        slip_idx = np.where(current <= i_slip)[0]
        if len(slip_idx) > 0:
            slip_time = time[slip_idx[0]]
            ax.axvline(x=slip_time, color="g", linestyle=":", alpha=0.7, label=f"Slip @ t={slip_time:.2f}s")

        ax.set_ylabel("Current (A)")
        ax.set_title("Phase 2: Slip Test Profile")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Object velocity
        ax = axes[1]
        velocity_magnitude = np.linalg.norm(object_velocity, axis=-1) if object_velocity.ndim > 1 else np.abs(object_velocity)
        ax.plot(time, velocity_magnitude, "g-", linewidth=2, label="Object Velocity")
        ax.axhline(y=0.001, color="r", linestyle="--", alpha=0.5, label="Slip Threshold")

        if len(slip_idx) > 0:
            ax.axvline(x=slip_time, color="g", linestyle=":", alpha=0.7)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.legend(loc="upper right")
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

        params = self.get_friction_params()
        trials = self.get_trial_history()
        i_slip = self.get_i_slip_comparison()

        lines = [
            "=" * 50,
            "Phase 2: Friction & Contact Calibration Summary",
            "=" * 50,
            f"Number of optimization trials: {len(trials)}",
            "",
            "Calibrated Parameters:",
            f"  Static Friction (μ_s): {params['static_friction']:.4f}",
            f"  Dynamic Friction (μ_d): {params['dynamic_friction']:.4f}",
            f"  Contact Stiffness: {params['contact_stiffness']:.2e}",
            f"  Contact Damping: {params['contact_damping']:.2e}",
            f"  Contact Offset: {params['contact_offset']:.6f}",
            "",
            "Slip Current Comparison:",
            f"  Real I_slip: {i_slip['real_i_slip']:.4f} A",
            f"  Sim I_slip: {i_slip['sim_i_slip']:.4f} A",
            f"  Error: {i_slip['error']:.4f} A",
            "=" * 50,
        ]

        return "\n".join(lines)
