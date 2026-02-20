# Phase 1 Analysis: Current-Torque Calibration Results

import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from scipy import stats


class Phase1Analyzer:
    """Analyzer for Phase 1 (Current-Torque) calibration results."""

    def __init__(self, result_path: Optional[str] = None):
        """Initialize analyzer.

        Args:
            result_path: Path to Phase 1 calibration result YAML file
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

    def get_calibration_params(self, joint_name: str) -> Tuple[float, float, float]:
        """Get calibration parameters for a specific joint.

        Args:
            joint_name: Name of the joint

        Returns:
            Tuple of (k_gain, k_offset, r_squared)
        """
        if self.results is None:
            raise ValueError("No results loaded")

        params = self.results.get("parameters", {}).get("per_joint", {})
        joint_params = params.get(joint_name, {})

        return (
            joint_params.get("k_gain", 0.0),
            joint_params.get("k_offset", 0.0),
            joint_params.get("r_squared", 0.0),
        )

    def get_all_joints(self) -> list:
        """Get list of all calibrated joints.

        Returns:
            List of joint names
        """
        if self.results is None:
            raise ValueError("No results loaded")

        return list(self.results.get("parameters", {}).get("per_joint", {}).keys())

    def validate_calibration(
        self, current: np.ndarray, torque: np.ndarray, joint_name: str
    ) -> dict:
        """Validate calibration against new data.

        Args:
            current: Measured motor current array
            torque: Measured joint torque array
            joint_name: Name of the joint

        Returns:
            Validation metrics dictionary
        """
        k_gain, k_offset, _ = self.get_calibration_params(joint_name)

        # Predict torque from current
        predicted_torque = k_gain * current + k_offset

        # Compute errors
        error = torque - predicted_torque
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))

        # Compute R²
        ss_res = np.sum(error**2)
        ss_tot = np.sum((torque - np.mean(torque)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "rmse": float(rmse),
            "mae": float(mae),
            "r_squared": float(r_squared),
            "max_error": float(np.max(np.abs(error))),
        }

    def plot_current_torque_relationship(
        self,
        current_data: dict,
        torque_data: dict,
        joint_name: str,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot current vs torque relationship with calibration line.

        Args:
            current_data: Dict with 'real' and 'sim' current arrays
            torque_data: Dict with 'real' and 'sim' torque arrays
            joint_name: Name of the joint
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        k_gain, k_offset, r_squared = self.get_calibration_params(joint_name)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot data points
        if "real" in current_data:
            ax.scatter(
                current_data["real"],
                torque_data.get("real", []),
                alpha=0.6,
                label="Real Data",
                c="blue",
            )

        if "sim" in current_data:
            ax.scatter(
                current_data["sim"],
                torque_data.get("sim", []),
                alpha=0.6,
                label="Sim Data",
                c="orange",
            )

        # Plot calibration line
        all_currents = np.concatenate(
            [current_data.get("real", []), current_data.get("sim", [])]
        )
        if len(all_currents) > 0:
            current_range = np.linspace(
                np.min(all_currents), np.max(all_currents), 100
            )
            torque_pred = k_gain * current_range + k_offset
            ax.plot(
                current_range,
                torque_pred,
                "r-",
                linewidth=2,
                label=f"Calibration: τ = {k_gain:.4f}×I + {k_offset:.4f}",
            )

        ax.set_xlabel("Motor Current (A)")
        ax.set_ylabel("Joint Torque (N·m)")
        ax.set_title(f"Phase 1: Current-Torque Calibration - {joint_name}\nR² = {r_squared:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_all_joints(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot calibration parameters for all joints.

        Args:
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        joints = self.get_all_joints()
        n_joints = len(joints)

        if n_joints == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No joint data available", ha="center", va="center")
            return fig

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        k_gains = []
        k_offsets = []
        r_squareds = []

        for joint in joints:
            k_gain, k_offset, r_squared = self.get_calibration_params(joint)
            k_gains.append(k_gain)
            k_offsets.append(k_offset)
            r_squareds.append(r_squared)

        x = np.arange(n_joints)

        # k_gain plot
        axes[0].bar(x, k_gains, color="steelblue")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(joints, rotation=45, ha="right")
        axes[0].set_ylabel("k_gain (N·m/A)")
        axes[0].set_title("Torque Gain per Joint")
        axes[0].grid(True, alpha=0.3)

        # k_offset plot
        axes[1].bar(x, k_offsets, color="coral")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(joints, rotation=45, ha="right")
        axes[1].set_ylabel("k_offset (N·m)")
        axes[1].set_title("Torque Offset per Joint")
        axes[1].grid(True, alpha=0.3)

        # R² plot
        axes[2].bar(x, r_squareds, color="green")
        axes[2].axhline(y=0.9, color="r", linestyle="--", label="Threshold (0.9)")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(joints, rotation=45, ha="right")
        axes[2].set_ylabel("R²")
        axes[2].set_title("Calibration Quality (R²)")
        axes[2].set_ylim(0, 1.1)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

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

        joints = self.get_all_joints()

        lines = [
            "=" * 50,
            "Phase 1: Current-Torque Calibration Summary",
            "=" * 50,
            f"Number of joints calibrated: {len(joints)}",
            "",
            "Per-Joint Parameters (τ = k_gain × I + k_offset):",
        ]

        for joint in joints:
            k_gain, k_offset, r_squared = self.get_calibration_params(joint)
            lines.append(
                f"  {joint}: k_gain={k_gain:.4f}, k_offset={k_offset:.4f}, R²={r_squared:.4f}"
            )

        # Check quality
        all_r2 = [self.get_calibration_params(j)[2] for j in joints]
        passed = sum(r2 >= 0.9 for r2 in all_r2)
        lines.extend(
            [
                "",
                f"Quality Check: {passed}/{len(joints)} joints passed R² >= 0.9",
                "=" * 50,
            ]
        )

        return "\n".join(lines)
