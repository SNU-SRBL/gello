# Copyright (c) 2025, SRBL
# Phase 1: Paired Sim-Real Current-Torque Calibration Model

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression

if TYPE_CHECKING:
    from ...envs.phase1_current_torque import CurrentTorqueEnv


@dataclass
class Phase1Config:
    """Configuration for Phase 1 calibration."""

    # Minimum data points for regression
    min_data_points: int = 10

    # R-squared threshold for valid calibration
    r_squared_threshold: float = 0.9

    # Force threshold for valid data (N)
    min_force_threshold: float = 0.5

    # Outlier rejection
    outlier_std_threshold: float = 3.0

    # Velocity threshold to classify as static (rad/s)
    static_velocity_threshold: float = 0.01


@dataclass
class JointCalibrationResult:
    """Calibration result for a single joint."""

    joint_idx: int
    joint_name: str
    k_t: float  # Torque constant [Nm/A]
    offset: float  # Regression intercept (absorbs gravity/friction diff)
    r_squared: float
    num_data_points: int
    success: bool

    # Validation metrics
    jacobian_consistency: float = 0.0  # J^T*F vs tau_sim match
    k_t_std: float = 0.0  # Std of k_t across configs

    error_msg: str = ""


@dataclass
class FingerCalibrationResult:
    """Calibration result for a finger (4 joints)."""

    finger_name: str
    joints: list[JointCalibrationResult] = field(default_factory=list)

    @property
    def mean_r_squared(self) -> float:
        if not self.joints:
            return 0.0
        successful = [j.r_squared for j in self.joints if j.success]
        return float(np.mean(successful)) if successful else 0.0

    @property
    def all_success(self) -> bool:
        return all(j.success for j in self.joints)

    @property
    def k_t_values(self) -> list[float]:
        return [j.k_t for j in self.joints]

    @property
    def offset_values(self) -> list[float]:
        return [j.offset for j in self.joints]


class JacobianCurrentTorqueModel:
    """Paired Sim-Real current-to-torque calibration model.

    Calibration Method:
    Given matched data from real robot and simulation:
        - Real: (q, qdot, I_real, F_ext) at various conditions
        - Sim: τ_sim at same (q, F_ext) conditions

    Per-joint linear regression:
        τ_sim_j = k_t_j × I_real_j + b_j

    Where:
        - k_t_j: torque constant [Nm/A]
        - b_j: offset absorbing gravity/friction differences between sim and real

    Jacobian validation (independent):
        J^T(q) × F_ext ≈ τ_sim - g(q) - friction

    Attributes:
        num_joints: Number of hand joints (default: 20).
        k_t: Per-joint torque constants [Nm/A].
        offset: Per-joint regression offsets.
    """

    def __init__(self, num_joints: int = 20, cfg: Phase1Config | None = None):
        self.num_joints = num_joints
        self.cfg = cfg or Phase1Config()

        # Per-joint torque constants
        self.k_t = np.zeros(num_joints)

        # Per-joint regression offsets (absorbs gravity/friction diff)
        self.offset = np.zeros(num_joints)

        # Validation metrics
        self.r_squared = np.zeros(num_joints)
        self.jacobian_consistency = np.zeros(num_joints)

        # Finger names and joint mapping
        self.finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        self.finger_joint_indices = {
            "thumb": [0, 1, 2, 3],
            "index": [4, 5, 6, 7],
            "middle": [8, 9, 10, 11],
            "ring": [12, 13, 14, 15],
            "pinky": [16, 17, 18, 19],
        }

        # Calibration results
        self.finger_results: dict[str, FingerCalibrationResult] = {}

    # =========================================================================
    # Core: Paired Sim-Real Calibration
    # =========================================================================

    def calibrate_paired(
        self,
        tau_sim: np.ndarray,
        I_real: np.ndarray,
        finger_idx: np.ndarray | None = None,
    ) -> dict[str, FingerCalibrationResult]:
        """Paired sim-real calibration.

        For each joint j:
            τ_sim_j = k_t_j × I_real_j + b_j

        Uses all data points across conditions (configs, directions, fingers).
        The offset b absorbs gravity/friction differences between sim and real.

        Args:
            tau_sim: Simulated joint torques (N, 20).
            I_real: Real motor currents (N, 20).
            finger_idx: Finger index per sample (N,), optional.
                If provided, calibrates per-finger.
                If None, calibrates all joints together.

        Returns:
            Dictionary of finger calibration results.
        """
        print(f"\nCalibrating with {len(tau_sim)} paired data points...")

        results = {}
        for finger_name in self.finger_names:
            joint_indices = self.finger_joint_indices[finger_name]
            finger_int_idx = self.finger_names.index(finger_name)

            # Filter per finger if finger_idx provided
            if finger_idx is not None:
                mask = finger_idx == finger_int_idx
                if not mask.any():
                    # No data for this finger - create empty result
                    result = FingerCalibrationResult(finger_name=finger_name)
                    for j_idx in joint_indices:
                        result.joints.append(JointCalibrationResult(
                            joint_idx=j_idx,
                            joint_name=self._get_joint_name(j_idx),
                            k_t=0.0, offset=0.0, r_squared=0.0,
                            num_data_points=0, success=False,
                            error_msg="No data for this finger",
                        ))
                    results[finger_name] = result
                    self.finger_results[finger_name] = result
                    continue

                tau_finger = tau_sim[mask]
                I_finger = I_real[mask]
            else:
                tau_finger = tau_sim
                I_finger = I_real

            # Calibrate each joint in this finger
            result = FingerCalibrationResult(finger_name=finger_name)
            for j_idx in joint_indices:
                joint_result = self._calibrate_joint_paired(
                    j_idx, tau_finger[:, j_idx], I_finger[:, j_idx],
                )
                result.joints.append(joint_result)

            results[finger_name] = result
            self.finger_results[finger_name] = result

            if result.all_success:
                print(f"  {finger_name}: k_t = {[f'{v:.4f}' for v in result.k_t_values]}, "
                      f"R² = {result.mean_r_squared:.3f}")
            else:
                failed = [j.joint_name for j in result.joints if not j.success]
                print(f"  {finger_name}: R² = {result.mean_r_squared:.3f} "
                      f"(failed: {failed})")

        return results

    def _calibrate_joint_paired(
        self,
        joint_idx: int,
        tau_sim: np.ndarray,
        I_real: np.ndarray,
    ) -> JointCalibrationResult:
        """Calibrate a single joint using paired sim-real data.

        Fits: τ_sim = k_t × I_real + b

        Args:
            joint_idx: Global joint index.
            tau_sim: Sim torques for this joint (N,).
            I_real: Real currents for this joint (N,).
        """
        joint_name = self._get_joint_name(joint_idx)
        N = len(tau_sim)

        if N < self.cfg.min_data_points:
            return JointCalibrationResult(
                joint_idx=joint_idx, joint_name=joint_name,
                k_t=0.0, offset=0.0, r_squared=0.0, num_data_points=N,
                success=False,
                error_msg=f"Insufficient data: {N} < {self.cfg.min_data_points}",
            )

        # Remove outliers
        tau_sim, I_real = self._remove_outliers_paired(tau_sim, I_real)
        N = len(tau_sim)

        if N < self.cfg.min_data_points:
            return JointCalibrationResult(
                joint_idx=joint_idx, joint_name=joint_name,
                k_t=0.0, offset=0.0, r_squared=0.0, num_data_points=N,
                success=False,
                error_msg=f"Insufficient data after outlier removal: {N}",
            )

        # Linear regression: τ_sim = k_t × I_real + b
        reg = LinearRegression(fit_intercept=True)
        reg.fit(I_real.reshape(-1, 1), tau_sim)

        k_t = float(reg.coef_[0])
        offset = float(reg.intercept_)
        r_squared = float(reg.score(I_real.reshape(-1, 1), tau_sim))

        # Store
        self.k_t[joint_idx] = k_t
        self.offset[joint_idx] = offset
        self.r_squared[joint_idx] = r_squared

        return JointCalibrationResult(
            joint_idx=joint_idx,
            joint_name=joint_name,
            k_t=k_t,
            offset=offset,
            r_squared=r_squared,
            num_data_points=N,
            success=r_squared >= self.cfg.r_squared_threshold,
        )

    def _remove_outliers_paired(
        self, tau: np.ndarray, I: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove outliers based on residual from initial fit."""
        threshold = self.cfg.outlier_std_threshold

        if len(tau) < self.cfg.min_data_points:
            return tau, I

        # Quick initial fit
        reg = LinearRegression(fit_intercept=True)
        reg.fit(I.reshape(-1, 1), tau)
        residuals = tau - reg.predict(I.reshape(-1, 1))

        std_res = np.std(residuals)
        if std_res < 1e-8:
            return tau, I

        inlier_mask = np.abs(residuals) < threshold * std_res
        return tau[inlier_mask], I[inlier_mask]

    # =========================================================================
    # Jacobian Validation (sim only)
    # =========================================================================

    def validate_jacobian_consistency(
        self,
        tau_sim: np.ndarray,
        F_ext: np.ndarray,
        jacobians: np.ndarray,
        finger_idx: np.ndarray,
    ) -> dict[str, float]:
        """Validate Jacobian model: compare tau_sim vs J^T * F_ext.

        In simulation, we know tau_sim directly. If J^T * F_ext matches
        tau_sim (after removing gravity), the kinematic model is correct.

        This is independent of k_t estimation — it validates the kinematics.

        Args:
            tau_sim: Simulated joint torques (M, 20).
            F_ext: External FT wrench (M, 6).
            jacobians: Fingertip Jacobians (M, 6, 4) per finger.
            finger_idx: Finger index per sample (M,).

        Returns:
            Dictionary of consistency metrics per finger.
        """
        metrics = {}

        for finger_name in self.finger_names:
            joint_indices = self.finger_joint_indices[finger_name]
            finger_int_idx = self.finger_names.index(finger_name)

            mask = finger_idx == finger_int_idx
            if not mask.any():
                continue

            tau_finger = tau_sim[mask]
            F_finger = F_ext[mask]
            J_finger = jacobians[mask]

            # J^T @ F_ext -> (N, 4)
            tau_jac = np.einsum('nij,ni->nj', J_finger, F_finger)

            # Compare with actual sim torques for this finger's joints
            tau_actual = tau_finger[:, joint_indices]

            # RMSE per joint
            rmse_per_joint = np.sqrt(np.mean((tau_actual - tau_jac) ** 2, axis=0))
            rel_error = rmse_per_joint / (np.std(tau_actual, axis=0) + 1e-8)

            metrics[finger_name] = {
                "rmse_per_joint": rmse_per_joint.tolist(),
                "relative_error": rel_error.tolist(),
                "mean_rmse": float(np.mean(rmse_per_joint)),
                "mean_relative_error": float(np.mean(rel_error)),
            }

            # Store consistency score
            for local_idx, j_idx in enumerate(joint_indices):
                self.jacobian_consistency[j_idx] = 1.0 - min(rel_error[local_idx], 1.0)

            print(f"  {finger_name}: Jacobian RMSE = {rmse_per_joint}, "
                  f"rel_error = {rel_error}")

        return metrics

    def calibrate_from_sim_only(
        self,
        tau_sim: np.ndarray,
        F_ext: np.ndarray,
        jacobians: np.ndarray,
        finger_idx: np.ndarray,
    ) -> dict[str, FingerCalibrationResult]:
        """Sim-only calibration (validation / dry run).

        Uses J^T * F_ext as proxy. k_t is set to 1.0 (identity mapping in sim).

        Args:
            tau_sim: Simulated torques (M, 20).
            F_ext: External FT wrench (M, 6).
            jacobians: Fingertip Jacobians (M, 6, 4).
            finger_idx: Finger index (M,).
        """
        # Validate Jacobian consistency
        print("\nValidating Jacobian model (sim)...")
        self.validate_jacobian_consistency(tau_sim, F_ext, jacobians, finger_idx)

        # Set k_t = 1.0 as placeholder (sim has direct torque access)
        self.k_t[:] = 1.0
        self.offset[:] = 0.0
        self.r_squared[:] = 1.0

        results = {}
        for finger_name in self.finger_names:
            joint_indices = self.finger_joint_indices[finger_name]
            result = FingerCalibrationResult(finger_name=finger_name)

            for j_idx in joint_indices:
                result.joints.append(JointCalibrationResult(
                    joint_idx=j_idx,
                    joint_name=self._get_joint_name(j_idx),
                    k_t=1.0, offset=0.0, r_squared=1.0, num_data_points=0,
                    success=True,
                    jacobian_consistency=float(self.jacobian_consistency[j_idx]),
                ))

            results[finger_name] = result
            self.finger_results[finger_name] = result

        print("Note: Sim-only calibration (k_t=1.0). Real robot data needed.")
        return results

    # =========================================================================
    # Full Calibration Pipeline
    # =========================================================================

    def calibrate_all(
        self,
        sim_matched_file: str | None = None,
        real_contact_file: str | None = None,
        sim_contact_file: str | None = None,
    ) -> dict[str, FingerCalibrationResult]:
        """Run full calibration pipeline.

        Primary path: sim_matched + real_contact → calibrate_paired()
        Fallback: sim_contact only → validate Jacobian, k_t=1.0

        Args:
            sim_matched_file: Sim matched τ_sim data (.npz) from sim_from_real mode.
            real_contact_file: Real contact data (.npz) with I_real.
            sim_contact_file: Sim contact data (.npz) for Jacobian validation.

        Returns:
            Dictionary of finger calibration results.
        """
        # Primary: Paired sim-real calibration
        if sim_matched_file is not None and real_contact_file is not None:
            print("\n--- Paired Sim-Real Calibration ---")
            sim_data = np.load(sim_matched_file, allow_pickle=True)
            real_data = np.load(real_contact_file, allow_pickle=True)

            tau_sim = sim_data["tau_sim"]  # (N, 20)
            I_real = real_data["I_motor"]  # (N, 20)
            finger_idx = real_data.get("finger_idx")

            results = self.calibrate_paired(tau_sim, I_real, finger_idx)

            # Optional: Validate Jacobian if sim_contact data available
            if sim_contact_file is not None:
                print("\n--- Jacobian Validation (sim) ---")
                sim_c = np.load(sim_contact_file, allow_pickle=True)
                self.validate_jacobian_consistency(
                    tau_sim=sim_c["tau_applied"],
                    F_ext=sim_c["F_ext"],
                    jacobians=sim_c["jacobian"],
                    finger_idx=sim_c["finger_idx"],
                )

            return results

        # Fallback: Sim-only validation (k_t = 1.0)
        if sim_contact_file is not None:
            print("\n--- Sim-Only Calibration (no real data) ---")
            sim_data = np.load(sim_contact_file, allow_pickle=True)
            return self.calibrate_from_sim_only(
                tau_sim=sim_data["tau_applied"],
                F_ext=sim_data["F_ext"],
                jacobians=sim_data["jacobian"],
                finger_idx=sim_data["finger_idx"],
            )

        print("Error: No data files provided for calibration.")
        return {}

    # =========================================================================
    # Inference
    # =========================================================================

    def current_to_torque(self, currents: np.ndarray) -> np.ndarray:
        """Convert motor currents to joint torques.

        τ = k_t * I + offset

        Args:
            currents: Motor currents, shape (..., num_joints).

        Returns:
            Joint torques, same shape as input.
        """
        return self.k_t * currents + self.offset

    def torque_to_current(self, torques: np.ndarray) -> np.ndarray:
        """Convert joint torques to motor currents.

        I = (τ - offset) / k_t

        Args:
            torques: Joint torques, shape (..., num_joints).

        Returns:
            Motor currents, same shape as input.
        """
        k_t_safe = np.where(np.abs(self.k_t) > 1e-8, self.k_t, 1e-8)
        return (torques - self.offset) / k_t_safe

    def get_calibration_tensor(self, device: str = "cuda:0") -> dict[str, torch.Tensor]:
        """Get calibration parameters as tensors for use in sim.

        Returns:
            Dictionary with k_t and offset tensors.
        """
        return {
            "k_t": torch.from_numpy(self.k_t).float().to(device),
            "offset": torch.from_numpy(self.offset).float().to(device),
        }

    # =========================================================================
    # I/O
    # =========================================================================

    def save(self, filepath: str):
        """Save calibration to file."""
        np.savez(
            filepath,
            k_t=self.k_t,
            offset=self.offset,
            r_squared=self.r_squared,
            jacobian_consistency=self.jacobian_consistency,
        )
        print(f"Saved calibration to {filepath}")

    def load(self, filepath: str):
        """Load calibration from file."""
        data = np.load(filepath)
        self.k_t = data["k_t"]
        self.r_squared = data["r_squared"]
        if "offset" in data:
            self.offset = data["offset"]
        if "jacobian_consistency" in data:
            self.jacobian_consistency = data["jacobian_consistency"]

        # Legacy compatibility
        if "baseline_offset" in data and "offset" not in data:
            self.offset = data["baseline_offset"]

        print(f"Loaded calibration from {filepath}")

    def to_yaml_dict(self) -> dict:
        """Convert calibration to YAML-serializable dictionary."""
        result = {
            "method": "paired_sim_real",
            "global": {
                "mean_r_squared": float(np.mean(self.r_squared)),
                "mean_k_t": float(np.mean(np.abs(self.k_t))),
                "mean_offset": float(np.mean(self.offset)),
                "num_joints": self.num_joints,
            },
            "per_joint": {},
            "per_finger": {},
        }

        for j_idx in range(self.num_joints):
            result["per_joint"][self._get_joint_name(j_idx)] = {
                "k_t": float(self.k_t[j_idx]),
                "offset": float(self.offset[j_idx]),
                "r_squared": float(self.r_squared[j_idx]),
                "jacobian_consistency": float(self.jacobian_consistency[j_idx]),
            }

        for finger_name, finger_result in self.finger_results.items():
            result["per_finger"][finger_name] = {
                "mean_r_squared": float(finger_result.mean_r_squared),
                "all_success": finger_result.all_success,
                "k_t_values": [float(v) for v in finger_result.k_t_values],
                "offset_values": [float(v) for v in finger_result.offset_values],
                "joints": [
                    {
                        "name": j.joint_name,
                        "k_t": j.k_t,
                        "offset": j.offset,
                        "r_squared": j.r_squared,
                        "success": j.success,
                    }
                    for j in finger_result.joints
                ],
            }

        return result

    # =========================================================================
    # Utilities
    # =========================================================================

    def _get_joint_name(self, joint_idx: int) -> str:
        """Get joint name from index."""
        for finger, indices in self.finger_joint_indices.items():
            if joint_idx in indices:
                local_idx = indices.index(joint_idx)
                return f"{finger}_j{local_idx}"
        return f"joint_{joint_idx}"

    def summary(self) -> str:
        """Get calibration summary string."""
        lines = ["=" * 50]
        lines.append("Phase 1 Paired Sim-Real Calibration Summary")
        lines.append("=" * 50)
        lines.append(f"Method: Paired (τ_sim = k_t × I_real + b)")
        lines.append(f"Mean R²: {np.mean(self.r_squared):.3f}")
        lines.append(f"Mean |k_t|: {np.mean(np.abs(self.k_t)):.4f} Nm/A")
        lines.append(f"Mean |offset|: {np.mean(np.abs(self.offset)):.4f} Nm")
        lines.append("")

        for finger_name, result in self.finger_results.items():
            status = "OK" if result.all_success else "FAIL"
            lines.append(f"{finger_name} [{status}]: R² = {result.mean_r_squared:.3f}")
            for j in result.joints:
                j_status = "OK" if j.success else "FAIL"
                lines.append(
                    f"  {j.joint_name} [{j_status}]: "
                    f"k_t={j.k_t:.4f} Nm/A, offset={j.offset:.4f}, "
                    f"R²={j.r_squared:.3f}, J_cons={j.jacobian_consistency:.3f}"
                )

        lines.append("=" * 50)
        return "\n".join(lines)
