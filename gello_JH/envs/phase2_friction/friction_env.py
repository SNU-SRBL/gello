# Copyright (c) 2025, SRBL
# Phase 2: Friction & Contact Calibration Environment

from __future__ import annotations

import torch
import numpy as np
from typing import Any
from dataclasses import dataclass

from isaaclab.assets import RigidObject

from ..base.real2sim_base_env import Real2SimBaseEnv
from .friction_env_cfg import FrictionEnvCfg
from .slip_detection import (
    VisionSlipDetector,
    TactileSlipDetector,
    SlipEvent,
    DynamicFrictionData,
)


@dataclass
class Phase2SlipData:
    """Data structure for Phase 2 slip test results."""

    # Time information
    timestamps: np.ndarray
    """Timestamps (T,)."""

    # Grip state
    q_grip: np.ndarray
    """Joint positions during test (T, 20)."""

    tau_grip: np.ndarray
    """Joint torques during test (T, 20)."""

    grip_position_normalized: np.ndarray
    """Normalized grip position during test (T,)."""

    # Force measurements
    F_tangential: np.ndarray
    """Tangential forces (T,)."""

    F_normal: np.ndarray
    """Normal forces (T,)."""

    # Object tracking (ArUco)
    object_position: np.ndarray
    """Object position (T, 3)."""

    object_velocity: np.ndarray
    """Object velocity (T, 3)."""

    object_acceleration: np.ndarray
    """Object acceleration (T, 3)."""

    # Slip information
    slip_detected: bool
    """Whether slip was detected."""

    t_slip: float
    """Timestamp when slip occurred."""

    q_slip: np.ndarray
    """Joint positions at slip (20,)."""

    F_slip_tangential: float
    """Tangential force at slip (N)."""

    F_slip_normal: float
    """Normal force at slip (N)."""

    # Friction coefficients
    static_friction: float
    """Computed static friction coefficient (μ_s)."""

    dynamic_friction: float
    """Computed dynamic friction coefficient (μ_d)."""

    dynamic_friction_valid: bool
    """Whether dynamic friction measurement is valid."""

    # Metadata
    metadata: dict
    """Additional metadata."""


class FrictionEnv(Real2SimBaseEnv):
    """Friction & Contact Calibration Environment for Phase 2.

    This environment calibrates friction and contact parameters using
    position control to perform slip tests.

    Workflow:
    1. Grip object with position control (tight grip)
    2. Gradually loosen grip (position sweep)
    3. Detect when slip occurs (via object motion tracking)
    4. Measure μ_s (static friction) at slip onset
    5. Track post-slip motion to compute μ_d (dynamic friction)
    6. Optimize friction parameters to match real measurements

    Key insight: Slip occurs when grip_force < μ × normal_force + gravity
    """

    cfg: FrictionEnvCfg

    def __init__(self, cfg: FrictionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Create vision-based slip detector with ArUco tracking
        self._slip_detector = VisionSlipDetector(
            marker_id=cfg.aruco_marker_id,
            marker_size=cfg.aruco_marker_size,
            history_length=cfg.tracking_history_length,
            tracking_fps=cfg.tracking_fps,
            velocity_threshold=cfg.slip_velocity_threshold,
            displacement_threshold=cfg.slip_displacement_threshold,
            acceleration_threshold=cfg.slip_acceleration_threshold,
            device=self.device,
            num_envs=self.num_envs,
        )

        # Create tactile slip detector for force-based detection
        self._tactile_detector = TactileSlipDetector(
            force_ratio_threshold=0.8,
            force_derivative_threshold=0.5,
            device=self.device,
            num_envs=self.num_envs,
        )

        # Current grip position (normalized 0~1)
        self._grip_position = torch.ones(self.num_envs, device=self.device) * cfg.initial_grip_position
        self._sweep_timer = torch.zeros(self.num_envs, device=self.device)
        self._settle_timer = torch.zeros(self.num_envs, device=self.device)
        self._in_settle_phase = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # Post-slip tracking
        self._post_slip_timer = torch.zeros(self.num_envs, device=self.device)
        self._tracking_post_slip = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Slip detection results
        self._slip_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._q_slip = torch.zeros(self.num_envs, 20, device=self.device)
        self._tau_slip = torch.zeros(self.num_envs, 20, device=self.device)
        self._grip_pos_at_slip = torch.zeros(self.num_envs, device=self.device)
        self._F_tangential_at_slip = torch.zeros(self.num_envs, device=self.device)
        self._F_normal_at_slip = torch.zeros(self.num_envs, device=self.device)

        # Friction results
        self._static_friction = torch.zeros(self.num_envs, device=self.device)
        self._dynamic_friction = torch.zeros(self.num_envs, device=self.device)
        self._dynamic_friction_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Test object reference (set in _setup_scene)
        self._test_object: RigidObject | None = None

        # Data recording
        self._recorded_data: list[dict] = []

        # Load Phase 0/1 calibration if provided
        self._load_phase0_calibration()
        self._load_phase1_calibration()

    def _load_phase0_calibration(self):
        """Load Phase 0 calibration results and apply to actuators."""
        if self.cfg.phase0_result_file is None:
            print("Phase 0 result not provided - using default stiffness/damping")
            return

        try:
            import yaml
            from pathlib import Path

            result_path = Path(self.cfg.phase0_result_file)
            if not result_path.exists():
                print(f"Warning: Phase 0 result file not found: {result_path}")
                return

            with open(result_path, "r") as f:
                result = yaml.safe_load(f)

            # Extract stiffness and damping
            params = result.get("parameters", result.get("calibrated_params", {}))

            if "joint_stiffness" in params:
                stiffness = params["joint_stiffness"]
                if isinstance(stiffness, list):
                    self.cfg.hand_stiffness = stiffness
                    print(f"Loaded Phase 0 stiffness: {len(stiffness)} joints")

            if "joint_damping" in params:
                damping = params["joint_damping"]
                if isinstance(damping, list):
                    self.cfg.hand_damping = damping
                    print(f"Loaded Phase 0 damping: {len(damping)} joints")

            print(f"Phase 0 calibration loaded from: {result_path}")

        except Exception as e:
            print(f"Warning: Failed to load Phase 0 calibration: {e}")

    def _load_phase1_calibration(self):
        """Load Phase 1 calibration results (I→τ mapping)."""
        if self.cfg.phase1_result_file is None:
            print("Phase 1 result not provided - using default I→τ mapping")
            return

        try:
            import yaml
            from pathlib import Path

            result_path = Path(self.cfg.phase1_result_file)
            if not result_path.exists():
                print(f"Warning: Phase 1 result file not found: {result_path}")
                return

            with open(result_path, "r") as f:
                result = yaml.safe_load(f)

            # Extract k_gain and k_offset for I→τ conversion
            params = result.get("parameters", {})
            per_joint = params.get("per_joint", {})

            if per_joint:
                joint_names = list(per_joint.keys())
                k_gains = [per_joint[j].get("k_gain", 0.15) for j in joint_names]
                k_offsets = [per_joint[j].get("k_offset", 0.02) for j in joint_names]

                k_gain_tensor = torch.tensor(k_gains, device=self.device)
                k_offset_tensor = torch.tensor(k_offsets, device=self.device)

                self.set_current_torque_calibration(k_gain_tensor, k_offset_tensor)
                print(f"Phase 1 calibration loaded: k_gain mean={k_gain_tensor.mean():.4f}")

            print(f"Phase 1 calibration loaded from: {result_path}")

        except Exception as e:
            print(f"Warning: Failed to load Phase 1 calibration: {e}")

    def _setup_scene(self):
        """Set up scene with test object."""
        super()._setup_scene()

        # Add test object
        self._test_object = RigidObject(self.cfg.test_object)
        self.scene.rigid_objects["test_object"] = self._test_object

    def _apply_action(self):
        """Apply grip position based on sweep progress."""
        current_time = self.sim.current_time

        # Lock UR5 arm in place
        if hasattr(self, "_ur5e_arm") and self._ur5e_arm is not None:
            self._ur5e_arm.set_joint_position_target(self._ur5e_initial_pos)

        # Handle settle phase (before starting sweep)
        if self._in_settle_phase.any():
            self._settle_timer[self._in_settle_phase] += self.physics_dt
            settle_done = self._settle_timer >= self.cfg.settle_time_before_sweep
            self._in_settle_phase &= ~settle_done

        # Update grip position for environments not in settle phase and not slipped
        sweep_mask = ~self._in_settle_phase & ~self._slip_detected & ~self._tracking_post_slip
        if sweep_mask.any():
            self._sweep_timer[sweep_mask] += self.physics_dt

            # Compute sweep progress
            alpha = torch.clamp(
                self._sweep_timer / self.cfg.sweep_duration,
                0.0, 1.0
            )

            # Linear interpolation from initial to final grip position
            self._grip_position[sweep_mask] = (
                self.cfg.initial_grip_position * (1 - alpha[sweep_mask]) +
                self.cfg.final_grip_position * alpha[sweep_mask]
            )

        # Post-slip tracking timer
        if self._tracking_post_slip.any():
            self._post_slip_timer[self._tracking_post_slip] += self.physics_dt

        # Convert normalized grip position to joint targets
        grip_joint_targets = self._compute_grip_joint_targets()

        # Apply position control to hand
        if hasattr(self, "_tesollo_hand") and self._tesollo_hand is not None:
            self._tesollo_hand.set_joint_position_target(grip_joint_targets)

    def _compute_grip_joint_targets(self) -> torch.Tensor:
        """Compute joint position targets from normalized grip position.

        Returns:
            Joint position targets, shape (num_envs, 20).
        """
        # Get joint limits
        if hasattr(self, "_tesollo_hand") and self._tesollo_hand is not None:
            joint_limits = self._tesollo_hand.data.soft_joint_pos_limits
            lower = joint_limits[..., 0]  # (num_envs, 20)
            upper = joint_limits[..., 1]  # (num_envs, 20)
        else:
            # Default limits if hand not available
            lower = torch.zeros(self.num_envs, 20, device=self.device)
            upper = torch.ones(self.num_envs, 20, device=self.device) * 1.57

        # Get grip finger joint indices
        grip_indices = []
        for finger in self.cfg.grip_fingers:
            grip_indices.extend(self.cfg.finger_joint_indices[finger])

        # Compute joint targets
        joint_targets = torch.zeros(self.num_envs, 20, device=self.device)

        for idx in grip_indices:
            # Interpolate between lower and upper based on grip position
            joint_targets[:, idx] = (
                lower[:, idx] + self._grip_position * (upper[:, idx] - lower[:, idx])
            )

        return joint_targets

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute observations and check for slip."""
        obs = super()._get_observations()
        current_time = self.sim.current_time

        # Update slip detector with object state
        if self._test_object is not None:
            obj_pos = self._test_object.data.root_pos_w
            obj_vel = self._test_object.data.root_lin_vel_w

            # Update vision-based detector (includes ArUco tracking)
            self._slip_detector.update(obj_pos, obj_vel, current_time)

            # Check for slip (only for environments not already slipped)
            check_mask = ~self._slip_detected
            if check_mask.any():
                slip = self._slip_detector.check_slip(current_time)
                newly_slipped = slip & check_mask

                # Record slip data for newly slipped environments
                if newly_slipped.any():
                    self._record_slip_event(newly_slipped, current_time)

        # Update tactile detector if available
        if hasattr(self, "get_fingertip_forces"):
            fingertip_forces = self.get_fingertip_forces()
            if fingertip_forces is not None:
                F_tangential, F_normal = self._tactile_detector.update(fingertip_forces)

        # Record data
        if self._step_count % self.cfg.record_frequency == 0:
            self._record_step_data(current_time)

        # Check for post-slip tracking completion
        if self._tracking_post_slip.any():
            tracking_done = self._post_slip_timer >= self.cfg.post_slip_tracking_duration
            if tracking_done.any():
                # Compute dynamic friction for completed tracking
                self._compute_dynamic_friction(torch.where(tracking_done)[0])
                self._tracking_post_slip[tracking_done] = False

        return obs

    def _record_slip_event(self, env_ids: torch.Tensor, timestamp: float):
        """Record slip event data for specified environments.

        Args:
            env_ids: Environment indices where slip just occurred.
            timestamp: Current simulation time.
        """
        # Record joint state at slip
        if hasattr(self, "_tesollo_hand") and self._tesollo_hand is not None:
            self._q_slip[env_ids] = self._tesollo_hand.data.joint_pos[env_ids].clone()
            self._tau_slip[env_ids] = self._tesollo_hand.data.applied_torque[env_ids].clone()

        # Record grip position at slip
        self._grip_pos_at_slip[env_ids] = self._grip_position[env_ids].clone()

        # Record forces at slip (from tactile detector)
        F_tangential, F_normal = self._tactile_detector.get_forces_at_slip()
        self._F_tangential_at_slip[env_ids] = F_tangential[env_ids]
        self._F_normal_at_slip[env_ids] = F_normal[env_ids]

        # Compute static friction
        self._static_friction[env_ids] = (
            self._F_tangential_at_slip[env_ids] /
            (self._F_normal_at_slip[env_ids] + 1e-6)
        )

        # Mark as slipped and start post-slip tracking
        self._slip_detected[env_ids] = True
        self._tracking_post_slip[env_ids] = True
        self._post_slip_timer[env_ids] = 0.0

        print(f"Slip detected at t={timestamp:.3f}s for envs {env_ids.tolist()}")
        print(f"  Static friction (μ_s): {self._static_friction[env_ids].mean():.4f}")

    def _compute_dynamic_friction(self, env_ids: torch.Tensor):
        """Compute dynamic friction from post-slip tracking.

        Args:
            env_ids: Environment indices to compute for.
        """
        for env_idx in env_ids:
            idx = env_idx.item()

            # Get normal force (average during slip)
            F_normal = self._F_normal_at_slip[idx].item()

            # Compute dynamic friction from tracker
            mu_d, valid = self._slip_detector.compute_dynamic_friction(
                env_idx=idx,
                object_mass=self.cfg.object_mass,
                normal_force=F_normal,
                gravity=self.cfg.gravity,
            )

            self._dynamic_friction[idx] = mu_d
            self._dynamic_friction_valid[idx] = valid

            if valid:
                print(f"Env {idx}: Dynamic friction (μ_d) = {mu_d:.4f}")
            else:
                print(f"Env {idx}: Dynamic friction measurement invalid (insufficient motion)")

    def _record_step_data(self, timestamp: float):
        """Record data for current step.

        Args:
            timestamp: Current simulation time.
        """
        data = {
            "timestamp": timestamp,
            "grip_position": self._grip_position.cpu().numpy().copy(),
        }

        # Joint state
        if hasattr(self, "_tesollo_hand") and self._tesollo_hand is not None:
            data["q"] = self._tesollo_hand.data.joint_pos.cpu().numpy().copy()
            data["tau"] = self._tesollo_hand.data.applied_torque.cpu().numpy().copy()

        # Object state
        if self._test_object is not None:
            data["object_pos"] = self._test_object.data.root_pos_w.cpu().numpy().copy()
            data["object_vel"] = self._test_object.data.root_lin_vel_w.cpu().numpy().copy()

        # Force data
        data["F_tangential"] = self._tactile_detector._F_tangential_at_slip.cpu().numpy().copy()
        data["F_normal"] = self._tactile_detector._F_normal_at_slip.cpu().numpy().copy()

        self._recorded_data.append(data)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        # Terminate after post-slip tracking is complete
        terminated = self._slip_detected & ~self._tracking_post_slip

        # Truncate if sweep complete without slip
        sweep_complete = self._sweep_timer >= self.cfg.sweep_duration
        time_done = self.episode_length_buf >= self.max_episode_length
        truncated = (sweep_complete & ~self._slip_detected) | time_done

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments."""
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset grip state
        self._grip_position[env_ids] = self.cfg.initial_grip_position
        self._sweep_timer[env_ids] = 0.0
        self._settle_timer[env_ids] = 0.0
        self._in_settle_phase[env_ids] = True
        self._post_slip_timer[env_ids] = 0.0
        self._tracking_post_slip[env_ids] = False

        # Reset slip detection
        self._slip_detected[env_ids] = False
        self._q_slip[env_ids] = 0.0
        self._tau_slip[env_ids] = 0.0
        self._grip_pos_at_slip[env_ids] = 0.0
        self._F_tangential_at_slip[env_ids] = 0.0
        self._F_normal_at_slip[env_ids] = 0.0

        # Reset friction results
        self._static_friction[env_ids] = 0.0
        self._dynamic_friction[env_ids] = 0.0
        self._dynamic_friction_valid[env_ids] = False

        # Reset detectors
        self._slip_detector.reset(env_ids)
        self._tactile_detector.reset(env_ids)

        # Reset object position (grasp pose)
        if self._test_object is not None:
            default_pos = self._test_object.data.default_root_state[env_ids, :3]
            default_quat = self._test_object.data.default_root_state[env_ids, 3:7]
            self._test_object.write_root_pose_to_sim(
                torch.cat([default_pos, default_quat], dim=-1),
                env_ids=env_ids,
            )

        # Clear recorded data
        self._recorded_data.clear()

    # ===== Calibration Methods =====

    def run_slip_test(self) -> dict[str, Any]:
        """Run a complete slip test and return results.

        Returns:
            Dictionary with slip test results.
        """
        # Reset and run until done
        self.reset()

        while True:
            obs, reward, terminated, truncated, info = self.step(
                torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
            )

            if terminated.all() or truncated.all():
                break

        return self.get_slip_test_results()

    def get_slip_test_results(self) -> dict[str, Any]:
        """Get comprehensive slip test results.

        Returns:
            Dictionary of results for analysis.
        """
        return {
            "slip_detected": self._slip_detected.cpu().numpy(),
            "static_friction": self._static_friction.cpu().numpy(),
            "dynamic_friction": self._dynamic_friction.cpu().numpy(),
            "dynamic_friction_valid": self._dynamic_friction_valid.cpu().numpy(),
            "q_slip": self._q_slip.cpu().numpy(),
            "tau_slip": self._tau_slip.cpu().numpy(),
            "grip_position_at_slip": self._grip_pos_at_slip.cpu().numpy(),
            "F_tangential_at_slip": self._F_tangential_at_slip.cpu().numpy(),
            "F_normal_at_slip": self._F_normal_at_slip.cpu().numpy(),
            "sweep_time": self._sweep_timer.cpu().numpy(),
        }

    def get_static_friction(self) -> torch.Tensor:
        """Get static friction coefficient for all environments.

        Returns:
            Tensor of μ_s values.
        """
        return self._static_friction.clone()

    def get_dynamic_friction(self) -> torch.Tensor:
        """Get dynamic friction coefficient for all environments.

        Returns:
            Tensor of μ_d values.
        """
        return self._dynamic_friction.clone()

    def compute_friction_matching_loss(
        self,
        real_static_friction: float | torch.Tensor,
        real_dynamic_friction: float | torch.Tensor | None = None,
        static_weight: float = 1.0,
        dynamic_weight: float = 0.5,
    ) -> torch.Tensor:
        """Compute loss based on friction coefficient matching.

        Args:
            real_static_friction: Real robot's static friction coefficient.
            real_dynamic_friction: Real robot's dynamic friction coefficient (optional).
            static_weight: Weight for static friction loss.
            dynamic_weight: Weight for dynamic friction loss.

        Returns:
            Loss tensor (lower is better).
        """
        if isinstance(real_static_friction, float):
            real_static_friction = torch.tensor(real_static_friction, device=self.device)

        loss = torch.zeros(self.num_envs, device=self.device)

        # Static friction loss
        slipped = self._slip_detected
        if slipped.any():
            static_loss = torch.abs(self._static_friction[slipped] - real_static_friction)
            loss[slipped] += static_weight * static_loss

        # Dynamic friction loss (if provided and valid)
        if real_dynamic_friction is not None:
            if isinstance(real_dynamic_friction, float):
                real_dynamic_friction = torch.tensor(real_dynamic_friction, device=self.device)

            valid = self._dynamic_friction_valid
            if valid.any():
                dynamic_loss = torch.abs(self._dynamic_friction[valid] - real_dynamic_friction)
                loss[valid] += dynamic_weight * dynamic_loss

        # Large penalty for no slip
        loss[~slipped] = 1e6

        return loss.mean()

    def set_friction_params(
        self,
        static_friction: float | None = None,
        dynamic_friction: float | None = None,
        contact_stiffness: float | None = None,
        contact_damping: float | None = None,
        contact_offset: float | None = None,
    ):
        """Set friction and contact parameters.

        Args:
            static_friction: Static friction coefficient.
            dynamic_friction: Dynamic friction coefficient.
            contact_stiffness: Contact stiffness.
            contact_damping: Contact damping.
            contact_offset: Contact offset distance.
        """
        # Update internal parameters
        if static_friction is not None:
            self._calibration_params["static_friction"] = torch.tensor(
                [[static_friction]], device=self.device
            ).expand(self.num_envs, 1)

        if dynamic_friction is not None:
            self._calibration_params["dynamic_friction"] = torch.tensor(
                [[dynamic_friction]], device=self.device
            ).expand(self.num_envs, 1)

        # Apply to physics materials
        self._apply_friction_to_simulation(
            static_friction, dynamic_friction,
            contact_stiffness, contact_damping, contact_offset
        )

    def _apply_friction_to_simulation(
        self,
        static_friction: float | None,
        dynamic_friction: float | None,
        contact_stiffness: float | None,
        contact_damping: float | None,
        contact_offset: float | None,
    ):
        """Apply friction parameters to simulation.

        This requires modifying physics materials at runtime.
        Implementation depends on Isaac Lab/PhysX version.
        """
        # TODO: Implement runtime physics material modification
        # This is complex and may require direct PhysX API access
        pass

    def export_slip_data(self, filepath: str):
        """Export recorded slip test data to file.

        Args:
            filepath: Output file path (.npz format).
        """
        import numpy as np

        # Convert recorded data to arrays
        if not self._recorded_data:
            print("Warning: No data recorded to export")
            return

        timestamps = np.array([d["timestamp"] for d in self._recorded_data])
        grip_positions = np.array([d["grip_position"] for d in self._recorded_data])

        data_dict = {
            "timestamps": timestamps,
            "grip_positions": grip_positions,
            "slip_detected": self._slip_detected.cpu().numpy(),
            "static_friction": self._static_friction.cpu().numpy(),
            "dynamic_friction": self._dynamic_friction.cpu().numpy(),
            "dynamic_friction_valid": self._dynamic_friction_valid.cpu().numpy(),
            "q_slip": self._q_slip.cpu().numpy(),
            "tau_slip": self._tau_slip.cpu().numpy(),
            "F_tangential_at_slip": self._F_tangential_at_slip.cpu().numpy(),
            "F_normal_at_slip": self._F_normal_at_slip.cpu().numpy(),
        }

        # Add optional arrays if available
        if "q" in self._recorded_data[0]:
            data_dict["joint_positions"] = np.array([d["q"] for d in self._recorded_data])
        if "tau" in self._recorded_data[0]:
            data_dict["joint_torques"] = np.array([d["tau"] for d in self._recorded_data])
        if "object_pos" in self._recorded_data[0]:
            data_dict["object_positions"] = np.array([d["object_pos"] for d in self._recorded_data])
        if "object_vel" in self._recorded_data[0]:
            data_dict["object_velocities"] = np.array([d["object_vel"] for d in self._recorded_data])

        np.savez(filepath, **data_dict)
        print(f"Slip test data exported to: {filepath}")

    @property
    def sweep_progress(self) -> torch.Tensor:
        """Get current sweep progress (0.0 to 1.0)."""
        return torch.clamp(
            self._sweep_timer / self.cfg.sweep_duration,
            0.0, 1.0
        )

    @property
    def is_slip_detected(self) -> torch.Tensor:
        """Get slip detection status."""
        return self._slip_detected
