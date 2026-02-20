# Copyright (c) 2025, SRBL
# Phase 1: Current-Torque Calibration Environment

from __future__ import annotations

import torch
import numpy as np
from typing import Any
from enum import Enum

from isaaclab.assets import RigidObject

from ..base.real2sim_base_env import Real2SimBaseEnv
from .current_torque_env_cfg import CurrentTorqueEnvCfg


class CalibrationState(Enum):
    """Calibration protocol state machine.

    Phase 1A (Gravity/Friction Baseline, sim only):
        GRAVITY_BASELINE_MOVING → GRAVITY_BASELINE_RECORDING
        → GRAVITY_BASELINE_VELOCITY (friction sweep)

    Phase 1B (Contact Calibration, sim pressing against FT sensor):
        MOVING_TO_CONTACT_POSITION → PRESSING → SETTLING → RECORDING
        → RETRACTING → REPOSITION_FT_SENSOR (next direction)

    Sim-From-Real (direct force application from real data):
        SFR_SETTING_STATE → SFR_APPLYING_FORCE → SFR_SETTLING
        → SFR_RECORDING → SFR_NEXT_POINT
    """
    IDLE = "idle"
    # Phase 1A: Gravity/Friction Baseline
    GRAVITY_BASELINE_MOVING = "gravity_baseline_moving"
    GRAVITY_BASELINE_RECORDING = "gravity_baseline_recording"
    GRAVITY_BASELINE_VELOCITY = "gravity_baseline_velocity"
    # Phase 1B: Contact Calibration
    MOVING_TO_CONTACT_POSITION = "moving_to_contact_position"
    PRESSING = "pressing"
    SETTLING = "settling"
    RECORDING = "recording"
    RETRACTING = "retracting"
    REPOSITION_FT_SENSOR = "reposition_ft_sensor"
    # Sim-From-Real mode
    SFR_SETTING_STATE = "sfr_setting_state"
    SFR_APPLYING_FORCE = "sfr_applying_force"
    SFR_SETTLING = "sfr_settling"
    SFR_RECORDING = "sfr_recording"
    SFR_NEXT_POINT = "sfr_next_point"
    COMPLETED = "completed"


class CurrentTorqueEnv(Real2SimBaseEnv):
    """Current-Torque Calibration Environment.

    Supports three operating modes:

    1. sim_full/sim_baseline/sim_contact: Standard simulation data collection
       (pressing against FT sensor fixture in sim)

    2. sim_from_real: Takes real robot data (q, F_ext) and computes τ_sim
       by directly applying F_ext to fingertips via set_external_force_and_torque().
       No physical contact simulation needed.

    3. calibrate: Offline, uses JacobianCurrentTorqueModel.
    """

    cfg: CurrentTorqueEnvCfg

    def __init__(self, cfg: CurrentTorqueEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # State machine
        self._calibration_state = CalibrationState.IDLE
        self._current_finger_idx = 0
        self._current_trial = 0
        self._current_step = 0
        self._current_config = 0
        self._current_direction = 0
        self._phase = "idle"

        # Timing
        self._state_timer = 0.0
        self._record_counter = 0

        # Position control targets
        self._ur5_target_pos: torch.Tensor | None = None
        self._hand_target_pos: torch.Tensor | None = None
        self._hand_neutral_pos: torch.Tensor | None = None

        # Gravity baseline configs (generated at start)
        self._gravity_configs: list[torch.Tensor] = []

        # Phase 1A data
        self._baseline_data: dict[str, list] = {
            "q_positions": [],
            "qdot": [],
            "tau_applied": [],
            "gravity_torques": [],
            "F_internal": [],
            "finger_idx": [],
        }

        # Phase 1B data
        self._contact_data: dict[str, list] = {
            "q": [],
            "qdot": [],
            "tau_applied": [],
            "F_ext": [],
            "F_internal": [],
            "jacobian": [],
            "finger_idx": [],
            "config_idx": [],
            "direction_idx": [],
        }

        # Sim-from-real data
        self._sfr_real_data: dict[str, np.ndarray] | None = None
        self._sfr_current_idx = 0
        self._sfr_results: dict[str, list] = {
            "tau_sim": [],
            "q_actual": [],
            "q_cmd": [],
            "q_deflection": [],
            "F_applied": [],
            "F_internal": [],
            "finger_idx": [],
        }

        # Fingertip body indices
        self._fingertip_body_indices: dict[str, int] = {}

        # FT sensor fixture (external)
        self._ft_fixture: RigidObject | None = None

    def _setup_scene(self):
        """Set up scene with external FT sensor fixture."""
        super()._setup_scene()

        # Add external FT sensor fixture
        self._ft_fixture = RigidObject(self.cfg.ft_fixture)
        self.scene.rigid_objects["ft_fixture"] = self._ft_fixture

        # Get fingertip body indices
        self._setup_fingertip_indices()

        # Store initial positions
        self._store_initial_positions()

        # Load and apply Phase 0 calibration if provided
        self._load_phase0_calibration()

    def _load_phase0_calibration(self):
        """Load Phase 0 calibration results and apply to actuators."""
        if self.cfg.phase0_result_file is None:
            print("Phase 0 result not provided - using default stiffness/damping")
            return

        from pathlib import Path

        phase0_path = Path(self.cfg.phase0_result_file)
        if not phase0_path.exists():
            print(f"Warning: Phase 0 result file not found: {phase0_path}")
            return

        try:
            from ...data.storage import CalibrationResultStorage

            storage = CalibrationResultStorage(phase0_path.parent)
            result = storage.load(phase0_path.name)

            print(f"Loading Phase 0 calibration from: {phase0_path}")

            if hasattr(result, 'joint_stiffness') and result.joint_stiffness is not None:
                stiffness = result.joint_stiffness
                if isinstance(stiffness, np.ndarray):
                    stiffness = torch.from_numpy(stiffness).float().to(self.device)
                self._apply_joint_stiffness(stiffness)
                print(f"  Applied joint stiffness: mean={stiffness.mean():.2f}")

            if hasattr(result, 'joint_damping') and result.joint_damping is not None:
                damping = result.joint_damping
                if isinstance(damping, np.ndarray):
                    damping = torch.from_numpy(damping).float().to(self.device)
                self._apply_joint_damping(damping)
                print(f"  Applied joint damping: mean={damping.mean():.2f}")

            self._phase0_calibration = result
            print("Phase 0 calibration applied successfully")

        except Exception as e:
            print(f"Warning: Failed to load Phase 0 calibration: {e}")
            self._phase0_calibration = None

    def _setup_fingertip_indices(self):
        """Set up body indices for fingertips."""
        body_names = self._tesollo_hand.body_names

        finger_tip_map = {
            "thumb": "ll_dg_1_4",
            "index": "ll_dg_2_4",
            "middle": "ll_dg_3_4",
            "ring": "ll_dg_4_4",
            "pinky": "ll_dg_5_4",
        }

        for finger, tip_name in finger_tip_map.items():
            if tip_name in body_names:
                self._fingertip_body_indices[finger] = body_names.index(tip_name)

    def _store_initial_positions(self):
        """Store initial joint positions for UR5 and hand."""
        if self._ur5e_arm is not None:
            self._ur5_target_pos = self._ur5e_arm.data.default_joint_pos.clone()

        self._hand_neutral_pos = self._tesollo_hand.data.default_joint_pos.clone()
        self._hand_target_pos = self._hand_neutral_pos.clone()

    # =========================================================================
    # Jacobian & Gravity API
    # =========================================================================

    def _get_fingertip_jacobian(self, finger: str) -> torch.Tensor:
        """Get fingertip Jacobian for a specific finger.

        Returns:
            Jacobian tensor, shape (num_envs, 6, 4).
        """
        all_jacobians = self._tesollo_hand.root_physx_view.get_jacobians()

        tip_body_idx = self._fingertip_body_indices[finger] - 1
        finger_joint_ids = self.cfg.finger_joint_indices[finger]

        jacobian = all_jacobians[:, tip_body_idx, :, :][:, :, finger_joint_ids]
        return jacobian

    def _get_gravity_torques(self) -> torch.Tensor:
        """Get gravity compensation torques from IsaacLab."""
        return self._tesollo_hand.root_physx_view.get_gravity_compensation_forces()

    # =========================================================================
    # Sim-From-Real Mode
    # =========================================================================

    def start_sim_from_real(self, real_data_path: str):
        """Start sim-from-real mode with q_cmd replay.

        Replays the real robot's control scenario in simulation:
        1. Set initial joint state to q_cmd (pre-force position, same as real before F_ext)
        2. Set PD target to q_cmd (same controller target as real)
        3. Apply real F_ext directly to fingertip via PhysX API
        4. Let physics settle → sim deflects from q_cmd to q_sim (as real deflects to q_actual)
        5. Record τ_sim at equilibrium, compare q_sim vs q_actual

        Args:
            real_data_path: Path to real contact data (.npz).
                Required keys: q (N,20), q_cmd (N,20), F_ext (N,6), finger_idx (N,)
                Optional: qdot (N,20), I_motor (N,20), F_internal (N,30)
        """
        data = np.load(real_data_path, allow_pickle=True)

        # Validate q_cmd presence
        if "q_cmd" not in data:
            raise ValueError(
                "Real data must contain 'q_cmd' (position command) for q_cmd replay mode. "
                "Record both q (actual position) and q_cmd (position target) from real robot."
            )

        self._sfr_real_data = {
            "q": data["q"],  # (N, 20) actual joint position (after deflection)
            "q_cmd": data["q_cmd"],  # (N, 20) position command
            "F_ext": data["F_ext"],  # (N, 6)
            "finger_idx": data["finger_idx"],  # (N,)
        }
        if "qdot" in data:
            self._sfr_real_data["qdot"] = data["qdot"]
        if "F_internal" in data:
            self._sfr_real_data["F_internal"] = data["F_internal"]

        self._sfr_current_idx = 0
        self._sfr_results = {
            "tau_sim": [],
            "q_actual": [],
            "q_cmd": [],
            "q_deflection": [],
            "F_applied": [],
            "F_internal": [],
            "finger_idx": [],
        }

        N = len(data["q"])
        print(f"Sim-from-real (q_cmd replay): {N} data points to process")
        print(f"  Fingers: {np.unique(data['finger_idx'])}")
        q_diff = np.abs(data["q"] - data["q_cmd"]).mean()
        print(f"  Mean |q_actual - q_cmd|: {q_diff:.4f} rad")

        # Enter state machine
        self._phase = "sfr"
        self._calibration_state = CalibrationState.SFR_SETTING_STATE
        self._state_timer = 0.0

    def _sfr_set_joint_state(self):
        """Set sim joint state for q_cmd replay.

        Initializes joint position to q_cmd (pre-force position, same as real
        robot before F_ext is applied) and sets PD target to q_cmd. Then F_ext
        is applied, causing the sim to deflect to q_sim — just as the real robot
        deflects from q_cmd to q_actual under the same force.

        After settling, we compare:
            - q_sim (sim deflected position) vs q_actual (real deflected position)
            - τ_sim (sim PD torque) vs I_real (real motor current)
        """
        idx = self._sfr_current_idx
        q_cmd = self._sfr_real_data["q_cmd"][idx]  # (20,) position command (pre-force)

        q_cmd_tensor = torch.from_numpy(q_cmd).float().to(self.device).unsqueeze(0)
        q_cmd_tensor = q_cmd_tensor.expand(self.num_envs, -1)
        vel = torch.zeros_like(q_cmd_tensor)

        # Set initial joint state to q_cmd (pre-force position, same starting point as real)
        self._tesollo_hand.write_joint_state_to_sim(q_cmd_tensor, vel)

        # Set PD target to q_cmd (same controller target as real robot)
        self._hand_target_pos = q_cmd_tensor.clone()

    def _sfr_apply_external_force(self):
        """Apply real F_ext directly to fingertip body."""
        idx = self._sfr_current_idx
        F_ext = self._sfr_real_data["F_ext"][idx]  # (6,)
        finger_int_idx = int(self._sfr_real_data["finger_idx"][idx])

        finger_name = self.finger_names[finger_int_idx]
        if finger_name not in self._fingertip_body_indices:
            return

        body_idx = self._fingertip_body_indices[finger_name]

        # Prepare force/torque tensors for all bodies
        forces = torch.zeros(
            self.num_envs, self._tesollo_hand.num_bodies, 3, device=self.device,
        )
        torques = torch.zeros_like(forces)

        # Apply F_ext [Fx, Fy, Fz, Tx, Ty, Tz]
        F_tensor = torch.from_numpy(F_ext).float().to(self.device)
        forces[:, body_idx, :] = F_tensor[:3]
        torques[:, body_idx, :] = F_tensor[3:]

        self._tesollo_hand.set_external_force_and_torque(forces, torques)

    def _sfr_clear_external_force(self):
        """Clear any applied external forces."""
        forces = torch.zeros(
            self.num_envs, self._tesollo_hand.num_bodies, 3, device=self.device,
        )
        torques = torch.zeros_like(forces)
        self._tesollo_hand.set_external_force_and_torque(forces, torques)

    def _sfr_record_torque(self):
        """Record sim torque at current state."""
        idx = self._sfr_current_idx

        tau_sim = self._tesollo_hand.data.applied_torque.clone()
        q_sim_actual = self._tesollo_hand.data.joint_pos.clone()

        # Record internal FT sensors
        F_internal = self._get_fingertip_forces_resultant()

        # Compute deflection: how much sim settled position differs from real actual
        q_real_actual = self._sfr_real_data["q"][idx]
        q_deflection = q_sim_actual[0].cpu().numpy() - q_real_actual

        self._sfr_results["tau_sim"].append(tau_sim[0].cpu().numpy())
        self._sfr_results["q_actual"].append(q_sim_actual[0].cpu().numpy())
        self._sfr_results["q_cmd"].append(self._sfr_real_data["q_cmd"][idx])
        self._sfr_results["q_deflection"].append(q_deflection)
        self._sfr_results["F_applied"].append(self._sfr_real_data["F_ext"][idx])
        self._sfr_results["F_internal"].append(F_internal[0].cpu().numpy())
        self._sfr_results["finger_idx"].append(self._sfr_real_data["finger_idx"][idx])

    @property
    def finger_names(self) -> list[str]:
        return self.cfg.fingers_to_calibrate

    # =========================================================================
    # Calibration Protocol
    # =========================================================================

    def start_calibration(self, mode: str = "full"):
        """Start the calibration protocol.

        Args:
            mode: "full" (1A + 1B), "baseline" (1A only), "contact" (1B only).
        """
        self._current_finger_idx = 0
        self._current_trial = 0
        self._current_step = 0
        self._current_config = 0
        self._current_direction = 0
        self._state_timer = 0.0
        self._clear_all_data()

        if mode in ("full", "baseline"):
            self._phase = "1A"
            self._generate_gravity_configs()
            self._calibration_state = CalibrationState.GRAVITY_BASELINE_MOVING
            print("Starting Phase 1A: Gravity/Friction Baseline")
            print(f"  Configs: {self.cfg.num_gravity_configs}, "
                  f"Velocity sweeps: {self.cfg.num_velocity_sweeps}")
        elif mode == "contact":
            self._phase = "1B"
            self._calibration_state = CalibrationState.MOVING_TO_CONTACT_POSITION
            self._position_ft_sensor_for_direction(0)
            print("Starting Phase 1B: Contact Calibration")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"Fingers to calibrate: {self.cfg.fingers_to_calibrate}")

    def _generate_gravity_configs(self):
        """Generate random joint configurations for gravity baseline."""
        joint_limits_low = self._tesollo_hand.data.soft_joint_pos_limits[0, :, 0]
        joint_limits_high = self._tesollo_hand.data.soft_joint_pos_limits[0, :, 1]

        self._gravity_configs = []
        for _ in range(self.cfg.num_gravity_configs):
            alpha = torch.rand(20, device=self.device) * 0.8 + 0.1
            config = joint_limits_low + alpha * (joint_limits_high - joint_limits_low)
            self._gravity_configs.append(config.unsqueeze(0).expand(self.num_envs, -1))

    def _get_current_finger(self) -> str:
        return self.cfg.fingers_to_calibrate[self._current_finger_idx]

    def _get_finger_joint_indices(self, finger: str) -> list[int]:
        return self.cfg.finger_joint_indices.get(finger, [])

    # =========================================================================
    # FT Sensor Positioning
    # =========================================================================

    def _position_ft_sensor_for_finger(self, finger: str, direction_idx: int = 0):
        """Position FT sensor for finger calibration with given direction."""
        if self._ft_fixture is None:
            return

        directions = self.cfg.ft_directions
        if direction_idx >= len(directions):
            direction_idx = 0

        direction = directions[direction_idx]
        base_pos = self.cfg.ft_sensor_positions.get(finger, (0.0, 0.15, 0.05))
        offset = direction.get("offset", (0.0, 0.0, 0.0))
        pos = (
            base_pos[0] + offset[0],
            base_pos[1] + offset[1],
            base_pos[2] + offset[2],
        )

        pos_tensor = torch.tensor([pos], device=self.device).expand(self.num_envs, -1)
        rot = torch.tensor([direction.get("rot", (1.0, 0.0, 0.0, 0.0))],
                           device=self.device).expand(self.num_envs, -1)

        self._ft_fixture.write_root_pose_to_sim(
            root_pos=pos_tensor,
            root_quat=rot,
        )

    def _position_ft_sensor_for_direction(self, direction_idx: int):
        """Convenience: position FT sensor for current finger + direction."""
        finger = self._get_current_finger()
        self._position_ft_sensor_for_finger(finger, direction_idx)

    # =========================================================================
    # State Machine
    # =========================================================================

    def step(self, action: torch.Tensor | None = None) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Override step to run calibration protocol."""
        self._update_calibration_state()
        self._apply_position_control()

        dummy_action = torch.zeros(self.num_envs, 20, device=self.device)
        obs, reward, terminated, truncated, info = super().step(dummy_action)

        return obs, reward, terminated, truncated, info

    def _update_calibration_state(self):
        """Update calibration state machine."""
        self._state_timer += self.physics_dt

        if self._calibration_state == CalibrationState.IDLE:
            return

        # =================================================================
        # Sim-From-Real Mode
        # =================================================================

        elif self._calibration_state == CalibrationState.SFR_SETTING_STATE:
            # Set joint state to real q
            self._sfr_set_joint_state()
            self._calibration_state = CalibrationState.SFR_APPLYING_FORCE
            self._state_timer = 0.0

        elif self._calibration_state == CalibrationState.SFR_APPLYING_FORCE:
            # Apply F_ext to fingertip
            self._sfr_apply_external_force()
            self._calibration_state = CalibrationState.SFR_SETTLING
            self._state_timer = 0.0

        elif self._calibration_state == CalibrationState.SFR_SETTLING:
            # Wait for physics to settle (quasi-static equilibrium)
            self._sfr_apply_external_force()  # Keep applying force each step

            if self._state_timer >= self.cfg.settle_time:
                self._calibration_state = CalibrationState.SFR_RECORDING
                self._state_timer = 0.0
                self._record_counter = 0

        elif self._calibration_state == CalibrationState.SFR_RECORDING:
            # Record τ_sim (average over a few samples)
            self._sfr_apply_external_force()  # Keep applying force
            self._record_counter += 1
            self._sfr_record_torque()

            if self._record_counter >= 5:  # Average 5 samples
                self._sfr_clear_external_force()
                self._calibration_state = CalibrationState.SFR_NEXT_POINT
                self._state_timer = 0.0

        elif self._calibration_state == CalibrationState.SFR_NEXT_POINT:
            # Move to next data point
            self._sfr_current_idx += 1
            total = len(self._sfr_real_data["q"])

            if self._sfr_current_idx >= total:
                self._calibration_state = CalibrationState.COMPLETED
                print(f"Sim-from-real: Processed {total} data points")
            else:
                if self._sfr_current_idx % 50 == 0:
                    print(f"  Processing: {self._sfr_current_idx}/{total} "
                          f"({100 * self._sfr_current_idx / total:.0f}%)")
                self._calibration_state = CalibrationState.SFR_SETTING_STATE
                self._state_timer = 0.0

        # =================================================================
        # Phase 1A: Gravity/Friction Baseline
        # =================================================================

        elif self._calibration_state == CalibrationState.GRAVITY_BASELINE_MOVING:
            if self._current_config < len(self._gravity_configs):
                self._hand_target_pos = self._gravity_configs[self._current_config].clone()

            if self._state_timer >= self.cfg.gravity_settle_time:
                self._calibration_state = CalibrationState.GRAVITY_BASELINE_RECORDING
                self._state_timer = 0.0
                self._record_counter = 0

        elif self._calibration_state == CalibrationState.GRAVITY_BASELINE_RECORDING:
            self._record_counter += 1
            self._record_baseline_data()

            if self._record_counter >= self.cfg.gravity_record_samples:
                self._current_config += 1

                if self._current_config >= self.cfg.num_gravity_configs:
                    self._calibration_state = CalibrationState.GRAVITY_BASELINE_VELOCITY
                    self._state_timer = 0.0
                    self._current_config = 0
                    print("Phase 1A: Gravity recording complete, starting velocity sweep")
                else:
                    self._calibration_state = CalibrationState.GRAVITY_BASELINE_MOVING
                    self._state_timer = 0.0

        elif self._calibration_state == CalibrationState.GRAVITY_BASELINE_VELOCITY:
            self._record_baseline_data()

            t = self._state_timer
            sweep_duration = self.cfg.velocity_sweep_duration
            if t < sweep_duration:
                freq = 0.5 + self._current_config * 0.3
                amplitude = 0.3
                joint_limits_low = self._tesollo_hand.data.soft_joint_pos_limits[0, :, 0]
                joint_limits_high = self._tesollo_hand.data.soft_joint_pos_limits[0, :, 1]
                mid = (joint_limits_low + joint_limits_high) / 2
                span = (joint_limits_high - joint_limits_low) / 2
                target = mid + amplitude * span * torch.sin(
                    torch.tensor(2 * np.pi * freq * t, device=self.device)
                )
                self._hand_target_pos = target.unsqueeze(0).expand(self.num_envs, -1)
            else:
                self._current_config += 1
                self._state_timer = 0.0

                if self._current_config >= self.cfg.num_velocity_sweeps:
                    if self._phase == "1A":
                        self._calibration_state = CalibrationState.COMPLETED
                        print("Phase 1A: Baseline collection complete!")
                    else:
                        self._phase = "1B"
                        self._current_finger_idx = 0
                        self._current_config = 0
                        self._current_direction = 0
                        self._current_trial = 0
                        self._calibration_state = CalibrationState.MOVING_TO_CONTACT_POSITION
                        self._position_ft_sensor_for_direction(0)
                        print("Phase 1A complete. Starting Phase 1B: Contact Calibration")

        # =================================================================
        # Phase 1B: Contact Calibration
        # =================================================================

        elif self._calibration_state == CalibrationState.MOVING_TO_CONTACT_POSITION:
            self._update_approach_position()

            if self._state_timer >= 1.0:
                self._calibration_state = CalibrationState.PRESSING
                self._state_timer = 0.0
                self._current_step = 0

        elif self._calibration_state == CalibrationState.PRESSING:
            step_duration = self.cfg.ramp_duration / self.cfg.position_steps

            if self._state_timer >= step_duration:
                self._state_timer = 0.0
                self._current_step += 1
                self._update_press_position()

                contact_force = self._get_external_contact_force()
                if contact_force > self.cfg.max_contact_force or \
                   self._current_step >= self.cfg.position_steps:
                    self._calibration_state = CalibrationState.SETTLING
                    self._state_timer = 0.0

        elif self._calibration_state == CalibrationState.SETTLING:
            if self._state_timer >= self.cfg.settle_time:
                self._calibration_state = CalibrationState.RECORDING
                self._state_timer = 0.0
                self._record_counter = 0

        elif self._calibration_state == CalibrationState.RECORDING:
            self._record_counter += 1
            if self._record_counter % self.cfg.record_frequency == 0:
                self._record_contact_data()

            if self._record_counter >= self.cfg.contact_record_samples:
                self._calibration_state = CalibrationState.RETRACTING
                self._state_timer = 0.0

        elif self._calibration_state == CalibrationState.RETRACTING:
            self._hand_target_pos = self._hand_neutral_pos.clone()

            if self._state_timer >= 0.5:
                self._advance_contact_calibration()

        elif self._calibration_state == CalibrationState.REPOSITION_FT_SENSOR:
            if self._state_timer >= 0.5:
                self._calibration_state = CalibrationState.MOVING_TO_CONTACT_POSITION
                self._state_timer = 0.0

    def _update_approach_position(self):
        """Set hand to approach position before contact."""
        finger = self._get_current_finger()
        joint_indices = self._get_finger_joint_indices(finger)
        if not joint_indices:
            return

        joint_limits_low = self._tesollo_hand.data.soft_joint_pos_limits[0, :, 0]
        joint_limits_high = self._tesollo_hand.data.soft_joint_pos_limits[0, :, 1]

        alpha = self.cfg.press_start_position + self._current_config * 0.1

        for j_idx in joint_indices:
            j_low = joint_limits_low[j_idx]
            j_high = joint_limits_high[j_idx]
            self._hand_target_pos[:, j_idx] = j_low + alpha * (j_high - j_low)

    def _update_press_position(self):
        """Update hand target position for current press step."""
        finger = self._get_current_finger()
        joint_indices = self._get_finger_joint_indices(finger)
        if not joint_indices:
            return

        progress = self._current_step / self.cfg.position_steps
        start_pos = self.cfg.press_start_position + self._current_config * 0.1
        end_pos = self.cfg.press_end_position
        target_normalized = start_pos + progress * (end_pos - start_pos)

        joint_limits_low = self._tesollo_hand.data.soft_joint_pos_limits[0, :, 0]
        joint_limits_high = self._tesollo_hand.data.soft_joint_pos_limits[0, :, 1]

        for j_idx in joint_indices:
            j_low = joint_limits_low[j_idx]
            j_high = joint_limits_high[j_idx]
            self._hand_target_pos[:, j_idx] = j_low + target_normalized * (j_high - j_low)

    def _advance_contact_calibration(self):
        """Advance to next contact calibration condition."""
        self._current_trial += 1
        total_trials = self.cfg.num_contact_configs * self.cfg.num_force_directions

        if self._current_trial >= total_trials:
            self._current_trial = 0
            self._current_config = 0
            self._current_direction = 0
            self._current_finger_idx += 1

            if self._current_finger_idx >= len(self.cfg.fingers_to_calibrate):
                self._calibration_state = CalibrationState.COMPLETED
                print("Phase 1B: Contact calibration complete!")
                return

            finger = self._get_current_finger()
            print(f"Moving to finger: {finger}")
        else:
            self._current_config = self._current_trial // self.cfg.num_force_directions
            new_direction = self._current_trial % self.cfg.num_force_directions

            if new_direction != self._current_direction:
                self._current_direction = new_direction
                self._position_ft_sensor_for_direction(self._current_direction)
                self._calibration_state = CalibrationState.REPOSITION_FT_SENSOR
                self._state_timer = 0.0
                return

        self._hand_target_pos = self._hand_neutral_pos.clone()
        self._current_step = 0
        self._position_ft_sensor_for_direction(self._current_direction)
        self._calibration_state = CalibrationState.MOVING_TO_CONTACT_POSITION
        self._state_timer = 0.0

    # =========================================================================
    # Position Control
    # =========================================================================

    def _apply_position_control(self):
        """Apply position control to UR5 and hand."""
        if self._ur5e_arm is not None and self._ur5_target_pos is not None:
            self._ur5e_arm.set_joint_position_target(self._ur5_target_pos)

        if self._hand_target_pos is not None:
            self._tesollo_hand.set_joint_position_target(self._hand_target_pos)

    def _apply_action(self):
        """Override to use position control."""
        pass

    # =========================================================================
    # Force Measurement
    # =========================================================================

    def _get_external_contact_force(self) -> float:
        """Get contact force magnitude with external FT sensor."""
        if self._ft_fixture is None:
            return 0.0

        ft_forces = self._get_fingertip_forces_resultant()
        finger = self._get_current_finger()
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        if finger not in finger_names:
            return 0.0

        finger_idx = finger_names.index(finger)
        force_start = finger_idx * 6
        force = ft_forces[0, force_start:force_start + 3]
        return force.norm().item()

    def _get_external_ft_reading(self) -> torch.Tensor:
        """Get external FT sensor 6D wrench reading.

        Returns:
            6D wrench (num_envs, 6): [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        ft_forces = self._get_fingertip_forces_resultant()
        finger = self._get_current_finger()
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        finger_idx = finger_names.index(finger)

        start_idx = finger_idx * 6
        return ft_forces[:, start_idx:start_idx + 6]

    # =========================================================================
    # Data Recording
    # =========================================================================

    def _record_baseline_data(self):
        """Record Phase 1A baseline data point."""
        q_actual = self._tesollo_hand.data.joint_pos.clone()
        qdot = self._tesollo_hand.data.joint_vel.clone()
        tau_applied = self._tesollo_hand.data.applied_torque.clone()
        gravity_torques = self._get_gravity_torques()
        F_internal = self._get_fingertip_forces_resultant()

        self._baseline_data["q_positions"].append(q_actual.cpu().numpy())
        self._baseline_data["qdot"].append(qdot.cpu().numpy())
        self._baseline_data["tau_applied"].append(tau_applied.cpu().numpy())
        self._baseline_data["gravity_torques"].append(gravity_torques.cpu().numpy())
        self._baseline_data["F_internal"].append(F_internal.cpu().numpy())

        finger_idx = self._current_finger_idx if self._phase == "1A" else -1
        self._baseline_data["finger_idx"].append(finger_idx)

    def _record_contact_data(self):
        """Record Phase 1B contact calibration data point."""
        finger = self._get_current_finger()

        q_actual = self._tesollo_hand.data.joint_pos.clone()
        qdot = self._tesollo_hand.data.joint_vel.clone()
        tau_applied = self._tesollo_hand.data.applied_torque.clone()
        F_ext = self._get_external_ft_reading()

        # Get Jacobian for current finger
        jacobian = self._get_fingertip_jacobian(finger)

        # Get internal FT sensors (always record)
        F_internal = self._get_fingertip_forces_resultant()

        self._contact_data["q"].append(q_actual.cpu().numpy())
        self._contact_data["qdot"].append(qdot.cpu().numpy())
        self._contact_data["tau_applied"].append(tau_applied.cpu().numpy())
        self._contact_data["F_ext"].append(F_ext.cpu().numpy())
        self._contact_data["F_internal"].append(F_internal.cpu().numpy())
        self._contact_data["jacobian"].append(jacobian.cpu().numpy())
        self._contact_data["finger_idx"].append(self._current_finger_idx)
        self._contact_data["config_idx"].append(self._current_config)
        self._contact_data["direction_idx"].append(self._current_direction)

    def _clear_all_data(self):
        """Clear all calibration data."""
        for key in self._baseline_data:
            self._baseline_data[key] = []
        for key in self._contact_data:
            self._contact_data[key] = []

    # =========================================================================
    # Data Export
    # =========================================================================

    def get_baseline_data(self) -> dict[str, np.ndarray]:
        """Get Phase 1A baseline data as numpy arrays."""
        return {
            key: np.array(value) if value else np.array([])
            for key, value in self._baseline_data.items()
        }

    def get_contact_data(self) -> dict[str, np.ndarray]:
        """Get Phase 1B contact data as numpy arrays."""
        return {
            key: np.array(value) if value else np.array([])
            for key, value in self._contact_data.items()
        }

    def get_sfr_results(self) -> dict[str, np.ndarray]:
        """Get sim-from-real results as numpy arrays."""
        return {
            key: np.array(value) if value else np.array([])
            for key, value in self._sfr_results.items()
        }

    def export_calibration_data(self, filepath: str):
        """Export all calibration data to file."""
        from pathlib import Path
        base = Path(filepath).parent
        stem = Path(filepath).stem

        # Export baseline data
        baseline = self.get_baseline_data()
        if len(baseline["q_positions"]) > 0:
            baseline_path = base / f"{stem}_baseline.npz"
            np.savez(str(baseline_path), **baseline)
            print(f"Exported {len(baseline['q_positions'])} baseline samples to {baseline_path}")

        # Export contact data
        contact = self.get_contact_data()
        if len(contact["q"]) > 0:
            contact_path = base / f"{stem}_contact.npz"
            np.savez(str(contact_path), **contact)
            print(f"Exported {len(contact['q'])} contact samples to {contact_path}")

        # Export combined for backward compatibility
        combined = {}
        combined.update({f"baseline_{k}": v for k, v in baseline.items() if len(v) > 0})
        combined.update({f"contact_{k}": v for k, v in contact.items() if len(v) > 0})
        if combined:
            np.savez(filepath, **combined)
            print(f"Exported combined data to {filepath}")

    def export_sfr_data(self, filepath: str):
        """Export sim-from-real results (q_cmd replay mode).

        Output keys:
            tau_sim: (N, 20) - sim torques at each real condition
            q_actual: (N, 20) - actual sim joint positions after settling
            q_cmd: (N, 20) - position command (PD target)
            q_deflection: (N, 20) - q_sim_settled - q_real_actual
            F_applied: (N, 6) - applied external force
            F_internal: (N, 30) - internal FT sensor readings
            finger_idx: (N,) - finger index
        """
        results = self.get_sfr_results()

        # Average over recording samples for each data point
        # Results are stored as repeated entries per point (5 samples per point)
        N_raw = len(results["tau_sim"])
        samples_per_point = 5
        N_points = N_raw // samples_per_point

        if N_points == 0:
            print("Warning: No sim-from-real data to export")
            return

        averaged = {}
        for key in ["tau_sim", "q_actual", "q_deflection", "F_internal"]:
            data = results[key][:N_points * samples_per_point]
            reshaped = data.reshape(N_points, samples_per_point, -1)
            averaged[key] = reshaped.mean(axis=1)

        # Constant fields within each point (take first sample)
        averaged["q_cmd"] = results["q_cmd"][::samples_per_point][:N_points]
        averaged["F_applied"] = results["F_applied"][::samples_per_point][:N_points]
        averaged["finger_idx"] = results["finger_idx"][::samples_per_point][:N_points]

        np.savez(filepath, **averaged)

        # Print diagnostics
        mean_deflection = np.abs(averaged["q_deflection"]).mean()
        print(f"Exported {N_points} sim-from-real data points to {filepath}")
        print(f"  Mean |q_sim_settled - q_real_actual|: {mean_deflection:.4f} rad")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def calibration_progress(self) -> float:
        """Get calibration progress (0.0 to 1.0)."""
        if self._calibration_state == CalibrationState.COMPLETED:
            return 1.0

        if self._phase == "sfr":
            if self._sfr_real_data is None:
                return 0.0
            total = len(self._sfr_real_data["q"])
            return self._sfr_current_idx / total if total > 0 else 0.0

        if self._phase == "1A":
            total = self.cfg.num_gravity_configs + self.cfg.num_velocity_sweeps
            completed = self._current_config
            if self._calibration_state == CalibrationState.GRAVITY_BASELINE_VELOCITY:
                completed = self.cfg.num_gravity_configs + self._current_config
            phase_1a_progress = completed / total if total > 0 else 0.0

            if self._phase == "1A":
                return phase_1a_progress
            else:
                return phase_1a_progress * 0.5

        elif self._phase == "1B":
            total_fingers = len(self.cfg.fingers_to_calibrate)
            total_trials = self.cfg.num_contact_configs * self.cfg.num_force_directions
            completed = self._current_finger_idx * total_trials + self._current_trial
            total = total_fingers * total_trials
            phase_1b_progress = completed / total if total > 0 else 0.0

            return 0.5 + phase_1b_progress * 0.5

        return 0.0

    @property
    def is_calibration_complete(self) -> bool:
        return self._calibration_state == CalibrationState.COMPLETED

    # =========================================================================
    # Results
    # =========================================================================

    def get_result(self) -> dict:
        """Get calibration result for YAML export."""
        from ...data.storage import Phase1Result

        baseline = self.get_baseline_data()
        contact = self.get_contact_data()

        num_baseline = len(baseline["q_positions"]) if len(baseline["q_positions"]) > 0 else 0
        num_contact = len(contact["q"]) if len(contact["q"]) > 0 else 0

        # Include SFR data if available
        sfr = self.get_sfr_results()
        num_sfr = len(sfr["tau_sim"]) if len(sfr["tau_sim"]) > 0 else 0

        if num_baseline == 0 and num_contact == 0 and num_sfr == 0:
            return Phase1Result(
                phase=1,
                parameters={},
                loss_history=[],
                validation_metrics={},
                metadata={"num_measurements": 0},
            )

        per_joint = {}
        for finger_idx, finger in enumerate(self.cfg.fingers_to_calibrate):
            joint_indices = self._get_finger_joint_indices(finger)

            if num_contact > 0:
                mask = contact["finger_idx"] == finger_idx
                if isinstance(mask, np.ndarray) and mask.any():
                    per_joint[finger] = {
                        "num_measurements": int(mask.sum()),
                        "joint_indices": joint_indices,
                        "has_jacobian": True,
                    }

        return Phase1Result(
            phase=1,
            parameters={"per_joint": per_joint},
            loss_history=[],
            validation_metrics={
                "num_baseline_measurements": num_baseline,
                "num_contact_measurements": num_contact,
                "num_sfr_measurements": num_sfr,
                "fingers_calibrated": self.cfg.fingers_to_calibrate,
            },
            metadata={
                "method": "paired_sim_real",
                "num_baseline_measurements": num_baseline,
                "num_contact_measurements": num_contact,
                "num_sfr_measurements": num_sfr,
            },
        )
