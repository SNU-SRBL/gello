"""Phase 1 Real Data Collection: Tesollo DG5F Hand.

Standalone script that controls the Tesollo DG5F hand directly (no UR/GELLO)
to collect current-torque calibration data for the Real2Sim pipeline.

Protocol:
  1. Send q_cmd to all 20 joints → PID holds position
  2. User manually pushes fingers → FT sensor detects contact
  3. Record (q_actual, q_cmd, I_motor, F_ext, finger_idx) per contact event
  4. Save as real_contact.npz compatible with run_phase1.py

Usage:
  python scripts/collect_phase1_real.py --num_positions 5
  python scripts/collect_phase1_real.py --position_mode per_finger --num_positions 3
  python scripts/collect_phase1_real.py --dry_run
"""

import argparse
import signal
import sys
import time
import select
import enum
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# SDK path setup (follows SRBL_Tesollo.py pattern)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(
    0, str(_PROJECT_ROOT / "gello_HD" / "gello" / "robots" / "delto_py" / "src")
)

from dgsdk import (
    DGGripper,
    GripperSystemSetting,
    GripperSetting,
    ControlMode,
    CommunicationMode,
    DGModel,
    DGResult,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TESOLLO_IP = "169.254.186.73"
TESOLLO_PORT = 502
NUM_JOINTS = 20
NUM_FINGERS = 5
JOINTS_PER_FINGER = 4
JOINT_RANGE_DEG = (0.0, 30.0)

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_JOINT_SLICES = {
    "thumb": slice(0, 4),
    "index": slice(4, 8),
    "middle": slice(8, 12),
    "ring": slice(12, 16),
    "pinky": slice(16, 20),
}
FINGER_FT_SLICES = {
    "thumb": slice(0, 6),
    "index": slice(6, 12),
    "middle": slice(12, 18),
    "ring": slice(18, 24),
    "pinky": slice(24, 30),
}

# Unit conversions
DEG2RAD = np.pi / 180.0
MA_TO_A = 1.0 / 1000.0
FT_FORCE_SCALE = 0.1    # 0.1 N → N
FT_TORQUE_SCALE = 0.001  # 1 mNm → Nm

# Default thresholds
DEFAULT_CONTACT_THRESHOLD = 0.3     # N
DEFAULT_RELEASE_THRESHOLD = 0.15    # N
DEFAULT_STEADY_VEL_THRESHOLD = 2.0  # RPM
DEFAULT_SETTLE_TIME = 1.5           # seconds
DEFAULT_POLL_RATE = 60              # Hz


class CollectionState(enum.Enum):
    MOVING_TO_POSITION = "moving"
    WAITING_FOR_CONTACT = "waiting"
    RECORDING = "recording"
    SAMPLE_COMPLETE = "sample_done"
    DONE = "done"


# ---------------------------------------------------------------------------
# TesolloPhase1Collector
# ---------------------------------------------------------------------------
class TesolloPhase1Collector:
    """Standalone Phase 1 data collector for Tesollo DG5F hand."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.gripper: Optional[DGGripper] = None
        self.collected_samples: list[dict] = []
        self._current_q_cmd_deg: Optional[np.ndarray] = None

        # Thresholds
        self.contact_threshold = args.contact_threshold
        self.release_threshold = args.release_threshold
        self.steady_vel_threshold = args.steady_state_velocity
        self.settle_time = args.settle_time
        self.poll_rate = args.poll_rate
        self.min_samples_per_contact = args.min_samples_per_contact

    # =====================================================================
    # Connection
    # =====================================================================

    def connect(self) -> None:
        """Initialize and connect to Tesollo DG5F."""
        self.gripper = DGGripper()

        system_setting = GripperSystemSetting.create(
            ip=self.args.tesollo_ip,
            port=self.args.tesollo_port,
            control_mode=ControlMode.DEVELOPER,
            communication_mode=CommunicationMode.ETHERNET,
            read_timeout=1000,
            slave_id=1,
            baudrate=115200,
        )
        result = self.gripper.set_gripper_system(system_setting)
        if result != DGResult.NONE:
            raise RuntimeError(f"System setting failed: {result.name}")

        self._connected = False

        def on_connected():
            self._connected = True
            print("[INFO] Connection callback received")

        self.gripper.on_connected(on_connected)
        self.gripper.on_disconnected(lambda: print("[INFO] Disconnected callback"))

        result = self.gripper.connect()
        if result != DGResult.NONE:
            raise RuntimeError(f"Connection failed: {result.name}")
        print("[OK] Gripper connected")

        hand_model = (
            DGModel.DG_5F_LEFT
            if self.args.hand_model == "left"
            else DGModel.DG_5F_RIGHT
        )
        gripper_setting = GripperSetting.create(
            model=hand_model,
            joint_count=NUM_JOINTS,
            finger_count=NUM_FINGERS,
            moving_inpose=0.5,
            received_data_type=[1, 2, 0, 4, 5, 0],  # JOINT, CURRENT, VELOCITY, FT
        )
        result = self.gripper.set_gripper_option(gripper_setting)
        if result != DGResult.NONE:
            raise RuntimeError(f"Gripper option failed: {result.name}")

        time.sleep(0.5)

        result = self.gripper.start()
        if result != DGResult.NONE:
            raise RuntimeError(f"System start failed: {result.name}")

        time.sleep(0.5)
        print("[OK] Tesollo DG5F initialized (DEVELOPER mode, all 20 joints)")

    def disconnect(self) -> None:
        """Safe shutdown: neutral position → stop → disconnect."""
        if self.gripper is None:
            return
        try:
            neutral = [15.0] * NUM_JOINTS
            self.gripper.move_joint_all(neutral)
            print("[INFO] Moving to neutral position...")
            time.sleep(2.0)
        except Exception as e:
            print(f"[WARN] Failed to move to neutral: {e}")
        finally:
            try:
                self.gripper.stop()
                self.gripper.disconnect()
            except Exception:
                pass
        print("[OK] Disconnected")

    # =====================================================================
    # Sensor Reading (with unit conversion)
    # =====================================================================

    def read_gripper_state(self) -> dict:
        """Read joint positions, currents, velocities in SI units.

        Returns:
            dict with keys:
                q_actual_rad: (20,) joint positions in radians
                I_motor_A: (20,) motor currents in Amperes
                velocity_rpm: (20,) joint velocities in RPM
                target_arrived: bool
        """
        data = self.gripper.get_gripper_data()
        q_deg = np.array([float(data.joint[i]) for i in range(NUM_JOINTS)])
        current_mA = np.array([float(data.current[i]) for i in range(NUM_JOINTS)])
        velocity = np.array([float(data.velocity[i]) for i in range(NUM_JOINTS)])

        return {
            "q_actual_rad": q_deg * DEG2RAD,
            "I_motor_A": current_mA * MA_TO_A,
            "velocity_rpm": velocity,
            "target_arrived": bool(data.targetArrived),
        }

    def read_ft_sensors(self) -> np.ndarray:
        """Read all 5 fingertip FT sensors in SI units.

        Returns:
            (30,) array: [Fx,Fy,Fz,Tx,Ty,Tz] × 5 fingers in N and Nm
        """
        data = self.gripper.get_fingertip_sensor_data()
        raw = np.array([float(data.forceTorque[i]) for i in range(30)])

        converted = np.zeros(30)
        for finger_i in range(NUM_FINGERS):
            base = finger_i * 6
            # Forces (0.1N → N)
            converted[base : base + 3] = raw[base : base + 3] * FT_FORCE_SCALE
            # Torques (1mNm → Nm)
            converted[base + 3 : base + 6] = raw[base + 3 : base + 6] * FT_TORQUE_SCALE
        return converted

    def zero_ft_sensors(self) -> None:
        """Zero all fingertip FT sensors (compensate gravity at current pose)."""
        self.gripper.set_fingertip_data_zero()
        time.sleep(0.3)

    # =====================================================================
    # Active Finger Detection
    # =====================================================================

    def detect_active_finger(self, ft_si: np.ndarray) -> tuple:
        """Detect which finger is being pushed from FT data.

        Args:
            ft_si: (30,) FT data in SI units

        Returns:
            (finger_idx, force_magnitude) or (-1, 0.0) if no contact
        """
        magnitudes = np.zeros(NUM_FINGERS)
        for i in range(NUM_FINGERS):
            base = i * 6
            fx, fy, fz = ft_si[base], ft_si[base + 1], ft_si[base + 2]
            magnitudes[i] = np.sqrt(fx**2 + fy**2 + fz**2)

        max_finger = int(np.argmax(magnitudes))
        max_force = magnitudes[max_finger]

        if max_force < self.contact_threshold:
            return (-1, 0.0)

        # Warn if multiple fingers have significant force
        sorted_mags = np.sort(magnitudes)[::-1]
        if sorted_mags[1] > 0.5 * sorted_mags[0] and sorted_mags[1] > self.contact_threshold:
            second_finger = int(np.argsort(magnitudes)[::-1][1])
            print(
                f"  [WARN] Multiple fingers: {FINGER_NAMES[max_finger]}={max_force:.2f}N, "
                f"{FINGER_NAMES[second_finger]}={sorted_mags[1]:.2f}N"
            )

        return (max_finger, max_force)

    # =====================================================================
    # Motion
    # =====================================================================

    def send_q_cmd(self, q_cmd_deg: np.ndarray) -> None:
        """Send position command with safety clamping."""
        q_clamped = np.clip(q_cmd_deg, JOINT_RANGE_DEG[0], JOINT_RANGE_DEG[1])
        if not np.allclose(q_clamped, q_cmd_deg):
            print(f"  [WARN] q_cmd clamped to [{JOINT_RANGE_DEG[0]}, {JOINT_RANGE_DEG[1]}] deg")
        self.gripper.move_joint_all(q_clamped.tolist())
        self._current_q_cmd_deg = q_clamped.copy()

    def wait_for_arrival(self, timeout: float = 5.0) -> bool:
        """Wait until targetArrived or timeout."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            state = self.read_gripper_state()
            if state["target_arrived"]:
                return True
            time.sleep(0.05)
        print(f"  [WARN] Target not arrived after {timeout}s, continuing anyway")
        return False

    # =====================================================================
    # q_cmd Position Generation
    # =====================================================================

    def generate_q_cmd_positions(self) -> list:
        """Generate q_cmd positions based on mode."""
        mode = self.args.position_mode
        n = self.args.num_positions

        if self.args.position_file:
            positions_deg = np.load(self.args.position_file)
            print(f"Loaded {len(positions_deg)} positions from {self.args.position_file}")
            return [positions_deg[i] for i in range(len(positions_deg))]

        if mode == "grid":
            positions = []
            for val in np.linspace(0.0, 30.0, n):
                positions.append(np.full(NUM_JOINTS, val))
            return positions

        elif mode == "per_finger":
            positions = []
            neutral = 0.0
            # Parse --fingers argument
            if self.args.fingers == "all":
                finger_indices = list(range(NUM_FINGERS))
            else:
                finger_indices = [
                    FINGER_NAMES.index(f.strip())
                    for f in self.args.fingers.split(",")
                ]
            for finger_idx in finger_indices:
                for val in np.linspace(0.0, 30.0, n):
                    q_cmd = np.full(NUM_JOINTS, neutral)
                    s = FINGER_JOINT_SLICES[FINGER_NAMES[finger_idx]]
                    q_cmd[s] = val
                    positions.append(q_cmd)
            return positions

        elif mode == "random":
            return [np.random.uniform(0.0, 30.0, size=NUM_JOINTS) for _ in range(n)]

        elif mode == "manual":
            return []  # Will be collected interactively in run_collection

        else:
            raise ValueError(f"Unknown position mode: {mode}")

    # =====================================================================
    # Non-blocking keyboard input
    # =====================================================================

    @staticmethod
    def _check_key() -> Optional[str]:
        """Non-blocking check for keyboard input (Linux)."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline().strip()
        return None

    @staticmethod
    def _prompt_q_cmd_manual() -> Optional[np.ndarray]:
        """Prompt user for manual q_cmd input."""
        print("\nEnter q_cmd (single value for all joints, or 20 comma-separated values):")
        print("  Press Enter with no input to skip, 'q' to finish collection")
        line = input("> ").strip()
        if not line:
            return None
        if line.lower() == "q":
            return None
        try:
            values = [float(x) for x in line.replace(",", " ").split()]
            if len(values) == 1:
                return np.full(NUM_JOINTS, values[0])
            elif len(values) == NUM_JOINTS:
                return np.array(values)
            else:
                print(f"  [ERROR] Expected 1 or {NUM_JOINTS} values, got {len(values)}")
                return None
        except ValueError as e:
            print(f"  [ERROR] Invalid input: {e}")
            return None

    # =====================================================================
    # Main Collection Loop
    # =====================================================================

    def run_collection(self) -> None:
        """Main data collection loop with state machine."""
        positions = self.generate_q_cmd_positions()

        if self.args.position_mode == "manual":
            self._run_manual_collection()
            return

        print(f"\n{'=' * 60}")
        print(f"Phase 1 Data Collection: {len(positions)} positions")
        print(f"Contact threshold: {self.contact_threshold}N")
        print(f"Release threshold: {self.release_threshold}N")
        print(f"Poll rate: {self.poll_rate}Hz")
        print(f"{'=' * 60}")
        print("Controls: [Enter]=next position, [q]=quit\n")

        for pos_idx, q_cmd_deg in enumerate(positions):
            print(f"\n--- Position {pos_idx + 1}/{len(positions)} ---")
            print(f"  q_cmd (deg): [{', '.join(f'{v:.1f}' for v in q_cmd_deg[:8])}...]")

            samples = self._run_single_position(q_cmd_deg, pos_idx)
            self.collected_samples.extend(samples)
            print(f"  Collected {len(samples)} samples at this position "
                  f"(total: {len(self.collected_samples)})")

        print(f"\n{'=' * 60}")
        print(f"Collection complete: {len(self.collected_samples)} total samples")

    def _run_manual_collection(self) -> None:
        """Interactive manual collection mode."""
        print(f"\n{'=' * 60}")
        print("Phase 1 Data Collection: MANUAL mode")
        print("Enter q_cmd values at each prompt")
        print(f"{'=' * 60}\n")

        pos_idx = 0
        while True:
            q_cmd_deg = self._prompt_q_cmd_manual()
            if q_cmd_deg is None:
                break
            print(f"\n--- Manual Position {pos_idx + 1} ---")
            samples = self._run_single_position(q_cmd_deg, pos_idx)
            self.collected_samples.extend(samples)
            print(f"  Collected {len(samples)} samples (total: {len(self.collected_samples)})")
            pos_idx += 1

    def _run_single_position(self, q_cmd_deg: np.ndarray, position_idx: int) -> list:
        """Collect data at a single q_cmd position.

        Returns:
            List of aggregated sample dicts
        """
        samples = []

        # Move to position
        self.send_q_cmd(q_cmd_deg)
        self.wait_for_arrival(timeout=5.0)
        time.sleep(self.settle_time)

        # Zero FT sensors at this position (gravity compensation)
        self.zero_ft_sensors()
        print("  FT sensors zeroed. Push any finger to record data...")

        q_cmd_rad = q_cmd_deg * DEG2RAD
        state = CollectionState.WAITING_FOR_CONTACT
        raw_contact_samples = []
        active_finger = -1
        release_start_time = None
        poll_interval = 1.0 / self.poll_rate

        while state != CollectionState.DONE:
            loop_start = time.time()

            # Read sensors
            gripper_state = self.read_gripper_state()
            ft_si = self.read_ft_sensors()
            finger_idx, force_mag = self.detect_active_finger(ft_si)

            if state == CollectionState.WAITING_FOR_CONTACT:
                # Display live status
                self._print_waiting_status(ft_si, len(samples))

                if finger_idx >= 0:
                    active_finger = finger_idx
                    raw_contact_samples = []
                    release_start_time = None
                    state = CollectionState.RECORDING
                    print(f"\n  >>> Contact detected: {FINGER_NAMES[active_finger]} "
                          f"({force_mag:.2f}N)")

                # Check for keyboard input
                key = self._check_key()
                if key is not None:
                    if key.lower() == "q":
                        state = CollectionState.DONE
                    elif key == "" or key.lower() == "n":
                        state = CollectionState.DONE

            elif state == CollectionState.RECORDING:
                # Record sample
                raw_sample = {
                    "q_actual_rad": gripper_state["q_actual_rad"].copy(),
                    "q_cmd_rad": q_cmd_rad.copy(),
                    "I_motor_A": gripper_state["I_motor_A"].copy(),
                    "F_ext_6": ft_si[active_finger * 6 : active_finger * 6 + 6].copy(),
                    "F_internal_30": ft_si.copy(),
                    "finger_idx": active_finger,
                    "force_magnitude": force_mag,
                    "velocity_rpm": gripper_state["velocity_rpm"].copy(),
                    "position_idx": position_idx,
                }
                raw_contact_samples.append(raw_sample)

                # Print recording status
                deflection = np.abs(gripper_state["q_actual_rad"] - q_cmd_rad)
                max_defl_deg = np.max(deflection) / DEG2RAD
                print(
                    f"\r  Recording: {len(raw_contact_samples)} samples, "
                    f"F={force_mag:.2f}N ({FINGER_NAMES[active_finger]}), "
                    f"max_defl={max_defl_deg:.2f}deg",
                    end="", flush=True,
                )

                # Check if force released
                if force_mag < self.release_threshold:
                    if release_start_time is None:
                        release_start_time = time.time()
                    elif time.time() - release_start_time > 0.2:
                        state = CollectionState.SAMPLE_COMPLETE
                else:
                    release_start_time = None

            elif state == CollectionState.SAMPLE_COMPLETE:
                print()  # newline after recording
                if len(raw_contact_samples) >= self.min_samples_per_contact:
                    aggregated = self._aggregate_contact(raw_contact_samples)
                    samples.append(aggregated)
                    self._print_sample_stats(aggregated, len(samples))
                else:
                    print(f"  [SKIP] Only {len(raw_contact_samples)} samples "
                          f"(min: {self.min_samples_per_contact})")

                raw_contact_samples = []
                active_finger = -1

                # Re-send q_cmd to restore position (finger doesn't return on its own)
                print("  Restoring q_cmd position...")
                self.send_q_cmd(q_cmd_deg)
                self.wait_for_arrival(timeout=3.0)
                time.sleep(0.5)
                # Re-zero FT sensors (position changed, gravity offset may differ)
                self.zero_ft_sensors()

                state = CollectionState.WAITING_FOR_CONTACT

            # Rate limiting
            elapsed = time.time() - loop_start
            if elapsed < poll_interval:
                time.sleep(poll_interval - elapsed)

        return samples

    # =====================================================================
    # Sample Aggregation
    # =====================================================================

    def _aggregate_contact(self, raw_samples: list) -> dict:
        """Aggregate raw contact samples into one data point.

        Filters to samples with force >= 50% of peak force, then averages.
        This excludes both the initial ramp-up and the release tail.
        """
        peak_force = max(s["force_magnitude"] for s in raw_samples)
        threshold = 0.5 * peak_force
        steady = [s for s in raw_samples if s["force_magnitude"] >= threshold]

        if len(steady) < 3:
            # Fallback: use all samples above release threshold
            steady = [s for s in raw_samples if s["force_magnitude"] >= self.release_threshold]
        if len(steady) < 1:
            steady = raw_samples  # last resort

        return {
            "q_actual_rad": np.mean([s["q_actual_rad"] for s in steady], axis=0),
            "q_cmd_rad": steady[0]["q_cmd_rad"],
            "I_motor_A": np.mean([s["I_motor_A"] for s in steady], axis=0),
            "F_ext_6": np.mean([s["F_ext_6"] for s in steady], axis=0),
            "F_internal_30": np.mean([s["F_internal_30"] for s in steady], axis=0),
            "finger_idx": steady[0]["finger_idx"],
            "force_magnitude": np.mean([s["force_magnitude"] for s in steady]),
            "position_idx": steady[0]["position_idx"],
        }

    # =====================================================================
    # Display
    # =====================================================================

    def _print_waiting_status(self, ft_si: np.ndarray, sample_count: int) -> None:
        """Print live FT readings while waiting for contact."""
        parts = []
        for i, name in enumerate(FINGER_NAMES):
            base = i * 6
            mag = np.sqrt(ft_si[base]**2 + ft_si[base + 1]**2 + ft_si[base + 2]**2)
            parts.append(f"{name[0].upper()}:{mag:.2f}N")
        status = " | ".join(parts)
        print(f"\r  [{sample_count} samples] FT: {status}    ", end="", flush=True)

    def _print_sample_stats(self, sample: dict, total: int) -> None:
        """Print stats for a completed sample."""
        deflection = sample["q_actual_rad"] - sample["q_cmd_rad"]
        finger_name = FINGER_NAMES[sample["finger_idx"]]
        finger_joints = FINGER_JOINT_SLICES[finger_name]
        finger_defl_deg = np.abs(deflection[finger_joints]) / DEG2RAD

        finger_current = sample["I_motor_A"][finger_joints]

        print(f"  [Sample {total}] {finger_name}: "
              f"F={sample['force_magnitude']:.2f}N, "
              f"defl={finger_defl_deg.max():.2f}deg, "
              f"I={np.abs(finger_current).max():.3f}A")

    # =====================================================================
    # Dry Run
    # =====================================================================

    def run_dry_run(self) -> None:
        """Sensor verification mode: print readings without recording."""
        print(f"\n{'=' * 60}")
        print("DRY RUN: Sensor verification mode")
        print("Press Ctrl-C to exit")
        print(f"{'=' * 60}\n")

        # Move to neutral
        neutral = np.full(NUM_JOINTS, 15.0)
        self.send_q_cmd(neutral)
        self.wait_for_arrival(timeout=5.0)
        time.sleep(1.0)
        self.zero_ft_sensors()
        print("FT sensors zeroed at neutral position\n")

        try:
            while True:
                state = self.read_gripper_state()
                ft = self.read_ft_sensors()

                # Joint positions
                q_deg = state["q_actual_rad"] / DEG2RAD
                print(f"\rJoints (deg): [{', '.join(f'{v:5.1f}' for v in q_deg)}]")

                # Currents
                I_mA = state["I_motor_A"] / MA_TO_A  # back to mA for display
                print(f"Current (mA): [{', '.join(f'{v:6.1f}' for v in I_mA)}]")

                # FT per finger
                for i, name in enumerate(FINGER_NAMES):
                    base = i * 6
                    fx, fy, fz = ft[base], ft[base + 1], ft[base + 2]
                    tx, ty, tz = ft[base + 3], ft[base + 4], ft[base + 5]
                    mag = np.sqrt(fx**2 + fy**2 + fz**2)
                    print(f"  {name:7s}: F=[{fx:6.3f}, {fy:6.3f}, {fz:6.3f}]N  "
                          f"T=[{tx:7.4f}, {ty:7.4f}, {tz:7.4f}]Nm  |F|={mag:.3f}N")

                print(f"  targetArrived: {state['target_arrived']}")
                # Move cursor up for overwrite
                print(f"\033[{NUM_FINGERS + 4}A", end="")

                time.sleep(0.1)
        except KeyboardInterrupt:
            # Move cursor down to avoid overwriting
            print(f"\n" * (NUM_FINGERS + 5))
            print("[OK] Dry run ended")

    # =====================================================================
    # Save Data
    # =====================================================================

    def save_data(self, samples: list, output_path: str) -> None:
        """Save collected samples as real_contact.npz."""
        N = len(samples)
        if N == 0:
            print("[WARN] No samples to save")
            return

        q = np.stack([s["q_actual_rad"] for s in samples])        # (N, 20)
        q_cmd = np.stack([s["q_cmd_rad"] for s in samples])       # (N, 20)
        I_motor = np.stack([s["I_motor_A"] for s in samples])     # (N, 20)
        F_ext = np.stack([s["F_ext_6"] for s in samples])         # (N, 6)
        finger_idx = np.array([s["finger_idx"] for s in samples]) # (N,)

        # Validate
        assert q.shape == (N, NUM_JOINTS), f"q shape mismatch: {q.shape}"
        assert q_cmd.shape == (N, NUM_JOINTS), f"q_cmd shape mismatch: {q_cmd.shape}"
        assert I_motor.shape == (N, NUM_JOINTS), f"I_motor shape mismatch: {I_motor.shape}"
        assert F_ext.shape == (N, 6), f"F_ext shape mismatch: {F_ext.shape}"
        assert not np.any(np.isnan(q)), "NaN in q"
        assert not np.any(np.isnan(I_motor)), "NaN in I_motor"
        assert np.all((finger_idx >= 0) & (finger_idx < NUM_FINGERS)), "Invalid finger_idx"

        save_dict = {
            "q": q,
            "q_cmd": q_cmd,
            "I_motor": I_motor,
            "F_ext": F_ext,
            "finger_idx": finger_idx,
        }

        # Optional: full FT internal data
        if "F_internal_30" in samples[0]:
            F_internal = np.stack([s["F_internal_30"] for s in samples])
            save_dict["F_internal"] = F_internal

        # Optional: config index (which q_cmd position)
        if "position_idx" in samples[0]:
            save_dict["config_idx"] = np.array([s["position_idx"] for s in samples])

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **save_dict)

        print(f"\n[OK] Saved {N} samples to {output_path}")
        print(f"  q:          {q.shape}")
        print(f"  q_cmd:      {q_cmd.shape}")
        print(f"  I_motor:    {I_motor.shape}")
        print(f"  F_ext:      {F_ext.shape}")
        print(f"  finger_idx: {finger_idx.shape}")

        # Per-finger distribution
        unique, counts = np.unique(finger_idx, return_counts=True)
        dist = {FINGER_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)}
        print(f"  Per-finger: {dist}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 Real Data Collection: Tesollo DG5F Hand",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/collect_phase1_real.py --num_positions 5
  python scripts/collect_phase1_real.py --position_mode per_finger --num_positions 3
  python scripts/collect_phase1_real.py --dry_run
  python scripts/collect_phase1_real.py --output data/real_data/phase1_contact.npz
        """,
    )

    # Output
    parser.add_argument(
        "--output", type=str, default="data/real_data/phase1_contact.npz",
        help="Output .npz file path (default: data/real_data/phase1_contact.npz)",
    )

    # Position generation
    parser.add_argument(
        "--position_mode", type=str, choices=["grid", "per_finger", "random", "manual"],
        default="grid", help="How to generate q_cmd positions",
    )
    parser.add_argument(
        "--num_positions", type=int, default=4,
        help="Number of q_cmd positions (grid/per_finger/random mode)",
    )
    parser.add_argument(
        "--position_file", type=str, default=None,
        help="Load q_cmd positions from .npy file (K, 20) in degrees",
    )
    parser.add_argument(
        "--fingers", type=str, default="all",
        help="Fingers to test in per_finger mode. Comma-separated: thumb,index,middle,ring,pinky or 'all'",
    )

    # Contact detection
    parser.add_argument(
        "--contact_threshold", type=float, default=DEFAULT_CONTACT_THRESHOLD,
        help="Force magnitude (N) to detect contact onset",
    )
    parser.add_argument(
        "--release_threshold", type=float, default=DEFAULT_RELEASE_THRESHOLD,
        help="Force magnitude (N) to detect contact release",
    )
    parser.add_argument(
        "--min_samples_per_contact", type=int, default=5,
        help="Minimum raw samples during contact to count as valid",
    )

    # Recording
    parser.add_argument(
        "--poll_rate", type=int, default=DEFAULT_POLL_RATE,
        help="Polling rate in Hz",
    )
    parser.add_argument(
        "--settle_time", type=float, default=DEFAULT_SETTLE_TIME,
        help="Wait time after move_joint_all before accepting contacts (s)",
    )
    parser.add_argument(
        "--steady_state_velocity", type=float, default=DEFAULT_STEADY_VEL_THRESHOLD,
        help="Max velocity (RPM) to consider quasi-static",
    )

    # Tesollo connection
    parser.add_argument(
        "--tesollo_ip", type=str, default=TESOLLO_IP,
        help="Tesollo gripper IP address",
    )
    parser.add_argument(
        "--tesollo_port", type=int, default=TESOLLO_PORT,
        help="Tesollo gripper port",
    )
    parser.add_argument(
        "--hand_model", type=str, default="left", choices=["left", "right"],
        help="Hand model",
    )

    # External FT sensor (optional)
    parser.add_argument(
        "--use_external_ft", action="store_true",
        help="Also record from external RFT64 FT sensor",
    )
    parser.add_argument(
        "--ft_port", type=str, default="/dev/ttyUSB2",
        help="Serial port for external FT sensor",
    )

    # Resume
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from partial .npz file",
    )

    # Modes
    parser.add_argument("--dry_run", action="store_true", help="Sensor check only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
_collector: Optional[TesolloPhase1Collector] = None
_output_path: Optional[str] = None


def _signal_handler(signum, frame):
    """Save partial data on interrupt."""
    print("\n\n[INTERRUPT] Saving collected data...")
    if _collector and _collector.collected_samples and _output_path:
        emergency_path = _output_path.replace(".npz", "_partial.npz")
        _collector.save_data(_collector.collected_samples, emergency_path)
    if _collector:
        _collector.disconnect()
    sys.exit(0)


def main():
    global _collector, _output_path

    args = parse_args()

    # Resolve output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _PROJECT_ROOT / output_path
    _output_path = str(output_path)

    # Load partial data if resuming
    resumed_samples = []
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            data = np.load(str(resume_path), allow_pickle=True)
            N = len(data["q"])
            for i in range(N):
                resumed_samples.append({
                    "q_actual_rad": data["q"][i],
                    "q_cmd_rad": data["q_cmd"][i],
                    "I_motor_A": data["I_motor"][i],
                    "F_ext_6": data["F_ext"][i],
                    "finger_idx": int(data["finger_idx"][i]),
                    "F_internal_30": data["F_internal"][i] if "F_internal" in data else np.zeros(30),
                    "position_idx": int(data["config_idx"][i]) if "config_idx" in data else 0,
                    "force_magnitude": float(np.linalg.norm(data["F_ext"][i][:3])),
                })
            print(f"[OK] Resumed {N} samples from {resume_path}")

    collector = TesolloPhase1Collector(args)
    _collector = collector

    if resumed_samples:
        collector.collected_samples = resumed_samples

    # Signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        collector.connect()

        if args.dry_run:
            collector.run_dry_run()
            return

        collector.run_collection()

        if collector.collected_samples:
            collector.save_data(collector.collected_samples, str(output_path))
        else:
            print("[WARN] No samples collected")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        # Save partial data
        if collector.collected_samples:
            emergency_path = str(output_path).replace(".npz", "_partial.npz")
            collector.save_data(collector.collected_samples, emergency_path)
    finally:
        collector.disconnect()

    # Print next steps
    if collector.collected_samples:
        print(f"\n{'=' * 60}")
        print("Next steps:")
        print(f"  1. Run sim_from_real:")
        print(f"     ./isaaclab.sh -p scripts/run_phase1.py --mode sim_from_real \\")
        print(f"         --real_contact {output_path}")
        print(f"  2. Run calibration:")
        print(f"     ./isaaclab.sh -p scripts/run_phase1.py --mode calibrate \\")
        print(f"         --sim_matched results/phase1/sim_matched.npz \\")
        print(f"         --real_contact {output_path}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
