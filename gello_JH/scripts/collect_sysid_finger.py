"""Finger Sys-ID Data Collection — Tesollo DG-5F

각 손가락에 chirp (주파수 스윕) excitation 신호를 인가하면서
control command / fingertip TCP / 관절 각도 / 전류 / FT 센서를 100Hz로 기록한다.

실행 순서 (각 그룹 --duration 초, 기본 20초 × 2 = 40초):
  Group 0: thumb  (CMC joint 0은 0–10°로 제한)
  Group 1: index + middle + ring + pinky  (동시 excitation, pinky joint 0은 -20–0°로 제한)

Abduction 관절:
  - 기본: 0° 고정 (충돌 방지)
  - --include_abduction: 별도 진폭으로 excitation 포함

Usage:
    python scripts/collect_sysid_finger.py
    python scripts/collect_sysid_finger.py --duration 30
    python scripts/collect_sysid_finger.py --include_abduction --amp_abduction 3
"""

import argparse
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# SDK path setup (follows collect_phase1_real.py pattern)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(
    0, str(_PROJECT_ROOT / "gello" / "gello_HD" / "gello" / "robots" / "delto_py" / "src")
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
TESOLLO_IP = "192.168.4.73"
TESOLLO_PORT = 502

NUM_JOINTS = 20
NUM_FINGERS = 5
JOINTS_PER_FINGER = 4

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_JOINT_SLICES = {
    "thumb":  slice(0,  4),
    "index":  slice(4,  8),
    "middle": slice(8,  12),
    "ring":   slice(12, 16),
    "pinky":  slice(16, 20),
}

JOINT_RANGE_DEG = (0.0, 30.0)           # non-thumb flexion
THUMB_JOINT_RANGE_DEG = (-30.0, 0.0)   # thumb joints that curl negatively
# Thumb joints 0,2,3: negative for flexion; joint 1: positive
THUMB_NEGATIVE_JOINTS = [0, 2, 3]
# Abduction joint indices (non-thumb):
#   index=4, middle=8, ring=12: 각 손가락의 첫 번째 관절 (joint 0)
#   pinky=17:                   새끼손가락은 두 번째 관절 (joint 1)
ABDUCTION_JOINT_INDICES = [4, 8, 12, 17]

# Per-joint limits
THUMB_ABD_MAX_DEG = 10.0      # thumb CMC (joint 0): clamped to [-10°, 0°]
PINKY_JOINT0_MAX_DEG = 20.0   # pinky joint 0 (index 16): clamped to [0°, 20°]

# Unit conversions
MA_TO_A = 1.0 / 1000.0
FT_FORCE_SCALE = 0.1    # 0.1 N → N   (DEVELOPER mode, streaming type 0x05)
FT_TORQUE_SCALE = 0.1   # 0.1 Nm → Nm (DEVELOPER mode, streaming type 0x05)

RECORD_RATE_HZ = 100    # 기록 주파수


# ---------------------------------------------------------------------------
# SysIdFingerCollector
# ---------------------------------------------------------------------------
class SysIdFingerCollector:
    """Tesollo DG-5F 손가락 Sys-ID 데이터 수집기."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.gripper: Optional[DGGripper] = None

        # Servo thread
        self._servo_running = False
        self._servo_thread: Optional[threading.Thread] = None
        self._servo_lock = threading.Lock()
        self._servo_rate_hz = 100
        self._current_q_cmd_deg: Optional[np.ndarray] = None

    # =========================================================================
    # Connection / Disconnection
    # =========================================================================

    def connect(self) -> None:
        """Initialize and connect to Tesollo DG5F (same pattern as collect_phase1_real.py)."""
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

        self.gripper.on_connected(lambda: print("[INFO] Connection callback"))
        self.gripper.on_disconnected(lambda: print("[INFO] Disconnected callback"))

        result = self.gripper.connect()
        if result != DGResult.NONE:
            raise RuntimeError(f"Connection failed: {result.name}")
        print("[OK] Gripper connected")

        hand_model = DGModel.DG_5F_LEFT if self.args.hand_model == "left" else DGModel.DG_5F_RIGHT
        gripper_setting = GripperSetting.create(
            model=hand_model,
            joint_count=NUM_JOINTS,
            finger_count=NUM_FINGERS,
            moving_inpose=0.5,
            received_data_type=[1, 2, 0, 4, 5, 0],  # JOINT, CURRENT, _, VELOCITY, FT_SENSOR, _
        )
        for i in range(NUM_JOINTS):
            gripper_setting.jointInpose[i] = 6.0
        result = self.gripper.set_gripper_option(gripper_setting)
        if result != DGResult.NONE:
            raise RuntimeError(f"Gripper option failed: {result.name}")

        time.sleep(0.5)

        result = self.gripper.start()
        if result != DGResult.NONE:
            raise RuntimeError(f"System start failed: {result.name}")

        self.gripper.set_low_pass_filter(is_used=1, alpha=0.3)
        time.sleep(0.5)

        self._start_servo()
        time.sleep(2.0)  # 안정화 대기

        self.gripper.set_fingertip_data_zero()
        print("[OK] Tesollo DG5F initialized (DEVELOPER mode)")

    def disconnect(self) -> None:
        """Safe shutdown."""
        self._servo_running = False
        if self._servo_thread is not None:
            self._servo_thread.join(timeout=1.0)
            self._servo_thread = None

        if self.gripper is None:
            return
        try:
            self.gripper.move_joint_all([0.0] * NUM_JOINTS)
            print("[INFO] Moving to neutral...")
            time.sleep(2.0)
        except Exception as e:
            print(f"[WARN] {e}")
        finally:
            try:
                self.gripper.stop()
                self.gripper.disconnect()
            except Exception:
                pass
        print("[OK] Disconnected")

    # =========================================================================
    # Servo Thread
    # =========================================================================

    def _servo_loop(self) -> None:
        """100Hz servo loop — move_servo_joint 연속 전송."""
        interval = 1.0 / self._servo_rate_hz
        while self._servo_running:
            t0 = time.time()
            with self._servo_lock:
                q = self._current_q_cmd_deg
            if q is not None and self.gripper is not None:
                try:
                    self.gripper.move_servo_joint(q.tolist())
                except Exception:
                    pass
            rem = interval - (time.time() - t0)
            if rem > 0:
                time.sleep(rem)

    def _start_servo(self) -> None:
        if not self._servo_running:
            self._servo_running = True
            self._servo_thread = threading.Thread(target=self._servo_loop, daemon=True)
            self._servo_thread.start()

    def send_q_cmd(self, q_cmd_deg: np.ndarray) -> None:
        """Update servo target with clamping and abduction locking."""
        q = q_cmd_deg.copy()

        # Allow both directions (backward bending = negative values permitted)
        q = np.clip(q, -30.0, 30.0)

        # Non-thumb abduction joints: lock at 0° unless --include_abduction
        if not self.args.include_abduction:
            q[ABDUCTION_JOINT_INDICES] = 0.0

        with self._servo_lock:
            self._current_q_cmd_deg = q.copy()
        self.gripper.move_servo_joint(q.tolist())  # 즉시 한 번 전송

    # =========================================================================
    # Data Reading
    # =========================================================================

    def read_data(self) -> dict:
        """Read gripper state + FT sensor in 2 SDK calls.

        Returns dict with:
            q_actual_deg: (20,) degree
            I_motor_A: (20,) Ampere
            tcp: (30,) [x,y,z,rx,ry,rz] × 5 fingers, mm/deg
            ft_sensor: (30,) [Fx,Fy,Fz,Tx,Ty,Tz] × 5 fingers, N/Nm
        """
        gdata = self.gripper.get_gripper_data()
        ftdata = self.gripper.get_fingertip_sensor_data()

        q_deg = np.array([float(gdata.joint[i]) for i in range(NUM_JOINTS)])
        current_mA = np.array([float(gdata.current[i]) for i in range(NUM_JOINTS)])
        tcp = np.array([float(gdata.TCP[i]) for i in range(30)])

        ft_raw = np.array([float(ftdata.forceTorque[i]) for i in range(30)])
        ft = np.zeros(30)
        for fi in range(NUM_FINGERS):
            b = fi * 6
            ft[b:b+3] = ft_raw[b:b+3] * FT_FORCE_SCALE
            ft[b+3:b+6] = ft_raw[b+3:b+6] * FT_TORQUE_SCALE

        return {
            "q_actual_deg": q_deg,
            "I_motor_A": current_mA * MA_TO_A,
            "tcp": tcp,
            "ft_sensor": ft,
        }

    # =========================================================================
    # Excitation Signal
    # =========================================================================

    @staticmethod
    def chirp_signal(t: float, T: float, f0: float, f1: float) -> float:
        """Linear chirp: sin(2π*(f0 + (f1-f0)/(2T)*t)*t)."""
        phase = 2.0 * np.pi * (f0 + (f1 - f0) / (2.0 * T) * t) * t
        return float(np.sin(phase))

    def make_q_cmd(self, finger_idx: int, flex_val: float, abd_val: float = 0.0) -> np.ndarray:
        """Build full 20-joint q_cmd for excitation of one finger.

        Abduction 관절 처리:
          - Thumb (finger 0): 모든 4관절이 굽힘에 참여 (sign convention 적용)
          - Non-thumb: 관절 0 = abd_val (include_abduction 여부에 따라)
                        관절 1,2,3 = flex_val

        Args:
            finger_idx: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky
            flex_val:   flexion 명령값 (degree, positive). 내부에서 sign 처리.
            abd_val:    abduction 명령값 (degree). include_abduction=True 시 사용.
        """
        q = np.zeros(NUM_JOINTS)

        if finger_idx == 0:
            # Thumb: joint 0,2,3 → negative for flexion, joint 1 → positive for flexion
            # Positive flex_val = flexion, negative = extension (backward) — no lower clamp
            clamped = min(flex_val, THUMB_ABD_MAX_DEG)  # clamp positive side only (CMC ≤ 10°)
            q[0] = -clamped   # CMC
            q[1] = max(flex_val, 0.0)   # MCP: positive only
            q[2] = -flex_val  # IP1
            q[3] = -flex_val  # IP2
        elif finger_idx == 4:
            # Pinky: abduction = joint 1 (index 17), flexion = joints 0,2,3
            start = FINGER_JOINT_SLICES["pinky"].start  # 16
            q[start + 0] = max(min(flex_val, 0.0), -PINKY_JOINT0_MAX_DEG)  # joint 16: limited to [-20°, 0°]
            q[start + 1] = abd_val if self.args.include_abduction else 0.0  # joint 17: abduction
            q[start + 2] = flex_val                    # joint 18: flexion
            q[start + 3] = flex_val                    # joint 19: flexion
        else:
            # Index, middle, ring: abduction = joint 0, flexion = joints 1,2,3
            s = FINGER_JOINT_SLICES[FINGER_NAMES[finger_idx]]
            start = s.start
            q[start + 0] = abd_val if self.args.include_abduction else 0.0  # abduction
            q[start + 1] = max(flex_val, 0.0)  # MCP: positive only
            q[start + 2] = flex_val  # PIP
            q[start + 3] = flex_val  # DIP

        return q

    # =========================================================================
    # Per-Finger Excitation
    # =========================================================================

    def run_finger_excitation(self, finger_idx: int) -> tuple:
        """Chirp excitation on one finger + 100Hz recording.

        Returns:
            (timestamps, q_cmds, q_actuals, I_motors, tcps, ft_sensors)
        """
        fname = FINGER_NAMES[finger_idx]
        duration = self.args.duration
        f0 = self.args.freq_start
        f1 = self.args.freq_end
        amp = self.args.amp
        center = self.args.center
        amp_abd = self.args.amp_abduction if self.args.include_abduction else 0.0

        print(
            f"  Chirp: {f0}→{f1} Hz, amp={amp}°, center={center}°"
            + (f", abd_amp={amp_abd}°" if self.args.include_abduction else "")
        )

        record_interval = 1.0 / RECORD_RATE_HZ

        timestamps = []
        q_cmds = []
        q_actuals = []
        I_motors = []
        tcps = []
        ft_sensors = []

        t_start = time.time()

        while True:
            t_loop = time.time()
            t_elapsed = t_loop - t_start

            if t_elapsed >= duration:
                break

            # Chirp value ∈ [-1, +1]
            c = self.chirp_signal(t_elapsed, duration, f0, f1)
            flex_val = center + amp * c
            abd_val = amp_abd * c  # centered at 0 for abduction

            q_cmd = self.make_q_cmd(finger_idx, flex_val, abd_val)
            self.send_q_cmd(q_cmd)

            obs = self.read_data()

            timestamps.append(t_elapsed)
            q_cmds.append(q_cmd.copy())
            q_actuals.append(obs["q_actual_deg"])
            I_motors.append(obs["I_motor_A"])
            tcps.append(obs["tcp"])
            ft_sensors.append(obs["ft_sensor"])

            rem = record_interval - (time.time() - t_loop)
            if rem > 0:
                time.sleep(rem)

        n = len(timestamps)
        actual_rate = n / timestamps[-1] if n > 1 else 0.0
        print(f"  → {n} samples ({actual_rate:.1f} Hz avg)")

        return (
            np.array(timestamps),
            np.array(q_cmds),
            np.array(q_actuals),
            np.array(I_motors),
            np.array(tcps),
            np.array(ft_sensors),
        )

    def run_group_excitation(self, finger_indices: list, group_label: str) -> tuple:
        """Chirp excitation on multiple fingers simultaneously + 100Hz recording.

        All fingers in the group receive the same chirp signal at each time step.
        q_cmd is the sum of individual make_q_cmd() outputs (non-overlapping joints).

        Returns:
            (timestamps, q_cmds, q_actuals, I_motors, tcps, ft_sensors)
        """
        duration = self.args.duration
        f0 = self.args.freq_start
        f1 = self.args.freq_end
        amp = self.args.amp
        center = self.args.center
        amp_abd = self.args.amp_abduction if self.args.include_abduction else 0.0

        names_str = "+".join(FINGER_NAMES[fi] for fi in finger_indices)
        print(
            f"  Group [{group_label}] fingers={names_str}"
            f"  |  Chirp: {f0}→{f1} Hz, amp={amp}°, center={center}°"
            + (f", abd_amp={amp_abd}°" if self.args.include_abduction else "")
        )

        record_interval = 1.0 / RECORD_RATE_HZ

        timestamps = []
        q_cmds = []
        q_actuals = []
        I_motors = []
        tcps = []
        ft_sensors = []

        t_start = time.time()

        while True:
            t_loop = time.time()
            t_elapsed = t_loop - t_start

            if t_elapsed >= duration:
                break

            c = self.chirp_signal(t_elapsed, duration, f0, f1)
            flex_val = center + amp * c
            abd_val = amp_abd * c

            # Combine q_cmd for all fingers in group (joints are non-overlapping)
            q_cmd = np.zeros(NUM_JOINTS)
            for fi in finger_indices:
                q_cmd += self.make_q_cmd(fi, flex_val, abd_val)
            self.send_q_cmd(q_cmd)

            obs = self.read_data()

            timestamps.append(t_elapsed)
            q_cmds.append(q_cmd.copy())
            q_actuals.append(obs["q_actual_deg"])
            I_motors.append(obs["I_motor_A"])
            tcps.append(obs["tcp"])
            ft_sensors.append(obs["ft_sensor"])

            rem = record_interval - (time.time() - t_loop)
            if rem > 0:
                time.sleep(rem)

        n = len(timestamps)
        actual_rate = n / timestamps[-1] if n > 1 else 0.0
        print(f"  → {n} samples ({actual_rate:.1f} Hz avg)")

        return (
            np.array(timestamps),
            np.array(q_cmds),
            np.array(q_actuals),
            np.array(I_motors),
            np.array(tcps),
            np.array(ft_sensors),
        )

    # =========================================================================
    # Main Run Loop
    # =========================================================================

    def run(self) -> dict:
        """Fixed group excitation: thumb → index+middle+ring+pinky (each --duration sec).

        Group 0: thumb                    (20 joints total, but only thumb joints excited)
        Group 1: index+middle+ring+pinky  (same chirp applied to all four simultaneously)
        """
        GROUPS = [
            ([0],          "0: thumb"),
            ([1, 2, 3, 4], "1: index+middle+ring+pinky"),
        ]

        all_ts, all_q_cmd, all_q_actual = [], [], []
        all_I, all_tcp, all_ft, all_gidx = [], [], [], []

        t_offset = 0.0  # global timestamp offset for continuity

        for gidx, (finger_indices, group_label) in enumerate(GROUPS):
            print(f"\n{'='*55}")
            print(f"  Group {group_label}  ({gidx+1}/{len(GROUPS)})")
            print(f"{'='*55}")

            # Move to neutral and settle
            self.send_q_cmd(np.zeros(NUM_JOINTS))
            print(f"  Settling ({self.args.settle_time}s)...")
            time.sleep(self.args.settle_time)

            ts, q_cmd, q_act, I, tcp, ft = self.run_group_excitation(finger_indices, group_label)

            all_ts.append(ts + t_offset)
            t_offset += ts[-1] if len(ts) > 0 else 0.0
            all_q_cmd.append(q_cmd)
            all_q_actual.append(q_act)
            all_I.append(I)
            all_tcp.append(tcp)
            all_ft.append(ft)
            all_gidx.append(np.full(len(ts), gidx, dtype=int))

        # Return to neutral
        self.send_q_cmd(np.zeros(NUM_JOINTS))
        time.sleep(1.0)

        return {
            "timestamp":  np.concatenate(all_ts),           # (N,)   sec
            "q_cmd":      np.concatenate(all_q_cmd),         # (N,20) deg
            "q_actual":   np.concatenate(all_q_actual),      # (N,20) deg
            "I_motor":    np.concatenate(all_I),             # (N,20) A
            "tcp":        np.concatenate(all_tcp),           # (N,30) mm/deg
            "ft_sensor":  np.concatenate(all_ft),            # (N,30) N/Nm
            "group_idx":  np.concatenate(all_gidx),          # (N,)   int (0=thumb,1=IMRP)
        }

    # =========================================================================
    # Save
    # =========================================================================

    def save(self, data: dict) -> str:
        """Save collected data to NPZ file."""
        output = self.args.output
        if output is None:
            ts_str = datetime.now().strftime("%m%d_%H%M%S")
            out_dir = Path("data")
            out_dir.mkdir(parents=True, exist_ok=True)
            output = str(out_dir / f"sysid_finger_{ts_str}.npz")

        np.savez(output, **data)

        N = len(data["timestamp"])
        total_t = data["timestamp"][-1] if N > 0 else 0.0
        print(f"\n{'='*55}")
        print(f"[SAVED] {output}")
        print(f"  Samples: {N},  Total time: {total_t:.1f} s")
        for k, v in data.items():
            shape = v.shape if hasattr(v, "shape") else "?"
            print(f"  {k:12s}: {shape}")
        print(f"{'='*55}")
        return output


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tesollo DG-5F finger sys-id: chirp excitation + data collection"
    )
    parser.add_argument(
        "--duration", type=float, default=20.0,
        help="Excitation duration per finger (seconds)"
    )
    parser.add_argument(
        "--freq_start", type=float, default=0.1,
        help="Chirp start frequency (Hz)"
    )
    parser.add_argument(
        "--freq_end", type=float, default=2.0,
        help="Chirp end frequency (Hz)"
    )
    parser.add_argument(
        "--amp", type=float, default=30.0,
        help="Flexion amplitude (degrees, applied around --center)"
    )
    parser.add_argument(
        "--center", type=float, default=0.0,
        help="Flexion center position (degrees)"
    )
    parser.add_argument(
        "--include_abduction", action="store_true",
        help="Also excite abduction joint (non-thumb). Default: locked at 0°."
    )
    parser.add_argument(
        "--amp_abduction", type=float, default=5.0,
        help="Abduction amplitude (degrees). Used only with --include_abduction."
    )
    parser.add_argument(
        "--settle_time", type=float, default=1.0,
        help="Settle time (seconds) after moving to neutral between fingers"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output NPZ path (default: data/sysid_finger_MMDD_HHMMSS.npz)"
    )
    parser.add_argument("--tesollo_ip", type=str, default=TESOLLO_IP)
    parser.add_argument("--tesollo_port", type=int, default=TESOLLO_PORT)
    parser.add_argument(
        "--hand_model", type=str, default="left", choices=["left", "right"]
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    collector = SysIdFingerCollector(args)

    def signal_handler(sig, frame):
        print("\n[INTERRUPT] Stopping...")
        collector.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        collector.connect()
        data = collector.run()
        collector.save(data)
    finally:
        collector.disconnect()


if __name__ == "__main__":
    main()
