"""Index Finger Single-Joint Sin-Wave Excitation — Tesollo DG-5F (Developer Mode)

Index finger의 PIP (motor 7, joint index 6)과 DIP (motor 8, joint index 7)에 대해
고정 주파수 sin파 excitation을 인가하면서 100Hz로 데이터를 기록한다.

collect_sysid_finger.py와 동일한 구조 (Developer 모드, move_servo_joint).

Usage:
    python scripts/collect_two_finger_sin.py
    python scripts/collect_two_finger_sin.py --amp 30 --duration 10
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
# SDK path setup (collect_sysid_finger.py와 동일)
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

INDEX_PIP = 6   # motor 7
INDEX_DIP = 7   # motor 8

FREQUENCIES_HZ = [0.25, 0.5, 1.0, 2.0]
TARGET_JOINTS = [INDEX_PIP, INDEX_DIP]

MA_TO_A = 1.0 / 1000.0
FT_FORCE_SCALE = 0.1
FT_TORQUE_SCALE = 0.1
RECORD_RATE_HZ = 100


# ---------------------------------------------------------------------------
# TwoFingerSinCollector 
# ---------------------------------------------------------------------------
class TwoFingerSinCollector:

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
            received_data_type=[1, 2, 0, 4, 5, 0],
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
        time.sleep(2.0)

        self.gripper.set_fingertip_data_zero()
        print("[OK] Tesollo DG5F initialized (DEVELOPER mode)")

    def disconnect(self) -> None:
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
        q = np.clip(q_cmd_deg.copy(), -90.0, 90.0)
        with self._servo_lock:
            self._current_q_cmd_deg = q.copy()
        self.gripper.move_servo_joint(q.tolist())

    # =========================================================================
    # Data Reading
    # =========================================================================

    def read_data(self) -> dict:
        gdata = self.gripper.get_gripper_data()
        ftdata = self.gripper.get_fingertip_sensor_data()

        q_deg = np.array([float(gdata.joint[i]) for i in range(NUM_JOINTS)])
        current_mA = np.array([float(gdata.current[i]) for i in range(NUM_JOINTS)])
        tcp = np.array([float(gdata.TCP[i]) for i in range(30)])

        ft_raw = np.array([float(ftdata.forceTorque[i]) for i in range(30)])
        ft = np.zeros(30)
        for fi in range(NUM_FINGERS):
            b = fi * 6
            ft[b:b + 3] = ft_raw[b:b + 3] * FT_FORCE_SCALE
            ft[b + 3:b + 6] = ft_raw[b + 3:b + 6] * FT_TORQUE_SCALE

        return {
            "q_actual_deg": q_deg,
            "I_motor_A": current_mA * MA_TO_A,
            "tcp": tcp,
            "ft_sensor": ft,
        }

    # =========================================================================
    # Single-Joint Sin-Wave Excitation
    # =========================================================================

    def run_single_excitation(self, joint_idx: int, freq_hz: float, amp: float) -> dict:
        duration = self.args.duration
        record_interval = 1.0 / RECORD_RATE_HZ

        joint_name = "PIP" if joint_idx == INDEX_PIP else "DIP"
        print(f"  Sin wave: joint {joint_idx} ({joint_name}), {freq_hz} Hz, ±{amp}°, {duration}s")

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

            val = amp * np.sin(2.0 * np.pi * freq_hz * t_elapsed)

            q_cmd = np.zeros(NUM_JOINTS)
            q_cmd[joint_idx] = val
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

        return {
            "timestamp": np.array(timestamps),
            "q_cmd": np.array(q_cmds),
            "q_actual": np.array(q_actuals),
            "I_motor": np.array(I_motors),
            "tcp": np.array(tcps),
            "ft_sensor": np.array(ft_sensors),
            "joint_idx": np.array(joint_idx),
            "frequency": np.array(freq_hz),
        }

    # =========================================================================
    # Main Run Loop
    # =========================================================================

    def run(self) -> None:
        ts_str = datetime.now().strftime("%m%d_%H%M%S")
        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        amps = self.args.amp  # list of amplitudes
        total_runs = len(TARGET_JOINTS) * len(FREQUENCIES_HZ) * len(amps)
        run_count = 0

        for joint_idx in TARGET_JOINTS:
            for amp in amps:
                for freq_hz in FREQUENCIES_HZ:
                    run_count += 1
                    joint_name = "PIP" if joint_idx == INDEX_PIP else "DIP"

                    print(f"\n{'=' * 55}")
                    print(f"  Run {run_count}/{total_runs}: joint {joint_idx} ({joint_name}), ±{amp}°, {freq_hz} Hz")
                    print(f"{'=' * 55}")

                    self.send_q_cmd(np.zeros(NUM_JOINTS))
                    print(f"  Settling ({self.args.settle_time}s)...")
                    time.sleep(self.args.settle_time)

                    data = self.run_single_excitation(joint_idx, freq_hz, amp)

                    # excitation 끝나면 즉시 0°로 복귀
                    self.send_q_cmd(np.zeros(NUM_JOINTS))

                    fname = f"two_finger_sin_j{joint_idx}_{joint_name}_{amp:.0f}deg_{freq_hz:.2f}Hz_{ts_str}.npz"
                    fpath = out_dir / fname
                    np.savez(str(fpath), **data)

                    N = len(data["timestamp"])
                    print(f"  [SAVED] {fpath}  ({N} samples)")

        self.send_q_cmd(np.zeros(NUM_JOINTS))
        time.sleep(1.0)

        print(f"\n{'=' * 55}")
        print(f"[DONE] {total_runs} files saved to {out_dir}/")
        print(f"{'=' * 55}")


# ---------------------------------------------------------------------------
# Args / Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--amp", type=float, nargs="+", default=[45.0, 80.0])
    parser.add_argument("--settle_time", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--tesollo_ip", type=str, default=TESOLLO_IP)
    parser.add_argument("--tesollo_port", type=int, default=TESOLLO_PORT)
    parser.add_argument("--hand_model", type=str, default="left", choices=["left", "right"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collector = TwoFingerSinCollector(args)

    def signal_handler(sig, frame):
        print("\n[INTERRUPT] Stopping...")
        collector.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        collector.connect()
        collector.run()
    finally:
        collector.disconnect()


if __name__ == "__main__":
    main()
