"""Passive Force Recording — Tesollo DG-5F

모든 관절을 0도로 servo hold한 상태에서 사용자가 손가락에 힘을 가하는 동안
관절 위치 / 전류 / 속도 / FT 센서를 100Hz로 연속 기록한다.

Usage:
    python scripts/collect_passive_force.py                   # 60초 녹화
    python scripts/collect_passive_force.py --duration 30     # 30초 녹화
    python scripts/collect_passive_force.py --output my.npz   # 출력 파일 지정
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
# SDK path setup
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

# Unit conversions
MA_TO_A = 1.0 / 1000.0
FT_FORCE_SCALE = 0.1    # 0.1 N → N   (DEVELOPER mode, streaming type 0x05)
FT_TORQUE_SCALE = 0.1   # 0.1 Nm → Nm (DEVELOPER mode, streaming type 0x05)


# ---------------------------------------------------------------------------
# PassiveForceRecorder
# ---------------------------------------------------------------------------
class PassiveForceRecorder:
    """0도 hold + 연속 녹화."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.gripper: Optional[DGGripper] = None

        # Servo thread
        self._servo_running = False
        self._servo_thread: Optional[threading.Thread] = None
        self._servo_rate_hz = 100
        self._q_cmd_deg = [0.0] * NUM_JOINTS  # 항상 0도

    # =========================================================================
    # Connection / Disconnection
    # =========================================================================

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
    # Servo Thread — 0도 hold
    # =========================================================================

    def _servo_loop(self) -> None:
        """100Hz servo loop — move_servo_joint([0]*20) 연속 전송."""
        interval = 1.0 / self._servo_rate_hz
        while self._servo_running:
            t0 = time.time()
            if self.gripper is not None:
                try:
                    self.gripper.move_servo_joint(self._q_cmd_deg)
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

    # =========================================================================
    # Data Reading
    # =========================================================================

    def read_data(self) -> dict:
        """Read gripper state + FT sensor.

        Returns dict with:
            q_actual_deg: (20,) degree
            I_motor_A: (20,) Ampere
            velocity_rpm: (20,) RPM
            ft_sensor: (30,) [Fx,Fy,Fz,Tx,Ty,Tz] × 5 fingers, N/Nm
        """
        gdata = self.gripper.get_gripper_data()
        ftdata = self.gripper.get_fingertip_sensor_data()

        q_deg = np.array([float(gdata.joint[i]) for i in range(NUM_JOINTS)])
        current_mA = np.array([float(gdata.current[i]) for i in range(NUM_JOINTS)])
        velocity = np.array([float(gdata.velocity[i]) for i in range(NUM_JOINTS)])

        ft_raw = np.array([float(ftdata.forceTorque[i]) for i in range(30)])
        ft = np.zeros(30)
        for fi in range(NUM_FINGERS):
            b = fi * 6
            ft[b:b+3] = ft_raw[b:b+3] * FT_FORCE_SCALE
            ft[b+3:b+6] = ft_raw[b+3:b+6] * FT_TORQUE_SCALE

        return {
            "q_actual_deg": q_deg,
            "I_motor_A": current_mA * MA_TO_A,
            "velocity_rpm": velocity,
            "ft_sensor": ft,
        }

    # =========================================================================
    # Recording
    # =========================================================================

    def record(self) -> dict:
        """0도 hold 상태에서 duration 초 동안 연속 녹화."""
        duration = self.args.duration
        record_interval = 1.0 / self.args.record_rate

        timestamps = []
        q_cmds = []
        q_actuals = []
        I_motors = []
        velocities = []
        ft_sensors = []

        q_cmd_arr = np.zeros(NUM_JOINTS)  # 항상 0도

        print(f"\n{'='*55}")
        print(f"  Recording: {duration:.0f}s @ {self.args.record_rate}Hz")
        print(f"  All joints held at 0 deg")
        print(f"  Press Ctrl+C to stop early")
        print(f"{'='*55}\n")

        last_print = 0.0
        t_start = time.time()

        while True:
            t_loop = time.time()
            t_elapsed = t_loop - t_start

            if t_elapsed >= duration:
                break

            obs = self.read_data()

            timestamps.append(t_elapsed)
            q_cmds.append(q_cmd_arr.copy())
            q_actuals.append(obs["q_actual_deg"])
            I_motors.append(obs["I_motor_A"])
            velocities.append(obs["velocity_rpm"])
            ft_sensors.append(obs["ft_sensor"])

            # 1초마다 상태 출력
            if t_elapsed - last_print >= 1.0:
                n = len(timestamps)
                # 가장 큰 FT force magnitude 표시
                ft = obs["ft_sensor"]
                max_force = 0.0
                max_finger = ""
                for fi in range(NUM_FINGERS):
                    b = fi * 6
                    mag = np.sqrt(ft[b]**2 + ft[b+1]**2 + ft[b+2]**2)
                    if mag > max_force:
                        max_force = mag
                        max_finger = FINGER_NAMES[fi]
                print(
                    f"  [{t_elapsed:5.1f}s] {n} samples | "
                    f"max force: {max_finger} {max_force:.2f}N"
                )
                last_print = t_elapsed

            rem = record_interval - (time.time() - t_loop)
            if rem > 0:
                time.sleep(rem)

        n = len(timestamps)
        actual_rate = n / timestamps[-1] if n > 1 else 0.0
        print(f"\n  Done: {n} samples ({actual_rate:.1f} Hz avg)")

        return {
            "timestamp":     np.array(timestamps),       # (N,)
            "q_cmd_deg":     np.array(q_cmds),            # (N, 20)
            "q_actual_deg":  np.array(q_actuals),         # (N, 20)
            "I_motor_A":     np.array(I_motors),          # (N, 20)
            "velocity_rpm":  np.array(velocities),        # (N, 20)
            "ft_sensor":     np.array(ft_sensors),        # (N, 30)
        }

    # =========================================================================
    # Save
    # =========================================================================

    def save(self, data: dict) -> str:
        """Save collected data to NPZ file."""
        output = self.args.output
        if output is None:
            ts_str = datetime.now().strftime("%m%d_%H%M%S")
            out_dir = Path(_PROJECT_ROOT / "data")
            out_dir.mkdir(parents=True, exist_ok=True)
            output = str(out_dir / f"passive_force_{ts_str}.npz")

        np.savez(output, **data)

        N = len(data["timestamp"])
        total_t = data["timestamp"][-1] if N > 0 else 0.0
        print(f"\n{'='*55}")
        print(f"[SAVED] {output}")
        print(f"  Samples: {N},  Total time: {total_t:.1f} s")
        for k, v in data.items():
            shape = v.shape if hasattr(v, "shape") else "?"
            print(f"  {k:15s}: {shape}")
        print(f"{'='*55}")
        return output


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tesollo DG-5F passive force recording: 0 deg hold + continuous data capture"
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Recording duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--record_rate", type=int, default=100,
        help="Recording rate in Hz (default: 100)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output NPZ path (default: data/passive_force_MMDD_HHMMSS.npz)"
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
    recorder = PassiveForceRecorder(args)

    def signal_handler(sig, frame):
        print("\n[INTERRUPT] Stopping early...")
        recorder.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        recorder.connect()
        data = recorder.record()
        recorder.save(data)
    finally:
        recorder.disconnect()


if __name__ == "__main__":
    main()
