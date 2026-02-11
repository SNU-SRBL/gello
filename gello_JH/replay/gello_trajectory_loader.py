# Copyright (c) 2025, SRBL
# Gello_HD Trajectory Loader for Real2Sim Replay

"""Gello_HD PKL 데이터 로더.

gello_HD/Real_data에서 기록된 UR5e + Tesollo 데이터를 로드하고
시뮬레이션에서 재생할 수 있는 형식으로 변환합니다.

Usage:
    loader = GelloTrajectoryLoader()
    traj = loader.load_session("gello_HD/Real_data/0202_152340")
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle representation to quaternion (wxyz).

    Args:
        axis_angle: Rotation vector [rx, ry, rz] where magnitude is angle.

    Returns:
        Quaternion [w, x, y, z].
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = axis_angle / angle
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


@dataclass
class GelloTrajectoryData:
    """Gello_HD에서 로드된 trajectory 데이터."""

    # Timestamps (seconds from start)
    timestamps: np.ndarray  # (T,)

    # UR5e robot arm (6 DOF)
    ur5e_joint_pos: np.ndarray  # (T, 6) radians
    ur5e_joint_vel: np.ndarray  # (T, 6) rad/s
    ur5e_current: np.ndarray  # (T, 6) A

    # UR5e end-effector pose (TCP)
    ur5e_ee_pos: np.ndarray  # (T, 3) meters [x, y, z]
    ur5e_ee_rot: np.ndarray  # (T, 4) quaternion [w, x, y, z]

    # Tesollo Index finger (3 joints - abduction not recorded)
    index_joint_pos: np.ndarray  # (T, 3) radians (converted from 0.1 degree)
    index_joint_vel: np.ndarray  # (T, 3) rad/s (converted from RPM)
    index_current: np.ndarray  # (T, 3) A

    # Fingertip FT sensor
    fingertip_ft: np.ndarray  # (T, 6) [Fx, Fy, Fz, Tx, Ty, Tz]

    # Control commands
    control: np.ndarray  # (T, 7)

    # Gripper position (normalized 0-1)
    gripper_pos: np.ndarray  # (T,)

    @property
    def num_frames(self) -> int:
        """Total number of frames."""
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0.0

    @property
    def avg_dt(self) -> float:
        """Average time step in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.duration / (len(self.timestamps) - 1)


class GelloTrajectoryLoader:
    """Gello_HD PKL 세션 로더."""

    # Unit conversion constants
    DEG_TO_RAD = np.pi / 180.0  # degree → radians
    RPM_TO_RADS = 2.0 * np.pi / 60.0  # RPM → rad/s

    def load_session(self, session_dir: str | Path) -> GelloTrajectoryData:
        """세션 디렉토리의 모든 pkl 파일을 로드.

        Args:
            session_dir: gello_HD/Real_data/<session_id> 경로

        Returns:
            GelloTrajectoryData: 변환된 trajectory 데이터
        """
        session_path = Path(session_dir)
        if not session_path.exists():
            raise FileNotFoundError(f"Session directory not found: {session_path}")

        # PKL 파일 목록
        pkl_files = list(session_path.glob("*.pkl"))
        if not pkl_files:
            raise ValueError(f"No PKL files found in {session_path}")

        # 타임스탬프로 정렬
        frames = []
        for pkl_file in pkl_files:
            timestamp = self._parse_timestamp(pkl_file.name)
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            frames.append((timestamp, data))

        frames.sort(key=lambda x: x[0])

        # 데이터 변환
        return self._convert_to_trajectory(frames)

    def _parse_timestamp(self, filename: str) -> float:
        stem = filename.replace(".pkl", "")
        # 기존: dt_str = stem.replace("T", " ").replace("_", ":")
        # 수정: 공백과 언더스코어 둘 다 처리
        dt_str = stem.replace("T", " ").replace("_", ":").replace(" ", ":")
        # 날짜-시간 구분자 복원 (첫 번째 공백만)
        dt_str = dt_str.replace(":", " ", 1)
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
        return dt.timestamp()

    def _convert_to_trajectory(
        self, frames: list[tuple[float, dict]]
    ) -> GelloTrajectoryData:
        """프레임 데이터를 GelloTrajectoryData로 변환.

        Args:
            frames: (timestamp, data) 튜플 리스트

        Returns:
            GelloTrajectoryData
        """
        num_frames = len(frames)

        # Initialize arrays
        timestamps = np.zeros(num_frames)
        ur5e_joint_pos = np.zeros((num_frames, 6))
        ur5e_joint_vel = np.zeros((num_frames, 6))
        ur5e_current = np.zeros((num_frames, 6))
        ur5e_ee_pos = np.zeros((num_frames, 3))
        ur5e_ee_rot = np.zeros((num_frames, 4))
        index_joint_pos = np.zeros((num_frames, 3))
        index_joint_vel = np.zeros((num_frames, 3))
        index_current = np.zeros((num_frames, 3))
        fingertip_ft = np.zeros((num_frames, 6))
        control = np.zeros((num_frames, 7))
        gripper_pos = np.zeros(num_frames)

        # First timestamp for relative time
        t0 = frames[0][0]

        for i, (ts, data) in enumerate(frames):
            # Timestamp (relative to start)
            timestamps[i] = ts - t0

            # UR5e joint positions (radians) - first 6 elements
            joint_pos = np.array(data["joint_positions"])
            ur5e_joint_pos[i] = joint_pos[:6]
            gripper_pos[i] = joint_pos[6] if len(joint_pos) > 6 else 0.0

            # UR5e joint velocities (rad/s)
            ur5e_joint_vel[i] = np.array(data["robot_velocity"])

            # UR5e motor currents (A)
            ur5e_current[i] = np.array(data["robot_current"])

            # UR5e end-effector pose (TCP)
            if "ee_pose" in data:
                ee_pose = np.array(data["ee_pose"])
                ur5e_ee_pos[i] = ee_pose[:3]  # [x, y, z] meters
                axis_angle = ee_pose[3:]  # [rx, ry, rz] rotation vector
                ur5e_ee_rot[i] = axis_angle_to_quat(axis_angle)

            # Tesollo index finger positions (convert degree → radians)
            finger_pos_raw = np.array(data["finger_positions"])
            index_joint_pos[i] = finger_pos_raw * self.DEG_TO_RAD

            # Tesollo index finger velocities (convert RPM → rad/s)
            finger_vel_raw = np.array(data["finger_velocity"])
            index_joint_vel[i] = finger_vel_raw * self.RPM_TO_RADS

            # Tesollo index finger currents (A)
            index_current[i] = np.array(data["finger_current"])

            # Fingertip FT sensor
            fingertip_ft[i] = np.array(data["fingertip_sensor"])

            # Control commands
            control[i] = np.array(data["control"])

        return GelloTrajectoryData(
            timestamps=timestamps,
            ur5e_joint_pos=ur5e_joint_pos,
            ur5e_joint_vel=ur5e_joint_vel,
            ur5e_current=ur5e_current,
            ur5e_ee_pos=ur5e_ee_pos,
            ur5e_ee_rot=ur5e_ee_rot,
            index_joint_pos=index_joint_pos,
            index_joint_vel=index_joint_vel,
            index_current=index_current,
            fingertip_ft=fingertip_ft,
            control=control,
            gripper_pos=gripper_pos,
        )

    def get_sim_joint_targets(
        self, traj: GelloTrajectoryData, frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """시뮬레이션용 joint targets 반환.

        Args:
            traj: GelloTrajectoryData
            frame_idx: 프레임 인덱스

        Returns:
            (ur5e_targets, tesollo_targets) 튜플
            - ur5e_targets: (6,) UR5e joint positions
            - tesollo_targets: (20,) Tesollo hand joint positions (only index finger set)
        """
        # UR5e targets (direct)
        ur5e_targets = traj.ur5e_joint_pos[frame_idx]

        # Tesollo targets (20 joints, only index finger [4:8] is set)
        tesollo_targets = np.zeros(20)
        # Index finger: lj_dg_2_1 (abduction, idx 4) = 0
        #               lj_dg_2_2, lj_dg_2_3, lj_dg_2_4 (idx 5, 6, 7)
        tesollo_targets[5] = traj.index_joint_pos[frame_idx, 0]
        tesollo_targets[6] = traj.index_joint_pos[frame_idx, 1]
        tesollo_targets[7] = traj.index_joint_pos[frame_idx, 2]

        return ur5e_targets, tesollo_targets

    @staticmethod
    def lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        """Linear interpolation.

        Args:
            a: Start value
            b: End value
            alpha: Interpolation factor [0, 1]

        Returns:
            Interpolated value
        """
        return (1.0 - alpha) * a + alpha * b
