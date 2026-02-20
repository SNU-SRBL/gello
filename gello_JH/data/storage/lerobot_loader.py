# Copyright (c) 2025, SRBL
# LeRobot v3.0 Dataset Loader for Real2Sim

"""LeRobot v3.0 데이터셋 로더.

LeRobot 형식의 데이터셋을 로드하고 Real2Sim calibration에
사용할 수 있는 형식으로 변환합니다.

Usage:
    loader = LeRobotLoader("data/lerobot/HD_0204")
    traj = loader.load_trajectory()  # Returns TrajectoryData

    # Or load as numpy arrays directly
    data = loader.load_as_dict()
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .trajectory_storage import TrajectoryData


@dataclass
class LeRobotDatasetInfo:
    """LeRobot 데이터셋 메타데이터."""

    robot_type: str
    fps: int
    total_episodes: int
    total_frames: int
    features: dict[str, dict]
    source: dict[str, Any]


class LeRobotLoader:
    """LeRobot v3.0 데이터셋 로더."""

    def __init__(self, dataset_dir: str | Path):
        """Initialize loader.

        Args:
            dataset_dir: Path to LeRobot dataset directory.
        """
        self.dataset_dir = Path(dataset_dir)

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        # Load metadata
        self.info = self._load_info()

    def _load_info(self) -> LeRobotDatasetInfo:
        """Load meta/info.json."""
        info_path = self.dataset_dir / "meta" / "info.json"

        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found: {info_path}")

        with open(info_path, "r") as f:
            data = json.load(f)

        return LeRobotDatasetInfo(
            robot_type=data.get("robot_type", "unknown"),
            fps=data.get("fps", 30),
            total_episodes=data.get("total_episodes", 1),
            total_frames=data.get("total_frames", 0),
            features=data.get("features", {}),
            source=data.get("source", {}),
        )

    def load_stats(self) -> dict[str, dict]:
        """Load meta/stats.json."""
        stats_path = self.dataset_dir / "meta" / "stats.json"

        if not stats_path.exists():
            return {}

        with open(stats_path, "r") as f:
            return json.load(f)

    def load_dataframe(self, episode_index: int | None = None) -> pd.DataFrame:
        """Load data as pandas DataFrame.

        Args:
            episode_index: Optional episode index to filter. If None, load all.

        Returns:
            DataFrame with all data.
        """
        data_dir = self.dataset_dir / "data"

        # Find all parquet files
        parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        # Load and concatenate
        dfs = []
        for pq_file in parquet_files:
            df = pd.read_parquet(pq_file)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        # Filter by episode if specified
        if episode_index is not None and "episode_index" in df.columns:
            df = df[df["episode_index"] == episode_index]

        return df

    def load_as_dict(self, episode_index: int | None = None) -> dict[str, np.ndarray]:
        """Load data as dictionary of numpy arrays.

        Args:
            episode_index: Optional episode index to filter.

        Returns:
            Dictionary with numpy arrays for each feature.
        """
        df = self.load_dataframe(episode_index)

        result = {}

        for col in df.columns:
            values = df[col].values

            # Convert list columns to numpy arrays
            if len(values) > 0 and isinstance(values[0], (list, np.ndarray)):
                result[col] = np.array([np.array(v) for v in values])
            else:
                result[col] = np.array(values)

        return result

    def load_trajectory(self, episode_index: int = 0) -> TrajectoryData:
        """Load data as TrajectoryData for Real2Sim.

        Args:
            episode_index: Episode index to load.

        Returns:
            TrajectoryData object compatible with Real2Sim.
        """
        data = self.load_as_dict(episode_index)

        # Map LeRobot columns to TrajectoryData fields
        timestamps = data.get("timestamp", np.array([]))

        # UR5e data
        ur5_joint_positions = data.get("observation.state.ur5e_joint_pos", np.array([]))
        ur5_joint_velocities = data.get("observation.state.ur5e_joint_vel", np.array([]))
        ur5_ee_position = data.get("observation.state.ur5e_ee_pos", np.array([]))
        ur5_ee_orientation = data.get("observation.state.ur5e_ee_rot", np.array([]))

        # Index finger data (map to hand_joint_positions indices 5,6,7)
        index_joint_pos = data.get("observation.state.index_joint_pos", np.array([]))
        index_joint_vel = data.get("observation.state.index_joint_vel", np.array([]))

        # Create hand_joint_positions with 20 joints (only index finger set)
        num_frames = len(timestamps)
        if len(index_joint_pos) > 0:
            hand_joint_positions = np.zeros((num_frames, 20))
            hand_joint_velocities = np.zeros((num_frames, 20))
            # Index finger joints at indices 5, 6, 7 (lj_dg_2_2, lj_dg_2_3, lj_dg_2_4)
            hand_joint_positions[:, 5:8] = index_joint_pos
            hand_joint_velocities[:, 5:8] = index_joint_vel
        else:
            hand_joint_positions = np.array([])
            hand_joint_velocities = np.array([])

        # Motor currents
        ur5e_current = data.get("observation.state.ur5e_current", np.array([]))
        index_current = data.get("observation.state.index_current", np.array([]))

        if len(ur5e_current) > 0 and len(index_current) > 0:
            motor_currents = np.hstack([ur5e_current, index_current])
        elif len(ur5e_current) > 0:
            motor_currents = ur5e_current
        else:
            motor_currents = np.array([])

        # FT sensor
        ft_sensor_data = {}
        fingertip_ft = data.get("observation.sensor.fingertip_ft", np.array([]))
        if len(fingertip_ft) > 0:
            ft_sensor_data["index"] = fingertip_ft

        # Actions
        actions = data.get("action", np.array([]))

        # Legacy joint_positions (combined UR5e + hand for backward compatibility)
        if len(ur5_joint_positions) > 0 and len(hand_joint_positions) > 0:
            joint_positions = np.hstack([ur5_joint_positions, hand_joint_positions])
            joint_velocities = np.hstack([ur5_joint_velocities, hand_joint_velocities])
        elif len(ur5_joint_positions) > 0:
            joint_positions = ur5_joint_positions
            joint_velocities = ur5_joint_velocities
        else:
            joint_positions = np.array([])
            joint_velocities = np.array([])

        return TrajectoryData(
            timestamps=timestamps,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            ur5_joint_positions=ur5_joint_positions,
            ur5_joint_velocities=ur5_joint_velocities,
            ur5_ee_position=ur5_ee_position,
            ur5_ee_orientation=ur5_ee_orientation,
            hand_joint_positions=hand_joint_positions,
            hand_joint_velocities=hand_joint_velocities,
            motor_currents=motor_currents,
            ft_sensor_data=ft_sensor_data,
            actions=actions,
            metadata={
                "source": "lerobot",
                "robot_type": self.info.robot_type,
                "fps": self.info.fps,
                "episode_index": episode_index,
            },
        )

    def load_index_finger_data(self, episode_index: int = 0) -> dict[str, np.ndarray]:
        """Load only index finger related data for calibration.

        Args:
            episode_index: Episode index to load.

        Returns:
            Dictionary with index finger data.
        """
        data = self.load_as_dict(episode_index)

        return {
            "timestamps": data.get("timestamp", np.array([])),
            "joint_pos": data.get("observation.state.index_joint_pos", np.array([])),
            "joint_vel": data.get("observation.state.index_joint_vel", np.array([])),
            "current": data.get("observation.state.index_current", np.array([])),
            "fingertip_ft": data.get("observation.sensor.fingertip_ft", np.array([])),
            # Include UR5e data for context
            "ur5e_joint_pos": data.get("observation.state.ur5e_joint_pos", np.array([])),
            "ur5e_ee_pos": data.get("observation.state.ur5e_ee_pos", np.array([])),
            "ur5e_ee_rot": data.get("observation.state.ur5e_ee_rot", np.array([])),
        }

    def get_episode_count(self) -> int:
        """Get number of episodes in dataset."""
        return self.info.total_episodes

    def get_frame_count(self) -> int:
        """Get total number of frames in dataset."""
        return self.info.total_frames

    def __repr__(self) -> str:
        return (
            f"LeRobotLoader(dataset_dir={self.dataset_dir}, "
            f"episodes={self.info.total_episodes}, "
            f"frames={self.info.total_frames}, "
            f"fps={self.info.fps})"
        )
