#!/usr/bin/env python3
# Copyright (c) 2025, SRBL
# Convert Gello_HD PKL data to LeRobot v3.0 format

"""Convert Gello_HD PKL session to LeRobot v3.0 dataset format.

LeRobot v3.0 format:
    dataset/
    ├── meta/
    │   ├── info.json
    │   ├── stats.json
    │   └── episodes/chunk-000/file-000.parquet
    ├── data/chunk-000/file-000.parquet
    └── (videos/ - optional)

Usage:
    # Auto output directory (data/lerobot/<session_name>)
    python scripts/convert_gello_to_lerobot.py \
        --session_dir gello_HD/Real_data/HD_0204

    # Custom output directory
    python scripts/convert_gello_to_lerobot.py \
        --session_dir gello_HD/Real_data/HD_0204 \
        --output_dir /custom/path
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gello_HD.replay.gello_trajectory_loader import GelloTrajectoryLoader, GelloTrajectoryData


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Gello_HD PKL session to LeRobot v3.0 format"
    )
    parser.add_argument(
        "--session_dir",
        type=str,
        required=True,
        help="Path to Gello_HD session directory containing PKL files",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="data/lerobot",
        help="Base output directory (default: data/lerobot). "
             "Output will be <output_base>/<session_name>",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override full output directory path (optional)",
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=0,
        help="Episode index for this session (default: 0)",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="ur5e_tesollo",
        help="Robot type identifier (default: ur5e_tesollo)",
    )
    return parser.parse_args()


def create_info_json(
    output_dir: Path,
    traj: GelloTrajectoryData,
    robot_type: str,
    session_name: str,
) -> dict:
    """Create meta/info.json file."""
    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "fps": round(1.0 / traj.avg_dt) if traj.avg_dt > 0 else 30,
        "total_episodes": 1,
        "total_frames": traj.num_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None,
        "features": {
            "observation.state.ur5e_joint_pos": {"dtype": "float64", "shape": [6]},
            "observation.state.ur5e_joint_vel": {"dtype": "float64", "shape": [6]},
            "observation.state.ur5e_current": {"dtype": "float64", "shape": [6]},
            "observation.state.ur5e_ee_pos": {"dtype": "float64", "shape": [3]},
            "observation.state.ur5e_ee_rot": {"dtype": "float64", "shape": [4]},
            "observation.state.index_joint_pos": {"dtype": "float64", "shape": [3]},
            "observation.state.index_joint_vel": {"dtype": "float64", "shape": [3]},
            "observation.state.index_current": {"dtype": "float64", "shape": [3]},
            "observation.sensor.fingertip_ft": {"dtype": "float64", "shape": [6]},
            "observation.state.gripper_pos": {"dtype": "float64", "shape": []},
            "action": {"dtype": "float64", "shape": [9]},
            "timestamp": {"dtype": "float64", "shape": []},
            "frame_index": {"dtype": "int64", "shape": []},
            "episode_index": {"dtype": "int64", "shape": []},
        },
        "source": {
            "session_name": session_name,
            "converted_at": datetime.now().isoformat(),
            "original_format": "gello_hd_pkl",
        },
    }

    # Write info.json
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    return info


def create_stats_json(output_dir: Path, traj: GelloTrajectoryData) -> dict:
    """Create meta/stats.json file with aggregated statistics."""

    def compute_stats(arr: np.ndarray, name: str) -> dict:
        """Compute min, max, mean, std for an array."""
        if arr.ndim == 1:
            return {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }
        else:
            # Per-dimension stats
            return {
                "min": np.min(arr, axis=0).tolist(),
                "max": np.max(arr, axis=0).tolist(),
                "mean": np.mean(arr, axis=0).tolist(),
                "std": np.std(arr, axis=0).tolist(),
            }

    # Convert action for stats computation (same as in create_data_parquet)
    action = convert_gripper_to_finger_action(traj.control)

    stats = {
        "observation.state.ur5e_joint_pos": compute_stats(traj.ur5e_joint_pos, "ur5e_joint_pos"),
        "observation.state.ur5e_joint_vel": compute_stats(traj.ur5e_joint_vel, "ur5e_joint_vel"),
        "observation.state.ur5e_current": compute_stats(traj.ur5e_current, "ur5e_current"),
        "observation.state.ur5e_ee_pos": compute_stats(traj.ur5e_ee_pos, "ur5e_ee_pos"),
        "observation.state.ur5e_ee_rot": compute_stats(traj.ur5e_ee_rot, "ur5e_ee_rot"),
        "observation.state.index_joint_pos": compute_stats(traj.index_joint_pos, "index_joint_pos"),
        "observation.state.index_joint_vel": compute_stats(traj.index_joint_vel, "index_joint_vel"),
        "observation.state.index_current": compute_stats(traj.index_current, "index_current"),
        "observation.sensor.fingertip_ft": compute_stats(traj.fingertip_ft, "fingertip_ft"),
        "observation.state.gripper_pos": compute_stats(traj.gripper_pos, "gripper_pos"),
        "action": compute_stats(action, "action"),
    }

    meta_dir = output_dir / "meta"
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def create_episodes_parquet(output_dir: Path, traj: GelloTrajectoryData, episode_index: int):
    """Create meta/episodes/chunk-000/file-000.parquet with episode metadata."""
    episodes_dir = output_dir / "meta" / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    episode_data = {
        "episode_index": [episode_index],
        "chunk_index": [0],
        "file_index": [0],
        "from_index": [0],
        "to_index": [traj.num_frames],
        "length": [traj.num_frames],
        "task_index": [0],
    }

    df = pd.DataFrame(episode_data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, episodes_dir / "file-000.parquet")


def create_tasks_parquet(output_dir: Path):
    """Create meta/tasks.parquet with task definitions."""
    meta_dir = output_dir / "meta"

    task_data = {
        "task_index": [0],
        "task": ["real2sim_calibration"],
    }

    df = pd.DataFrame(task_data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, meta_dir / "tasks.parquet")


def convert_gripper_to_finger_action(control: np.ndarray) -> np.ndarray:
    """Convert GELLO gripper normalized [0,1] to finger joint commands (radians).

    GELLO control[6] is normalized [0,1], which SRBL_Tesollo.move() converts to [0,30] degrees.
    The same command is sent to all 3 finger joints (MCP, PIP, DIP).

    Args:
        control: (N, 7) array with UR5e 6 joints + gripper normalized

    Returns:
        action: (N, 9) array with UR5e 6 joints + finger 3 joints (radians)
    """
    FINGER_LOWER_LIMIT = 0.0   # degrees
    FINGER_UPPER_LIMIT = 30.0  # degrees

    num_frames = control.shape[0]
    action = np.zeros((num_frames, 9))

    # UR5e joint targets (radians) - pass through
    action[:, 0:6] = control[:, 0:6]

    # gripper normalized [0,1] → degrees [0,30] → radians
    gripper_deg = control[:, 6] * (FINGER_UPPER_LIMIT - FINGER_LOWER_LIMIT) + FINGER_LOWER_LIMIT
    finger_cmd_rad = np.deg2rad(gripper_deg)

    # Same command for all 3 finger joints (MCP, PIP, DIP)
    action[:, 6] = finger_cmd_rad  # lj_dg_2_2 (MCP)
    action[:, 7] = finger_cmd_rad  # lj_dg_2_3 (PIP)
    action[:, 8] = finger_cmd_rad  # lj_dg_2_4 (DIP)

    return action


def create_data_parquet(output_dir: Path, traj: GelloTrajectoryData, episode_index: int):
    """Create data/chunk-000/file-000.parquet with all frame data."""
    data_dir = output_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    num_frames = traj.num_frames

    # Convert gripper normalized to finger joint commands
    action = convert_gripper_to_finger_action(traj.control)

    # Build data dictionary
    data = {
        "timestamp": traj.timestamps,
        "frame_index": np.arange(num_frames, dtype=np.int64),
        "episode_index": np.full(num_frames, episode_index, dtype=np.int64),
        "observation.state.ur5e_joint_pos": [traj.ur5e_joint_pos[i].tolist() for i in range(num_frames)],
        "observation.state.ur5e_joint_vel": [traj.ur5e_joint_vel[i].tolist() for i in range(num_frames)],
        "observation.state.ur5e_current": [traj.ur5e_current[i].tolist() for i in range(num_frames)],
        "observation.state.ur5e_ee_pos": [traj.ur5e_ee_pos[i].tolist() for i in range(num_frames)],
        "observation.state.ur5e_ee_rot": [traj.ur5e_ee_rot[i].tolist() for i in range(num_frames)],
        "observation.state.index_joint_pos": [traj.index_joint_pos[i].tolist() for i in range(num_frames)],
        "observation.state.index_joint_vel": [traj.index_joint_vel[i].tolist() for i in range(num_frames)],
        "observation.state.index_current": [traj.index_current[i].tolist() for i in range(num_frames)],
        "observation.sensor.fingertip_ft": [traj.fingertip_ft[i].tolist() for i in range(num_frames)],
        "observation.state.gripper_pos": traj.gripper_pos,
        "action": [action[i].tolist() for i in range(num_frames)],
    }

    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, data_dir / "file-000.parquet")


def main():
    args = parse_args()

    session_dir = Path(args.session_dir)
    session_name = session_dir.name

    # Auto-generate output_dir if not specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.output_base) / session_name

    print("=" * 70)
    print("Gello_HD PKL to LeRobot v3.0 Converter")
    print("=" * 70)
    print(f"  Session: {session_dir}")
    print(f"  Output: {output_dir}")

    # Load PKL data
    print("\n[Step 1] Loading PKL data...")
    loader = GelloTrajectoryLoader()
    traj = loader.load_session(session_dir)

    print(f"  Frames: {traj.num_frames}")
    print(f"  Duration: {traj.duration:.2f} s")
    print(f"  Avg dt: {traj.avg_dt * 1000:.2f} ms ({1.0 / traj.avg_dt:.1f} Hz)")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create LeRobot v3.0 structure
    print("\n[Step 2] Creating LeRobot v3.0 structure...")

    # meta/info.json
    print("  Creating meta/info.json...")
    create_info_json(output_dir, traj, args.robot_type, session_name)

    # meta/stats.json
    print("  Creating meta/stats.json...")
    create_stats_json(output_dir, traj)

    # meta/episodes/chunk-000/file-000.parquet
    print("  Creating meta/episodes/chunk-000/file-000.parquet...")
    create_episodes_parquet(output_dir, traj, args.episode_index)

    # meta/tasks.parquet
    print("  Creating meta/tasks.parquet...")
    create_tasks_parquet(output_dir)

    # data/chunk-000/file-000.parquet
    print("  Creating data/chunk-000/file-000.parquet...")
    create_data_parquet(output_dir, traj, args.episode_index)

    print("\n[Step 3] Conversion complete!")
    print(f"\n  LeRobot dataset: {output_dir}")
    print("\n  Directory structure:")
    print(f"    {output_dir}/")
    print(f"    ├── meta/")
    print(f"    │   ├── info.json")
    print(f"    │   ├── stats.json")
    print(f"    │   ├── tasks.parquet")
    print(f"    │   └── episodes/chunk-000/file-000.parquet")
    print(f"    └── data/chunk-000/file-000.parquet")

    # Print sample data and action conversion verification
    action = convert_gripper_to_finger_action(traj.control)
    print("\n  Sample data (frame 0):")
    print(f"    UR5e joint pos: {traj.ur5e_joint_pos[0]}")
    print(f"    UR5e EE pos: {traj.ur5e_ee_pos[0]}")
    print(f"    Index joint pos: {traj.index_joint_pos[0]}")

    print(f"\n  Action conversion verification:")
    print(f"    control[6] range: [{traj.control[:, 6].min():.4f}, {traj.control[:, 6].max():.4f}]")
    print(f"    action[6] range (rad): [{action[:, 6].min():.4f}, {action[:, 6].max():.4f}]")
    print(f"    action[6] range (deg): [{np.degrees(action[:, 6].min()):.2f}, {np.degrees(action[:, 6].max()):.2f}]")
    print(f"\n    Example frame 0:")
    print(f"      control[6]: {traj.control[0, 6]:.4f} → action[6:9]: {action[0, 6:9]} ({np.degrees(action[0, 6:9])} deg)")


if __name__ == "__main__":
    main()
