#!/usr/bin/env python3
# Copyright (c) 2025, SRBL
# Gello_HD Data Replay Script

"""Gello_HD 데이터를 World0.usd 시뮬레이션에서 재생.

gello_HD/Real_data에서 기록된 UR5e + Tesollo 데이터를
World0.usd (UR5e + Tesollo 결합 asset)에서 재생합니다.

Usage:
    ./isaaclab.sh -p scripts/replay_gello_data.py --session_dir gello_HD/Real_data/0202_152340
    ./isaaclab.sh -p scripts/replay_gello_data.py --session_dir gello_HD/Real_data/0202_152340 --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add Real2Sim root to path
REAL2SIM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REAL2SIM_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Replay Gello_HD data in World0.usd")

    parser.add_argument(
        "--session_dir",
        type=str,
        required=True,
        help="Path to gello_HD session directory (e.g., gello_HD/Real_data/0202_152340)",
    )
    parser.add_argument(
        "--playback_speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (default: 1.0)",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


def main():
    args = parse_args()

    # Isaac Sim 초기화
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Isaac Sim 초기화 후 import
    import torch
    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sim import SimulationContext

    # Gello trajectory loader
    from gello_HD.replay import GelloTrajectoryLoader

    print("=" * 70)
    print("Gello Data Replay (UR5e + Tesollo)")
    print("=" * 70)

    # ===== Step 1: Load Trajectory =====
    print("\n[Step 1] Loading trajectory...")

    session_path = REAL2SIM_DIR / args.session_dir
    loader = GelloTrajectoryLoader()
    traj = loader.load_session(session_path)

    print(f"  Session: {session_path}")
    print(f"  Frames: {traj.num_frames}")
    print(f"  Duration: {traj.duration:.2f} s")
    print(f"  Avg dt: {traj.avg_dt * 1000:.2f} ms ({1.0 / traj.avg_dt:.1f} Hz)")

    # Print initial joint values
    print("\n  Initial UR5e joints (rad):")
    print(f"    {traj.ur5e_joint_pos[0]}")
    print("\n  Initial Index finger joints (rad):")
    print(f"    {traj.index_joint_pos[0]}")

    # ===== Step 2: Create Simulation Context =====
    print("\n[Step 2] Creating Simulation Context...")

    sim_cfg = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = SimulationContext(sim_cfg)
    physics_dt = sim_cfg.dt

    # ===== Step 3: Load World0.usd =====
    print("\n[Step 3] Loading World0.usd...")

    WORLD0_USD = str(REAL2SIM_DIR / "assets" / "robots" / "World0.usd")

    from pxr import Usd, UsdLux

    stage = sim.stage
    world_prim = stage.OverridePrim("/World")
    world_prim.GetReferences().AddReference(WORLD0_USD)

    print(f"  Asset: {WORLD0_USD}")

    # Add default lighting
    print("  Adding default lighting...")
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(500.0)

    distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant_light.CreateIntensityAttr(1500.0)
    distant_light.CreateAngleAttr(0.53)

    # Set camera view
    sim.set_camera_view(eye=[1.5, 1.5, 2], target=[0.0, 0.0, 1])

    # Prim path (UR5e + Tesollo combined articulation)
    ROBOT_PRIM_PATH = "/World/simulation_env/ur5e"

    # ===== Step 4: Create Combined Articulation =====
    print("\n[Step 4] Creating Combined Articulation (UR5e + Tesollo)...")
    print("  Using USD default drive properties")

    # Single articulation for UR5e + Tesollo (26 DOF total)
    # stiffness=None, damping=None → use USD default values
    robot_cfg = ArticulationCfg(
        prim_path=ROBOT_PRIM_PATH,
        spawn=None,  # Already in scene
        actuators={
            "ur5e_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*_joint"],  # UR5e 6 joints
                stiffness=None,  # Use USD values
                damping=None,    # Use USD values
            ),
            "tesollo_fingers": ImplicitActuatorCfg(
                joint_names_expr=["lj_dg_.*"],  # Tesollo 20 joints
                stiffness=None,
                damping=None,
            ),
        },
    )
    robot = Articulation(robot_cfg)

    # ===== Step 6: Initialize Simulation =====
    print("\n[Step 5] Initializing Simulation...")

    sim.reset()

    print(f"\n  Combined Robot:")
    print(f"    Num joints: {robot.num_joints}")
    print(f"    Joint names: {robot.joint_names}")

    # Find joint indices for UR5e and Tesollo
    ur5e_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    ur5e_indices = [robot.joint_names.index(name) for name in ur5e_joint_names]

    # Tesollo index finger joints: lj_dg_2_1, lj_dg_2_2, lj_dg_2_3, lj_dg_2_4
    tesollo_index_names = ["lj_dg_2_1", "lj_dg_2_2", "lj_dg_2_3", "lj_dg_2_4"]
    tesollo_index_indices = [robot.joint_names.index(name) for name in tesollo_index_names]

    print(f"\n  UR5e joint indices: {ur5e_indices}")
    print(f"  Tesollo index finger indices: {tesollo_index_indices}")

    # ===== Step 7: Set Initial Pose =====
    print("\n[Step 6] Setting Initial Pose...")

    # Update articulation data
    robot.update(sim.cfg.dt)

    # Print default joint positions from USD
    default_pos = robot.data.joint_pos[0].cpu().numpy()
    print(f"  Default USD joint positions (first 6 - UR5e):")
    print(f"    {default_pos[:6]}")

    # Get initial joint positions from trajectory
    ur5e_init = traj.ur5e_joint_pos[0].copy()
    finger_init = traj.index_joint_pos[0]  # Already in radians

    # Handle wrist_3_joint π offset (USD default is ~π, real data is near 0)
    # Add π to wrist_3_joint to match USD convention
    WRIST3_OFFSET = np.pi
    ur5e_init[5] += WRIST3_OFFSET
    print(f"  Target UR5e initial (with π offset on wrist_3): {ur5e_init}")
    print(f"  Target Index finger initial (rad): {finger_init}")

    # Use default joint positions as starting point, only modify what we need
    robot_target = robot.data.joint_pos.clone()

    # Set UR5e joints
    for i, idx in enumerate(ur5e_indices):
        robot_target[0, idx] = ur5e_init[i]

    # Set Tesollo index finger joints
    robot_target[0, tesollo_index_indices[1]] = finger_init[0]  # lj_dg_2_2
    robot_target[0, tesollo_index_indices[2]] = finger_init[1]  # lj_dg_2_3
    robot_target[0, tesollo_index_indices[3]] = finger_init[2]  # lj_dg_2_4

    # Gradually move to target using position control (no teleportation)
    print("\n  Moving to initial pose with position control (500 steps)...")
    for step in range(500):
        robot.set_joint_position_target(robot_target)
        robot.write_data_to_sim()
        sim.step()
        sim.render()
        robot.update(sim.cfg.dt)

        if (step + 1) % 100 == 0:
            actual = robot.data.joint_pos[0, ur5e_indices].cpu().numpy()
            error = np.abs(actual - ur5e_init).max()
            print(f"    Step {step+1}: max_error = {error:.4f} rad")

    actual = robot.data.joint_pos[0, ur5e_indices].cpu().numpy()
    print(f"\n  After settling - UR5e actual: {actual}")
    print(f"  After settling - UR5e target: {ur5e_init}")
    print(f"  After settling - Error: {np.abs(actual - ur5e_init).max():.4f} rad")

    # ===== Step 8: Replay Trajectory =====
    print("\n" + "=" * 70)
    print("[Step 7] Replaying Trajectory...")
    print("=" * 70)

    print(f"\n  Total frames: {traj.num_frames}")
    print(f"  Playback speed: {args.playback_speed}x")
    print(f"  Physics dt: {physics_dt * 1000:.2f} ms")

    sim_time = 0.0
    total_physics_steps = 0
    progress_interval = max(1, traj.num_frames // 10)  # Print every 10%
    render_interval = 8 # 120/8 = 15 FPS
    # Use current joint positions as base (don't zero other joints)
    robot_target = robot.data.joint_pos.clone()

    for frame_idx in range(traj.num_frames - 1):
        t_start = traj.timestamps[frame_idx]
        t_end = traj.timestamps[frame_idx + 1]

        # Get joint targets directly from trajectory
        ur5e_start = traj.ur5e_joint_pos[frame_idx].copy()
        ur5e_end = traj.ur5e_joint_pos[frame_idx + 1].copy()
        finger_start = traj.index_joint_pos[frame_idx]
        finger_end = traj.index_joint_pos[frame_idx + 1]

        # Apply wrist_3 offset to trajectory data
        ur5e_start[5] += WRIST3_OFFSET
        ur5e_end[5] += WRIST3_OFFSET

        # Interpolate through this frame interval
        while sim_time < t_end / args.playback_speed:
            # Calculate interpolation factor
            frame_duration = (t_end - t_start) / args.playback_speed
            if frame_duration > 0:
                alpha = (sim_time - t_start / args.playback_speed) / frame_duration
                alpha = max(0.0, min(1.0, alpha))
            else:
                alpha = 1.0

            # Interpolate targets
            ur5e_target = loader.lerp(ur5e_start, ur5e_end, alpha)
            finger_target = loader.lerp(finger_start, finger_end, alpha)

            # Update only the joints we control (keep others at current position)
            for i, idx in enumerate(ur5e_indices):
                robot_target[0, idx] = ur5e_target[i]
            robot_target[0, tesollo_index_indices[1]] = finger_target[0]  # lj_dg_2_2
            robot_target[0, tesollo_index_indices[2]] = finger_target[1]  # lj_dg_2_3
            robot_target[0, tesollo_index_indices[3]] = finger_target[2]  # lj_dg_2_4

            # Apply position control
            robot.set_joint_position_target(robot_target)
            robot.write_data_to_sim()

            # Step simulation and render
            sim.step()
            if total_physics_steps % render_interval == 0:
                sim.render()
            robot.update(sim.cfg.dt)

            sim_time += physics_dt
            total_physics_steps += 1

        # Print progress
        if (frame_idx + 1) % progress_interval == 0:
            progress = 100.0 * (frame_idx + 1) / traj.num_frames
            actual_ur5e = robot.data.joint_pos[0, ur5e_indices].cpu().numpy()
            actual_finger = robot.data.joint_pos[0, tesollo_index_indices[1:4]].cpu().numpy()
            ur5e_errors = np.abs(actual_ur5e - ur5e_end)
            finger_errors = np.abs(actual_finger - finger_end)
            print(
                f"\n  Frame {frame_idx + 1:4d}/{traj.num_frames} ({progress:5.1f}%) | sim: {sim_time:.2f}s"
            )
            print(f"    UR5e error (rad):   {ur5e_errors}")
            print(f"    Finger error (rad): {finger_errors}")

    # ===== Step 9: Summary =====
    print("\n" + "=" * 70)
    print("[Step 8] Replay Complete!")
    print("=" * 70)

    print(f"\n  Total frames replayed: {traj.num_frames}")
    print(f"  Total physics steps: {total_physics_steps}")
    print(f"  Total simulation time: {sim_time:.2f} s")
    print(f"  Real data duration: {traj.duration:.2f} s")

    # Final error check
    ur5e_final = traj.ur5e_joint_pos[-1].copy()
    ur5e_final[5] += WRIST3_OFFSET  # Apply offset for comparison
    finger_final = traj.index_joint_pos[-1]

    actual_ur5e = robot.data.joint_pos[0, ur5e_indices].cpu().numpy()
    actual_finger = robot.data.joint_pos[0, tesollo_index_indices[1:4]].cpu().numpy()

    ur5e_error = np.abs(actual_ur5e - ur5e_final).max()
    tesollo_error = np.abs(actual_finger - finger_final).max()

    print(f"\n  Final tracking errors:")
    print(f"    UR5e max error: {ur5e_error:.4f} rad ({np.degrees(ur5e_error):.2f} deg)")
    print(f"    Index finger max error: {tesollo_error:.4f} rad ({np.degrees(tesollo_error):.2f} deg)")

    # Keep window open if not headless
    if not args.headless:
        print("\n  Press Ctrl+C to exit...")
        try:
            while simulation_app.is_running():
                sim.step()
        except KeyboardInterrupt:
            pass

    simulation_app.close()


if __name__ == "__main__":
    main()
