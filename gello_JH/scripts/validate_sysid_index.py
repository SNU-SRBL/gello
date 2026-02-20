#!/usr/bin/env python3
# Copyright (c) 2025, SRBL
# Validate Index Finger System Identification Results

"""Validate Index Finger SysID calibration by comparing fingertip TCP tracking.

Runs trajectory replay with and without calibrated parameters,
comparing Real vs Sim fingertip poses (translation and rotation errors).

Usage:
    # Run without calibration (baseline)
    ./isaaclab.sh -p scripts/validate_sysid_index.py \
        --trajectory_dir data/lerobot/HD_0204

    # Run with calibration
    ./isaaclab.sh -p scripts/validate_sysid_index.py \
        --trajectory_dir data/lerobot/HD_0204 \
        --calibration_file results/sysid_index/sysid_index_result.yaml
"""

from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path

# Add parent directory to path
REAL2SIM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REAL2SIM_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Index Finger SysID Calibration"
    )

    parser.add_argument(
        "--trajectory_dir",
        type=str,
        required=True,
        help="Path to LeRobot dataset directory",
    )
    parser.add_argument(
        "--calibration_file",
        type=str,
        default=None,
        help="Path to SysID calibration YAML file (optional, for comparison)",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=30.0,
        help="Max trajectory time to validate (default: 30.0 seconds)",
    )

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


def main():
    args = parse_args()

    # Launch Isaac Sim
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Now import Isaac Lab modules
    import torch
    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sim import SimulationContext

    # Import local modules
    from data.storage import LeRobotLoader
    from calibration.utils.loss_functions import simpler_sysid_loss

    # === World-to-base frame transform helpers (quaternion wxyz format) ===
    # USD model's base frame is rotated 180° around Z vs UR5e native base frame.
    RZ180 = np.array([0.0, 0.0, 0.0, 1.0])       # Rz(180°) wxyz
    RZ180_CONJ = np.array([0.0, 0.0, 0.0, -1.0])  # conj(Rz(180°))

    def quat_conjugate(q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    def quat_rotate_inverse(q, v):
        q_conj = quat_conjugate(q)
        v_quat = np.array([0.0, v[0], v[1], v[2]])
        result = quat_multiply(quat_multiply(q_conj, v_quat), q)
        return result[1:]

    def world_to_base_pos(pos_w, base_pos_w, base_quat_w):
        pos_base = quat_rotate_inverse(base_quat_w, pos_w - base_pos_w)
        return np.array([-pos_base[0], -pos_base[1], pos_base[2]])

    def world_to_base_quat(quat_w, base_quat_w):
        quat_base = quat_multiply(quat_conjugate(base_quat_w), quat_w)
        return quat_multiply(quat_multiply(RZ180, quat_base), RZ180_CONJ)

    print("=" * 70)
    print("Index Finger SysID Validation")
    print("=" * 70)

    # Load calibration if provided
    calibration = None
    if args.calibration_file:
        cal_path = Path(args.calibration_file)
        if cal_path.exists():
            with open(cal_path, "r") as f:
                calibration = yaml.safe_load(f)
            print(f"\n  Calibration file: {cal_path}")
        else:
            print(f"\n  [WARNING] Calibration file not found: {cal_path}")

    # ===== Step 1: Load Trajectory Data =====
    print("\n[Step 1] Loading LeRobot dataset...")

    trajectory_dir = Path(args.trajectory_dir)
    loader = LeRobotLoader(trajectory_dir)
    data = loader.load_as_dict(episode_index=0)

    timestamps = data["timestamp"]
    real_fingertip_pos = data["observation.state.index_fingertip_pos"]
    real_fingertip_rot = data["observation.state.index_fingertip_rot"]
    actions = data["action"]
    ur5e_joint_pos = data["observation.state.ur5e_joint_pos"]

    num_frames = len(timestamps)
    duration = timestamps[-1] - timestamps[0]
    avg_dt = duration / (num_frames - 1) if num_frames > 1 else 0

    print(f"  Dataset: {trajectory_dir}")
    print(f"  Frames: {num_frames}")
    print(f"  Duration: {duration:.2f} s")
    print(f"  Avg dt: {avg_dt * 1000:.2f} ms ({1.0 / avg_dt:.1f} Hz)")

    # Limit trajectory length
    max_frames = min(num_frames, int(args.max_time / avg_dt))
    print(f"  Using first {args.max_time}s ({max_frames} frames)")

    # ===== Step 2: Create Simulation =====
    print("\n[Step 2] Creating Simulation...")

    sim_cfg = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = SimulationContext(sim_cfg)
    physics_dt = sim_cfg.dt

    # Load World0.usd (UR5e + Tesollo combined asset)
    WORLD0_USD = str(REAL2SIM_DIR / "assets" / "robots" / "World0.usd")
    ROBOT_PRIM_PATH = "/World/simulation_env/ur5e"

    from pxr import Usd, UsdLux

    stage = sim.stage
    world_prim = stage.OverridePrim("/World")
    world_prim.GetReferences().AddReference(WORLD0_USD)

    # Add default lighting
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(600.0)

    # ===== Step 3: Create Robot =====
    print("\n[Step 3] Creating robot articulation...")

    UR5E_JOINTS = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    INDEX_FINGER_JOINTS = ["lj_dg_2_2", "lj_dg_2_3", "lj_dg_2_4"]
    FINGERTIP_BODY_NAME = "lk_dg_2_4"  # Index finger DIP link (fingertip)
    WRIST3_OFFSET = np.pi

    steps_per_action = max(1, int(avg_dt / physics_dt))

    robot_cfg = ArticulationCfg(
        prim_path=ROBOT_PRIM_PATH,
        spawn=None,
        actuators={
            "ur5e_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*_joint"],
                stiffness=None,
                damping=None,
            ),
            "tesollo_fingers": ImplicitActuatorCfg(
                joint_names_expr=["lj_dg_.*"],
                stiffness=None,
                damping=None,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    sim.reset()
    robot.update(sim.cfg.dt)

    ur5e_indices = [robot.joint_names.index(name) for name in UR5E_JOINTS]
    finger_indices = [robot.joint_names.index(name) for name in INDEX_FINGER_JOINTS]
    fingertip_body_idx = robot.find_bodies(FINGERTIP_BODY_NAME)[0][0]

    print(f"  Total joints: {robot.num_joints}")
    print(f"  Finger indices: {finger_indices}")
    print(f"  Fingertip body index: {fingertip_body_idx}")

    # Capture robot base frame (for world→UR5e base transform)
    base_pos_w = robot.data.root_pos_w[0].cpu().numpy()
    base_quat_w = robot.data.root_quat_w[0].cpu().numpy()  # wxyz

    # Initial pose
    ur5e_init = ur5e_joint_pos[0].copy()
    ur5e_init[5] += WRIST3_OFFSET

    # ===== Helper: Run trajectory =====
    def run_trajectory(apply_calibration: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Run open-loop trajectory and return fingertip poses."""
        sim.reset()
        robot.update(sim.cfg.dt)

        # Apply calibration if requested
        if apply_calibration and calibration:
            full_stiffness = robot.root_physx_view.get_dof_stiffnesses().clone()
            full_damping = robot.root_physx_view.get_dof_dampings().clone()

            stiffness = calibration["parameters"]["stiffness"]
            damping = calibration["parameters"]["damping"]

            for i, idx in enumerate(finger_indices):
                full_stiffness[0, idx] = stiffness[i]
                full_damping[0, idx] = damping[i]

            robot.write_joint_stiffness_to_sim(full_stiffness)
            robot.write_joint_damping_to_sim(full_damping)

        # Set initial pose
        robot_target = robot.data.joint_pos.clone()
        for i, idx in enumerate(ur5e_indices):
            robot_target[0, idx] = ur5e_init[i]

        for _ in range(50):
            robot.set_joint_position_target(robot_target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.cfg.dt)

        # Save settled position as base target (non-controlled joints hold here)
        init_target = robot.data.joint_pos.clone()

        # Run trajectory
        fingertip_pos_traj = np.zeros((max_frames, 3))
        fingertip_rot_traj = np.zeros((max_frames, 4))

        for t in range(max_frames):
            action = actions[t]
            robot_target = init_target.clone()

            for i, idx in enumerate(ur5e_indices):
                target_pos = action[i]
                if i == 5:
                    target_pos += WRIST3_OFFSET
                robot_target[0, idx] = target_pos

            for i, idx in enumerate(finger_indices):
                robot_target[0, idx] = action[6 + i]

            for _ in range(steps_per_action):
                robot.set_joint_position_target(robot_target)
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.cfg.dt)

            ft_pos_w = robot.data.body_pos_w[0, fingertip_body_idx, :].cpu().numpy()
            ft_rot_w = robot.data.body_quat_w[0, fingertip_body_idx, :].cpu().numpy()
            fingertip_pos_traj[t] = world_to_base_pos(ft_pos_w, base_pos_w, base_quat_w)
            fingertip_rot_traj[t] = world_to_base_quat(ft_rot_w, base_quat_w)

        return fingertip_pos_traj, fingertip_rot_traj

    # ===== Step 4: Run Baseline (no calibration) =====
    print("\n[Step 4] Running baseline (no calibration)...")
    baseline_fingertip_pos, baseline_fingertip_rot = run_trajectory(apply_calibration=False)

    baseline_loss, baseline_components = simpler_sysid_loss(
        real_ee_pos=real_fingertip_pos[:max_frames],
        sim_ee_pos=baseline_fingertip_pos,
        real_ee_rot=real_fingertip_rot[:max_frames],
        sim_ee_rot=baseline_fingertip_rot,
    )

    # Per-frame translation error (mm)
    baseline_transl_errors = np.linalg.norm(real_fingertip_pos[:max_frames] - baseline_fingertip_pos, axis=1) * 1000

    print(f"  Baseline L_transl: {baseline_components['translation']:.6f} m²")
    print(f"  Baseline L_rot: {baseline_components['rotation']:.4f} rad")
    print(f"  Baseline L_total: {baseline_loss:.6f}")
    print(f"  Translation RMSE: {np.sqrt(baseline_components['translation']) * 1000:.2f} mm")
    print(f"  Translation Max Error: {np.max(baseline_transl_errors):.2f} mm")

    # ===== Step 5: Run with Calibration (if provided) =====
    PHYSX_TO_USD = np.pi / 180.0  # PhysX (radian) → USD (degree): ÷ 57.296

    if calibration:
        print("\n[Step 5] Running with calibration...")
        cal_k = calibration['parameters']['stiffness']
        cal_d = calibration['parameters']['damping']
        print(f"  Stiffness (PhysX): {cal_k}")
        print(f"  Stiffness (USD):   {[round(v * PHYSX_TO_USD, 2) for v in cal_k]}")
        print(f"  Damping (PhysX):   {cal_d}")
        print(f"  Damping (USD):     {[round(v * PHYSX_TO_USD, 4) for v in cal_d]}")

        calibrated_fingertip_pos, calibrated_fingertip_rot = run_trajectory(apply_calibration=True)

        calibrated_loss, calibrated_components = simpler_sysid_loss(
            real_ee_pos=real_fingertip_pos[:max_frames],
            sim_ee_pos=calibrated_fingertip_pos,
            real_ee_rot=real_fingertip_rot[:max_frames],
            sim_ee_rot=calibrated_fingertip_rot,
        )

        calibrated_transl_errors = np.linalg.norm(real_fingertip_pos[:max_frames] - calibrated_fingertip_pos, axis=1) * 1000

        print(f"  Calibrated L_transl: {calibrated_components['translation']:.6f} m²")
        print(f"  Calibrated L_rot: {calibrated_components['rotation']:.4f} rad")
        print(f"  Calibrated L_total: {calibrated_loss:.6f}")
        print(f"  Translation RMSE: {np.sqrt(calibrated_components['translation']) * 1000:.2f} mm")
        print(f"  Translation Max Error: {np.max(calibrated_transl_errors):.2f} mm")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    print(f"\n  Trajectory: {max_frames} frames ({args.max_time}s)")

    # Compute per-frame errors for baseline
    baseline_pos_errors = np.linalg.norm(real_fingertip_pos[:max_frames] - baseline_fingertip_pos, axis=1) * 1000  # mm
    baseline_rot_errors = []
    for i in range(max_frames):
        dot = np.abs(np.dot(real_fingertip_rot[i] / np.linalg.norm(real_fingertip_rot[i]),
                           baseline_fingertip_rot[i] / np.linalg.norm(baseline_fingertip_rot[i])))
        baseline_rot_errors.append(2 * np.arccos(np.clip(dot, 0, 1)))
    baseline_rot_errors = np.array(baseline_rot_errors)

    print(f"\n  Baseline (no calibration):")
    print(f"    L_transl:          {baseline_components['translation']:.6f} m²")
    print(f"    L_rot:             {baseline_components['rotation']:.4f} rad ({np.degrees(baseline_components['rotation']):.2f} deg)")
    print(f"    Translation RMSE:  {np.sqrt(baseline_components['translation']) * 1000:.2f} mm")
    print(f"    Translation Max:   {np.max(baseline_pos_errors):.2f} mm")
    print(f"    Translation Mean:  {np.mean(baseline_pos_errors):.2f} mm")
    print(f"    Rotation Max:      {np.degrees(np.max(baseline_rot_errors)):.2f} deg")
    print(f"    Rotation Mean:     {np.degrees(np.mean(baseline_rot_errors)):.2f} deg")

    if calibration:
        # Compute per-frame errors for calibrated
        calibrated_pos_errors = np.linalg.norm(real_fingertip_pos[:max_frames] - calibrated_fingertip_pos, axis=1) * 1000  # mm
        calibrated_rot_errors = []
        for i in range(max_frames):
            dot = np.abs(np.dot(real_fingertip_rot[i] / np.linalg.norm(real_fingertip_rot[i]),
                               calibrated_fingertip_rot[i] / np.linalg.norm(calibrated_fingertip_rot[i])))
            calibrated_rot_errors.append(2 * np.arccos(np.clip(dot, 0, 1)))
        calibrated_rot_errors = np.array(calibrated_rot_errors)

        improvement_transl = (baseline_components['translation'] - calibrated_components['translation']) / baseline_components['translation'] * 100
        improvement_rot = (baseline_components['rotation'] - calibrated_components['rotation']) / baseline_components['rotation'] * 100
        improvement_total = (baseline_loss - calibrated_loss) / baseline_loss * 100
        improvement_max_pos = (np.max(baseline_pos_errors) - np.max(calibrated_pos_errors)) / np.max(baseline_pos_errors) * 100
        improvement_max_rot = (np.max(baseline_rot_errors) - np.max(calibrated_rot_errors)) / np.max(baseline_rot_errors) * 100

        print(f"\n  With Calibration:")
        print(f"    L_transl:          {calibrated_components['translation']:.6f} m² ({improvement_transl:+.1f}%)")
        print(f"    L_rot:             {calibrated_components['rotation']:.4f} rad ({np.degrees(calibrated_components['rotation']):.2f} deg) ({improvement_rot:+.1f}%)")
        print(f"    Translation RMSE:  {np.sqrt(calibrated_components['translation']) * 1000:.2f} mm")
        print(f"    Translation Max:   {np.max(calibrated_pos_errors):.2f} mm ({improvement_max_pos:+.1f}%)")
        print(f"    Translation Mean:  {np.mean(calibrated_pos_errors):.2f} mm")
        print(f"    Rotation Max:      {np.degrees(np.max(calibrated_rot_errors)):.2f} deg ({improvement_max_rot:+.1f}%)")
        print(f"    Rotation Mean:     {np.degrees(np.mean(calibrated_rot_errors)):.2f} deg")
        print(f"\n  Total Loss Improvement: {improvement_total:+.1f}%")
    else:
        print("\n  [INFO] Run with --calibration_file to compare before/after")

    print("\n" + "=" * 70)

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()
