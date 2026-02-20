#!/usr/bin/env python3
# Copyright (c) 2025, SRBL
# Index Finger System Identification (SIMPLER Style)

"""SIMPLER-Style System Identification for Index Finger (Tesollo).

Uses fingertip TCP tracking loss (L_transl + L_rot) to calibrate Index finger PD parameters.
Executes action trajectories in open-loop and compares Real vs Sim fingertip poses.

Loss Functions (SIMPLER style):
    L_transl = (1/T) * sum(||fingertip_pos_real - fingertip_pos_sim||^2)
    L_rot = (1/T) * sum(geodesic_distance(fingertip_rot_real, fingertip_rot_sim))
    L_sysid = L_transl + L_rot

Optimization:
    - 3-round Simulated Annealing
    - Each round narrows search range around best parameters

Usage:
    ./isaaclab.sh -p scripts/run_sysid_index.py \
        --trajectory_dir data/lerobot/HD_0204 \
        --output_dir results/sysid_index \
        --num_rounds 3 \
        --trials_per_round 100 \
        --headless
"""

from __future__ import annotations

import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Add parent directory to path
REAL2SIM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REAL2SIM_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="SIMPLER-Style System Identification for Index Finger"
    )

    parser.add_argument(
        "--trajectory_dir",
        type=str,
        required=True,
        help="Path to LeRobot dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/sysid_index",
        help="Output directory for calibration results",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of SA rounds (default: 3)",
    )
    parser.add_argument(
        "--trials_per_round",
        type=int,
        default=100,
        help="Trials per SA round (default: 100)",
    )
    # NOTE: Bounds are in PhysX (radian) units. PhysX = USD × (180/π ≈ 57.296).
    # USD defaults: Tesollo finger K=100, D=2
    # These defaults correspond to USD K=[1, 200], D=[0.01, 10].
    parser.add_argument(
        "--stiffness_min",
        type=float,
        default=57.3,
        help="Minimum stiffness bound in PhysX units (≈ USD 1.0)",
    )
    parser.add_argument(
        "--stiffness_max",
        type=float,
        default=11459.2,
        help="Maximum stiffness bound in PhysX units (≈ USD 200.0)",
    )
    parser.add_argument(
        "--damping_min",
        type=float,
        default=0.573,
        help="Minimum damping bound in PhysX units (≈ USD 0.01)",
    )
    parser.add_argument(
        "--damping_max",
        type=float,
        default=572.96,
        help="Maximum damping bound in PhysX units (≈ USD 10.0)",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=20.0,
        help="Max trajectory time to use for SysID (default: 20.0 seconds)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for SA optimization",
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
    from calibration.phase0.simulated_annealing import SimulatedAnnealingOptimizer, SAConfig
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
    print("SIMPLER-Style System Identification for Index Finger")
    print("=" * 70)

    # ===== Step 1: Load Trajectory Data =====
    print("\n[Step 1] Loading LeRobot dataset...")

    trajectory_dir = Path(args.trajectory_dir)
    loader = LeRobotLoader(trajectory_dir)
    data = loader.load_as_dict(episode_index=0)

    timestamps = data["timestamp"]
    actions = data["action"]
    ur5e_joint_pos = data["observation.state.ur5e_joint_pos"]

    # Fingertip TCP data (Real)
    real_fingertip_pos = data["observation.state.index_fingertip_pos"]  # (T, 3)
    real_fingertip_rot = data["observation.state.index_fingertip_rot"]  # (T, 4)

    num_frames = len(timestamps)
    duration = timestamps[-1] - timestamps[0]
    avg_dt = duration / (num_frames - 1) if num_frames > 1 else 0

    print(f"  Dataset: {trajectory_dir}")
    print(f"  Frames: {num_frames}")
    print(f"  Duration: {duration:.2f} s")
    print(f"  Avg dt: {avg_dt * 1000:.2f} ms ({1.0 / avg_dt:.1f} Hz)")
    print(f"  Action shape: {actions.shape}")
    print(f"  Real fingertip pos shape: {real_fingertip_pos.shape}")
    print(f"  Real fingertip rot shape: {real_fingertip_rot.shape}")

    # Limit trajectory length for speed
    max_frames = min(num_frames, int(args.max_time / avg_dt))
    print(f"  Using first {args.max_time}s ({max_frames} frames) for SysID")

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

    # ===== Step 3: Setup Calibration Parameters =====
    print("\n[Step 3] Setting up calibration parameters...")

    # Joint names
    UR5E_JOINTS = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
    ]
    INDEX_FINGER_JOINTS = ["lj_dg_2_2", "lj_dg_2_3", "lj_dg_2_4"]
    FINGERTIP_BODY_NAME = "lk_dg_2_4"  # Index finger DIP link (fingertip)

    # Wrist3 offset for UR5e
    WRIST3_OFFSET = np.pi

    # Steps per action (match data recording rate)
    steps_per_action = max(1, int(avg_dt / physics_dt))

    print(f"  Index finger joints: {INDEX_FINGER_JOINTS}")
    print(f"  Fingertip body: {FINGERTIP_BODY_NAME}")
    print(f"  Stiffness bounds: [{args.stiffness_min}, {args.stiffness_max}]")
    print(f"  Damping bounds: [{args.damping_min}, {args.damping_max}]")
    print(f"  Steps per action: {steps_per_action}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Step 4: Create Robot =====
    print("\n[Step 4] Creating robot articulation...")

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
                stiffness=None,  # Will be set during optimization
                damping=None,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    sim.reset()
    robot.update(sim.cfg.dt)

    # Get joint indices
    ur5e_indices = [robot.joint_names.index(name) for name in UR5E_JOINTS]
    finger_indices = [robot.joint_names.index(name) for name in INDEX_FINGER_JOINTS]

    # Get fingertip body index
    fingertip_body_idx = robot.find_bodies(FINGERTIP_BODY_NAME)[0][0]

    print(f"  Total joints: {robot.num_joints}")
    print(f"  UR5e indices: {ur5e_indices}")
    print(f"  Finger indices: {finger_indices}")
    print(f"  Fingertip body index: {fingertip_body_idx}")

    # Capture robot base frame (for world→UR5e base transform)
    base_pos_w = robot.data.root_pos_w[0].cpu().numpy()
    base_quat_w = robot.data.root_quat_w[0].cpu().numpy()  # wxyz

    # Pre-compute initial poses with WRIST3_OFFSET
    ur5e_init = ur5e_joint_pos[0].copy()
    ur5e_init[5] += WRIST3_OFFSET

    # ===== Step 5: Define Loss Function =====
    print("\n[Step 5] Defining SIMPLER loss function for fingertip TCP...")

    trial_counter = [0]

    def compute_index_sysid_loss(params: dict[str, np.ndarray]) -> float:
        """Compute SIMPLER-style SysID loss for Index finger PD parameters."""
        stiffness = params["stiffness"]  # (3,)
        damping = params["damping"]      # (3,)

        trial_counter[0] += 1

        try:
            # Reset simulation
            sim.reset()
            robot.update(sim.cfg.dt)

            # Apply PD parameters to Index finger joints
            full_stiffness = robot.root_physx_view.get_dof_stiffnesses().clone()
            full_damping = robot.root_physx_view.get_dof_dampings().clone()

            for i, idx in enumerate(finger_indices):
                full_stiffness[0, idx] = stiffness[i]
                full_damping[0, idx] = damping[i]

            robot.write_joint_stiffness_to_sim(full_stiffness)
            robot.write_joint_damping_to_sim(full_damping)

            # Set initial pose (UR5e)
            robot_target = robot.data.joint_pos.clone()
            for i, idx in enumerate(ur5e_indices):
                robot_target[0, idx] = ur5e_init[i]

            # Settle to initial pose
            for _ in range(50):
                robot.set_joint_position_target(robot_target)
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.cfg.dt)

            # Save settled position as base target (non-controlled joints hold here)
            init_target = robot.data.joint_pos.clone()

            # Run open-loop trajectory, record fingertip TCP poses
            sim_fingertip_pos_traj = np.zeros((max_frames, 3))
            sim_fingertip_rot_traj = np.zeros((max_frames, 4))

            for t in range(max_frames):
                # Set joint targets from action (UR5e 6 joints + finger 3 joints)
                action = actions[t]
                robot_target = init_target.clone()

                # UR5e joints (action[0:6]) with WRIST3_OFFSET
                for i, idx in enumerate(ur5e_indices):
                    target_pos = action[i]
                    if i == 5:  # wrist_3_joint
                        target_pos += WRIST3_OFFSET
                    robot_target[0, idx] = target_pos

                # Finger joints (action[6:9])
                for i, idx in enumerate(finger_indices):
                    robot_target[0, idx] = action[6 + i]

                # Step simulation
                for _ in range(steps_per_action):
                    robot.set_joint_position_target(robot_target)
                    robot.write_data_to_sim()
                    sim.step()
                    robot.update(sim.cfg.dt)

                # Record fingertip TCP pose (convert world → base frame)
                ft_pos_w = robot.data.body_pos_w[0, fingertip_body_idx, :].cpu().numpy()
                ft_rot_w = robot.data.body_quat_w[0, fingertip_body_idx, :].cpu().numpy()
                sim_fingertip_pos_traj[t] = world_to_base_pos(ft_pos_w, base_pos_w, base_quat_w)
                sim_fingertip_rot_traj[t] = world_to_base_quat(ft_rot_w, base_quat_w)

            # Compute SIMPLER loss (fingertip TCP tracking)
            total_loss, loss_components = simpler_sysid_loss(
                real_ee_pos=real_fingertip_pos[:max_frames],
                sim_ee_pos=sim_fingertip_pos_traj,
                real_ee_rot=real_fingertip_rot[:max_frames],
                sim_ee_rot=sim_fingertip_rot_traj,
            )

            if trial_counter[0] % 10 == 0:
                print(f"    Trial {trial_counter[0]}: "
                      f"L_transl={loss_components['translation']:.6f}, "
                      f"L_rot={loss_components['rotation']:.4f}, "
                      f"total={total_loss:.6f}")

            return float(total_loss)

        except Exception as e:
            print(f"    Error in loss computation: {e}")
            return float("inf")

    # ===== Step 6: Run 3-Round Simulated Annealing =====
    print("\n[Step 6] Running 3-round Simulated Annealing optimization...")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Trials per round: {args.trials_per_round}")

    # Parameter bounds (tuple format for SA optimizer)
    param_bounds = {
        "stiffness": (
            np.full(3, args.stiffness_min),
            np.full(3, args.stiffness_max),
        ),
        "damping": (
            np.full(3, args.damping_min),
            np.full(3, args.damping_max),
        ),
    }

    PHYSX_TO_USD = np.pi / 180.0  # PhysX (radian) → USD (degree): ÷ 57.296

    print(f"  Stiffness bounds (PhysX): [{args.stiffness_min:.1f}, {args.stiffness_max:.1f}]")
    print(f"  Stiffness bounds (USD):   [{args.stiffness_min * PHYSX_TO_USD:.2f}, {args.stiffness_max * PHYSX_TO_USD:.2f}]")
    print(f"  Damping bounds (PhysX):   [{args.damping_min:.3f}, {args.damping_max:.2f}]")
    print(f"  Damping bounds (USD):     [{args.damping_min * PHYSX_TO_USD:.4f}, {args.damping_max * PHYSX_TO_USD:.4f}]")

    def on_round_complete(round_idx, bounds, best_params, best_loss):
        print(f"\n  [Round {round_idx + 1} complete] Best loss: {best_loss:.6f}")
        for name, (low, high) in bounds.items():
            if isinstance(low, np.ndarray):
                low_usd = low * PHYSX_TO_USD
                high_usd = high * PHYSX_TO_USD
                print(f"    {name} bounds (USD): [{', '.join(f'{v:.2f}' for v in low_usd)}] ~ [{', '.join(f'{v:.2f}' for v in high_usd)}]")
            else:
                print(f"    {name} bounds (USD): [{low * PHYSX_TO_USD:.4f}, {high * PHYSX_TO_USD:.4f}]")

    # Setup SA optimizer with built-in 3-round functionality
    sa_cfg = SAConfig(
        num_rounds=args.num_rounds,
        trials_per_round=args.trials_per_round,
        temp_initial=1.0,
        temp_final=0.01,
        shrink_factor=0.5,  # SIMPLER: shrink bounds by 0.5x each round
        seed=args.seed,
    )

    optimizer = SimulatedAnnealingOptimizer(
        param_names=["stiffness", "damping"],
        param_bounds=param_bounds,
        cfg=sa_cfg,
    )

    # Run optimization with random initialization
    best_params = optimizer.optimize(
        loss_fn=compute_index_sysid_loss,
        initial_params=None,  # Random initialization
        round_callback=on_round_complete,
    )

    # Get optimization summary
    opt_summary = optimizer.get_optimization_summary()
    best_loss = opt_summary["best_loss"]

    # ===== Step 7: Final Evaluation =====
    print("\n[Step 7] Final evaluation with best parameters...")

    final_loss = compute_index_sysid_loss(best_params)

    # Run one more time to get loss components
    sim.reset()
    robot.update(sim.cfg.dt)

    full_stiffness = robot.root_physx_view.get_dof_stiffnesses().clone()
    full_damping = robot.root_physx_view.get_dof_dampings().clone()

    for i, idx in enumerate(finger_indices):
        full_stiffness[0, idx] = best_params["stiffness"][i]
        full_damping[0, idx] = best_params["damping"][i]

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
    sim_fingertip_pos_final = np.zeros((max_frames, 3))
    sim_fingertip_rot_final = np.zeros((max_frames, 4))

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
        sim_fingertip_pos_final[t] = world_to_base_pos(ft_pos_w, base_pos_w, base_quat_w)
        sim_fingertip_rot_final[t] = world_to_base_quat(ft_rot_w, base_quat_w)

    final_total, final_components = simpler_sysid_loss(
        real_ee_pos=real_fingertip_pos[:max_frames],
        sim_ee_pos=sim_fingertip_pos_final,
        real_ee_rot=real_fingertip_rot[:max_frames],
        sim_ee_rot=sim_fingertip_rot_final,
    )

    # ===== Step 8: Save Results =====
    print("\n[Step 8] Saving results...")

    result = {
        "method": "simpler",
        "loss_type": "fingertip_tcp",
        "joint_names": INDEX_FINGER_JOINTS,
        "fingertip_body": FINGERTIP_BODY_NAME,
        "parameters": {
            "stiffness": best_params["stiffness"].tolist(),
            "damping": best_params["damping"].tolist(),
        },
        "final_loss": {
            "L_transl": final_components["translation"],
            "L_rot": final_components["rotation"],
            "L_total": final_total,
        },
        "optimization": {
            "num_rounds": args.num_rounds,
            "trials_per_round": args.trials_per_round,
            "stiffness_bounds": [args.stiffness_min, args.stiffness_max],
            "damping_bounds": [args.damping_min, args.damping_max],
            "total_evaluations": opt_summary["num_evaluations"],
        },
        "trajectory_info": {
            "dataset": str(trajectory_dir),
            "max_time_used": args.max_time,
            "frames_used": max_frames,
        },
        "timestamp": datetime.now().isoformat(),
    }

    result_file = output_dir / "sysid_index_result.yaml"
    with open(result_file, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Results saved to: {result_file}")

    # ===== Summary =====
    print("\n" + "=" * 70)
    print("Index Finger SysID Complete!")
    print("=" * 70)

    print(f"\n  Final Loss (Fingertip TCP tracking):")
    print(f"    L_transl: {final_components['translation']:.6f} m²")
    print(f"    L_rot:    {final_components['rotation']:.4f} rad")
    print(f"    L_total:  {final_total:.6f}")

    # Compute translation RMSE for readability
    transl_rmse_mm = np.sqrt(final_components['translation']) * 1000
    print(f"\n  Translation RMSE: {transl_rmse_mm:.2f} mm")

    print(f"\n  Calibrated Index Finger PD Parameters:")
    for i, name in enumerate(INDEX_FINGER_JOINTS):
        print(f"    {name}:")
        print(f"      stiffness: {best_params['stiffness'][i]:.2f}")
        print(f"      damping:   {best_params['damping'][i]:.2f}")

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()
