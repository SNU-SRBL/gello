#!/usr/bin/env python3
# Copyright (c) 2025, SRBL
# Phase 0 I/O Test Script using World0.usd (UR5e + Tesollo)

"""Phase 0 I/O Test Script with World0.usd

World0.usd (UR5e + Tesollo 결합 asset)를 직접 로드하여
Joint position 입력 → Fingertip position 출력 I/O를 테스트합니다.

World0.usd 구조:
  /World/simulation_env/ur5e                                    # UR5e (6 DOF)
  /World/simulation_env/ur5e/dg5f_L_final/dg5f_left_flattened   # Tesollo (20 DOF)

Usage:
    ./isaaclab.sh -p scripts/test_phase0_io_world0.py --num_steps 100
    ./isaaclab.sh -p scripts/test_phase0_io_world0.py --headless --num_steps 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add Real2Sim root to path
REAL2SIM_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REAL2SIM_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 0 I/O Test with World0.usd")

    parser.add_argument("--num_steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--flex_angle", type=float, default=0.3, help="Flex angle for finger joints (radians)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")

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

    print("=" * 70)
    print("Phase 0 I/O Test: World0.usd (UR5e + Tesollo)")
    print("=" * 70)

    # Asset paths
    WORLD0_USD = str(REAL2SIM_DIR / "assets" / "robots" / "World0.usd")
    print(f"\nAsset: {WORLD0_USD}")

    # prim paths (from World0.usd structure)
    UR5E_PRIM_PATH = "/World/simulation_env/ur5e"
    TESOLLO_PRIM_PATH = "/World/simulation_env/ur5e/dg5f_L_final/dg5f_left_flattened"

    # Finger joint configuration
    FINGER_JOINTS = {
        "thumb":  ["lj_dg_1_1", "lj_dg_1_2", "lj_dg_1_3", "lj_dg_1_4"],
        "index":  ["lj_dg_2_1", "lj_dg_2_2", "lj_dg_2_3", "lj_dg_2_4"],
        "middle": ["lj_dg_3_1", "lj_dg_3_2", "lj_dg_3_3", "lj_dg_3_4"],
        "ring":   ["lj_dg_4_1", "lj_dg_4_2", "lj_dg_4_3", "lj_dg_4_4"],
        "pinky":  ["lj_dg_5_1", "lj_dg_5_2", "lj_dg_5_3", "lj_dg_5_4"],
    }

    # ===== Step 1: Create Simulation Context =====
    print("\n[Step 1] Creating Simulation Context...")

    sim_cfg = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1.0 / 120.0,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = SimulationContext(sim_cfg)

    # ===== Step 2: Load World0.usd =====
    print("\n[Step 2] Loading World0.usd...")

    # Add World0.usd to the stage
    from pxr import Usd, UsdGeom
    stage = sim.stage

    # Reference World0.usd
    world_prim = stage.OverridePrim("/World")
    world_prim.GetReferences().AddReference(WORLD0_USD)

    print(f"  World0.usd loaded at /World")

    # ===== Step 3: Create Articulation for Tesollo Hand =====
    print("\n[Step 3] Creating Tesollo Hand Articulation...")

    tesollo_cfg = ArticulationCfg(
        prim_path=TESOLLO_PRIM_PATH,
        spawn=None,  # Don't spawn - already in scene
        actuators={
            "tesollo_fingers": ImplicitActuatorCfg(
                joint_names_expr=["lj_dg_.*"],
                stiffness=100.0,
                damping=10.0,
                friction=0.1,
                effort_limit_sim=10.0,
                velocity_limit_sim=10.0,
            ),
        },
    )

    tesollo_hand = Articulation(tesollo_cfg)

    # ===== Step 4: Create Articulation for UR5e Arm =====
    print("\n[Step 4] Creating UR5e Arm Articulation...")

    ur5e_cfg = ArticulationCfg(
        prim_path=UR5E_PRIM_PATH,
        spawn=None,  # Don't spawn - already in scene
        actuators={
            "ur5e_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*_joint"],
                stiffness=10000.0,  # High stiffness to hold position
                damping=1000.0,
                effort_limit_sim=150.0,
                velocity_limit_sim=3.14,
            ),
        },
    )

    ur5e_arm = Articulation(ur5e_cfg)

    # ===== Step 5: Initialize Simulation =====
    print("\n[Step 5] Initializing Simulation...")

    sim.reset()

    # Get articulation info
    print(f"\n  Tesollo Hand:")
    print(f"    Num joints: {tesollo_hand.num_joints}")
    print(f"    Joint names: {tesollo_hand.joint_names[:5]}... (showing first 5)")

    print(f"\n  UR5e Arm:")
    print(f"    Num joints: {ur5e_arm.num_joints}")
    print(f"    Joint names: {ur5e_arm.joint_names}")

    # ===== Step 6: Read Initial State =====
    print("\n" + "=" * 70)
    print("[Step 6] Initial State")
    print("=" * 70)

    # Update articulation data
    tesollo_hand.update(sim.cfg.dt)
    ur5e_arm.update(sim.cfg.dt)

    initial_hand_pos = tesollo_hand.data.joint_pos.clone()
    initial_arm_pos = ur5e_arm.data.joint_pos.clone()

    print(f"\n  Hand joint positions shape: {initial_hand_pos.shape}")
    print(f"  Arm joint positions shape: {initial_arm_pos.shape}")

    print("\n  Initial Hand Joint Positions:")
    for finger, joints in FINGER_JOINTS.items():
        finger_idx = list(FINGER_JOINTS.keys()).index(finger)
        start_idx = finger_idx * 4
        vals = initial_hand_pos[0, start_idx:start_idx+4].cpu().numpy()
        print(f"    {finger:8s}: [{vals[0]:6.3f}, {vals[1]:6.3f}, {vals[2]:6.3f}, {vals[3]:6.3f}] rad")

    # ===== Step 7: Apply Dummy Joint Positions =====
    print("\n" + "=" * 70)
    print("[Step 7] Apply Dummy Joint Positions (flex fingers)")
    print("=" * 70)

    dummy_hand_pos = initial_hand_pos.clone()

    # Flex each finger
    print(f"\n  Applying flex_angle = {args.flex_angle} rad")
    for finger_idx, (finger, joints) in enumerate(FINGER_JOINTS.items()):
        start_idx = finger_idx * 4
        dummy_hand_pos[0, start_idx] = args.flex_angle       # First joint
        dummy_hand_pos[0, start_idx + 1] = args.flex_angle * 0.7  # Second joint

    print("\n  Target Hand Joint Positions:")
    for finger_idx, finger in enumerate(FINGER_JOINTS.keys()):
        start_idx = finger_idx * 4
        vals = dummy_hand_pos[0, start_idx:start_idx+4].cpu().numpy()
        print(f"    {finger:8s}: [{vals[0]:6.3f}, {vals[1]:6.3f}, {vals[2]:6.3f}, {vals[3]:6.3f}] rad")

    # Write state to sim
    tesollo_hand.write_joint_state_to_sim(
        dummy_hand_pos,
        torch.zeros_like(dummy_hand_pos),
    )

    # ===== Step 8: Run Simulation =====
    print("\n" + "=" * 70)
    print(f"[Step 8] Running {args.num_steps} Simulation Steps")
    print("=" * 70)

    for step in range(args.num_steps):
        # Position control
        tesollo_hand.set_joint_position_target(dummy_hand_pos)
        ur5e_arm.set_joint_position_target(initial_arm_pos)  # Hold arm position

        # Step simulation
        sim.step()

        # Update articulation data
        tesollo_hand.update(sim.cfg.dt)
        ur5e_arm.update(sim.cfg.dt)

        # Print progress
        if (step + 1) % 20 == 0:
            current_pos = tesollo_hand.data.joint_pos[0, 0].item()
            target_pos = dummy_hand_pos[0, 0].item()
            error = abs(current_pos - target_pos)
            print(f"  Step {step+1:4d}: thumb_j1 = {current_pos:.4f} rad "
                  f"(target: {target_pos:.4f}, error: {error:.6f})")

    # ===== Step 9: Final State =====
    print("\n" + "=" * 70)
    print("[Step 9] Final State")
    print("=" * 70)

    final_hand_pos = tesollo_hand.data.joint_pos.clone()

    print("\n  Final Hand Joint Positions:")
    for finger_idx, finger in enumerate(FINGER_JOINTS.keys()):
        start_idx = finger_idx * 4
        vals = final_hand_pos[0, start_idx:start_idx+4].cpu().numpy()
        print(f"    {finger:8s}: [{vals[0]:6.3f}, {vals[1]:6.3f}, {vals[2]:6.3f}, {vals[3]:6.3f}] rad")

    # ===== Step 10: Verification =====
    print("\n" + "=" * 70)
    print("[Step 10] Verification")
    print("=" * 70)

    # Calculate errors
    total_error = 0.0
    max_error = 0.0
    for i in range(tesollo_hand.num_joints):
        target = dummy_hand_pos[0, i].item()
        actual = final_hand_pos[0, i].item()
        error = abs(target - actual)
        total_error += error
        max_error = max(max_error, error)

    avg_error = total_error / tesollo_hand.num_joints

    print(f"\n  Joint Position Tracking:")
    print(f"    Average error: {avg_error:.6f} rad")
    print(f"    Maximum error: {max_error:.6f} rad")

    # Pass/Fail
    joint_ok = max_error < 0.01
    print(f"\n  Result: {'✓ PASS' if joint_ok else '✗ FAIL'} (threshold: 0.01 rad)")

    # ===== Done =====
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()
