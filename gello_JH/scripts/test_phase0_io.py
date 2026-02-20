#!/usr/bin/env python3
# Copyright (c) 2025, SRBL
# Phase 0 I/O Test Script

"""Phase 0 I/O Test Script

Joint position을 입력하고 fingertip position을 출력하는 기본 I/O 테스트.
Trajectory 없이 더미 joint values로 시뮬레이션 I/O 흐름을 검증합니다.

Usage:
    # 기본 실행
    ./isaaclab.sh -p scripts/test_phase0_io.py --num_steps 100

    # Headless 모드
    ./isaaclab.sh -p scripts/test_phase0_io.py --headless --num_steps 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add Real2Sim root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 0 I/O Test: Joint → Fingertip")

    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of simulation steps",
    )
    parser.add_argument(
        "--flex_angle",
        type=float,
        default=0.3,
        help="Flex angle for first joints (radians)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode",
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

    from envs.phase0_sysid import SysIDEnv, SysIDEnvCfg
    from calibration.phase0.robot_config import FINGER_CONFIGS, RobotType

    print("=" * 60)
    print("Phase 0 I/O Test: Joint Position → Fingertip Position")
    print("=" * 60)

    # Environment config (trajectory 없이)
    cfg = SysIDEnvCfg(num_envs=args.num_envs)
    cfg.scene.num_envs = args.num_envs
    cfg.trajectory_dir = ""  # No trajectory
    cfg.control_mode = "position"  # Position control mode
    cfg.robot_type = RobotType.HAND

    print(f"\nConfiguration:")
    print(f"  num_envs: {args.num_envs}")
    print(f"  num_steps: {args.num_steps}")
    print(f"  flex_angle: {args.flex_angle} rad")
    print(f"  control_mode: {cfg.control_mode}")

    # Environment 생성
    print("\nCreating environment...")
    env = SysIDEnv(cfg)

    # Reset
    print("Resetting environment...")
    env.reset()

    # ===== Step 1: 기본 포즈 읽기 (초기 상태) =====
    print("\n" + "=" * 60)
    print("[Step 1] Initial State")
    print("=" * 60)

    initial_joint_pos = env._tesollo_hand.data.joint_pos.clone()
    initial_joint_vel = env._tesollo_hand.data.joint_vel.clone()

    print(f"\n  Joint positions shape: {initial_joint_pos.shape}")
    print(f"  Joint velocities shape: {initial_joint_vel.shape}")

    # Fingertip poses 읽기
    try:
        initial_fingertip_pos, initial_fingertip_rot = env._get_fingertip_poses()
        print(f"  Fingertip positions shape: {initial_fingertip_pos.shape}")
        print(f"  Fingertip orientations shape: {initial_fingertip_rot.shape}")

        print("\n  Initial Fingertip Positions:")
        for finger_name, finger_cfg in FINGER_CONFIGS.items():
            pos = initial_fingertip_pos[0, finger_cfg.ee_index, :].cpu().numpy()
            print(f"    {finger_name:8s}: [{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}] m")
    except Exception as e:
        print(f"\n  Warning: Could not get fingertip poses: {e}")
        print("  Using body states directly instead...")
        initial_fingertip_pos = None

    print("\n  Initial Joint Positions (first 4 joints per finger):")
    for finger_name, finger_cfg in FINGER_CONFIGS.items():
        indices = finger_cfg.joint_indices
        vals = initial_joint_pos[0, indices].cpu().numpy()
        print(f"    {finger_name:8s}: [{vals[0]:6.3f}, {vals[1]:6.3f}, {vals[2]:6.3f}, {vals[3]:6.3f}] rad")

    # ===== Step 2: 더미 Joint Position 적용 (손가락 굽히기) =====
    print("\n" + "=" * 60)
    print("[Step 2] Apply Dummy Joint Positions (flex fingers)")
    print("=" * 60)

    dummy_joint_pos = initial_joint_pos.clone()

    # 각 손가락의 joints 설정
    print(f"\n  Applying flex_angle = {args.flex_angle} rad to first two joints of each finger")

    for finger_name, finger_cfg in FINGER_CONFIGS.items():
        # 첫 번째와 두 번째 joint에 각도 적용
        dummy_joint_pos[:, finger_cfg.joint_indices[0]] = args.flex_angle
        dummy_joint_pos[:, finger_cfg.joint_indices[1]] = args.flex_angle * 0.7

    print("\n  Target Joint Positions:")
    for finger_name, finger_cfg in FINGER_CONFIGS.items():
        indices = finger_cfg.joint_indices
        vals = dummy_joint_pos[0, indices].cpu().numpy()
        print(f"    {finger_name:8s}: [{vals[0]:6.3f}, {vals[1]:6.3f}, {vals[2]:6.3f}, {vals[3]:6.3f}] rad")

    # Direct state write
    print("\n  Writing joint state to simulation...")
    env._tesollo_hand.write_joint_state_to_sim(
        dummy_joint_pos,
        torch.zeros_like(dummy_joint_pos),  # zero velocity
    )

    # ===== Step 3: 시뮬레이션 스텝 실행 =====
    print("\n" + "=" * 60)
    print("[Step 3] Running Simulation Steps")
    print("=" * 60)

    print(f"\n  Running {args.num_steps} simulation steps with position control...")

    for step in range(args.num_steps):
        # Position control로 유지
        env._tesollo_hand.set_joint_position_target(dummy_joint_pos)

        # Step
        action = torch.zeros(args.num_envs, cfg.action_space, device=env.device)
        env.step(action)

        # Progress 출력
        if (step + 1) % 20 == 0:
            current_pos = env._tesollo_hand.data.joint_pos[0, 0].item()
            target_pos = dummy_joint_pos[0, 0].item()
            error = abs(current_pos - target_pos)
            print(f"    Step {step+1:4d}: thumb_j1 = {current_pos:.4f} rad "
                  f"(target: {target_pos:.4f}, error: {error:.6f})")

    # ===== Step 4: 결과 읽기 =====
    print("\n" + "=" * 60)
    print("[Step 4] Final State After Simulation")
    print("=" * 60)

    final_joint_pos = env._tesollo_hand.data.joint_pos.clone()
    final_joint_vel = env._tesollo_hand.data.joint_vel.clone()

    print("\n  Final Joint Positions:")
    for finger_name, finger_cfg in FINGER_CONFIGS.items():
        indices = finger_cfg.joint_indices
        vals = final_joint_pos[0, indices].cpu().numpy()
        print(f"    {finger_name:8s}: [{vals[0]:6.3f}, {vals[1]:6.3f}, {vals[2]:6.3f}, {vals[3]:6.3f}] rad")

    # Fingertip poses 읽기
    try:
        final_fingertip_pos, final_fingertip_rot = env._get_fingertip_poses()

        print("\n  Final Fingertip Positions:")
        for finger_name, finger_cfg in FINGER_CONFIGS.items():
            pos = final_fingertip_pos[0, finger_cfg.ee_index, :].cpu().numpy()
            print(f"    {finger_name:8s}: [{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}] m")

        if initial_fingertip_pos is not None:
            print("\n  Fingertip Position Changes (Initial → Final):")
            for finger_name, finger_cfg in FINGER_CONFIGS.items():
                init_pos = initial_fingertip_pos[0, finger_cfg.ee_index, :].cpu().numpy()
                final_pos = final_fingertip_pos[0, finger_cfg.ee_index, :].cpu().numpy()
                delta = final_pos - init_pos
                dist = np.linalg.norm(delta)
                print(f"    {finger_name:8s}: Δ = [{delta[0]:7.4f}, {delta[1]:7.4f}, {delta[2]:7.4f}], |Δ| = {dist:.4f} m")
    except Exception as e:
        print(f"\n  Warning: Could not get final fingertip poses: {e}")

    # ===== Step 5: Joint → Fingertip 매핑 검증 =====
    print("\n" + "=" * 60)
    print("[Step 5] Joint → Fingertip Mapping Verification")
    print("=" * 60)

    print("\n  Joint Position Tracking Errors:")
    total_error = 0.0
    max_error = 0.0

    for finger_name, finger_cfg in FINGER_CONFIGS.items():
        for i, joint_idx in enumerate(finger_cfg.joint_indices):
            target = dummy_joint_pos[0, joint_idx].item()
            actual = final_joint_pos[0, joint_idx].item()
            error = abs(target - actual)
            total_error += error
            max_error = max(max_error, error)

    avg_error = total_error / 20  # 20 joints total
    print(f"    Average joint error: {avg_error:.6f} rad")
    print(f"    Maximum joint error: {max_error:.6f} rad")

    # 검증 결과
    print("\n  Verification Results:")
    joint_ok = max_error < 0.01
    print(f"    Joint tracking (max < 0.01 rad): {'✓ PASS' if joint_ok else '✗ FAIL'}")

    if initial_fingertip_pos is not None:
        # Fingertip이 움직였는지 확인
        total_movement = 0.0
        for finger_name, finger_cfg in FINGER_CONFIGS.items():
            init_pos = initial_fingertip_pos[0, finger_cfg.ee_index, :].cpu().numpy()
            final_pos = final_fingertip_pos[0, finger_cfg.ee_index, :].cpu().numpy()
            total_movement += np.linalg.norm(final_pos - init_pos)

        fingertip_ok = total_movement > 0.001  # At least 1mm total movement
        print(f"    Fingertip movement (> 1mm total): {'✓ PASS' if fingertip_ok else '✗ FAIL'}")
        print(f"      Total movement: {total_movement * 1000:.2f} mm")

    # ===== 종료 =====
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
