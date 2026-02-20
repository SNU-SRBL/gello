#!/usr/bin/env python3
# Copyright (c) 2025, SRBL
# Phase 2: Friction & Contact Calibration (Position Control Method)

"""
Phase 2: Friction & Contact Calibration

This script runs friction calibration using position control to perform slip tests.
The method gradually loosens the grip until slip occurs, then tracks post-slip
object motion to calculate both static and dynamic friction coefficients.

Usage:
    # Run simulation slip test
    ./isaaclab.sh -p scripts/run_phase2.py --mode sim --output_dir results/phase2

    # Calibrate with real robot measurements
    ./isaaclab.sh -p scripts/run_phase2.py --mode calibrate \
        --real_mu_s 0.6 --real_mu_d 0.4 \
        --phase0_result results/phase0/phase0_hand_result.yaml \
        --phase1_result results/phase1/phase1_result.yaml \
        --output_dir results/phase2

    # Full pipeline (sim + calibrate)
    ./isaaclab.sh -p scripts/run_phase2.py --mode both \
        --real_mu_s 0.6 --real_mu_d 0.4 \
        --output_dir results/phase2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Friction Calibration")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["sim", "calibrate", "both"],
        default="sim",
        help="Mode: 'sim' for simulation slip test, 'calibrate' for optimization, 'both' for full pipeline",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/phase2",
        help="Output directory for calibration results",
    )

    # Target friction coefficients (from real robot)
    parser.add_argument(
        "--real_mu_s",
        type=float,
        default=None,
        help="Real robot's static friction coefficient (for calibrate mode)",
    )
    parser.add_argument(
        "--real_mu_d",
        type=float,
        default=None,
        help="Real robot's dynamic friction coefficient (for calibrate mode, optional)",
    )
    parser.add_argument(
        "--real_q_slip",
        type=float,
        default=None,
        help="Real robot's grip position at slip (normalized, optional)",
    )

    # Phase 0/1 calibration results
    parser.add_argument(
        "--phase0_result",
        type=str,
        default=None,
        help="Path to Phase 0 calibration result YAML file (applies calibrated stiffness/damping)",
    )
    parser.add_argument(
        "--phase1_result",
        type=str,
        default=None,
        help="Path to Phase 1 calibration result YAML file (applies I→τ calibration)",
    )

    # Environment settings
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--object_mass",
        type=float,
        default=0.1,
        help="Test object mass in kg",
    )
    parser.add_argument(
        "--sweep_duration",
        type=float,
        default=10.0,
        help="Grip sweep duration (seconds)",
    )

    # Optimization settings
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--use_bayesian",
        action="store_true",
        help="Use Bayesian optimization (default: grid search)",
    )

    # ArUco tracking
    parser.add_argument(
        "--use_aruco_tracking",
        action="store_true",
        default=True,
        help="Use ArUco marker tracking for dynamic friction measurement",
    )
    parser.add_argument(
        "--marker_id",
        type=int,
        default=0,
        help="ArUco marker ID",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode",
    )

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


def run_simulation(args, output_dir: Path):
    """Run simulation slip test."""
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from envs.phase2_friction import FrictionEnv, FrictionEnvCfg

    print("=" * 60)
    print("Phase 2: Simulation Slip Test")
    print("=" * 60)

    # Create environment with custom config
    cfg = FrictionEnvCfg(num_envs=args.num_envs)
    cfg.scene.num_envs = args.num_envs
    cfg.object_mass = args.object_mass
    cfg.sweep_duration = args.sweep_duration
    cfg.use_aruco_tracking = args.use_aruco_tracking
    cfg.aruco_marker_id = args.marker_id

    # Apply Phase 0/1 calibration if provided
    if args.phase0_result is not None:
        cfg.phase0_result_file = args.phase0_result
        print(f"Using Phase 0 calibration: {args.phase0_result}")

    if args.phase1_result is not None:
        cfg.phase1_result_file = args.phase1_result
        print(f"Using Phase 1 calibration: {args.phase1_result}")

    env = FrictionEnv(cfg)

    # Run slip test
    print("\nStarting slip test...")
    print(f"  Object mass: {args.object_mass} kg")
    print(f"  Sweep duration: {args.sweep_duration} s")
    print(f"  ArUco tracking: {args.use_aruco_tracking}")

    results = env.run_slip_test()

    # Print results
    print("\n" + "=" * 60)
    print("Slip Test Results")
    print("=" * 60)

    if results["slip_detected"].any():
        slip_mask = results["slip_detected"]
        print(f"\nSlip detected: {slip_mask.sum()} / {len(slip_mask)} environments")
        print(f"\nStatic Friction (μ_s):")
        print(f"  Mean: {results['static_friction'][slip_mask].mean():.4f}")
        print(f"  Std:  {results['static_friction'][slip_mask].std():.4f}")

        if results["dynamic_friction_valid"].any():
            valid_mask = results["dynamic_friction_valid"]
            print(f"\nDynamic Friction (μ_d):")
            print(f"  Mean: {results['dynamic_friction'][valid_mask].mean():.4f}")
            print(f"  Std:  {results['dynamic_friction'][valid_mask].std():.4f}")
            print(f"  Valid measurements: {valid_mask.sum()}")
        else:
            print("\nDynamic Friction: No valid measurements")

        print(f"\nGrip Position at Slip:")
        print(f"  Mean: {results['grip_position_at_slip'][slip_mask].mean():.4f}")
    else:
        print("\nNo slip detected during test!")
        print("Consider adjusting sweep parameters or initial grip position.")

    # Export data
    sim_data_path = output_dir / "sim_slip_data.npz"
    env.export_slip_data(str(sim_data_path))

    # Save results to YAML
    import yaml
    yaml_result = {
        "phase": 2,
        "type": "slip_test",
        "slip_detected": bool(results["slip_detected"].any()),
        "static_friction": {
            "mean": float(results["static_friction"][results["slip_detected"]].mean()) if results["slip_detected"].any() else None,
            "std": float(results["static_friction"][results["slip_detected"]].std()) if results["slip_detected"].any() else None,
        },
        "dynamic_friction": {
            "mean": float(results["dynamic_friction"][results["dynamic_friction_valid"]].mean()) if results["dynamic_friction_valid"].any() else None,
            "std": float(results["dynamic_friction"][results["dynamic_friction_valid"]].std()) if results["dynamic_friction_valid"].any() else None,
            "valid_count": int(results["dynamic_friction_valid"].sum()),
        },
        "grip_position_at_slip": {
            "mean": float(results["grip_position_at_slip"][results["slip_detected"]].mean()) if results["slip_detected"].any() else None,
        },
        "config": {
            "object_mass": args.object_mass,
            "sweep_duration": args.sweep_duration,
            "use_aruco_tracking": args.use_aruco_tracking,
        },
    }

    yaml_path = output_dir / "phase2_sim_result.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_result, f, default_flow_style=False, allow_unicode=True)

    print(f"\nData exported to: {sim_data_path}")
    print(f"Results saved to: {yaml_path}")

    env.close()
    simulation_app.close()

    return results


def run_calibration(args, output_dir: Path):
    """Run friction parameter optimization."""
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from envs.phase2_friction import FrictionEnv, FrictionEnvCfg
    from calibration.phase2 import FrictionEstimator, FrictionEstimatorConfig

    print("=" * 60)
    print("Phase 2: Friction Calibration")
    print("=" * 60)

    # Check targets
    if args.real_mu_s is None:
        print("ERROR: --real_mu_s is required for calibration mode")
        return None

    print(f"\nTarget static friction (μ_s): {args.real_mu_s}")
    if args.real_mu_d is not None:
        print(f"Target dynamic friction (μ_d): {args.real_mu_d}")
    if args.real_q_slip is not None:
        print(f"Target grip position at slip: {args.real_q_slip}")

    # Create environment
    cfg = FrictionEnvCfg(num_envs=args.num_envs)
    cfg.scene.num_envs = args.num_envs
    cfg.object_mass = args.object_mass
    cfg.sweep_duration = args.sweep_duration
    cfg.use_aruco_tracking = args.use_aruco_tracking

    if args.phase0_result is not None:
        cfg.phase0_result_file = args.phase0_result
        print(f"Using Phase 0 calibration: {args.phase0_result}")

    if args.phase1_result is not None:
        cfg.phase1_result_file = args.phase1_result
        print(f"Using Phase 1 calibration: {args.phase1_result}")

    env = FrictionEnv(cfg)

    if args.use_bayesian:
        # Use Bayesian optimization
        from calibration.phase2 import BayesianOptimizer, BayesianOptimizerConfig

        opt_cfg = BayesianOptimizerConfig(n_trials=args.n_trials)
        optimizer = BayesianOptimizer(env, opt_cfg)

        def log_callback(trial_num: int, loss: float, params: dict):
            print(f"Trial {trial_num:3d}: Loss = {loss:.6f}, "
                  f"μ_s = {params['static_friction']:.3f}, "
                  f"μ_d = {params['dynamic_friction']:.3f}")

        print("\nStarting Bayesian optimization...")
        best_params = optimizer.optimize(
            real_static_friction=args.real_mu_s,
            real_dynamic_friction=args.real_mu_d,
            callback=log_callback,
        )
    else:
        # Use grid search
        est_cfg = FrictionEstimatorConfig()
        estimator = FrictionEstimator(env, est_cfg)

        print("\nStarting grid search optimization...")
        best_params = estimator.grid_search(
            real_static_friction=args.real_mu_s,
            real_dynamic_friction=args.real_mu_d,
            real_q_slip=args.real_q_slip,
        )

        # Refinement
        if best_params is not None:
            print("\nRefining search...")
            best_params = estimator.refine_search(
                real_static_friction=args.real_mu_s,
                real_dynamic_friction=args.real_mu_d,
                center_params=best_params,
            )

        print("\n" + estimator.summary())

    # Save results
    import yaml

    yaml_result = {
        "phase": 2,
        "type": "friction_calibration",
        "parameters": {
            "static_friction": float(best_params["static_friction"]),
            "dynamic_friction": float(best_params["dynamic_friction"]),
        },
        "simulation_results": {
            "static_friction": best_params.get("sim_mu_s"),
            "dynamic_friction": best_params.get("sim_mu_d"),
        },
        "targets": {
            "static_friction": args.real_mu_s,
            "dynamic_friction": args.real_mu_d,
        },
        "loss": float(best_params["loss"]),
        "config": {
            "n_trials": args.n_trials,
            "use_bayesian": args.use_bayesian,
            "object_mass": args.object_mass,
        },
    }

    yaml_path = output_dir / "phase2_result.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_result, f, default_flow_style=False, allow_unicode=True)

    print("\n" + "=" * 60)
    print("Calibration Complete!")
    print("=" * 60)
    print(f"\nBest Parameters:")
    print(f"  Static Friction (μ_s):  {best_params['static_friction']:.4f}")
    print(f"  Dynamic Friction (μ_d): {best_params['dynamic_friction']:.4f}")
    print(f"\nBest Loss: {best_params['loss']:.6f}")
    print(f"\nResults saved to: {yaml_path}")

    env.close()
    simulation_app.close()

    return best_params


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "sim":
        run_simulation(args, output_dir)

    elif args.mode == "calibrate":
        run_calibration(args, output_dir)

    elif args.mode == "both":
        print("Running full calibration pipeline...")
        print("\nStep 1/2: Simulation slip test")
        run_simulation(args, output_dir)

        if args.real_mu_s is not None:
            print("\n" + "=" * 60)
            print("\nStep 2/2: Calibration")
            run_calibration(args, output_dir)
        else:
            print("\nStep 2/2: Skipped (no target friction values provided)")
            print("Provide --real_mu_s and optionally --real_mu_d to complete calibration.")


if __name__ == "__main__":
    main()
