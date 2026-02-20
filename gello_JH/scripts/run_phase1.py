#!/usr/bin/env python3
# Copyright (c) 2025, SRBL
# Phase 1: Current-Torque Calibration

"""
Phase 1: Current-Torque Calibration (I_real ↔ τ_sim Mapping)

Usage:
    # Step 1: (Optional) Sim data collection (dry run / Jacobian validation)
    ./isaaclab.sh -p scripts/run_phase1.py --mode sim_full \
        --output_dir results/phase1

    # Step 2: Compute τ_sim from real robot data (direct F_ext application)
    ./isaaclab.sh -p scripts/run_phase1.py --mode sim_from_real \
        --real_contact data/real_data/phase1_contact.npz \
        --output_dir results/phase1

    # Step 3: Linear calibration (τ_sim = k_t × I_real + b)
    ./isaaclab.sh -p scripts/run_phase1.py --mode calibrate \
        --sim_matched results/phase1/sim_matched.npz \
        --real_contact data/real_data/phase1_contact.npz \
        --output_dir results/phase1

    # Step 4: (Optional) Learning-based calibration
    ./isaaclab.sh -p scripts/run_phase1.py --mode calibrate_learned \
        --sim_matched results/phase1/sim_matched.npz \
        --real_contact data/real_data/phase1_contact.npz \
        --output_dir results/phase1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Current-Torque Calibration")

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "sim_full", "sim_baseline", "sim_contact",
            "sim_from_real",
            "calibrate", "calibrate_learned",
        ],
        default="sim_full",
        help=(
            "Mode: 'sim_full/sim_baseline/sim_contact' for sim data collection, "
            "'sim_from_real' to compute τ_sim from real data, "
            "'calibrate' for linear calibration, "
            "'calibrate_learned' for learning-based calibration"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/phase1",
        help="Output directory for calibration data",
    )
    parser.add_argument(
        "--sim_matched",
        type=str,
        default=None,
        help="Path to sim matched τ_sim data file (for calibrate mode)",
    )
    parser.add_argument(
        "--sim_contact",
        type=str,
        default=None,
        help="Path to sim contact data file (for Jacobian validation)",
    )
    parser.add_argument(
        "--real_contact",
        type=str,
        default=None,
        help="Path to real robot contact data file",
    )
    parser.add_argument(
        "--fingers",
        type=str,
        default="thumb,index,middle,ring,pinky",
        help="Comma-separated list of fingers to calibrate",
    )
    parser.add_argument(
        "--phase0_result",
        type=str,
        default=None,
        help="Path to Phase 0 calibration result YAML file",
    )
    parser.add_argument(
        "--num_gravity_configs",
        type=int,
        default=20,
        help="Number of gravity baseline configurations (Phase 1A)",
    )
    parser.add_argument(
        "--num_contact_configs",
        type=int,
        default=5,
        help="Number of contact configurations per finger (Phase 1B)",
    )
    parser.add_argument(
        "--num_directions",
        type=int,
        default=3,
        help="Number of force directions per configuration (Phase 1B)",
    )
    parser.add_argument(
        "--ramp_duration",
        type=float,
        default=5.0,
        help="Duration of position ramp (seconds)",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode",
    )
    # Learning-based model options
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["mlp", "residual", "both"],
        default="both",
        help="Learning-based model type (for calibrate_learned mode)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Training epochs (for calibrate_learned mode)",
    )

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_args()


def _get_sim_mode(mode: str) -> str:
    """Convert CLI mode to env start_calibration mode."""
    if mode == "sim_full":
        return "full"
    elif mode == "sim_baseline":
        return "baseline"
    elif mode == "sim_contact":
        return "contact"
    return "full"


def run_simulation(args, output_dir: Path):
    """Run simulation to collect calibration data."""
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from envs.phase1_current_torque import CurrentTorqueEnv, CurrentTorqueEnvCfg
    from data.storage import CalibrationResultStorage

    print("=" * 60)
    print(f"Phase 1: Simulation Data Collection (mode={args.mode})")
    print("=" * 60)

    fingers = [f.strip() for f in args.fingers.split(",")]
    print(f"Fingers to calibrate: {fingers}")

    # Create environment
    cfg = CurrentTorqueEnvCfg(num_envs=args.num_envs)
    cfg.scene.num_envs = args.num_envs
    cfg.fingers_to_calibrate = fingers
    cfg.num_gravity_configs = args.num_gravity_configs
    cfg.num_contact_configs = args.num_contact_configs
    cfg.num_force_directions = args.num_directions
    cfg.ramp_duration = args.ramp_duration

    if args.phase0_result is not None:
        cfg.phase0_result_file = args.phase0_result
        print(f"Using Phase 0 calibration: {args.phase0_result}")

    env = CurrentTorqueEnv(cfg)

    # Start calibration with appropriate mode
    sim_mode = _get_sim_mode(args.mode)
    env.start_calibration(mode=sim_mode)

    # Run until complete
    step_count = 0
    max_steps = 200000

    while not env.is_calibration_complete and step_count < max_steps:
        env.step()
        step_count += 1

        if step_count % 1000 == 0:
            progress = env.calibration_progress * 100
            print(f"  Step {step_count}: {progress:.1f}% complete")

    # Export data
    sim_data_path = output_dir / "sim_data.npz"
    env.export_calibration_data(str(sim_data_path))

    # Save YAML result
    result = env.get_result()
    storage = CalibrationResultStorage(output_dir)
    yaml_path = storage.save(result, "phase1_sim_result.yaml", overwrite=True)

    print("\n" + "=" * 60)
    print("Simulation Data Collection Complete!")
    print("=" * 60)
    print(f"\nData exported to: {output_dir}")
    print(f"Results saved to: {yaml_path}")

    env.close()
    simulation_app.close()

    return sim_data_path


def run_sim_from_real(args, output_dir: Path):
    """Run sim-from-real: compute τ_sim for each real data point."""
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from envs.phase1_current_torque import CurrentTorqueEnv, CurrentTorqueEnvCfg
    from data.storage import CalibrationResultStorage

    real_contact_path = args.real_contact
    if real_contact_path is None:
        print("ERROR: --real_contact is required for sim_from_real mode")
        simulation_app.close()
        return None

    if not Path(real_contact_path).exists():
        print(f"ERROR: Real contact data not found: {real_contact_path}")
        simulation_app.close()
        return None

    print("=" * 60)
    print("Phase 1: Sim-From-Real (Direct Force Application)")
    print("=" * 60)
    print(f"Real data: {real_contact_path}")

    fingers = [f.strip() for f in args.fingers.split(",")]

    # Create environment
    cfg = CurrentTorqueEnvCfg(num_envs=args.num_envs)
    cfg.scene.num_envs = args.num_envs
    cfg.fingers_to_calibrate = fingers

    if args.phase0_result is not None:
        cfg.phase0_result_file = args.phase0_result
        print(f"Using Phase 0 calibration: {args.phase0_result}")

    env = CurrentTorqueEnv(cfg)

    # Start sim-from-real mode
    env.start_sim_from_real(real_contact_path)

    # Run until complete
    step_count = 0
    max_steps = 500000

    while not env.is_calibration_complete and step_count < max_steps:
        env.step()
        step_count += 1

        if step_count % 5000 == 0:
            progress = env.calibration_progress * 100
            print(f"  Step {step_count}: {progress:.1f}% complete")

    # Export sim-from-real results
    sfr_path = output_dir / "sim_matched.npz"
    env.export_sfr_data(str(sfr_path))

    # Save YAML result
    result = env.get_result()
    storage = CalibrationResultStorage(output_dir)
    yaml_path = storage.save(result, "phase1_sfr_result.yaml", overwrite=True)

    print("\n" + "=" * 60)
    print("Sim-From-Real Complete!")
    print("=" * 60)
    print(f"\nMatched τ_sim data: {sfr_path}")
    print(f"Results saved to: {yaml_path}")
    print("\nNext: Run calibration:")
    print(f"  ./isaaclab.sh -p scripts/run_phase1.py --mode calibrate \\")
    print(f"      --sim_matched {sfr_path} \\")
    print(f"      --real_contact {real_contact_path}")

    env.close()
    simulation_app.close()

    return sfr_path


def _plot_calibration_scatter(
    sim_matched_path: Path,
    real_contact_path: Path,
    model,
    output_dir: Path,
):
    """Plot per-joint I_real vs τ_sim scatter with regression line.

    Generates a 5×4 grid (5 fingers × 4 joints) of scatter plots.
    Each subplot shows data points and the fitted regression line.

    Args:
        sim_matched_path: Path to sim_matched.npz (contains τ_sim).
        real_contact_path: Path to real_contact.npz (contains I_motor).
        model: Fitted JacobianCurrentTorqueModel with k_t, offset, r_squared.
        output_dir: Directory to save the plot.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim_data = np.load(str(sim_matched_path), allow_pickle=True)
    real_data = np.load(str(real_contact_path), allow_pickle=True)

    tau_sim = sim_data["tau_sim"]   # (N, 20)
    I_real = real_data["I_motor"]   # (N, 20)

    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle("Phase 1: I_real vs τ_sim (per joint)", fontsize=16, y=0.98)

    for finger_i, finger_name in enumerate(model.finger_names):
        joint_indices = model.finger_joint_indices[finger_name]
        for local_j, j_idx in enumerate(joint_indices):
            ax = axes[finger_i, local_j]

            I_j = I_real[:, j_idx]
            tau_j = tau_sim[:, j_idx]

            # Scatter: data points
            ax.scatter(I_j, tau_j, alpha=0.4, s=10, c="steelblue", edgecolors="none")

            # Regression line
            k_t = model.k_t[j_idx]
            offset = model.offset[j_idx]
            r2 = model.r_squared[j_idx]

            if I_j.max() > I_j.min():
                I_range = np.linspace(I_j.min(), I_j.max(), 100)
                ax.plot(
                    I_range, k_t * I_range + offset, "r-", lw=2,
                    label=f"k_t={k_t:.4f}, b={offset:.4f}\nR²={r2:.3f}",
                )

            ax.set_title(f"{finger_name}_j{local_j}", fontsize=10)
            ax.set_xlabel("I_real (A)", fontsize=8)
            ax.set_ylabel("τ_sim (Nm)", fontsize=8)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = output_dir / "phase1_calibration_scatter.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nScatter plot saved to: {plot_path}")


def run_calibration(args, output_dir: Path):
    """Run linear calibration (τ_sim = k_t × I_real + b)."""
    from calibration.phase1 import JacobianCurrentTorqueModel, Phase1Config
    from data.storage import CalibrationResultStorage
    import yaml

    print("=" * 60)
    print("Phase 1: Paired Sim-Real Calibration")
    print("=" * 60)

    # Resolve file paths
    sim_matched_path = Path(args.sim_matched) if args.sim_matched else output_dir / "sim_matched.npz"
    real_contact_path = Path(args.real_contact) if args.real_contact else None
    sim_contact_path = Path(args.sim_contact) if args.sim_contact else None

    # Validate
    if not sim_matched_path.exists():
        print(f"ERROR: Sim matched data not found: {sim_matched_path}")
        print("Run --mode sim_from_real first.")
        return None

    if real_contact_path is None or not real_contact_path.exists():
        print(f"ERROR: Real contact data required: {real_contact_path}")
        return None

    print(f"\nSim matched:  {sim_matched_path}")
    print(f"Real contact: {real_contact_path}")
    if sim_contact_path:
        print(f"Sim contact:  {sim_contact_path} (Jacobian validation)")

    # Create model and run calibration
    cfg = Phase1Config()
    model = JacobianCurrentTorqueModel(num_joints=20, cfg=cfg)

    results = model.calibrate_all(
        sim_matched_file=str(sim_matched_path),
        real_contact_file=str(real_contact_path),
        sim_contact_file=str(sim_contact_path) if sim_contact_path and sim_contact_path.exists() else None,
    )

    # Print summary
    print("\n" + model.summary())

    # Save calibration
    calibration_path = output_dir / "calibration.npz"
    model.save(str(calibration_path))

    # Save YAML result
    yaml_result = model.to_yaml_dict()
    yaml_path = output_dir / "phase1_result.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_result, f, default_flow_style=False, allow_unicode=True)

    # Plot I vs τ scatter per joint
    try:
        _plot_calibration_scatter(sim_matched_path, real_contact_path, model, output_dir)
    except Exception as e:
        print(f"Warning: Failed to generate scatter plot: {e}")

    print("\n" + "=" * 60)
    print("Calibration Complete!")
    print("=" * 60)
    print(f"\nCalibration saved to: {calibration_path}")
    print(f"YAML result saved to: {yaml_path}")

    return calibration_path


def run_calibration_learned(args, output_dir: Path):
    """Run learning-based calibration."""
    from calibration.phase1 import JacobianCurrentTorqueModel, Phase1Config
    from calibration.phase1.learned_model import LearnedCurrentTorqueModel, LearnedModelConfig
    import numpy as np
    import yaml

    print("=" * 60)
    print("Phase 1: Learning-Based Calibration")
    print("=" * 60)

    # Resolve paths
    sim_matched_path = Path(args.sim_matched) if args.sim_matched else output_dir / "sim_matched.npz"
    real_contact_path = Path(args.real_contact) if args.real_contact else None

    if not sim_matched_path.exists():
        print(f"ERROR: Sim matched data not found: {sim_matched_path}")
        return None
    if real_contact_path is None or not real_contact_path.exists():
        print(f"ERROR: Real contact data required: {real_contact_path}")
        return None

    # Load data
    sim_data = np.load(str(sim_matched_path), allow_pickle=True)
    real_data = np.load(str(real_contact_path), allow_pickle=True)

    tau_sim = sim_data["tau_sim"]  # (N, 20)
    q = real_data["q"]  # (N, 20)
    qdot = real_data.get("qdot", np.zeros_like(q))
    I_real = real_data["I_motor"]  # (N, 20)
    F_internal = real_data.get("F_internal")  # (N, 30) or None

    print(f"\nData: {len(tau_sim)} samples")
    print(f"  q: {q.shape}, qdot: {qdot.shape}")
    print(f"  I_real: {I_real.shape}, τ_sim: {tau_sim.shape}")
    print(f"  F_internal: {F_internal.shape if F_internal is not None else 'None'}")

    # Step 1: Linear calibration (for initialization)
    print("\n--- Step 1: Linear Calibration (for initialization) ---")
    linear_cfg = Phase1Config()
    linear_model = JacobianCurrentTorqueModel(num_joints=20, cfg=linear_cfg)
    linear_model.calibrate_paired(tau_sim, I_real, real_data.get("finger_idx"))

    # Step 2: Learning-based calibration
    learned_cfg = LearnedModelConfig(
        num_epochs=args.num_epochs,
        use_finger_ft=F_internal is not None,
    )
    learned_model = LearnedCurrentTorqueModel(cfg=learned_cfg)

    if args.model_type in ("mlp", "both"):
        print("\n--- Step 2a: MLP Model ---")
        mlp_metrics = learned_model.train_mlp(q, qdot, F_internal, I_real, tau_sim)

    if args.model_type in ("residual", "both"):
        print("\n--- Step 2b: Residual Model ---")
        residual_metrics = learned_model.train_residual(
            q, qdot, F_internal, I_real, tau_sim,
            k_t_init=linear_model.k_t,
            offset_init=linear_model.offset,
        )

    # Save models
    linear_path = output_dir / "calibration.npz"
    linear_model.save(str(linear_path))

    learned_path = output_dir / "learned_model.pt"
    learned_model.save(str(learned_path))

    # Save combined YAML
    yaml_result = {
        "linear": linear_model.to_yaml_dict(),
        "learned": learned_model.to_yaml_dict(),
    }
    yaml_path = output_dir / "phase1_result.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_result, f, default_flow_style=False, allow_unicode=True)

    print("\n" + "=" * 60)
    print("Learning-Based Calibration Complete!")
    print("=" * 60)
    print(f"\nLinear model:  {linear_path}")
    print(f"Learned model: {learned_path}")
    print(f"YAML result:   {yaml_path}")

    # Print comparison
    print("\n" + linear_model.summary())
    print("\n" + learned_model.summary())

    return learned_path


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("sim_full", "sim_baseline", "sim_contact"):
        run_simulation(args, output_dir)

    elif args.mode == "sim_from_real":
        run_sim_from_real(args, output_dir)

    elif args.mode == "calibrate":
        run_calibration(args, output_dir)

    elif args.mode == "calibrate_learned":
        run_calibration_learned(args, output_dir)


if __name__ == "__main__":
    main()
