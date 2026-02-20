#!/usr/bin/env python3
"""Plot slip detection data from saved .npz file.

Usage:
    python plot_slip_data.py slip_data_20260205_091233.npz
    python plot_slip_data.py slip_data_20260205_091233.npz --output my_plot.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_ring_finger_data(data_path: Path, output_path: Path | None = None):
    """Plot ring finger time-series data from .npz file."""

    # Load data
    data = np.load(data_path)

    # Check available keys
    print(f"Available keys: {list(data.keys())}")

    # Extract ring finger data
    timestamps = data['ring_timestamps']
    joint_positions = data['ring_joint_positions']  # (T, 4)
    joint_velocities = data['ring_joint_velocities']  # (T, 4)
    motor_currents = data['ring_motor_currents']  # (T, 4) or (T,)
    ft_forces = data['ring_ft_forces'].copy()  # (T, 3)
    ft_torques = data['ring_ft_torques'].copy()  # (T, 3)
    relative_vel = data['ring_relative_vel']  # (T,)
    is_slipping = data['ring_is_slipping']  # (T,)

    # Flip Fz and Tz signs for display
    ft_forces[:, 2] = -ft_forces[:, 2]
    ft_torques[:, 2] = -ft_torques[:, 2]

    # Handle motor_currents shape (could be (T,) for old data or (T, 4) for new)
    if motor_currents.ndim == 1:
        motor_currents = motor_currents.reshape(-1, 1)
        single_effort = True
    else:
        single_effort = False

    # Create figure with 7 subplots
    fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'Ring Finger Slip Detection Data\n{data_path.name}', fontsize=14)

    joint_labels = ['Joint 1 (Abd)', 'Joint 2 (MCP)', 'Joint 3 (PIP)', 'Joint 4 (DIP)']
    colors = ['r', 'g', 'b', 'm']

    # 1. Joint Positions
    ax1 = axes[0]
    for j in range(min(4, joint_positions.shape[1])):
        ax1.plot(timestamps, np.rad2deg(joint_positions[:, j]), colors[j] + '-',
                 linewidth=0.8, label=joint_labels[j])
    ax1.set_ylabel('Position (deg)')
    ax1.set_title('Ring Finger Joint Positions')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8)

    # 2. Joint Velocities
    ax2 = axes[1]
    for j in range(min(4, joint_velocities.shape[1])):
        ax2.plot(timestamps, np.rad2deg(joint_velocities[:, j]), colors[j] + '-',
                 linewidth=0.8, label=joint_labels[j])
    ax2.set_ylabel('Velocity (deg/s)')
    ax2.set_title('Ring Finger Joint Velocities')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=8)

    # 3. Motor Effort (Joint Effort)
    ax3 = axes[2]
    if single_effort:
        ax3.plot(timestamps, motor_currents[:, 0], 'b-', linewidth=0.8, label='Motor Effort (mean)')
    else:
        for j in range(min(4, motor_currents.shape[1])):
            ax3.plot(timestamps, motor_currents[:, j], colors[j] + '-',
                     linewidth=0.8, label=joint_labels[j])
    ax3.set_ylabel('Joint Effort (Nm)')
    ax3.set_title('Ring Finger Motor Effort (All 4 Joints)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=8)

    # 4. F/T Sensor Forces
    ax4 = axes[3]
    force_mag = np.linalg.norm(ft_forces, axis=1)
    ax4.plot(timestamps, ft_forces[:, 0], 'r-', linewidth=0.8, alpha=0.7, label='Fx')
    ax4.plot(timestamps, ft_forces[:, 1], 'g-', linewidth=0.8, alpha=0.7, label='Fy')
    ax4.plot(timestamps, ft_forces[:, 2], 'b-', linewidth=0.8, alpha=0.7, label='-Fz')
    ax4.plot(timestamps, force_mag, 'k-', linewidth=1.2, label='|F|')
    ax4.set_ylabel('Force (N)')
    ax4.set_title('Ring Finger F/T Sensor - Forces')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')

    # 5. F/T Sensor Torques
    ax5 = axes[4]
    torque_mag = np.linalg.norm(ft_torques, axis=1)
    ax5.plot(timestamps, ft_torques[:, 0], 'r-', linewidth=0.8, alpha=0.7, label='Tx')
    ax5.plot(timestamps, ft_torques[:, 1], 'g-', linewidth=0.8, alpha=0.7, label='Ty')
    ax5.plot(timestamps, ft_torques[:, 2], 'b-', linewidth=0.8, alpha=0.7, label='-Tz')
    ax5.plot(timestamps, torque_mag, 'k-', linewidth=1.2, label='|T|')
    ax5.set_ylabel('Torque (Nm)')
    ax5.set_title('Ring Finger F/T Sensor - Torques')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right')

    # 6. Relative Velocity and Slip Detection
    ax6 = axes[5]
    ax6.plot(timestamps, relative_vel * 1000, 'b-', linewidth=0.8, label='Rel. Velocity')
    # Default threshold (can be customized)
    vel_threshold = 5.0  # mm/s
    ax6.axhline(y=vel_threshold, color='r', linestyle='--',
                linewidth=1, label=f'Threshold ({vel_threshold:.1f} mm/s)')

    # Highlight slip regions
    slip_start = None
    for i, slipping in enumerate(is_slipping):
        if slipping and slip_start is None:
            slip_start = timestamps[i]
        elif not slipping and slip_start is not None:
            ax6.axvspan(slip_start, timestamps[i], alpha=0.3, color='red', label='_nolegend_')
            slip_start = None
    if slip_start is not None:
        ax6.axvspan(slip_start, timestamps[-1], alpha=0.3, color='red', label='Slip')

    ax6.set_ylabel('Velocity (mm/s)')
    ax6.set_title('Relative Velocity & Slip Detection')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper right')

    # 7. Overlay: Force, Motor Effort, and Slip
    ax7 = axes[6]
    ax7_twin = ax7.twinx()

    # Plot force components + magnitude
    ax7.plot(timestamps, ft_forces[:, 0], 'r-', linewidth=0.8, alpha=0.6, label='Fx')
    ax7.plot(timestamps, ft_forces[:, 1], 'g-', linewidth=0.8, alpha=0.6, label='Fy')
    ax7.plot(timestamps, ft_forces[:, 2], 'b-', linewidth=0.8, alpha=0.6, label='-Fz')
    ax7.plot(timestamps, force_mag, 'k-', linewidth=1.2, label='|F|')

    # Motor efforts on secondary axis
    effort_colors = ['c', 'm', 'tab:orange', 'tab:brown']
    if single_effort:
        ax7_twin.plot(timestamps, motor_currents[:, 0], 'm-', linewidth=1.0, label='Motor Effort')
    else:
        for j in range(min(4, motor_currents.shape[1])):
            ax7_twin.plot(timestamps, motor_currents[:, j], color=effort_colors[j],
                          linestyle='-', linewidth=0.8, alpha=0.7, label=f'Effort J{j+1}')

    # Highlight slip regions
    slip_start = None
    for i, slipping in enumerate(is_slipping):
        if slipping and slip_start is None:
            slip_start = timestamps[i]
        elif not slipping and slip_start is not None:
            ax7.axvspan(slip_start, timestamps[i], alpha=0.25, color='yellow', label='_nolegend_')
            slip_start = None
    if slip_start is not None:
        ax7.axvspan(slip_start, timestamps[-1], alpha=0.25, color='yellow', label='Slip')

    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Force (N)')
    ax7_twin.set_ylabel('Motor Effort (Nm)')
    ax7.set_title('Overlay: F/T Forces, Motor Efforts & Slip')
    ax7.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_twin.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    plt.tight_layout()

    # Save or show
    if output_path is None:
        output_path = data_path.with_name(data_path.stem + '_plot.png')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

    # Print summary
    print(f"\nData Summary:")
    print(f"  Duration: {timestamps[-1] - timestamps[0]:.2f}s")
    print(f"  Samples: {len(timestamps)}")
    print(f"  Slip ratio: {is_slipping.mean() * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Plot slip detection data from .npz file")
    parser.add_argument("input", type=str, help="Input .npz file path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output plot path (.png)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        input_path = script_dir / args.input

    if not input_path.exists():
        print(f"Error: File not found: {args.input}")
        return

    output_path = Path(args.output) if args.output else None
    plot_ring_finger_data(input_path, output_path)


if __name__ == "__main__":
    main()
