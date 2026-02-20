# Trajectory Visualization Tools

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict


def plot_trajectory_comparison(
    real_trajectory: Dict[str, np.ndarray],
    sim_trajectory: Dict[str, np.ndarray],
    joint_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot comparison between real and simulated trajectories.

    Args:
        real_trajectory: Real robot trajectory with 'timestamps', 'joint_positions', 'joint_velocities'
        sim_trajectory: Simulated trajectory with same keys
        joint_names: Optional list of joint names for labeling
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    real_time = np.array(real_trajectory.get("timestamps", []))
    real_pos = np.array(real_trajectory.get("joint_positions", []))
    real_vel = real_trajectory.get("joint_velocities")

    sim_time = np.array(sim_trajectory.get("timestamps", []))
    sim_pos = np.array(sim_trajectory.get("joint_positions", []))
    sim_vel = sim_trajectory.get("joint_velocities")

    n_joints = real_pos.shape[1] if real_pos.ndim > 1 else 1

    if joint_names is None:
        joint_names = [f"Joint {i}" for i in range(n_joints)]

    # Determine layout
    has_velocity = real_vel is not None and sim_vel is not None
    n_rows = n_joints
    n_cols = 2 if has_velocity else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows), squeeze=False)

    for i in range(n_joints):
        # Position plot
        ax = axes[i, 0]
        ax.plot(real_time, real_pos[:, i], "b-", label="Real", linewidth=1.5)
        ax.plot(sim_time, sim_pos[:, i], "r--", label="Sim", linewidth=1.5)
        ax.set_ylabel(f"{joint_names[i]}\nPosition (rad)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title("Joint Position")

        if i == n_joints - 1:
            ax.set_xlabel("Time (s)")

        # Velocity plot
        if has_velocity:
            real_vel_arr = np.array(real_vel)
            sim_vel_arr = np.array(sim_vel)

            ax = axes[i, 1]
            ax.plot(real_time, real_vel_arr[:, i], "b-", label="Real", linewidth=1.5)
            ax.plot(sim_time, sim_vel_arr[:, i], "r--", label="Sim", linewidth=1.5)
            ax.set_ylabel(f"{joint_names[i]}\nVelocity (rad/s)")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.set_title("Joint Velocity")

            if i == n_joints - 1:
                ax.set_xlabel("Time (s)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_trajectory_error(
    real_trajectory: Dict[str, np.ndarray],
    sim_trajectory: Dict[str, np.ndarray],
    joint_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot trajectory tracking error over time.

    Args:
        real_trajectory: Real robot trajectory
        sim_trajectory: Simulated trajectory
        joint_names: Optional list of joint names
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    real_time = np.array(real_trajectory.get("timestamps", []))
    real_pos = np.array(real_trajectory.get("joint_positions", []))
    sim_pos = np.array(sim_trajectory.get("joint_positions", []))

    # Interpolate sim to real timestamps if needed
    if len(real_time) != len(sim_pos):
        sim_time = np.array(sim_trajectory.get("timestamps", []))
        sim_pos_interp = np.zeros_like(real_pos)
        for i in range(sim_pos.shape[1]):
            sim_pos_interp[:, i] = np.interp(real_time, sim_time, sim_pos[:, i])
        sim_pos = sim_pos_interp

    error = real_pos - sim_pos
    n_joints = error.shape[1] if error.ndim > 1 else 1

    if joint_names is None:
        joint_names = [f"Joint {i}" for i in range(n_joints)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Per-joint error over time
    ax = axes[0]
    for i in range(n_joints):
        ax.plot(real_time, error[:, i], label=joint_names[i], alpha=0.7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position Error (rad)")
    ax.set_title("Trajectory Tracking Error Over Time")
    ax.legend(loc="upper right", ncol=min(n_joints, 5))
    ax.grid(True, alpha=0.3)

    # RMSE per joint
    ax = axes[1]
    rmse = np.sqrt(np.mean(error**2, axis=0))
    x = np.arange(n_joints)
    bars = ax.bar(x, rmse, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(joint_names, rotation=45, ha="right")
    ax.set_ylabel("RMSE (rad)")
    ax.set_title(f"Position RMSE per Joint (Total RMSE: {np.mean(rmse):.6f} rad)")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, rmse):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_joint_phase_space(
    trajectory: Dict[str, np.ndarray],
    joint_idx: int = 0,
    joint_name: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot joint position vs velocity phase space.

    Args:
        trajectory: Trajectory with 'joint_positions' and 'joint_velocities'
        joint_idx: Index of joint to plot
        joint_name: Optional name for labeling
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    positions = np.array(trajectory["joint_positions"])[:, joint_idx]
    velocities = np.array(trajectory["joint_velocities"])[:, joint_idx]

    if joint_name is None:
        joint_name = f"Joint {joint_idx}"

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by time
    time = np.array(trajectory.get("timestamps", np.arange(len(positions))))
    scatter = ax.scatter(positions, velocities, c=time, cmap="viridis", s=5, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label="Time (s)")

    # Mark start and end
    ax.scatter(positions[0], velocities[0], c="green", s=100, marker="o", label="Start", zorder=5)
    ax.scatter(positions[-1], velocities[-1], c="red", s=100, marker="s", label="End", zorder=5)

    ax.set_xlabel("Position (rad)")
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_title(f"Phase Space: {joint_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
