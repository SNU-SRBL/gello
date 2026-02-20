# Force Visualization Tools

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict


def plot_force_comparison(
    real_forces: Dict[str, np.ndarray],
    sim_forces: Dict[str, np.ndarray],
    timestamps: Optional[np.ndarray] = None,
    finger_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot comparison between real and simulated contact forces.

    Args:
        real_forces: Dict with finger names as keys, force arrays as values (N, 6) for 6-axis F/T
        sim_forces: Dict with same structure
        timestamps: Optional time array
        finger_names: List of finger names to plot
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if finger_names is None:
        finger_names = list(real_forces.keys())

    n_fingers = len(finger_names)

    fig, axes = plt.subplots(n_fingers, 2, figsize=(14, 3 * n_fingers), squeeze=False)

    for i, finger in enumerate(finger_names):
        real_f = np.array(real_forces.get(finger, np.zeros((1, 6))))
        sim_f = np.array(sim_forces.get(finger, np.zeros((1, 6))))

        if timestamps is None:
            timestamps = np.arange(len(real_f))

        # Force magnitude
        real_force_mag = np.linalg.norm(real_f[:, :3], axis=1)
        sim_force_mag = np.linalg.norm(sim_f[:, :3], axis=1)

        ax = axes[i, 0]
        ax.plot(timestamps, real_force_mag, "b-", label="Real", linewidth=1.5)
        ax.plot(timestamps, sim_force_mag, "r--", label="Sim", linewidth=1.5)
        ax.set_ylabel(f"{finger}\nForce (N)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title("Force Magnitude")
        if i == n_fingers - 1:
            ax.set_xlabel("Time (s)")

        # Torque magnitude
        real_torque_mag = np.linalg.norm(real_f[:, 3:], axis=1)
        sim_torque_mag = np.linalg.norm(sim_f[:, 3:], axis=1)

        ax = axes[i, 1]
        ax.plot(timestamps, real_torque_mag, "b-", label="Real", linewidth=1.5)
        ax.plot(timestamps, sim_torque_mag, "r--", label="Sim", linewidth=1.5)
        ax.set_ylabel(f"{finger}\nTorque (N·m)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.set_title("Torque Magnitude")
        if i == n_fingers - 1:
            ax.set_xlabel("Time (s)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_force_components(
    forces: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    title: str = "Force/Torque Components",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot individual force and torque components.

    Args:
        forces: Force/torque array of shape (N, 6) - [Fx, Fy, Fz, Tx, Ty, Tz]
        timestamps: Optional time array
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    forces = np.atleast_2d(forces)

    if timestamps is None:
        timestamps = np.arange(len(forces))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    component_names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    units = ["N", "N", "N", "N·m", "N·m", "N·m"]
    colors = ["red", "green", "blue", "orange", "purple", "brown"]

    for i, (name, unit, color) in enumerate(zip(component_names, units, colors)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        ax.plot(timestamps, forces[:, i], color=color, linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{name} ({unit})")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_fingertip_forces_3d(
    forces: Dict[str, np.ndarray],
    positions: Dict[str, np.ndarray],
    scale: float = 0.01,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot 3D visualization of fingertip positions and force vectors.

    Args:
        forces: Dict with finger names as keys, (3,) force vectors as values
        positions: Dict with finger names as keys, (3,) position vectors as values
        scale: Scale factor for force arrows
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = {"thumb": "red", "index": "blue", "middle": "green", "ring": "orange", "pinky": "purple"}

    for finger in forces.keys():
        if finger not in positions:
            continue

        pos = positions[finger]
        force = forces[finger]
        color = colors.get(finger, "gray")

        # Plot position
        ax.scatter(*pos, c=color, s=100, label=finger)

        # Plot force vector
        ax.quiver(
            pos[0], pos[1], pos[2],
            force[0] * scale, force[1] * scale, force[2] * scale,
            color=color, arrow_length_ratio=0.1
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Fingertip Positions and Forces")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_current_torque_scatter(
    current: np.ndarray,
    torque: np.ndarray,
    k_gain: Optional[float] = None,
    k_offset: Optional[float] = None,
    joint_name: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot current vs torque scatter with optional calibration line.

    Args:
        current: Motor current array
        torque: Joint torque array
        k_gain: Optional calibration gain
        k_offset: Optional calibration offset
        joint_name: Name of joint for title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(current, torque, alpha=0.6, s=30, label="Data points")

    if k_gain is not None and k_offset is not None:
        current_line = np.linspace(np.min(current), np.max(current), 100)
        torque_line = k_gain * current_line + k_offset
        ax.plot(
            current_line, torque_line, "r-", linewidth=2,
            label=f"Fit: τ = {k_gain:.4f}×I + {k_offset:.4f}"
        )

    ax.set_xlabel("Motor Current (A)")
    ax.set_ylabel("Joint Torque (N·m)")
    ax.set_title(f"Current-Torque Relationship{': ' + joint_name if joint_name else ''}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
