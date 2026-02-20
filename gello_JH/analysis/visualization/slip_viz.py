# Slip Visualization Tools

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple


def plot_slip_events(
    time: np.ndarray,
    current: np.ndarray,
    object_velocity: np.ndarray,
    slip_events: Optional[List[Dict]] = None,
    velocity_threshold: float = 0.001,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot slip test with detected slip events.

    Args:
        time: Time array
        current: Motor current array
        object_velocity: Object velocity array (can be 1D magnitude or 3D vector)
        slip_events: Optional list of slip event dicts with 'time', 'current', 'type'
        velocity_threshold: Velocity threshold for slip detection
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Compute velocity magnitude if 3D
    if object_velocity.ndim > 1:
        velocity_mag = np.linalg.norm(object_velocity, axis=-1)
    else:
        velocity_mag = np.abs(object_velocity)

    # Current plot
    ax = axes[0]
    ax.plot(time, current, "b-", linewidth=2, label="Motor Current")
    ax.set_ylabel("Current (A)")
    ax.set_title("Slip Test: Current Ramp Down")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Mark slip events on current plot
    if slip_events:
        for event in slip_events:
            ax.axvline(x=event["time"], color="r", linestyle="--", alpha=0.7)
            ax.annotate(
                f"I={event['current']:.3f}A",
                xy=(event["time"], event["current"]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
            )

    # Velocity plot
    ax = axes[1]
    ax.plot(time, velocity_mag, "g-", linewidth=2, label="Object Velocity")
    ax.axhline(y=velocity_threshold, color="r", linestyle="--", alpha=0.5, label=f"Threshold ({velocity_threshold} m/s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Object Velocity Magnitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Mark slip events on velocity plot
    if slip_events:
        for event in slip_events:
            ax.axvline(x=event["time"], color="r", linestyle="--", alpha=0.7)

    # Slip indicator
    ax = axes[2]
    slip_indicator = (velocity_mag > velocity_threshold).astype(float)
    ax.fill_between(time, 0, slip_indicator, alpha=0.5, color="red", label="Slip Detected")
    ax.set_ylabel("Slip Status")
    ax.set_xlabel("Time (s)")
    ax.set_title("Slip Detection")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No Slip", "Slip"])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_slip_current_distribution(
    i_slip_values: np.ndarray,
    real_i_slip: Optional[float] = None,
    bins: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot distribution of simulated slip currents.

    Args:
        i_slip_values: Array of I_slip values from multiple trials
        real_i_slip: Optional real robot I_slip for comparison
        bins: Number of histogram bins
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(i_slip_values, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")

    # Statistics
    mean_i = np.mean(i_slip_values)
    std_i = np.std(i_slip_values)
    ax.axvline(x=mean_i, color="blue", linestyle="-", linewidth=2, label=f"Mean: {mean_i:.4f} A")
    ax.axvspan(mean_i - std_i, mean_i + std_i, alpha=0.2, color="blue", label=f"±1σ: {std_i:.4f} A")

    if real_i_slip is not None:
        ax.axvline(x=real_i_slip, color="red", linestyle="--", linewidth=2, label=f"Real I_slip: {real_i_slip:.4f} A")

    ax.set_xlabel("Slip Current I_slip (A)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Simulated Slip Currents")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_friction_sensitivity(
    friction_values: np.ndarray,
    i_slip_values: np.ndarray,
    param_name: str = "Friction Coefficient",
    real_i_slip: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot sensitivity of I_slip to friction parameter.

    Args:
        friction_values: Array of friction parameter values
        i_slip_values: Corresponding I_slip values
        param_name: Name of friction parameter
        real_i_slip: Optional real I_slip target
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(friction_values, i_slip_values, c="steelblue", s=50, alpha=0.7)

    # Fit trend line
    if len(friction_values) > 2:
        z = np.polyfit(friction_values, i_slip_values, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(np.min(friction_values), np.max(friction_values), 100)
        ax.plot(x_smooth, p(x_smooth), "b--", linewidth=2, alpha=0.7, label="Trend")

    if real_i_slip is not None:
        ax.axhline(y=real_i_slip, color="red", linestyle="--", linewidth=2, label=f"Target I_slip: {real_i_slip:.4f} A")

    ax.set_xlabel(param_name)
    ax.set_ylabel("Slip Current I_slip (A)")
    ax.set_title(f"Sensitivity Analysis: {param_name} vs I_slip")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_contact_forces_during_slip(
    time: np.ndarray,
    normal_force: np.ndarray,
    tangent_force: np.ndarray,
    slip_time: Optional[float] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot contact forces during slip test.

    Args:
        time: Time array
        normal_force: Normal force array
        tangent_force: Tangential force array (can be 1D or 2D)
        slip_time: Optional time when slip occurred
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Compute tangent magnitude if 2D
    if tangent_force.ndim > 1:
        tangent_mag = np.linalg.norm(tangent_force, axis=-1)
    else:
        tangent_mag = np.abs(tangent_force)

    # Normal force
    ax = axes[0]
    ax.plot(time, normal_force, "b-", linewidth=2, label="Normal Force")
    ax.set_ylabel("Normal Force (N)")
    ax.set_title("Contact Forces During Slip Test")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if slip_time is not None:
        ax.axvline(x=slip_time, color="r", linestyle="--", alpha=0.7, label="Slip")

    # Tangential force
    ax = axes[1]
    ax.plot(time, tangent_mag, "g-", linewidth=2, label="Tangential Force")
    ax.set_ylabel("Tangential Force (N)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if slip_time is not None:
        ax.axvline(x=slip_time, color="r", linestyle="--", alpha=0.7)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_friction_cone(
    normal_force: float,
    static_friction: float,
    dynamic_friction: float,
    tangent_force_history: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot friction cone visualization.

    Args:
        normal_force: Current normal force
        static_friction: Static friction coefficient
        dynamic_friction: Dynamic friction coefficient
        tangent_force_history: Optional history of tangent forces (N, 2)
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw friction cones
    theta = np.linspace(0, 2 * np.pi, 100)

    # Static friction cone
    r_static = static_friction * normal_force
    x_static = r_static * np.cos(theta)
    y_static = r_static * np.sin(theta)
    ax.plot(x_static, y_static, "b-", linewidth=2, label=f"Static (μ={static_friction})")
    ax.fill(x_static, y_static, alpha=0.1, color="blue")

    # Dynamic friction cone
    r_dynamic = dynamic_friction * normal_force
    x_dynamic = r_dynamic * np.cos(theta)
    y_dynamic = r_dynamic * np.sin(theta)
    ax.plot(x_dynamic, y_dynamic, "r--", linewidth=2, label=f"Dynamic (μ={dynamic_friction})")

    # Plot force history
    if tangent_force_history is not None:
        ax.scatter(
            tangent_force_history[:, 0],
            tangent_force_history[:, 1],
            c=np.arange(len(tangent_force_history)),
            cmap="viridis",
            s=20,
            alpha=0.7,
        )
        ax.scatter(
            tangent_force_history[-1, 0],
            tangent_force_history[-1, 1],
            c="red",
            s=100,
            marker="x",
            label="Current",
        )

    ax.set_xlabel("Tangent Force X (N)")
    ax.set_ylabel("Tangent Force Y (N)")
    ax.set_title(f"Friction Cone (Normal Force = {normal_force:.2f} N)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
