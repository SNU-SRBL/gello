# Copyright (c) 2025, SRBL
# Loss functions for calibration

from __future__ import annotations

import torch
import numpy as np


def mse_loss(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    """Mean squared error loss.

    Args:
        pred: Predicted values.
        target: Target values.

    Returns:
        MSE loss value.
    """
    if isinstance(pred, np.ndarray):
        return float(np.mean((pred - target) ** 2))
    return float(torch.mean((pred - target) ** 2).item())


def mae_loss(pred: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> float:
    """Mean absolute error loss.

    Args:
        pred: Predicted values.
        target: Target values.

    Returns:
        MAE loss value.
    """
    if isinstance(pred, np.ndarray):
        return float(np.mean(np.abs(pred - target)))
    return float(torch.mean(torch.abs(pred - target)).item())


def slip_matching_loss(
    sim_I_slip: float | torch.Tensor,
    real_I_slip: float | torch.Tensor,
    no_slip_penalty: float = 1e6,
) -> float:
    """Slip current matching loss.

    Args:
        sim_I_slip: Simulated slip current.
        real_I_slip: Real slip current.
        no_slip_penalty: Penalty if no slip detected.

    Returns:
        Loss value.
    """
    if sim_I_slip is None:
        return no_slip_penalty

    if isinstance(sim_I_slip, torch.Tensor):
        sim_I_slip = sim_I_slip.item() if sim_I_slip.numel() == 1 else sim_I_slip.mean().item()
    if isinstance(real_I_slip, torch.Tensor):
        real_I_slip = real_I_slip.item() if real_I_slip.numel() == 1 else real_I_slip.mean().item()

    return abs(sim_I_slip - real_I_slip)


def weighted_joint_loss(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    weights: torch.Tensor | np.ndarray | None = None,
) -> float:
    """Weighted loss across joints.

    Args:
        pred: Predicted values, shape (..., num_joints).
        target: Target values, same shape.
        weights: Per-joint weights, shape (num_joints,).

    Returns:
        Weighted loss value.
    """
    if weights is None:
        return mse_loss(pred, target)

    if isinstance(pred, np.ndarray):
        diff_sq = (pred - target) ** 2
        return float(np.mean(weights * diff_sq))

    diff_sq = (pred - target) ** 2
    return float(torch.mean(weights * diff_sq).item())


def trajectory_loss(
    sim_positions: torch.Tensor,
    real_positions: torch.Tensor,
    sim_velocities: torch.Tensor | None = None,
    real_velocities: torch.Tensor | None = None,
    pos_weight: float = 1.0,
    vel_weight: float = 0.1,
) -> float:
    """Combined trajectory tracking loss.

    Args:
        sim_positions: Simulated positions, shape (T, num_joints).
        real_positions: Real positions, same shape.
        sim_velocities: Simulated velocities (optional).
        real_velocities: Real velocities (optional).
        pos_weight: Position loss weight.
        vel_weight: Velocity loss weight.

    Returns:
        Combined loss value.
    """
    pos_loss = mse_loss(sim_positions, real_positions)

    if sim_velocities is not None and real_velocities is not None:
        vel_loss = mse_loss(sim_velocities, real_velocities)
    else:
        vel_loss = 0.0

    return pos_weight * pos_loss + vel_weight * vel_loss


def force_matching_loss(
    sim_forces: torch.Tensor | np.ndarray,
    real_forces: torch.Tensor | np.ndarray,
    threshold: float = 0.1,
) -> float:
    """Force matching loss with threshold.

    Args:
        sim_forces: Simulated forces.
        real_forces: Real forces.
        threshold: Minimum force threshold for matching.

    Returns:
        Loss value.
    """
    if isinstance(sim_forces, torch.Tensor):
        sim_forces = sim_forces.cpu().numpy()
    if isinstance(real_forces, torch.Tensor):
        real_forces = real_forces.cpu().numpy()

    # Only compare where forces are above threshold
    mask = (np.abs(real_forces) > threshold) | (np.abs(sim_forces) > threshold)

    if not mask.any():
        return 0.0

    return float(np.mean((sim_forces[mask] - real_forces[mask]) ** 2))


# ===== SIMPLER Style Loss Functions (Phase 0) =====

def simpler_translation_loss(
    real_ee_pos: np.ndarray,
    sim_ee_pos: np.ndarray,
) -> float:
    """SIMPLER Eq. 3: Translation loss.

    L_transl = (1/T) * sum(||x_i - x'_i||^2)

    Args:
        real_ee_pos: Real EE positions, shape (T, 3) or (T, num_ee, 3).
        sim_ee_pos: Simulated EE positions, same shape.

    Returns:
        Translation loss value.
    """
    diff = real_ee_pos - sim_ee_pos
    return float(np.mean(np.sum(diff ** 2, axis=-1)))


def simpler_rotation_loss(
    real_ee_rot: np.ndarray,
    sim_ee_rot: np.ndarray,
) -> float:
    """SIMPLER Eq. 4: Rotation loss using geodesic distance.

    L_rot = (1/T) * sum(arcsin(||R_i - R'_i||_F / (2*sqrt(2))))

    For quaternions, we use the angle between them:
    angle = 2 * arccos(|q1 · q2|)

    Args:
        real_ee_rot: Real EE rotations (quaternions wxyz), shape (T, 4) or (T, num_ee, 4).
        sim_ee_rot: Simulated EE rotations, same shape.

    Returns:
        Rotation loss value (in radians).
    """
    # Normalize quaternions
    real_norm = real_ee_rot / (np.linalg.norm(real_ee_rot, axis=-1, keepdims=True) + 1e-8)
    sim_norm = sim_ee_rot / (np.linalg.norm(sim_ee_rot, axis=-1, keepdims=True) + 1e-8)

    # Dot product between quaternions
    dot = np.abs(np.sum(real_norm * sim_norm, axis=-1))
    dot = np.clip(dot, 0.0, 1.0)

    # Geodesic distance (angle between rotations)
    angle = 2.0 * np.arccos(dot)

    return float(np.mean(angle))


def simpler_sysid_loss(
    real_ee_pos: np.ndarray,
    sim_ee_pos: np.ndarray,
    real_ee_rot: np.ndarray,
    sim_ee_rot: np.ndarray,
    real_joint_pos: np.ndarray | None = None,
    sim_joint_pos: np.ndarray | None = None,
    joint_weight: float = 0.1,
) -> tuple[float, dict[str, float]]:
    """SIMPLER Eq. 5: Combined system identification loss.

    L_sysid = L_transl + L_rot + lambda * L_joint

    Args:
        real_ee_pos: Real EE positions, shape (T, 3) or (T, num_ee, 3).
        sim_ee_pos: Simulated EE positions.
        real_ee_rot: Real EE rotations (quaternions).
        sim_ee_rot: Simulated EE rotations.
        real_joint_pos: Real joint positions (optional).
        sim_joint_pos: Simulated joint positions (optional).
        joint_weight: Weight for joint position loss.

    Returns:
        Tuple of (total_loss, loss_components dict).
    """
    l_transl = simpler_translation_loss(real_ee_pos, sim_ee_pos)
    l_rot = simpler_rotation_loss(real_ee_rot, sim_ee_rot)

    if real_joint_pos is not None and sim_joint_pos is not None:
        l_joint = mse_loss(real_joint_pos, sim_joint_pos)
    else:
        l_joint = 0.0

    total = l_transl + l_rot + joint_weight * l_joint

    return total, {
        "translation": l_transl,
        "rotation": l_rot,
        "joint": l_joint,
        "total": total,
    }


def per_finger_simpler_loss(
    finger_name: str,
    real_ee_pos: np.ndarray,
    sim_ee_pos: np.ndarray,
    real_ee_rot: np.ndarray,
    sim_ee_rot: np.ndarray,
    real_joint_pos: np.ndarray | None = None,
    sim_joint_pos: np.ndarray | None = None,
    joint_weight: float = 0.1,
) -> tuple[float, dict[str, float]]:
    """SIMPLER loss for a single finger.

    Args:
        finger_name: Name of the finger (for logging).
        real_ee_pos: Real fingertip position, shape (T, 3).
        sim_ee_pos: Simulated fingertip position.
        real_ee_rot: Real fingertip rotation (quaternion).
        sim_ee_rot: Simulated fingertip rotation.
        real_joint_pos: Real finger joint positions, shape (T, 4).
        sim_joint_pos: Simulated finger joint positions.
        joint_weight: Weight for joint position loss.

    Returns:
        Tuple of (total_loss, loss_components dict).
    """
    total, components = simpler_sysid_loss(
        real_ee_pos, sim_ee_pos,
        real_ee_rot, sim_ee_rot,
        real_joint_pos, sim_joint_pos,
        joint_weight,
    )

    # Add finger name prefix to components
    return total, {f"{finger_name}_{k}": v for k, v in components.items()}


def hand_simpler_loss(
    real_fingertip_pos: np.ndarray,
    sim_fingertip_pos: np.ndarray,
    real_fingertip_rot: np.ndarray,
    sim_fingertip_rot: np.ndarray,
    real_joint_pos: np.ndarray,
    sim_joint_pos: np.ndarray,
    finger_config: dict,
    joint_weight: float = 0.1,
) -> tuple[float, dict[str, float]]:
    """SIMPLER loss for entire hand with per-finger decomposition.

    Args:
        real_fingertip_pos: Real fingertip positions, shape (T, 5, 3).
        sim_fingertip_pos: Simulated fingertip positions.
        real_fingertip_rot: Real fingertip rotations, shape (T, 5, 4).
        sim_fingertip_rot: Simulated fingertip rotations.
        real_joint_pos: Real hand joint positions, shape (T, 20).
        sim_joint_pos: Simulated hand joint positions.
        finger_config: Dictionary mapping finger names to {joint_indices, ee_index}.
        joint_weight: Weight for joint position loss.

    Returns:
        Tuple of (total_loss, per-finger loss components).
    """
    total_loss = 0.0
    all_components = {}

    for finger_name, cfg in finger_config.items():
        joint_idx = cfg["joint_indices"] if isinstance(cfg, dict) else cfg.joint_indices
        ee_idx = cfg["ee_index"] if isinstance(cfg, dict) else cfg.ee_index

        # Extract finger-specific data
        finger_ee_pos = real_fingertip_pos[:, ee_idx, :]
        finger_ee_sim_pos = sim_fingertip_pos[:, ee_idx, :]
        finger_ee_rot = real_fingertip_rot[:, ee_idx, :]
        finger_ee_sim_rot = sim_fingertip_rot[:, ee_idx, :]
        finger_joints = real_joint_pos[:, joint_idx]
        finger_joints_sim = sim_joint_pos[:, joint_idx]

        # Compute per-finger loss
        finger_loss, components = per_finger_simpler_loss(
            finger_name,
            finger_ee_pos, finger_ee_sim_pos,
            finger_ee_rot, finger_ee_sim_rot,
            finger_joints, finger_joints_sim,
            joint_weight,
        )

        total_loss += finger_loss
        all_components.update(components)

    all_components["total"] = total_loss

    return total_loss, all_components
