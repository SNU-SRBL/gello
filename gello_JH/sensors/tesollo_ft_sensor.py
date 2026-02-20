# Copyright (c) 2025, SRBL
# Tesollo DG5F F/T Sensor for Isaac Lab

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import dataclass

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class TesolloFTSensorCfg:
    """Configuration for Tesollo DG5F F/T sensor."""

    prim_path: str = "{ENV_REGEX_NS}/Robot/dg5f_left"
    """Prim path to the Tesollo hand articulation."""

    fingers: list[str] = ["thumb", "index", "middle", "ring", "pinky"]
    """List of finger names."""

    pads_per_finger: int = 7
    """Number of tactile pads per finger."""

    update_period: float = 0.0
    """Update period in seconds. 0.0 means update every physics step."""


@dataclass
class TesolloFTSensorData:
    """Data class for Tesollo F/T sensor readings."""

    # Per-pad forces: (num_envs, num_fingers, pads_per_finger, 6)
    pad_forces: torch.Tensor | None = None

    # Per-finger resultant forces: (num_envs, num_fingers, 6)
    finger_forces: torch.Tensor | None = None

    # Total force on all fingers: (num_envs, 6)
    total_force: torch.Tensor | None = None


class TesolloFTSensorIsaacLab:
    """Tesollo DG5F F/T Sensor integration for Isaac Lab.

    Provides access to 35 F/T sensors (7 pads x 5 fingers).
    Based on: /home/Isaac/workspace/HD/Tesollo/callback/tesollo_ft_sensor.py

    The sensor reads joint forces from the articulation's pad joints and
    transforms them to a standard coordinate frame.
    """

    FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
    PADS_PER_FINGER = 7
    NUM_SENSORS = 35  # 5 fingers × 7 pads

    def __init__(
        self,
        cfg: TesolloFTSensorCfg,
        articulation: Articulation,
        device: str = "cuda:0",
    ):
        """Initialize the Tesollo F/T sensor.

        Args:
            cfg: Sensor configuration.
            articulation: The Tesollo hand articulation.
            device: Device for tensor operations.
        """
        self.cfg = cfg
        self.articulation = articulation
        self.device = device

        self._num_envs = articulation.num_envs

        # Joint indices for each finger's pads
        self._pad_joint_indices: dict[str, list[int]] = {}

        # Data container
        self.data = TesolloFTSensorData()

        # Initialize indices
        self._setup_indices()

    def _setup_indices(self):
        """Set up pad joint indices from articulation metadata."""
        joint_names = self.articulation.joint_names

        for finger in self.cfg.fingers:
            self._pad_joint_indices[finger] = []
            for pad_idx in range(1, self.cfg.pads_per_finger + 1):
                joint_name = f"pad_{pad_idx}_{finger}_joint"
                if joint_name in joint_names:
                    idx = joint_names.index(joint_name)
                    self._pad_joint_indices[finger].append(idx)

    def update(self):
        """Update sensor readings from simulation."""
        # Get joint forces from articulation PhysX view
        # Shape: (num_envs, num_links, 6) where 6 = [Fx, Fy, Fz, Tx, Ty, Tz]
        joint_forces = self.articulation.root_physx_view.get_link_incoming_joint_force()

        # Extract pad forces for each finger
        pad_forces_list = []
        finger_forces_list = []

        for finger in self.cfg.fingers:
            indices = self._pad_joint_indices.get(finger, [])

            if indices:
                # Get forces for this finger's pads
                # Shape: (num_envs, pads_per_finger, 6)
                finger_pad_forces = joint_forces[:, indices, :]
                pad_forces_list.append(finger_pad_forces)

                # Compute resultant force for this finger
                # Shape: (num_envs, 6)
                resultant = finger_pad_forces.sum(dim=1)
                resultant = self._transform_axes(resultant, finger)
                finger_forces_list.append(resultant)
            else:
                # No pads found for this finger
                pad_forces_list.append(
                    torch.zeros(self._num_envs, self.cfg.pads_per_finger, 6, device=self.device)
                )
                finger_forces_list.append(
                    torch.zeros(self._num_envs, 6, device=self.device)
                )

        # Stack into tensors
        # pad_forces: (num_envs, num_fingers, pads_per_finger, 6)
        self.data.pad_forces = torch.stack(pad_forces_list, dim=1)

        # finger_forces: (num_envs, num_fingers, 6)
        self.data.finger_forces = torch.stack(finger_forces_list, dim=1)

        # total_force: (num_envs, 6)
        self.data.total_force = self.data.finger_forces.sum(dim=1)

    def _transform_axes(self, force_torque: torch.Tensor, finger: str) -> torch.Tensor:
        """Transform F/T sensor axes to standard frame.

        The standard frame has +Z pointing along the pad normal (outward).

        Args:
            force_torque: Force/torque tensor, shape (num_envs, 6)
            finger: Finger name.

        Returns:
            Transformed force/torque tensor, shape (num_envs, 6)
        """
        if finger == "thumb":
            # Thumb: negate X and Y components
            transformed = torch.stack([
                -force_torque[:, 0],  # Fx = -old Fx
                -force_torque[:, 1],  # Fy = -old Fy
                force_torque[:, 2],   # Fz = old Fz
                -force_torque[:, 3],  # Tx = -old Tx
                -force_torque[:, 4],  # Ty = -old Ty
                force_torque[:, 5],   # Tz = old Tz
            ], dim=-1)
        else:
            # Other fingers: permute axes (Y→X, Z→Y, X→Z)
            transformed = torch.stack([
                force_torque[:, 1],   # Fx = old Fy
                force_torque[:, 2],   # Fy = old Fz
                force_torque[:, 0],   # Fz = old Fx
                force_torque[:, 4],   # Tx = old Ty
                force_torque[:, 5],   # Ty = old Tz
                force_torque[:, 3],   # Tz = old Tx
            ], dim=-1)

        return transformed

    # ===== Accessor Methods =====

    def get_finger_forces(self, finger: str) -> torch.Tensor:
        """Get F/T readings for a specific finger's pads.

        Args:
            finger: Finger name ('thumb', 'index', 'middle', 'ring', 'pinky').

        Returns:
            Tensor of shape (num_envs, 7, 6) with [Fx, Fy, Fz, Tx, Ty, Tz] for each pad.
        """
        if self.data.pad_forces is None:
            return torch.zeros(self._num_envs, self.cfg.pads_per_finger, 6, device=self.device)

        finger_idx = self.cfg.fingers.index(finger)
        return self.data.pad_forces[:, finger_idx, :, :]

    def get_all_forces(self) -> dict[str, torch.Tensor]:
        """Get F/T readings for all fingers.

        Returns:
            Dictionary mapping finger name to (num_envs, 7, 6) tensor.
        """
        forces = {}
        for finger in self.cfg.fingers:
            forces[finger] = self.get_finger_forces(finger)
        return forces

    def get_finger_resultant(self, finger: str) -> torch.Tensor:
        """Get resultant F/T for a specific finger.

        Args:
            finger: Finger name.

        Returns:
            Tensor of shape (num_envs, 6) with summed [Fx, Fy, Fz, Tx, Ty, Tz].
        """
        if self.data.finger_forces is None:
            return torch.zeros(self._num_envs, 6, device=self.device)

        finger_idx = self.cfg.fingers.index(finger)
        return self.data.finger_forces[:, finger_idx, :]

    def get_all_resultants(self) -> dict[str, torch.Tensor]:
        """Get resultant F/T for all fingers.

        Returns:
            Dictionary mapping finger name to (num_envs, 6) tensor.
        """
        resultants = {}
        for finger in self.cfg.fingers:
            resultants[finger] = self.get_finger_resultant(finger)
        return resultants

    def get_total_force(self) -> torch.Tensor:
        """Get total F/T across all fingers.

        Returns:
            Tensor of shape (num_envs, 6) with total [Fx, Fy, Fz, Tx, Ty, Tz].
        """
        if self.data.total_force is None:
            return torch.zeros(self._num_envs, 6, device=self.device)
        return self.data.total_force

    def get_flattened_forces(self) -> torch.Tensor:
        """Get all F/T readings as a flattened tensor.

        Returns:
            Tensor of shape (num_envs, 210) = (num_envs, 35 pads × 6 DOF).
        """
        if self.data.pad_forces is None:
            return torch.zeros(self._num_envs, self.NUM_SENSORS * 6, device=self.device)

        # (num_envs, 5, 7, 6) -> (num_envs, 210)
        return self.data.pad_forces.reshape(self._num_envs, -1)

    def get_flattened_resultants(self) -> torch.Tensor:
        """Get finger resultants as a flattened tensor.

        Returns:
            Tensor of shape (num_envs, 30) = (num_envs, 5 fingers × 6 DOF).
        """
        if self.data.finger_forces is None:
            return torch.zeros(self._num_envs, len(self.cfg.fingers) * 6, device=self.device)

        # (num_envs, 5, 6) -> (num_envs, 30)
        return self.data.finger_forces.reshape(self._num_envs, -1)

    # ===== Properties =====

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._num_envs

    @property
    def num_sensors(self) -> int:
        """Total number of sensors (35)."""
        return self.NUM_SENSORS

    @property
    def num_fingers(self) -> int:
        """Number of fingers (5)."""
        return len(self.cfg.fingers)
