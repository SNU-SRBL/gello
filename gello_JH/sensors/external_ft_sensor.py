# Copyright (c) 2025, SRBL
# External F/T Sensor for Phase 1 Calibration

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from dataclasses import dataclass

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject


@configclass
class ExternalFTSensorCfg:
    """Configuration for external F/T sensor fixture."""

    prim_path: str = "{ENV_REGEX_NS}/FTSensorFixture"
    """Prim path to the F/T sensor fixture rigid body."""

    update_period: float = 0.0
    """Update period in seconds. 0.0 means update every physics step."""

    # Coordinate frame
    frame_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of sensor frame from fixture origin."""

    frame_rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion rotation of sensor frame (w, x, y, z)."""


@dataclass
class ExternalFTSensorData:
    """Data class for external F/T sensor readings."""

    # Net contact force on the sensor: (num_envs, 3)
    contact_force: torch.Tensor | None = None

    # Net contact torque on the sensor: (num_envs, 3)
    contact_torque: torch.Tensor | None = None

    # Combined force/torque: (num_envs, 6)
    force_torque: torch.Tensor | None = None


class ExternalFTSensor:
    """External F/T sensor for Phase 1 calibration.

    This sensor reads contact forces from a fixed F/T sensor fixture
    in the simulation. Used for contact-free current-torque calibration.

    In the real experiment:
    - The robot arm presses the hand's fingertip against this sensor
    - The sensor measures the normal and tangential forces
    - This is compared with motor current to establish I-τ relationship
    """

    def __init__(
        self,
        cfg: ExternalFTSensorCfg,
        rigid_object: RigidObject,
        device: str = "cuda:0",
    ):
        """Initialize the external F/T sensor.

        Args:
            cfg: Sensor configuration.
            rigid_object: The F/T sensor fixture rigid body.
            device: Device for tensor operations.
        """
        self.cfg = cfg
        self.rigid_object = rigid_object
        self.device = device

        self._num_envs = rigid_object.num_envs

        # Data container
        self.data = ExternalFTSensorData()

        # Frame transformation
        self._setup_frame_transform()

    def _setup_frame_transform(self):
        """Set up coordinate frame transformation."""
        self._frame_offset = torch.tensor(
            self.cfg.frame_offset, device=self.device
        ).unsqueeze(0).expand(self._num_envs, 3)

        self._frame_rotation = torch.tensor(
            self.cfg.frame_rotation, device=self.device
        ).unsqueeze(0).expand(self._num_envs, 4)

    def update(self):
        """Update sensor readings from simulation."""
        # Get net contact force on the fixture
        # This uses the rigid body's contact force data
        contact_force = self.rigid_object.data.net_contact_force

        # Apply frame transformation if needed
        # For now, assume sensor frame is aligned with body frame
        self.data.contact_force = contact_force[:, :3]

        # Compute torque (simplified - from contact point to frame origin)
        # In practice, this would require contact point information
        self.data.contact_torque = torch.zeros(self._num_envs, 3, device=self.device)

        # Combined force/torque
        self.data.force_torque = torch.cat([
            self.data.contact_force,
            self.data.contact_torque,
        ], dim=-1)

    def get_force(self) -> torch.Tensor:
        """Get contact force reading.

        Returns:
            Tensor of shape (num_envs, 3) with [Fx, Fy, Fz].
        """
        if self.data.contact_force is None:
            return torch.zeros(self._num_envs, 3, device=self.device)
        return self.data.contact_force

    def get_torque(self) -> torch.Tensor:
        """Get contact torque reading.

        Returns:
            Tensor of shape (num_envs, 3) with [Tx, Ty, Tz].
        """
        if self.data.contact_torque is None:
            return torch.zeros(self._num_envs, 3, device=self.device)
        return self.data.contact_torque

    def get_force_torque(self) -> torch.Tensor:
        """Get combined force/torque reading.

        Returns:
            Tensor of shape (num_envs, 6) with [Fx, Fy, Fz, Tx, Ty, Tz].
        """
        if self.data.force_torque is None:
            return torch.zeros(self._num_envs, 6, device=self.device)
        return self.data.force_torque

    def get_normal_force(self) -> torch.Tensor:
        """Get normal (Z-axis) force component.

        Returns:
            Tensor of shape (num_envs,) with Fz.
        """
        return self.get_force()[:, 2]

    def get_tangential_force(self) -> torch.Tensor:
        """Get tangential (XY-plane) force magnitude.

        Returns:
            Tensor of shape (num_envs,) with sqrt(Fx² + Fy²).
        """
        force = self.get_force()
        return torch.norm(force[:, :2], dim=-1)

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._num_envs
