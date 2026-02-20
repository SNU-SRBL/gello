#!/usr/bin/env python3
# Copyright (c) 2025, SRBL
# Slip Detection Test Environment - Same structure as Tesollo demo

"""Slip Detection Test Environment

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab

    # Interactive mode
    python scripts/slip_test_env.py

    # Sweep test
    python scripts/slip_test_env.py --sweep_test --duration 5

    # With output file
    python scripts/slip_test_env.py --sweep_test --output slip_data.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Slip Detection Test Environment")
parser.add_argument("--sweep_test", action="store_true", help="Run automatic grip sweep test")
parser.add_argument("--duration", type=float, default=5.0, help="Test duration (s)")
parser.add_argument("--output", type=str, default="", help="Output file path (.npz)")
parser.add_argument("--force_threshold", type=float, default=0.5, help="Contact force threshold (N)")
parser.add_argument("--velocity_threshold", type=float, default=0.005, help="Slip velocity threshold (m/s)")
parser.add_argument("--headless", action="store_true", help="Run headless")
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdGeom, Gf, UsdShade, UsdPhysics

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import signal

REAL2SIM_DIR = Path(__file__).parent.parent
TESOLLO_DIR = Path("/home/Isaac/workspace/HD/test-finger-sensor/Tesollo")
sys.path.insert(0, str(TESOLLO_DIR))
sys.path.insert(0, str(REAL2SIM_DIR / "envs" / "phase2_friction"))

from callback.tesollo_ft_sensor import SRBL_TesolloFTSensor
from slip_detection import RelativeVelocitySlipDetector
from slip_data_collector import SlipDataCollector


class SlipTestEnv:
    """Slip detection test environment using Tesollo hand."""

    def __init__(self, args):
        self.args = args
        self.my_world = World(stage_units_in_meters=1.0, backend="numpy")
        self.stage = simulation_app.context.get_stage()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self.num_fingers = 5

        # Slip detection
        self.slip_detector = RelativeVelocitySlipDetector(
            contact_force_threshold=args.force_threshold,
            slip_velocity_threshold=args.velocity_threshold,
            device=self.device,
            num_envs=1,
            num_fingers=self.num_fingers,
        )

        # Data collection
        self.data_collector = SlipDataCollector(
            num_fingers=self.num_fingers,
            record_joint_state=True,
        )

        # Components
        self.robot = None
        self.cube = None
        self.ft_sensor = None

        # Velocity tracking via position history
        self.prev_fingertip_pos = None
        self.prev_object_pos = None
        self.prev_time = 0.0
        self.filtered_fingertip_vel = None
        self.filtered_object_vel = None
        self.velocity_filter_alpha = 0.3  # Lower = more smoothing (0.1~0.5)

        # Grip control
        self.grip_start = 0.8  # Starting grip level
        self.grip_level = self.grip_start
        self.sweep_active = args.sweep_test
        self.sweep_start_time = None
        self.sweep_duration = args.duration
        self.grip_max = 1.1  # Maximum grip level

        # Stats
        self.sim_time = 0.0
        self.step_count = 0

        # Ring finger time-series data for plotting
        self.ring_data = {
            'timestamps': [],
            'joint_positions': [],   # Ring finger joint positions [4 joints]
            'joint_velocities': [],  # Ring finger joint velocities [4 joints]
            'motor_currents': [],    # Joint efforts (approximated as current)
            'ft_forces': [],         # F/T sensor force [Fx, Fy, Fz]
            'ft_torques': [],        # F/T sensor torque [Tx, Ty, Tz]
            'relative_vel': [],      # Relative velocity magnitude
            'is_slipping': [],       # Slip status
        }

    def setup_scene(self):
        """Setup scene identical to Tesollo demo."""
        self.my_world._physics_context.set_gravity(-9.81)
        self.my_world.scene.add_default_ground_plane()

        set_camera_view(eye=[0.5, 0.5, 0.4], target=[0.0, 0.0, 0.15])

        # Load Tesollo hand USD
        usd_path = str(TESOLLO_DIR / "assets" / "dg5f_L_final.usd")
        print(f"Loading USD: {usd_path}")

        prim = add_reference_to_stage(usd_path=usd_path, prim_path="/World/Dexhand")

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(0.23, 0, 0.15))
        rotate_op = xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat)
        rotate_op.Set(Gf.Vec3d(0, -90, 0))  # palm up

        self.robot = self.my_world.scene.add(
            Robot(prim_path="/World/Dexhand", name="tesollo_hand")
        )

        # Create cube
        self.cube = DynamicCuboid(
            prim_path="/World/Cube",
            name="interactive_cube",
            position=np.array([0.125, 0.015, 0.195]),
            scale=np.array([0.03, 0.15, 0.03]),
            color=np.array([0.2, 0.6, 1.0]),
            mass=0.1,  # 100g
        )
        self.my_world.scene.add(self.cube)

        self._setup_cube_physics()

        # Set joint stiffness and damping via USD (before first reset)
        self._set_joint_drives(stiffness=4.0, damping=0.2)

        # Setup F/T sensor
        self.ft_sensor = SRBL_TesolloFTSensor("/World/Dexhand")
        self.ft_sensor.setup(self.my_world, self.robot)

        self._setup_contact_sensing()

        self.my_world.reset(soft=False)

        self.ft_sensor.post_reset()

        # Store fingertip prim paths for position tracking
        self.fingertip_prim_paths = {}
        for finger in self.finger_names:
            prim_path = f"/World/Dexhand/dg5f_left_flattened/pad_4_{finger}"
            if self.stage.GetPrimAtPath(prim_path):
                self.fingertip_prim_paths[finger] = prim_path
            else:
                print(f"[Warning] Prim not found: {prim_path}")

        print(f"[Info] Fingertip prims for velocity tracking: {list(self.fingertip_prim_paths.keys())}")

        # Initialize joint positions
        num_dof = self.robot.num_dof
        zero_positions = np.zeros(num_dof)
        self.robot.set_joint_positions(zero_positions)

        # Add physics callback
        self.my_world.add_physics_callback("slip_detection", callback_fn=self.physics_step)

        self.my_world.reset(soft=False)

        print(f"\n[Info] Robot DOFs: {self.robot.num_dof}")
        print(f"[Info] Joint names: {self.robot.dof_names[:5]}...")

    def _set_joint_drives(self, stiffness: float = 4.0, damping: float = 0.2):
        """Set joint drive stiffness and damping via USD.

        Args:
            stiffness: Joint position drive stiffness.
            damping: Joint velocity drive damping.
        """
        from pxr import Usd

        hand_prim = self.stage.GetPrimAtPath("/World/Dexhand")
        if not hand_prim:
            print("[Warning] Hand prim not found for joint drive setup")
            return

        count = 0
        # Iterate through all descendants
        for prim in Usd.PrimRange(hand_prim):
            # Find joint prims (revolute joints)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                # Apply drive API
                drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
                if not drive_api:
                    drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")

                drive_api.CreateStiffnessAttr(stiffness)
                drive_api.CreateDampingAttr(damping)
                count += 1

        print(f"[Info] Set drive for {count} joints (stiffness={stiffness}, damping={damping})")

    def _setup_cube_physics(self):
        """Configure cube physics."""
        cube_prim = self.stage.GetPrimAtPath("/World/Cube")

        material = UsdShade.Material.Define(self.stage, "/World/Materials/CubeMaterial")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(0.6)
        physics_material.CreateDynamicFrictionAttr(0.5)
        physics_material.CreateRestitutionAttr(0.2)

        bind_api = UsdShade.MaterialBindingAPI.Apply(cube_prim)
        bind_api.Bind(material)

    def _setup_contact_sensing(self):
        """Setup contact sensors."""
        self.contact_sensors = {}
        for finger in self.finger_names:
            try:
                sensor = RigidPrim(
                    prim_paths_expr=f"/World/Dexhand/dg5f_left_flattened/pad_[1-7]_{finger}",
                    name=f"{finger}_contact",
                    contact_filter_prim_paths_expr=["/World/Cube"],
                    max_contact_count=7 * 5,
                )
                self.my_world.scene.add(sensor)
                self.contact_sensors[finger] = sensor
            except Exception as e:
                print(f"[Warning] Contact sensor for {finger} failed: {e}")

    def get_fingertip_positions(self) -> torch.Tensor:
        """Get fingertip positions from stage prims."""
        positions = torch.zeros(1, self.num_fingers, 3, device=self.device)

        for i, finger in enumerate(self.finger_names):
            if finger in self.fingertip_prim_paths:
                prim = self.stage.GetPrimAtPath(self.fingertip_prim_paths[finger])
                if prim:
                    xform = UsdGeom.Xformable(prim)
                    world_transform = xform.ComputeLocalToWorldTransform(0)
                    translation = world_transform.ExtractTranslation()
                    positions[0, i, :] = torch.tensor(
                        [translation[0], translation[1], translation[2]],
                        device=self.device
                    )

        return positions

    def get_fingertip_velocities(self, dt: float) -> torch.Tensor:
        """Get fingertip velocities from position difference (no filter)."""
        current_pos = self.get_fingertip_positions()

        if self.prev_fingertip_pos is None:
            self.prev_fingertip_pos = current_pos.clone()
            return torch.zeros(1, self.num_fingers, 3, device=self.device)

        # Compute velocity from position difference
        velocity = (current_pos - self.prev_fingertip_pos) / max(dt, 1e-6)

        # Update history
        self.prev_fingertip_pos = current_pos.clone()

        return velocity

    def get_fingertip_forces(self) -> torch.Tensor:
        """Get fingertip forces from F/T sensor."""
        forces = torch.zeros(1, self.num_fingers, 6, device=self.device)

        if self.ft_sensor is not None:
            resultants = self.ft_sensor.get_all_resultants()
            for i, finger in enumerate(self.finger_names):
                force = resultants.get(finger, np.zeros(6))
                forces[0, i, :] = torch.tensor(force, device=self.device)

        return forces

    def get_object_velocity(self, dt: float) -> torch.Tensor:
        """Get cube velocity from position difference (no filter)."""
        pos, _ = self.cube.get_world_pose()
        if pos is None:
            return torch.zeros(1, 3, device=self.device)

        current_pos = torch.tensor(pos, device=self.device).unsqueeze(0)  # (1, 3)

        if self.prev_object_pos is None:
            self.prev_object_pos = current_pos.clone()
            return torch.zeros(1, 3, device=self.device)

        # Calculate velocity from position difference
        velocity = (current_pos - self.prev_object_pos) / max(dt, 1e-6)

        # Update history
        self.prev_object_pos = current_pos.clone()

        return velocity

    def get_object_position(self) -> np.ndarray:
        """Get cube position."""
        pos, _ = self.cube.get_world_pose()
        return pos if pos is not None else np.zeros(3)

    def physics_step(self, step_size):
        """Physics callback - called every simulation step."""
        if self.robot is None:
            return

        self.sim_time += step_size
        self.step_count += 1

        # Grip sweep control
        if self.sweep_active:
            if self.sweep_start_time is None:
                self.sweep_start_time = self.sim_time

            progress = (self.sim_time - self.sweep_start_time) / self.sweep_duration
            if progress >= 1.0:
                self.sweep_active = False
                self.grip_level = self.grip_max  # Hold at max grip
                print("\n[Info] Grip sweep complete! Holding at max grip...")
            else:
                # Sweep from grip_start to grip_max
                self.grip_level = self.grip_start + progress * (self.grip_max - self.grip_start)

        # Get data for slip detection
        fingertip_vel = self.get_fingertip_velocities(step_size)
        object_vel = self.get_object_velocity(step_size)
        contact_forces = self.get_fingertip_forces()

        # Detect slip
        rel_vel, is_contact, is_slipping = self.slip_detector.update(
            fingertip_vel=fingertip_vel,
            object_vel=object_vel,
            contact_forces=contact_forces,
            timestamp=self.sim_time,
        )

        # Skip first 240 steps (~4 seconds, simulation settling)
        if self.step_count >= 250:
            # Log data
            self.data_collector.log_step(
                timestamp=self.sim_time,
                fingertip_vel=fingertip_vel,
                object_vel=object_vel,
                relative_vel=rel_vel,
                contact_forces=contact_forces,
                is_contact=is_contact,
                is_slipping=is_slipping,
                joint_positions=torch.tensor(self.robot.get_joint_positions(), device=self.device),
                object_position=torch.tensor(self.get_object_position(), device=self.device),
            )

            # Record ring finger data for plotting
            ring_idx = 3
            # Use measured efforts (actual computed torques from physics)
            joint_efforts = self.robot.get_measured_joint_efforts()
            joint_positions = self.robot.get_joint_positions()
            joint_velocities = self.robot.get_joint_velocities()
            ring_joint_indices = self._get_ring_joint_indices()

            ring_efforts = joint_efforts[ring_joint_indices] if ring_joint_indices else np.zeros(4)
            ring_joint_pos = joint_positions[ring_joint_indices] if ring_joint_indices else np.zeros(4)
            ring_joint_vel = joint_velocities[ring_joint_indices] if ring_joint_indices else np.zeros(4)

            self.ring_data['timestamps'].append(self.sim_time)
            self.ring_data['joint_positions'].append(ring_joint_pos.copy())
            self.ring_data['joint_velocities'].append(ring_joint_vel.copy())
            self.ring_data['motor_currents'].append(ring_efforts.copy())  # Store all 4 joint efforts
            self.ring_data['ft_forces'].append(contact_forces[0, ring_idx, :3].cpu().numpy().copy())
            self.ring_data['ft_torques'].append(contact_forces[0, ring_idx, 3:6].cpu().numpy().copy())
            self.ring_data['relative_vel'].append(torch.norm(rel_vel[0, ring_idx, :]).item())
            self.ring_data['is_slipping'].append(is_slipping[0, ring_idx].item())

        # Print status every ~1 second
        if self.step_count % 60 == 0:
            contact_count = is_contact.sum().item()
            slip_count = is_slipping.sum().item()
            rel_vel_mag = torch.norm(rel_vel, dim=-1).max().item()

            # Ring finger (index 3) debug
            ring_idx = 3
            ring_fingertip_vel = fingertip_vel[0, ring_idx, :]
            ring_obj_vel = object_vel[0, :]
            ring_rel_vel = rel_vel[0, ring_idx, :]
            ring_force = contact_forces[0, ring_idx, :3]

            status = f"t={self.sim_time:5.1f}s | grip={self.grip_level:.2f} | "
            status += f"contact={contact_count}/{self.num_fingers} | "
            status += f"slip={slip_count}/{self.num_fingers} | "
            status += f"rel_vel={rel_vel_mag:.4f}m/s"

            if is_slipping.any():
                status += " [SLIP!]"

            print(status)
            print(f"  [ring] tip_vel={torch.norm(ring_fingertip_vel):.4f} | obj_vel={torch.norm(ring_obj_vel):.4f} | rel={torch.norm(ring_rel_vel):.4f} | F={torch.norm(ring_force):.2f}N")

    def compute_joint_targets(self) -> np.ndarray:
        """Compute joint targets from grip level.

        Joint naming: lj_dg_X_Y where X=finger(1-5), Y=joint(1-4)
          - X=1: thumb - fixed at current position
          - _1 (Y=1): abduction/adduction - kept at 0
          - _2, _3, _4: flexion joints - controlled by grip_level
        """
        num_dof = self.robot.num_dof
        targets = np.zeros(num_dof)

        dof_names = self.robot.dof_names
        for i, name in enumerate(dof_names):
            # Thumb (finger 1) - fixed at 0
            if "lj_dg_1_" in name:
                targets[i] = 0.0
            # Abduction joints (Y=1) - fixed at 0
            elif name.endswith("_1"):
                targets[i] = 0.0
            # Flexion joints (Y=2,3,4) - controlled by grip_level
            elif name.endswith("_2") or name.endswith("_3") or name.endswith("_4"):
                targets[i] = self.grip_level * 1.2  # 0 ~ 1.2 rad

        return targets

    def _get_ring_joint_indices(self) -> list:
        """Get joint indices for ring finger (finger 4 in naming convention)."""
        indices = []
        dof_names = self.robot.dof_names
        for i, name in enumerate(dof_names):
            if "lj_dg_4_" in name:  # Ring finger is finger 4
                indices.append(i)
        return indices

    def _save_ring_plots(self, output_dir: Path):
        """Save ring finger time-series plots (data is saved in main file)."""
        if len(self.ring_data['timestamps']) == 0:
            print("[Warning] No ring finger data to plot")
            return

        timestamps = np.array(self.ring_data['timestamps'])
        joint_positions = np.array(self.ring_data['joint_positions'])  # (T, 4)
        joint_velocities = np.array(self.ring_data['joint_velocities'])  # (T, 4)
        motor_currents = np.array(self.ring_data['motor_currents'])
        ft_forces = np.array(self.ring_data['ft_forces'])  # (T, 3)
        ft_forces[:, 2] = -ft_forces[:, 2]  # Flip Fz sign only
        ft_torques = np.array(self.ring_data['ft_torques'])  # (T, 3)
        ft_torques[:, 2] = -ft_torques[:, 2]  # Flip Tz sign only
        relative_vel = np.array(self.ring_data['relative_vel'])
        is_slipping = np.array(self.ring_data['is_slipping'])

        # Create figure with subplots (7 panels)
        fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
        fig.suptitle('Ring Finger Slip Detection Data', fontsize=14)

        # 1. Joint Positions
        ax1 = axes[0]
        joint_labels = ['Joint 1 (Abd)', 'Joint 2 (MCP)', 'Joint 3 (PIP)', 'Joint 4 (DIP)']
        colors = ['r', 'g', 'b', 'm']
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

        # 3. Motor Current (Joint Effort) - All 4 joints
        ax3 = axes[2]
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
        ax6.axhline(y=self.args.velocity_threshold * 1000, color='r', linestyle='--',
                    linewidth=1, label=f'Threshold ({self.args.velocity_threshold*1000:.1f} mm/s)')

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

        # 7. Overlay: Force (Fx, Fy, Fz, |F|), Motor Effort (4 joints), and Slip
        ax7 = axes[6]
        ax7_twin = ax7.twinx()

        # Plot all force components + magnitude
        ax7.plot(timestamps, ft_forces[:, 0], 'r-', linewidth=0.8, alpha=0.6, label='Fx')
        ax7.plot(timestamps, ft_forces[:, 1], 'g-', linewidth=0.8, alpha=0.6, label='Fy')
        ax7.plot(timestamps, ft_forces[:, 2], 'b-', linewidth=0.8, alpha=0.6, label='-Fz')
        ax7.plot(timestamps, force_mag, 'k-', linewidth=1.2, label='|F|')

        # Motor efforts on secondary axis (all 4 joints)
        effort_colors = ['c', 'm', 'tab:orange', 'tab:brown']
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
        ax7_twin.tick_params(axis='y')
        ax7.set_title('Overlay: F/T Forces (Fx,Fy,Fz,|F|), Motor Efforts (4 joints) & Slip')
        ax7.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax7.get_legend_handles_labels()
        lines2, labels2 = ax7_twin.get_legend_handles_labels()
        ax7.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

        plt.tight_layout()

        # Save plot only (data is saved in single file via _save_data)
        plot_path = output_dir / 'ring_finger_plot.png'
        print(f"[Info] Saving plot to {plot_path}...")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[Info] Saved ring finger plot to {plot_path}")
        plt.close()

    def run(self):
        """Main simulation loop."""
        print("\n" + "=" * 60)
        if self.sweep_active:
            print(f"Starting grip sweep test (duration: {self.sweep_duration}s)")
        else:
            print("Interactive mode - move cube manually")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")

        # Flag for graceful shutdown
        self._shutdown_requested = False

        def signal_handler(signum, frame):
            print("\n\n[Info] Ctrl+C received, saving data...")
            self._shutdown_requested = True

        # Register signal handler
        original_handler = signal.signal(signal.SIGINT, signal_handler)

        try:
            while simulation_app.is_running() and not self._shutdown_requested:
                if not self.my_world.is_playing():
                    self.my_world.step(render=True)
                    continue

                # Apply grip control
                targets = self.compute_joint_targets()
                action = ArticulationAction(joint_positions=targets)
                self.robot.apply_action(action)

                self.my_world.step(render=True)

                # Continue running after sweep completes (user can Ctrl+C to stop)

        except Exception as e:
            print(f"\n\nError during simulation: {e}")
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)

            # Always try to save data
            print("\n[Info] Saving data...")
            try:
                self._print_summary()
                self._save_data()
            except Exception as e:
                print(f"[Error] Failed to save: {e}")
                import traceback
                traceback.print_exc()

            simulation_app.close()

    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        summary = self.data_collector.get_summary()
        print(f"Total samples: {summary['num_samples']}")
        print(f"Duration: {summary.get('duration_s', 0):.2f}s")
        print(f"Slip ratio: {summary['slip_ratio']*100:.1f}%")
        print(f"Contact ratio: {summary['contact_ratio']*100:.1f}%")
        print(f"Slip events: {summary['num_slip_events']}")
        print(f"Mean slip velocity: {summary['mean_slip_velocity']:.4f} m/s")
        print(f"Max slip velocity: {summary['max_slip_velocity']:.4f} m/s")

        if summary['per_finger_slip_ratio']:
            print("\nPer-finger slip ratio:")
            for i, ratio in enumerate(summary['per_finger_slip_ratio']):
                print(f"  {self.finger_names[i]}: {ratio*100:.1f}%")

    def _save_data(self):
        """Save all collected data to a single file."""
        if self.data_collector.num_samples == 0:
            print("\nNo data to save")
            return

        if self.args.output:
            output_path = Path(self.args.output)
            if not output_path.suffix:
                output_path = output_path.with_suffix(".npz")
            output_dir = output_path.parent if output_path.parent.exists() else REAL2SIM_DIR / "data" / "slip_data"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = REAL2SIM_DIR / "data" / "slip_data"
            output_dir.mkdir(parents=True, exist_ok=True)

            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"slip_data_{timestamp}.npz"

        # Build unified data dict from data_collector
        data = self.data_collector.data
        n_samples = len(data)

        save_dict = {
            # Main slip detection data
            'timestamps': np.array([d.timestamp for d in data]),
            'fingertip_velocities': np.stack([d.fingertip_velocities for d in data]),
            'object_velocities': np.stack([d.object_velocity for d in data]),
            'relative_velocities': np.stack([d.relative_velocities for d in data]),
            'contact_forces': np.stack([d.contact_forces for d in data]),
            'is_contact': np.stack([d.is_contact for d in data]),
            'is_slipping': np.stack([d.is_slipping for d in data]),
            'num_fingers': self.num_fingers,
            'num_samples': n_samples,
        }

        # Optional data from data_collector
        if data[0].joint_positions is not None:
            save_dict['joint_positions'] = np.stack([d.joint_positions for d in data])
        if data[0].object_position is not None:
            save_dict['object_positions'] = np.stack([d.object_position for d in data])

        # Ring finger specific data
        if len(self.ring_data['timestamps']) > 0:
            save_dict['ring_timestamps'] = np.array(self.ring_data['timestamps'])
            save_dict['ring_joint_positions'] = np.array(self.ring_data['joint_positions'])
            save_dict['ring_joint_velocities'] = np.array(self.ring_data['joint_velocities'])
            save_dict['ring_motor_currents'] = np.array(self.ring_data['motor_currents'])
            save_dict['ring_ft_forces'] = np.array(self.ring_data['ft_forces'])
            save_dict['ring_ft_torques'] = np.array(self.ring_data['ft_torques'])
            save_dict['ring_relative_vel'] = np.array(self.ring_data['relative_vel'])
            save_dict['ring_is_slipping'] = np.array(self.ring_data['is_slipping'])

        # Save slip events summary
        if self.data_collector.slip_events:
            save_dict['slip_event_count'] = len(self.data_collector.slip_events)
            save_dict['slip_event_start_times'] = np.array([
                e.start_time for e in self.data_collector.slip_events
            ])
            save_dict['slip_event_fingers'] = np.array([
                e.finger_idx for e in self.data_collector.slip_events
            ])

        # Save all data to single file
        np.savez(output_path, **save_dict)
        print(f"[Info] Saved {n_samples} samples (all data) to {output_path}")

        # Save ring finger plots
        try:
            self._save_ring_plots(output_dir)
        except Exception as e:
            print(f"[Error] Failed to save ring finger plot: {e}")


def main():
    print("=" * 60)
    print("Slip Detection Test Environment")
    print("=" * 60)

    env = SlipTestEnv(args)
    env.setup_scene()
    env.run()


if __name__ == "__main__":
    main()
