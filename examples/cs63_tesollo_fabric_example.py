# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Example script for CS63 robot with Tesollo gripper fabric.

This demonstrates force-based gripper control where:
- Input: finger forces (instead of positions or PCA targets)
- Conversion: forces → joint torques via Jacobian transpose (τ = J^T * F)
- Control: direct joint space control (no PCA dimensionality reduction)

Example usage:
python cs63_tesollo_fabric_example.py --batch_size=1 --render
"""

# Standard Library
import os
import time
import argparse

# Third party
import torch
import numpy as np

# Fabrics imports
from fabrics_sim.fabrics.cs63_tesollo_fabric import CS63TesolloFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.utils import initialize_warp, capture_fabric
from fabrics_sim.visualization.robot_visualizer import RobotVisualizer
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

# Reduce print precision
torch.set_printoptions(precision=4)

# Settings
use_viz = False
render_spheres = False
cuda_graph = True
batch_size = 2

# Declare device for fabric
device_int = 1
device = 'cuda:' + str(device_int)

# Set the warp cache directory based on device int
initialize_warp(str(device_int))

# Create world model for collision avoidance
print('Importing world')
world_filename = 'floor'  # You can create your own world file
max_objects_per_env = 20
world_model = WorldMeshesModel(batch_size=batch_size,
                               max_objects_per_env=max_objects_per_env,
                               device=device,
                               world_filename=world_filename)

# Get object handles for collision avoidance
object_ids, object_indicator = world_model.get_object_ids()

# Control rate and time settings
control_rate = 60.
timestep = 1./control_rate
total_time = 60.

# Create CS63-Tesollo fabric with force-based gripper control
print('Creating CS63-Tesollo fabric...')
cs63_fabric = CS63TesolloFabric(
    batch_size=batch_size, 
    device=device, 
    timestep=timestep,
    num_arm_joints=6,
    num_gripper_joints=12,
    num_fingers=3,
    graph_capturable=cuda_graph
)
num_joints = cs63_fabric.num_joints
            
# Create integrator for the fabric dynamics
cs63_integrator = DisplacementIntegrator(cs63_fabric)

# Create starting states for the robot
# Format: [6 arm joints, 12 gripper joints (3 fingers x 4 joints each)]
q = torch.tensor([
    # Arm joints (6 joints)
    0.0, 0.0, -0.7853982, 0.0, 1.57, 0.0,
    # Finger 1 joints (4 joints)
    0.0, 0.0, 0.7853982, 0.5235988,
    # Finger 2 joints (4 joints)
    -0.43633232, 0.0, 0.7853982, 0.5235988,
    # Finger 3 joints (4 joints)
    0.43633232, 0.0, 0.7853982, 0.5235988
], device=device)

# Resize according to batch size
q = q.unsqueeze(0).repeat(batch_size, 1).contiguous()

# Start with zero initial velocities and accelerations
qd = torch.zeros(batch_size, num_joints, device=device)
qdd = torch.zeros(batch_size, num_joints, device=device)

# Initialize finger force targets
# Shape: (batch_size x num_fingers x 3) for 3D forces on each fingertip
# These forces will be converted to joint torques via Jacobian transpose
finger_forces = torch.zeros(batch_size, 3, 3, device=device)

# Example: Apply small downward force on all fingertips for grasping
finger_forces[:, :, 2] = -0.5  # -0.5 N in Z direction (downward)

# Palm target pose (position + Euler ZYX angles)
# Format: [x, y, z, euler_z, euler_y, euler_x]
palm_target = torch.tensor([-0.5, 0.0, 0.5, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(batch_size, 1).float().contiguous()

# Get body sphere radii for visualization
body_sphere_radii = cs63_fabric.get_sphere_radii()

# Get body sphere locations
sphere_position, _ = cs63_fabric.get_taskmap("body_points")(q.detach(), None)

# Create visualizer (if rendering enabled)
robot_visualizer = None
if use_viz:
    robot_dir_name = "cs63_tesollo_sim"  # Update based on your robot directory
    robot_name = "cs63_tesollo_sim"
    vertical_offset = 0.
    if render_spheres:
        robot_visualizer = RobotVisualizer(
            robot_dir_name, robot_name, batch_size, device,
            body_sphere_radii, sphere_position,
            world_model, vertical_offset, cs63_fabric.get_joint_names()
        )
    else:
        robot_visualizer = RobotVisualizer(
            robot_dir_name, robot_name, batch_size, device,
            None, None,
            world_model, vertical_offset, cs63_fabric.get_joint_names()
        )

# CUDA graph capture (optional, for performance)
g = None
q_new = None
qd_new = None
qdd_new = None
if cuda_graph:
    print('Capturing CUDA graph...')
    inputs = [
        finger_forces, palm_target, "euler_zyx",
        q.detach(), qd.detach(), object_ids, object_indicator
    ]
    g, q_new, qd_new, qdd_new = capture_fabric(
        cs63_fabric, q, qd, qdd, timestep, cs63_integrator, inputs, device
    )

# Main control loop
print('Starting control loop...')
start = time.time()

for i in range(int(control_rate * total_time)):
    # Every 2 seconds, change finger forces
    if i % 120 == 0:
        # Example 1: Alternate between grasping and releasing
        if (i // 120) % 2 == 0:
            # Grasp: inward forces
            finger_forces[:, :, :] = 0.0
            finger_forces[:, :, 2] = -1.0  # Downward force
        else:
            # Release: outward forces
            finger_forces[:, :, :] = 0.0
            finger_forces[:, :, 2] = 0.5  # Upward force
        
        # Update palm target randomly (position + orientation)
        palm_target[:, 0] = -0.5 - 0.2 * torch.rand(batch_size, device=device)
        palm_target[:, 1] = -0.5 + torch.rand(batch_size, device=device)
        palm_target[:, 2] = 0.3 + 0.4 * torch.rand(batch_size, device=device)
        palm_target[:, 3:] = 2. * np.pi * (torch.rand(batch_size, 3, device=device) - 0.5)
        
        print(f'\nTime {i*timestep:.2f}s: Updated targets')
        print(f'  Finger forces (first finger): {finger_forces[0, 0, :].cpu().numpy()}')
        print(f'  Palm target pos: {palm_target[0, :3].cpu().numpy()}')
        print(f'  Palm target ori: {palm_target[0, 3:].cpu().numpy()}')

    # Save off current joint states for rendering
    q_prev = q.detach()
    qd_prev = qd.detach()

    # Step the fabric forward in time
    if cuda_graph:
        # Replay through the graph with the above changed inputs
        g.replay()
        q.copy_(q_new)
        qd.copy_(qd_new)
        qdd.copy_(qdd_new)
    else: 
        # Set the targets (finger forces + palm pose)
        cs63_fabric.set_features(
            finger_forces, palm_target, "euler_zyx",
            q.detach(), qd.detach(),
            object_ids, object_indicator
        )
        
        # Integrate fabrics one step producing new position and velocity
        q, qd, qdd = cs63_integrator.step(q.detach(), qd.detach(), qdd.detach(), timestep)
    
    # Render at lower framerate
    if use_viz and (i % 4 == 0):
        if render_spheres:
            sphere_position = cs63_fabric.get_taskmap_position("body_points").detach().cpu()
            sphere_position = sphere_position.reshape(
                batch_size * len(body_sphere_radii), -1
            ).detach().cpu().numpy()
        else:
            sphere_position = None

        robot_visualizer.render(
            q_prev.detach().cpu().numpy(),
            qd_prev.detach().cpu().numpy() * 0.,  # Setting to 0 to avoid jitters
            sphere_position,
            palm_target.detach().cpu().numpy()
        )
    
    # Get diagnostics
    dist_to_upper_limit = cs63_fabric.get_taskmap_position("upper_joint_limit")
    dist_to_lower_limit = cs63_fabric.get_taskmap_position("lower_joint_limit")
    collision = cs63_fabric.collision_status.max().item()

    # Print status every second
    if i % int(control_rate) == 0:
        print(f'Time: {i*timestep:5.2f}s | '
              f'Wallclock: {time.time()-start:5.2f}s | '
              f'Upper lim: {dist_to_upper_limit.min().item():6.3f} | '
              f'Lower lim: {dist_to_lower_limit.min().item():6.3f} | '
              f'Collision: {collision} | '
              f'Min dist: {cs63_fabric.base_fabric_repulsion.signed_distance.min():6.3f}')

# Cleanup
if use_viz:
    print('Destroying visualizer')
    robot_visualizer.close()

print('Done')

