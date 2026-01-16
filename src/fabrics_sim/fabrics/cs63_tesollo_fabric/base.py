# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
from fabrics_sim.fabrics.fabric import BaseFabric
from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.fabric_terms.joint_limit_repulsion import JointLimitRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BodySphereRepulsion, BaseFabricRepulsion
from fabrics_sim.taskmaps.identity import IdentityMap
from fabrics_sim.taskmaps.upper_joint_limit import UpperJointLimitMap
from fabrics_sim.taskmaps.lower_joint_limit import LowerJointLimitMap
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from fabrics_sim.energy.euclidean_energy import EuclideanEnergy
from fabrics_sim.utils.path_utils import get_robot_urdf_path

class CS63TesolloBaseFabric(BaseFabric):
    """
    Base fabric for CS63 robot with Tesollo gripper.
    Contains shared components like palm pose control, joint limits, and collision avoidance.
    """
    def __init__(self, batch_size, device, timestep, default_joint_config,
                 fabric_params_filename,
                 num_arm_joints=6, num_gripper_joints=12, num_fingers=3, 
                 graph_capturable=True):
        super().__init__(device, batch_size, timestep, fabric_params_filename,
                         graph_capturable=graph_capturable)

        # Store robot configuration
        self.num_arm_joints = num_arm_joints
        self.num_gripper_joints = num_gripper_joints
        self.num_fingers = num_fingers
        self.joints_per_finger = num_gripper_joints // num_fingers
        self._num_joints = num_arm_joints + num_gripper_joints
        
        # URDF setup
        robot_dir_name = "cs63_tesollo"
        robot_name = "cs63_tesollo"
        self.urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        self.load_robot(robot_dir_name, robot_name, batch_size)
        
        # Process default_joint_config
        self.default_config = self._validate_config(default_joint_config, self._num_joints, batch_size, device)
        
        # Shared targets
        self._palm_position = torch.zeros(batch_size, 3, device=device)
        self._palm_rotation_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        self._palm_pose_target = torch.zeros(batch_size, 12, device=device)

        # Pre-allocate buffers for convert_transform_to_points to support CUDA Graph
        self._palm_transform_buffer = torch.zeros(batch_size, 4, 4, device=device)
        self._palm_transform_buffer[:, 3, 3] = 1.0
        offset = 0.1
        self._pts_local_buffer = torch.tensor([
            [0, 0, 0, 1], [offset, 0, 0, 1], [-offset, 0, 0, 1],
            [0, offset, 0, 1], [0, -offset, 0, 1], [0, 0, offset, 1], [0, 0, -offset, 1]
        ], device=device, dtype=torch.float32)

    def _validate_config(self, config, total_joints, batch_size, device):
        if isinstance(config, torch.Tensor):
            config_tensor = config.to(device)
        else:
            config_tensor = torch.tensor(config, device=device, dtype=torch.float32)
        
        if config_tensor.dim() == 1:
            assert config_tensor.shape[0] == total_joints
            return config_tensor.unsqueeze(0).repeat(batch_size, 1)
        elif config_tensor.dim() == 2:
            assert config_tensor.shape[0] == batch_size and config_tensor.shape[1] == total_joints
            return config_tensor
        raise ValueError(f"Invalid config shape {config_tensor.shape}")

    def add_joint_limit_repulsion(self):
        joints = self.urdfpy_robot.joints
        upper_limits = [j.limit.upper for j in joints if j.joint_type == 'revolute']
        lower_limits = [j.limit.lower for j in joints if j.joint_type == 'revolute']

        for name, limits, map_cls in [("upper", upper_limits, UpperJointLimitMap), 
                                      ("lower", lower_limits, LowerJointLimitMap)]:
            taskmap_name = f"{name}_joint_limit"
            self.add_taskmap(taskmap_name, map_cls(limits, self.batch_size, self.device), 
                             graph_capturable=self.graph_capturable)
            self.add_fabric(taskmap_name, "joint_limit_repulsion", 
                           JointLimitRepulsion(True, self.fabric_params['joint_limit_repulsion'],
                                               self.device, graph_capturable=self.graph_capturable))

    def add_palm_pose_attractor(self):
        taskmap_name = "palm"
        control_point_frames = ["palm_link", "palm_x", "palm_x_neg", "palm_y", "palm_y_neg", "palm_z", "palm_z_neg"]
        self.add_taskmap(taskmap_name, RobotFrameOriginsTaskMap(self.urdf_path, control_point_frames, 
                                                               self.batch_size, self.device),
                         graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "palm_attractor", 
                       Attractor(True, self.fabric_params['palm_attractor'],
                                 self.device, graph_capturable=self.graph_capturable))

    def add_body_repulsion(self):
        params = self.fabric_params['body_repulsion']
        frames = params['collision_sphere_frames']
        radii = torch.tensor(params['collision_sphere_radii'], device=self.device).repeat(self.batch_size, 1)
        
        # Collision matrix logic (simplified)
        collision_matrix = torch.zeros(len(frames), len(frames), dtype=int, device=self.device)
        pairs = params.get('collision_sphere_pairs', [])
        if not pairs:
            for p1, p2 in params.get('collision_link_prefix_pairs', []):
                for i, f1 in enumerate(frames):
                    for j, f2 in enumerate(frames):
                        if p1 in f1 and p2 in f2: collision_matrix[i, j] = 1
        else:
            for s1, s2 in pairs:
                collision_matrix[frames.index(s1), frames.index(s2)] = 1

        self.add_taskmap("body_points", RobotFrameOriginsTaskMap(self.urdf_path, frames, self.batch_size, self.device),
                         graph_capturable=self.graph_capturable)
        
        for name, forcing in [("repulsion", True), ("geom_repulsion", False)]:
            self.add_fabric("body_points", name, 
                           BodySphereRepulsion(forcing, params, self.batch_size, radii, collision_matrix, 
                                               self.device, graph_capturable=self.graph_capturable))
        self.base_fabric_repulsion = BaseFabricRepulsion(params, self.batch_size, radii, collision_matrix, self.device)

    def add_cspace_attractor(self, is_forcing):
        self.add_taskmap("identity", IdentityMap(self.device), graph_capturable=self.graph_capturable)
        name = "forcing_cspace_attractor" if is_forcing else "cspace_attractor"
        self.add_fabric("identity", name, Attractor(is_forcing, self.fabric_params[name],
                                                   self.device, graph_capturable=self.graph_capturable))

    def add_cspace_energy(self):
        self.add_taskmap("identity", IdentityMap(self.device), graph_capturable=self.graph_capturable)
        self.add_energy("identity", "euclidean", EuclideanEnergy(self.batch_size, self._num_joints, self.device))

    def construct_fabric(self):
        """
        Base construct_fabric that sets internal flags. 
        Subclasses should call this or implement similar logic.
        """
        active_flags = self.fabric_params.get('active_flags', {})
        self.use_cspace_attractor = active_flags.get('cspace_attractor', False)
        self.use_joint_limits = active_flags.get('joint_limit_repulsion', False)
        self.use_palm_pose = active_flags.get('palm_attractor', False)
        self.use_collision_avoidance = active_flags.get('body_repulsion', False)
        self.use_energy = active_flags.get('cspace_energy', False)
        # Note: use_gripper_force and use_gripper_attractor are variant-specific

    def convert_transform_to_points(self):
        # Use pre-allocated buffers to avoid CUDA Graph capture errors
        self._palm_transform_buffer[:, :3, :3] = self._palm_rotation_matrix
        self._palm_transform_buffer[:, :3, 3] = self._palm_position

        pts_world = torch.einsum("bij,pj->bpi", self._palm_transform_buffer, self._pts_local_buffer)
        return pts_world[:, :, :3].reshape(self.batch_size, -1)

    @staticmethod
    def _quat_xyzw_to_rotmat(q_xyzw: torch.Tensor) -> torch.Tensor:
        # Standard quat to rotmat conversion logic
        x, y, z, w = q_xyzw.unbind(dim=-1)
        inv_norm = torch.rsqrt((x**2 + y**2 + z**2 + w**2).clamp_min(1e-12))
        x, y, z, w = x*inv_norm, y*inv_norm, z*inv_norm, w*inv_norm
        return torch.stack([
            torch.stack([1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)], dim=-1),
            torch.stack([2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)], dim=-1),
            torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)], dim=-1)
        ], dim=-2)
