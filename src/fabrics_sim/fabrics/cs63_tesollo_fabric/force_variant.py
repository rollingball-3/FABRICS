# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
from fabrics_sim.taskmaps.gripper_only_taskmap import GripperOnlyTaskMap
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotKinematics
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from .base import CS63TesolloBaseFabric

class CS63TesolloForceFabric(CS63TesolloBaseFabric):
    """
    Original force-based control variant.
    """
    def __init__(self, batch_size, device, timestep, default_joint_config,
                 num_arm_joints=6, num_gripper_joints=12, num_fingers=3, 
                 graph_capturable=True):
        super().__init__(batch_size, device, timestep, default_joint_config,
                         "cs63_tesollo_params.yaml",
                         num_arm_joints, num_gripper_joints, num_fingers, graph_capturable)
        
        gripper_dir_name = "DG3F/urdf"
        gripper_name = "delto_gripper_3f"
        self.gripper_urdf_path = get_robot_urdf_path(gripper_dir_name, gripper_name)
        
        self.construct_fabric()
        self._finger_forces = torch.zeros(batch_size, num_fingers, 3, device=device)

    def add_gripper_force_fabric(self):
        self._fingertip_frames = ["tip1_force_frame", "tip2_force_frame", "tip3_force_frame"]
        taskmap = GripperOnlyTaskMap(self.gripper_urdf_path, self._fingertip_frames,
                                     self.batch_size, self.device,
                                     num_arm_joints=self.num_arm_joints,
                                     num_gripper_joints=self.num_gripper_joints)
        self.add_taskmap("fingertips", taskmap, graph_capturable=self.graph_capturable)

    def construct_fabric(self):
        super().construct_fabric() # Sets base flags
        active_flags = self.fabric_params.get('active_flags', {})
        self.use_gripper_force = active_flags.get('gripper_force', False)
        
        if self.use_cspace_attractor: self.add_cspace_attractor(False)
        if self.use_joint_limits: self.add_joint_limit_repulsion()
        if self.use_gripper_force: self.add_gripper_force_fabric()
        if self.use_palm_pose: self.add_palm_pose_attractor()
        if self.use_collision_avoidance: self.add_body_repulsion()
        if self.use_energy: self.add_cspace_energy()

    def set_features(self, finger_forces, palm_position, palm_matrix,
                     batched_cspace_position, batched_cspace_velocity,
                     object_ids, object_indicator,
                     cspace_damping_gain=None, force_scale=None):
        if self.use_cspace_attractor:
            self.fabrics_features["identity"]["cspace_attractor"] = self.default_config

        if self.use_gripper_force:
            if finger_forces is not None: self._finger_forces.copy_(finger_forces)
            scale = self.fabric_params.get("gripper_force", {}).get("fingertip_force_scale", 1.0)
            
            forces_tip = torch.clamp(self._finger_forces, -1.0, 1.0)
            fingertips_tm = self.get_taskmap("fingertips")
            q_gripper = batched_cspace_position[:, -self.num_gripper_joints:]
            link_transforms, _ = RobotKinematics.apply(q_gripper, fingertips_tm.robot_kinematics)
            tip_quats = link_transforms[:, fingertips_tm.link_indices, 3:7]
            tip_R_base = self._quat_xyzw_to_rotmat(tip_quats)
            
            forces_base = torch.einsum("bfij,bfj->bfi", tip_R_base, forces_tip)
            self.external_forces["fingertips"] = (scale * forces_base).reshape(self.batch_size, -1)

        if self.use_palm_pose:
            self._palm_position.copy_(palm_position)
            self._palm_rotation_matrix.copy_(palm_matrix)
            self._palm_pose_target[:, :3] = self._palm_position
            self._palm_pose_target[:, 3:] = self._palm_rotation_matrix.reshape(self.batch_size, 9)
            
            palm_points_target = self.convert_transform_to_points()
            self.fabrics_features["palm"]["palm_attractor"] = palm_points_target
            self.get_fabric_term("palm", "palm_attractor").damping_position = palm_points_target

        if self.use_collision_avoidance:
            pos, jac = self.get_taskmap("body_points")(batched_cspace_position, None)
            vel = torch.bmm(jac, batched_cspace_velocity.unsqueeze(2)).squeeze(2)
            self.base_fabric_repulsion.calculate_response(pos, vel, object_ids, object_indicator)
            self.fabrics_features["body_points"]["repulsion"] = self.base_fabric_repulsion
            self.fabrics_features["body_points"]["geom_repulsion"] = self.base_fabric_repulsion

        if cspace_damping_gain is not None:
            self.fabric_params["cspace_damping"]["gain"] = cspace_damping_gain
