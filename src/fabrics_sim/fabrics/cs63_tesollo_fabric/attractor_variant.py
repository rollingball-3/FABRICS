# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.taskmaps.linear_taskmap import LinearMap
from .base import CS63TesolloBaseFabric
from .synergy import build_dg3f_synergy_matrix

class _CS63TesolloGripperAttractorBase(CS63TesolloBaseFabric):
    """
    Base class for gripper-attractor variants.
    """
    _GRIPPER_ATTRACTOR_TASKMAP_NAME = "gripper_attractor_task"

    def __init__(self, batch_size, device, timestep, default_joint_config,
                 num_arm_joints=6, num_gripper_joints=12, num_fingers=3, 
                 graph_capturable=True):
        super().__init__(batch_size, device, timestep, default_joint_config,
                         "cs63_tesollo_gripper_attractor_params.yaml",
                         num_arm_joints, num_gripper_joints, num_fingers, graph_capturable)
        
        P = self._build_projection_matrix(device)
        self._gripper_latent_dim = P.shape[0]
        self._gripper_pca_matrix = P.clone()
        
        self.construct_fabric()

    def _build_projection_matrix(self, device):
        raise NotImplementedError

    def _add_gripper_attractor_task(self):
        P = self._gripper_pca_matrix.to(self.device)
        P_full = torch.cat([torch.zeros(P.shape[0], self.num_arm_joints, device=self.device), P], dim=1)
        
        self.add_taskmap(self._GRIPPER_ATTRACTOR_TASKMAP_NAME, LinearMap(P_full, self.device), 
                         graph_capturable=self.graph_capturable)
        self.add_fabric(self._GRIPPER_ATTRACTOR_TASKMAP_NAME, "gripper_attractor",
                       Attractor(True, self.fabric_params["gripper_attractor"], self.device,
                                 graph_capturable=self.graph_capturable))

    def construct_fabric(self):
        super().construct_fabric() # Sets base flags
        active_flags = self.fabric_params.get('active_flags', {})
        self.use_gripper_attractor = active_flags.get('gripper_attractor', False)

        if self.use_cspace_attractor: self.add_cspace_attractor(False)
        if self.use_joint_limits: self.add_joint_limit_repulsion()
        if self.use_gripper_attractor: self._add_gripper_attractor_task()
        if self.use_palm_pose: self.add_palm_pose_attractor()
        if self.use_collision_avoidance: self.add_body_repulsion()
        if self.use_energy: self.add_cspace_energy()

    def set_features(self, gripper_target, palm_position, palm_matrix,
                     batched_cspace_position, batched_cspace_velocity,
                     object_ids, object_indicator,
                     cspace_damping_gain=None, force_scale=None):
        if self.use_cspace_attractor:
            self.fabrics_features["identity"]["cspace_attractor"] = self.default_config

        if self.use_gripper_attractor:
            if gripper_target is not None:
                assert gripper_target.shape[1] == self._gripper_latent_dim
            self.fabrics_features[self._GRIPPER_ATTRACTOR_TASKMAP_NAME]["gripper_attractor"] = gripper_target

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

class CS63TesolloGripperLatentAttractorFabric(_CS63TesolloGripperAttractorBase):
    """
    Latent-space attractor variant (e.g. 4D synergy).
    """
    def __init__(self, *args, gripper_latent_dim=8, **kwargs):
        self._requested_latent_dim = gripper_latent_dim
        super().__init__(*args, **kwargs)

    def _build_projection_matrix(self, device):
        return build_dg3f_synergy_matrix(self.num_fingers, self.joints_per_finger, device, self._requested_latent_dim)

class CS63TesolloGripperJointAttractorFabric(_CS63TesolloGripperAttractorBase):
    """
    Joint-space attractor variant (12DoF).
    """
    def _build_projection_matrix(self, device):
        return torch.eye(self.num_gripper_joints, device=device)
