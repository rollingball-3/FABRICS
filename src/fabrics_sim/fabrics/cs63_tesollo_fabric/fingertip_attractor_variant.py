# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.taskmaps.gripper_only_taskmap import FingertipsRelativeToPalmTaskMap
from fabrics_sim.utils.path_utils import get_robot_urdf_path

from .base import CS63TesolloBaseFabric


class CS63TesolloFingertipAttractorFabric(CS63TesolloBaseFabric):
    """
    Fingertip position-attractor variant.

    - Task is fingertip positions relative to palm_link in the DG3F gripper URDF frame.
    - Jacobian is expanded to full (arm+gripper) columns with arm columns zeroed, so this
      attractor only affects gripper joints (no arm coupling).
    """

    _FINGERTIP_TASKMAP_NAME = "fingertips_rel"
    _FINGERTIP_ATTRACTOR_NAME = "fingertip_attractor"

    def __init__(
        self,
        batch_size,
        device,
        timestep,
        default_joint_config,
        num_arm_joints=6,
        num_gripper_joints=12,
        num_fingers=3,
        graph_capturable=True,
    ):
        super().__init__(
            batch_size,
            device,
            timestep,
            default_joint_config,
            "cs63_tesollo_params.yaml",
            num_arm_joints,
            num_gripper_joints,
            num_fingers,
            graph_capturable,
        )

        gripper_dir_name = "DG3F/urdf"
        gripper_name = "delto_gripper_3f"
        self.gripper_urdf_path = get_robot_urdf_path(gripper_dir_name, gripper_name)

        self._fingertip_targets = torch.zeros(batch_size, num_fingers, 3, device=device)
        self.construct_fabric()

    def _add_fingertip_attractor_task(self):
        fingertip_frames = ["tip1_force_frame", "tip2_force_frame", "tip3_force_frame"]
        taskmap = FingertipsRelativeToPalmTaskMap(
            self.gripper_urdf_path,
            fingertip_frame_names=fingertip_frames,
            batch_size=self.batch_size,
            device=self.device,
            palm_frame_name="palm_link",
            num_arm_joints=self.num_arm_joints,
            num_gripper_joints=self.num_gripper_joints,
        )
        self.add_taskmap(self._FINGERTIP_TASKMAP_NAME, taskmap, graph_capturable=self.graph_capturable)
        self.add_fabric(
            self._FINGERTIP_TASKMAP_NAME,
            self._FINGERTIP_ATTRACTOR_NAME,
            Attractor(
                True,
                self.fabric_params[self._FINGERTIP_ATTRACTOR_NAME],
                self.device,
                graph_capturable=self.graph_capturable,
            ),
        )

    def construct_fabric(self):
        super().construct_fabric()  # Sets base flags
        active_flags = self.fabric_params.get("active_flags", {})
        self.use_fingertip_attractor = active_flags.get("fingertip_attractor", False)

        if self.use_cspace_attractor:
            self.add_cspace_attractor(False)
        if self.use_joint_limits:
            self.add_joint_limit_repulsion()
        if self.use_fingertip_attractor:
            self._add_fingertip_attractor_task()
        if self.use_palm_pose:
            self.add_palm_pose_attractor()
        if self.use_collision_avoidance:
            self.add_body_repulsion()
        if self.use_energy:
            self.add_cspace_energy()

    def set_features(
        self,
        fingertip_targets,
        palm_position,
        palm_matrix,
        batched_cspace_position,
        batched_cspace_velocity,
        object_ids,
        object_indicator,
        cspace_damping_gain=None,
        force_scale=None,
    ):
        if self.use_cspace_attractor:
            self.fabrics_features["identity"]["cspace_attractor"] = self.default_config

        if self.use_fingertip_attractor:
            if fingertip_targets is None:
                # Explicitly disable for this step (avoid stale targets).
                self.fabrics_features[self._FINGERTIP_TASKMAP_NAME][self._FINGERTIP_ATTRACTOR_NAME] = None
            else:
                if isinstance(fingertip_targets, torch.Tensor):
                    tgt = fingertip_targets.to(self.device)
                else:
                    tgt = torch.tensor(fingertip_targets, device=self.device, dtype=torch.float32)

                if tgt.dim() == 2:
                    tgt = tgt.reshape(self.batch_size, self.num_fingers, 3)
                elif tgt.dim() != 3:
                    raise ValueError(
                        f"fingertip_targets must be (B, F, 3) or (B, F*3), got shape {tuple(tgt.shape)}"
                    )

                self._fingertip_targets.copy_(tgt)
                fingertip_target_flat = self._fingertip_targets.reshape(self.batch_size, -1)
                self.fabrics_features[self._FINGERTIP_TASKMAP_NAME][self._FINGERTIP_ATTRACTOR_NAME] = (
                    fingertip_target_flat
                )
                self.get_fabric_term(self._FINGERTIP_TASKMAP_NAME, self._FINGERTIP_ATTRACTOR_NAME).damping_position = (
                    fingertip_target_flat
                )

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

