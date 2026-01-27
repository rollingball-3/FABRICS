# Copyright (c) 2024
# Taskmap wrapper that isolates gripper DOFs from arm DOFs for fingertip control

import torch
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap

class GripperOnlyTaskMap(RobotFrameOriginsTaskMap):
    """
    Extends RobotFrameOriginsTaskMap to mask arm DOF columns in Jacobian.
    
    Purpose: Ensure fingertip forces ONLY affect gripper joints, not arm joints.
    
    Implementation:
    - Computes full robot forward kinematics (correct fingertip positions in world frame)
    - Zeros out Jacobian columns corresponding to arm joints before returning
    - Result: J^T * F only generates torques for gripper joints
    """
    def __init__(self, urdf_path, frame_names, batch_size, device, 
                 num_arm_joints=6, num_gripper_joints=12,
                 full_q_gripper_joint_names=None):
        """
        Args:
            urdf_path: Path to full robot URDF (arm + gripper)
            frame_names: Fingertip link names (e.g., ["F1_TIP", "F2_TIP", "F3_TIP"])
            batch_size: Batch size
            device: Device (cuda:0, cuda:1, etc.)
            num_arm_joints: Number of arm DOF to mask (default: 6)
            num_gripper_joints: Number of gripper DOF to keep (default: 12)
        """
        super().__init__(urdf_path, frame_names, batch_size, device)
        self.num_arm_joints = num_arm_joints
        self.num_gripper_joints = num_gripper_joints
        self.full_q_gripper_joint_names = full_q_gripper_joint_names

        # Optional: reorder incoming gripper joints (from full-q order) into this URDF's cspace order
        self._q_reorder_idx = None
        if self.full_q_gripper_joint_names is not None:
            # Build URDF cspace joint name order by index
            name2idx = self.robot_kinematics.urdf_info.cspace_name2index_map
            urdf_cspace_names = [name for (name, _) in sorted(name2idx.items(), key=lambda kv: kv[1])]
            # Restrict to joints that exist in the full-q gripper set
            urdf_cspace_names = [n for n in urdf_cspace_names if n in set(self.full_q_gripper_joint_names)]
            if len(urdf_cspace_names) != self.num_gripper_joints:
                raise ValueError(
                    f"GripperOnlyTaskMap joint reorder failed: expected {self.num_gripper_joints} joints, "
                    f"got {len(urdf_cspace_names)} after intersecting URDF cspace with full_q_gripper_joint_names."
                )
            self._q_reorder_idx = torch.tensor(
                [self.full_q_gripper_joint_names.index(n) for n in urdf_cspace_names],
                device=self.device,
                dtype=torch.long,
            )
    
    def forward_position(self, q, features=None):
        """Override to mask arm columns in Jacobian.
        
        Special handling for DG3F gripper: when using a gripper-only URDF that only has 12 joints,
        we need to:
        1. Extract only the gripper joints from the full q (which has 18 joints total)
        2. Pass only these 12 joints to the gripper URDF's kinematics
        3. Expand the resulting Jacobian to 18 columns, placing gripper Jacobian in columns 6-17
        """
        # Extract only the gripper joints from q (last 12 joints, in *full-q gripper order*)
        q_gripper = q[:, -self.num_gripper_joints:]
        # Reorder to this URDF's cspace order if mapping provided
        if self._q_reorder_idx is not None:
            q_gripper = q_gripper.index_select(1, self._q_reorder_idx)
        
        # Call parent with only gripper joints
        pos, jac_full = super().forward_position(q_gripper, features)
        
        # Create a new jacobian with the same number of columns as q
        batch_size, task_dim, _ = jac_full.shape
        jac_new = torch.zeros(batch_size, task_dim, q.shape[1], device=jac_full.device)
        
        # Place gripper jacobian in the appropriate columns (based on num_arm_joints)
        jac_new[:, :, self.num_arm_joints:self.num_arm_joints+self.num_gripper_joints] = jac_full
        
        return pos, jac_new


class FingertipsRelativeToPalmTaskMap(RobotFrameOriginsTaskMap):
    """
    Taskmap that returns fingertip positions expressed in the palm_link frame.

    Convention:
    - We use a gripper-only URDF (DG3F) that contains both palm_link and tip*_force_frame.
    - We compute positions for [palm_link, tip1_force_frame, tip2_force_frame, tip3_force_frame]
      in the gripper URDF base frame, then return:

        x = [ (p_tip1 - p_palm), (p_tip2 - p_palm), (p_tip3 - p_palm) ]  in palm coordinates.

      Since in DG3F URDF palm_link is a fixed joint with rpy=(0,0,0) relative to delto_base_link,
      palm axes coincide with the gripper base axes, so no rotation is required.

    Jacobian:
      J_rel = J_tip - J_palm (w.r.t gripper joints only), then expanded to full (arm+gripper) columns
      with arm columns zeroed. This keeps fingertip attractor from affecting arm joints.
    """

    def __init__(
        self,
        urdf_path: str,
        fingertip_frame_names: list[str],
        batch_size: int,
        device: str,
        palm_frame_name: str = "palm_link",
        num_arm_joints: int = 6,
        num_gripper_joints: int = 12,
        full_q_gripper_joint_names: list[str] | None = None,
    ):
        self.num_arm_joints = num_arm_joints
        self.num_gripper_joints = num_gripper_joints
        self.palm_frame_name = palm_frame_name
        self.fingertip_frame_names = fingertip_frame_names
        self.full_q_gripper_joint_names = full_q_gripper_joint_names

        # We query palm first, then fingertips
        frame_names = [palm_frame_name] + list(fingertip_frame_names)
        super().__init__(urdf_path, frame_names, batch_size, device)

        # Optional: reorder incoming gripper joints (from full-q order) into this URDF's cspace order
        self._q_reorder_idx = None
        if self.full_q_gripper_joint_names is not None:
            name2idx = self.robot_kinematics.urdf_info.cspace_name2index_map
            urdf_cspace_names = [name for (name, _) in sorted(name2idx.items(), key=lambda kv: kv[1])]
            urdf_cspace_names = [n for n in urdf_cspace_names if n in set(self.full_q_gripper_joint_names)]
            if len(urdf_cspace_names) != self.num_gripper_joints:
                raise ValueError(
                    f"FingertipsRelativeToPalmTaskMap joint reorder failed: expected {self.num_gripper_joints} joints, "
                    f"got {len(urdf_cspace_names)} after intersecting URDF cspace with full_q_gripper_joint_names."
                )
            self._q_reorder_idx = torch.tensor(
                [self.full_q_gripper_joint_names.index(n) for n in urdf_cspace_names],
                device=self.device,
                dtype=torch.long,
            )

    def forward_position(self, q, features=None):
        # Extract only the gripper joints from q (last 12 joints)
        q_gripper = q[:, -self.num_gripper_joints:]
        if self._q_reorder_idx is not None:
            q_gripper = q_gripper.index_select(1, self._q_reorder_idx)

        # Evaluate FK for palm + fingertips in the gripper URDF frame
        x_all, jac_all = super().forward_position(q_gripper, features)

        # x_all: (B, (1 + num_tips) * 3)
        # jac_all: (B, (1 + num_tips) * 3, num_gripper_joints)
        b = x_all.shape[0]
        num_tips = len(self.fingertip_frame_names)

        x_all = x_all.reshape(b, 1 + num_tips, 3)
        jac_all = jac_all.reshape(b, 1 + num_tips, 3, self.num_gripper_joints)

        palm_pos = x_all[:, 0:1, :]  # (B,1,3)
        tip_pos = x_all[:, 1:, :]    # (B,num_tips,3)
        rel_pos = (tip_pos - palm_pos).reshape(b, num_tips * 3)  # (B, num_tips*3)

        palm_jac = jac_all[:, 0:1, :, :]  # (B,1,3,Jg)
        tip_jac = jac_all[:, 1:, :, :]    # (B,num_tips,3,Jg)
        rel_jac_gripper = (tip_jac - palm_jac).reshape(b, num_tips * 3, self.num_gripper_joints)

        # Expand jacobian to full q columns: (B, task_dim, arm+gripper)
        jac_new = torch.zeros(b, num_tips * 3, q.shape[1], device=jac_all.device)
        jac_new[:, :, self.num_arm_joints : self.num_arm_joints + self.num_gripper_joints] = rel_jac_gripper

        return rel_pos, jac_new


class FramesRelativeToPalmTaskMap(RobotFrameOriginsTaskMap):
    """
    Taskmap that returns positions of arbitrary frames expressed relative to a palm frame.

    This is a generalized version of FingertipsRelativeToPalmTaskMap.

    Output:
      x = [ (p_f1 - p_palm), (p_f2 - p_palm), ... ] flattened as (B, num_frames*3)

    Jacobian:
      J_rel = J_frame - J_palm (w.r.t gripper joints only), then expanded to full (arm+gripper) columns
      with arm columns zeroed.
    """

    def __init__(
        self,
        urdf_path: str,
        tracked_frame_names: list[str],
        batch_size: int,
        device: str,
        palm_frame_name: str = "palm_link",
        num_arm_joints: int = 6,
        num_gripper_joints: int = 12,
        full_q_gripper_joint_names: list[str] | None = None,
    ):
        self.num_arm_joints = num_arm_joints
        self.num_gripper_joints = num_gripper_joints
        self.palm_frame_name = palm_frame_name
        self.tracked_frame_names = tracked_frame_names
        self.full_q_gripper_joint_names = full_q_gripper_joint_names

        # Query palm first, then tracked frames
        frame_names = [palm_frame_name] + list(tracked_frame_names)
        super().__init__(urdf_path, frame_names, batch_size, device)

        # Optional: reorder incoming gripper joints (from full-q order) into this URDF's cspace order
        self._q_reorder_idx = None
        if self.full_q_gripper_joint_names is not None:
            name2idx = self.robot_kinematics.urdf_info.cspace_name2index_map
            urdf_cspace_names = [name for (name, _) in sorted(name2idx.items(), key=lambda kv: kv[1])]
            urdf_cspace_names = [n for n in urdf_cspace_names if n in set(self.full_q_gripper_joint_names)]
            if len(urdf_cspace_names) != self.num_gripper_joints:
                raise ValueError(
                    "FramesRelativeToPalmTaskMap joint reorder failed: expected "
                    f"{self.num_gripper_joints} joints, got {len(urdf_cspace_names)} after intersecting "
                    "URDF cspace with full_q_gripper_joint_names."
                )
            self._q_reorder_idx = torch.tensor(
                [self.full_q_gripper_joint_names.index(n) for n in urdf_cspace_names],
                device=self.device,
                dtype=torch.long,
            )

    def forward_position(self, q, features=None):
        # Extract only the gripper joints from q (last 12 joints)
        q_gripper = q[:, -self.num_gripper_joints:]
        if self._q_reorder_idx is not None:
            q_gripper = q_gripper.index_select(1, self._q_reorder_idx)

        # Evaluate FK for palm + tracked frames in the gripper URDF frame
        x_all, jac_all = super().forward_position(q_gripper, features)

        # x_all: (B, (1 + num_frames) * 3)
        # jac_all: (B, (1 + num_frames) * 3, num_gripper_joints)
        b = x_all.shape[0]
        num_frames = len(self.tracked_frame_names)

        x_all = x_all.reshape(b, 1 + num_frames, 3)
        jac_all = jac_all.reshape(b, 1 + num_frames, 3, self.num_gripper_joints)

        palm_pos = x_all[:, 0:1, :]  # (B,1,3)
        frame_pos = x_all[:, 1:, :]  # (B,num_frames,3)
        rel_pos = (frame_pos - palm_pos).reshape(b, num_frames * 3)

        palm_jac = jac_all[:, 0:1, :, :]  # (B,1,3,Jg)
        frame_jac = jac_all[:, 1:, :, :]  # (B,num_frames,3,Jg)
        rel_jac_gripper = (frame_jac - palm_jac).reshape(b, num_frames * 3, self.num_gripper_joints)

        # Expand jacobian to full q columns: (B, task_dim, arm+gripper)
        jac_new = torch.zeros(b, num_frames * 3, q.shape[1], device=jac_all.device)
        jac_new[:, :, self.num_arm_joints : self.num_arm_joints + self.num_gripper_joints] = rel_jac_gripper

        return rel_pos, jac_new
