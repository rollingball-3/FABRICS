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
                 num_arm_joints=6, num_gripper_joints=12):
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
    
    def forward_position(self, q, features=None):
        """Override to mask arm columns in Jacobian.
        
        Special handling for DG3F gripper: when using a gripper-only URDF that only has 12 joints,
        we need to:
        1. Extract only the gripper joints from the full q (which has 18 joints total)
        2. Pass only these 12 joints to the gripper URDF's kinematics
        3. Expand the resulting Jacobian to 18 columns, placing gripper Jacobian in columns 6-17
        """
        # Extract only the gripper joints from q (last 12 joints)
        q_gripper = q[:, -self.num_gripper_joints:]
        
        # Call parent with only gripper joints
        pos, jac_full = super().forward_position(q_gripper, features)
        
        # Create a new jacobian with the same number of columns as q
        batch_size, task_dim, _ = jac_full.shape
        jac_new = torch.zeros(batch_size, task_dim, q.shape[1], device=jac_full.device)
        
        # Place gripper jacobian in the appropriate columns (based on num_arm_joints)
        jac_new[:, :, self.num_arm_joints:self.num_arm_joints+self.num_gripper_joints] = jac_full
        
        return pos, jac_new

