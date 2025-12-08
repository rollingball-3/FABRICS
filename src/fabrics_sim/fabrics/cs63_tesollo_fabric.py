# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from pathlib import Path
import torch
import yaml

from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.fabric_terms.joint_limit_repulsion import JointLimitRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BodySphereRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BaseFabricRepulsion
from fabrics_sim.fabrics.fabric import BaseFabric
from fabrics_sim.taskmaps.identity import IdentityMap
from fabrics_sim.taskmaps.upper_joint_limit import UpperJointLimitMap
from fabrics_sim.taskmaps.lower_joint_limit import LowerJointLimitMap
from fabrics_sim.taskmaps.linear_taskmap import LinearMap
from fabrics_sim.energy.euclidean_energy import EuclideanEnergy
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from fabrics_sim.taskmaps.gripper_only_taskmap import GripperOnlyTaskMap
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.utils.rotation_utils import euler_to_matrix, matrix_to_euler
from fabrics_sim.utils.rotation_utils import quaternion_to_matrix, matrix_to_quaternion

class CS63TesolloFabric(BaseFabric):
    """
    Creates a fabric for CS63 robot with Tesollo gripper.
    
    Key differences from KukaAllegroPoseFabric:
    - Input: finger forces (not positions or PCA targets)
    - Forces are converted to joint torques via Jacobian transpose: τ = J^T * F
    - Direct joint control for gripper (no PCA dimensionality reduction)
    - Includes palm pose control, joint limiting, and collision avoidance
    
    Force Isolation Design:
    - Uses GripperOnlyTaskMap for fingertip control, which masks arm DOF columns in Jacobian
    - This ensures fingertip forces ONLY affect gripper joints, preventing unwanted arm motion
    - Palm attractor (separate fabric term) controls arm motion independently
    - Result: Clean separation between arm control (palm pose) and gripper control (fingertip forces)
    """
    def __init__(self, batch_size, device, timestep, num_arm_joints=6, num_gripper_joints=12, 
                 num_fingers=3, graph_capturable=True):
        """
        Constructor. Specifies parameter file and constructs the fabric.
        
        Args:
            batch_size: size of the batch
            device: str that sets the device for the fabric
            timestep: control timestep
            num_arm_joints: number of arm joints (default: 6)
            num_gripper_joints: number of gripper joints (default: 12 for 3-finger gripper)
            num_fingers: number of fingers (default: 3)
            graph_capturable: whether fabric can be captured in CUDA graph
        """
        # Load parameters
        fabric_params_filename = "cs63_tesollo_params.yaml"
        super().__init__(device, batch_size, timestep, fabric_params_filename,
                         graph_capturable=graph_capturable)

        # Store robot configuration
        self.num_arm_joints = num_arm_joints
        self.num_gripper_joints = num_gripper_joints
        self.num_fingers = num_fingers
        self.joints_per_finger = num_gripper_joints // num_fingers
        
        # URDF filepath - you may need to adjust this based on your robot
        robot_dir_name = "cs63_tesollo"  # Update this to match your robot directory
        robot_name = "cs63_tesollo"      # Update this to match your robot name
        self.urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)
        
        gripper_dir_name = "DG3F/urdf"
        gripper_name = "delto_gripper_3f"
        self.gripper_urdf_path = get_robot_urdf_path(gripper_dir_name, gripper_name)
        # Directly use DG3F gripper URDF path for gripper-only taskmap
        
        self.load_robot(robot_dir_name, robot_name, batch_size)
        
        # Load default configuration from YAML file
        self.default_config = self._load_initial_joint_config()
        
        # Construct the fabric
        self.construct_fabric()
        
        # Allocate target tensors
        # Palm pose target: (b x 12) -> 3 for position + 9 for rotation matrix (row-major)
        self._palm_pose_target = torch.zeros(batch_size, 12, device=device)
        # Separate storage for palm position and rotation matrix to avoid confusion
        self._palm_position = torch.zeros(batch_size, 3, device=device)
        self._palm_rotation_matrix = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Finger force targets: (b x num_fingers x 3) for 3D forces on each fingertip
        # These will be converted to external_force in fingertip space (b x 9)
        self._finger_forces = torch.zeros(batch_size, num_fingers, 3, device=device)

    def add_joint_limit_repulsion(self):
        """
        Adds forcing joint repulsion to the fabric.
        """
        # Create upper joint limiting
        joints = self.urdfpy_robot.joints
        upper_joint_limits = []
        for i in range(len(joints)):
            if joints[i].joint_type == 'revolute':
                upper_joint_limits.append(joints[i].limit.upper)
        
        # Create taskmap and its container
        taskmap_name = "upper_joint_limit"
        taskmap = UpperJointLimitMap(upper_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Create geometric fabric term and add to taskmap container
        is_forcing = True
        fabric_name = "joint_limit_repulsion"
        fabric = JointLimitRepulsion(is_forcing, self.fabric_params['joint_limit_repulsion'],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, fabric_name, fabric)
        
        # Create lower joint limiting
        lower_joint_limits = []
        for i in range(len(joints)):
            if joints[i].joint_type == 'revolute':
                lower_joint_limits.append(joints[i].limit.lower)

        # Create taskmap and its container
        taskmap_name = "lower_joint_limit"
        taskmap = LowerJointLimitMap(lower_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Create geometric fabric term and add to taskmap container
        is_forcing = True
        fabric_name = "joint_limit_repulsion"
        fabric = JointLimitRepulsion(is_forcing, self.fabric_params['joint_limit_repulsion'],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, fabric_name, fabric)
    
    def add_cspace_attractor(self, is_forcing):
        """
        Add a cspace attractor to the fabric.
        
        Args:
            is_forcing: bool, indicates whether the fabric term will be forcing or geometric
        """
        # Create taskmap and its container
        taskmap_name = "identity"
        taskmap = IdentityMap(self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Create fabric term and add to taskmap container
        if not is_forcing:
            fabric_name = "cspace_attractor"
            fabric = Attractor(is_forcing, self.fabric_params['cspace_attractor'],
                               self.device, graph_capturable=self.graph_capturable)
            self.add_fabric(taskmap_name, fabric_name, fabric)
        else:
            fabric_name = "forcing_cspace_attractor"
            fabric = Attractor(is_forcing, self.fabric_params['forcing_cspace_attractor'],
                               self.device, graph_capturable=self.graph_capturable)
            self.add_fabric(taskmap_name, fabric_name, fabric)
    
    def add_gripper_force_fabric(self):
        """
        Add force-based control for gripper joints using external_force mechanism.
        
        Uses GripperOnlyTaskMap to ensure fingertip forces ONLY affect gripper joints.
        The Jacobian columns for arm joints are masked to zero, preventing force coupling.
        
        Design: fingertip forces are passed as external_force, which gets pulled back
        to joint space via J^T: fπ(a) = γ * J^T(qf) * clamp(a, -1, 1)
        
        No fabric term needed - external_force is automatically added to potential_force
        in the taskmap container, then pulled back via J^T.
        """
        # Fingertip frame names from URDF (3-finger Tesollo gripper)
        self._fingertip_frames = ["tip1_force_frame", "tip2_force_frame", "tip3_force_frame"]
        
        # Create taskmap for all fingertips with GRIPPER-ONLY Jacobian
        # This taskmap zeros out arm columns in the Jacobian, so forces only
        # affect gripper joints and do NOT propagate to arm joints
        taskmap_name = "fingertips"
        taskmap = GripperOnlyTaskMap(
            self.gripper_urdf_path,  # Use DG3F gripper URDF directly
            self._fingertip_frames,
            self.batch_size, 
            self.device,
            num_arm_joints=self.num_arm_joints,
            num_gripper_joints=self.num_gripper_joints
        )
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        
        # external_force will be set in set_features() and pulled back via J^T
    
    def add_palm_pose_attractor(self):
        """
        Creates a taskmap for palm_link with position-only (3D) tracking.
        
        Uses RobotFrameOriginsTaskMap to track only the palm position,
        without orientation control. This simplifies the control and avoids
        orientation error computation issues.
        """
        # Set name for taskmap, create it, and add to pool of taskmaps
        taskmap_name = "palm"
        # Use multiple points (origin + +/-x, +/-y, +/-z) like Kuka demo for orientation control
        control_point_frames = [
            "palm_link",
            "palm_x", "palm_x_neg",
            "palm_y", "palm_y_neg",
            "palm_z", "palm_z_neg",
        ]
        taskmap = RobotFrameOriginsTaskMap(
            self.urdf_path,
            control_point_frames,
            self.batch_size,
            self.device,
        )
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
            
        # Create forcing attractor in 3D point space (origin + axis points)
        fabric_name = "palm_attractor"
        is_forcing = True
        fabric = Attractor(is_forcing, self.fabric_params['palm_attractor'],
                           self.device, graph_capturable=self.graph_capturable)

        self.add_fabric(taskmap_name, fabric_name, fabric)
    
    def add_body_repulsion(self):
        """
        Creates body spheres for collision detection and adds repulsion terms
        for self-collision and environment collision avoidance.
        """
        # Create list of frames that will be used to place body spheres at their origins
        collision_sphere_frames = self.fabric_params['body_repulsion']['collision_sphere_frames']

        # List of sphere radii, one for each frame origin
        self.collision_sphere_radii = self.fabric_params['body_repulsion']['collision_sphere_radii']
        
        assert(len(collision_sphere_frames) == len(self.collision_sphere_radii)),\
                "length of link names does not equal length of radii"

        # Declare which body spheres need to avoid collision
        collision_sphere_pairs = self.fabric_params['body_repulsion']['collision_sphere_pairs']
        
        # Calculate the body collision matrix
        collision_matrix = torch.zeros(len(collision_sphere_frames), len(collision_sphere_frames), 
                                       dtype=int, device=self.device)

        # If frames for collision sphere pairs were not manually specified, 
        # use link prefix pairs to determine which spheres should avoid each other
        if len(collision_sphere_pairs) == 0:
            collision_link_prefix_pairs = self.fabric_params['body_repulsion']['collision_link_prefix_pairs']
            for prefix1, prefix2 in collision_link_prefix_pairs:
                frames_for_prefix1 = [s for s in collision_sphere_frames if prefix1 in s]
                frames_for_prefix2 = [s for s in collision_sphere_frames if prefix2 in s]

                for sphere1 in frames_for_prefix1:
                    for sphere2 in frames_for_prefix2:
                        collision_sphere_pairs.append([sphere1, sphere2])

        for sphere1, sphere2 in collision_sphere_pairs:
            collision_matrix[collision_sphere_frames.index(sphere1), 
                           collision_sphere_frames.index(sphere2)] = 1

        # Set name for taskmap, create it, and add to pool of taskmaps
        taskmap_name = "body_points"
        taskmap = RobotFrameOriginsTaskMap(self.urdf_path, collision_sphere_frames,
                                           self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        # Create forcing fabric term and add to taskmap container
        fabric_name = "repulsion"
        is_forcing = True
        sphere_radius = torch.tensor(self.collision_sphere_radii, device=self.device)
        sphere_radius = sphere_radius.repeat(self.batch_size, 1)
        fabric = BodySphereRepulsion(is_forcing, self.fabric_params['body_repulsion'],
            self.batch_size, sphere_radius, collision_matrix, self.device,
            graph_capturable=self.graph_capturable)

        self.add_fabric(taskmap_name, fabric_name, fabric)

        # Add geometric body repulsion
        fabric_geom = BodySphereRepulsion(False, self.fabric_params['body_repulsion'],
            self.batch_size, sphere_radius, collision_matrix, self.device,
            graph_capturable=self.graph_capturable)
        
        self.add_fabric(taskmap_name, "geom_repulsion", fabric_geom)

        # Create object that constructs base response and signed distance
        self.base_fabric_repulsion = BaseFabricRepulsion(
            self.fabric_params['body_repulsion'],
            self.batch_size,
            sphere_radius,
            collision_matrix,
            self.device
        )
        
    def add_cspace_energy(self):
        """
        Add a Euclidean cspace energy to the fabric.
        """
        taskmap_name = "identity"
        energy_name = "euclidean"
        taskmap = IdentityMap(self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        self.add_energy(taskmap_name, energy_name, 
                       EuclideanEnergy(self.batch_size, self._num_joints, self.device))

    def construct_fabric(self):
        """
        Construct the fabric by adding the various geometric, potential, and energy components.
        Enable/disable flags are read from config file instead of hardcoded.
        """
        # Read enable/disable flags from config file
        active_flags = self.fabric_params.get('active_flags', {})
        self.use_cspace_attractor = active_flags.get('cspace_attractor')
        self.use_joint_limits = active_flags.get('joint_limit_repulsion')
        self.use_gripper_force = active_flags.get('gripper_force')
        self.use_palm_pose = active_flags.get('palm_attractor')
        self.use_collision_avoidance = active_flags.get('body_repulsion')
        self.use_energy = active_flags.get('cspace_energy')
        
        if self.use_cspace_attractor:
            self.add_cspace_attractor(False)
        
        if self.use_joint_limits:
            self.add_joint_limit_repulsion()
        
        if self.use_gripper_force:
            self.add_gripper_force_fabric()
        
        if self.use_palm_pose:
            self.add_palm_pose_attractor()
        
        if self.use_collision_avoidance:
            self.add_body_repulsion()
        
        if self.use_energy:
            self.add_cspace_energy()
    
    def get_palm_pose_target(self):
        """
        Returns the palm pose target (position + rotation matrix).
        
        Returns:
            palm_target: (b x 12) tensor, [x, y, z, R_flattened_row_major]
        """
        return self._palm_pose_target
    
    def convert_transform_to_points(self):
        """
        Converts palm pose target to collection of target points.
        Returns 7 points in world frame:
        origin + +/-x, +/-y, +/-z directions of the palm.
        """
        # Build 4x4 transformation matrix from stored palm pose
        palm_transform = torch.zeros(self.batch_size, 4, 4, device=self.device)
        palm_transform[:, 3, 3] = 1.0
        rotation_matrix = self._palm_rotation_matrix
        palm_transform[:, :3, :3] = rotation_matrix
        palm_transform[:, :3, 3] = self._palm_position

        # Define offset points in palm frame (homogeneous coordinates)
        offset_distance = 0.25
        x_point = torch.zeros(self.batch_size, 4, device=self.device)
        x_point[:, 3] = 1.0
        x_point[:, 0] = offset_distance

        x_neg_point = torch.zeros(self.batch_size, 4, device=self.device)
        x_neg_point[:, 3] = 1.0
        x_neg_point[:, 0] = -offset_distance

        y_point = torch.zeros(self.batch_size, 4, device=self.device)
        y_point[:, 3] = 1.0
        y_point[:, 1] = offset_distance

        y_neg_point = torch.zeros(self.batch_size, 4, device=self.device)
        y_neg_point[:, 3] = 1.0
        y_neg_point[:, 1] = -offset_distance

        z_point = torch.zeros(self.batch_size, 4, device=self.device)
        z_point[:, 3] = 1.0
        z_point[:, 2] = offset_distance

        z_neg_point = torch.zeros(self.batch_size, 4, device=self.device)
        z_neg_point[:, 3] = 1.0
        z_neg_point[:, 2] = -offset_distance

        # Allocate space for 7 points (origin + 6 axis points)
        palm_targets = torch.zeros(self.batch_size, 7 * 3, device=self.device)

        # Origin
        palm_targets[:, :3] = self._palm_position

        # Transform and stack axis points
        palm_targets[:, 3:6] = torch.bmm(palm_transform, x_point.unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 6:9] = torch.bmm(palm_transform, x_neg_point.unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 9:12] = torch.bmm(palm_transform, y_point.unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 12:15] = torch.bmm(palm_transform, y_neg_point.unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 15:18] = torch.bmm(palm_transform, z_point.unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 18:21] = torch.bmm(palm_transform, z_neg_point.unsqueeze(2)).squeeze(2)[:, :3]

        return palm_targets
    
    
    def get_sphere_radii(self):
        """
        Returns the radii for the body collision spheres.
        
        Returns:
            collision_sphere_radii: list of floats containing the radii
        """
        return self.collision_sphere_radii
    
    @property
    def collision_status(self):
        """
        Returns the collision state for each body sphere of the robot.
        
        Returns:
            collision_status: bxn bool tensor, b is batch size, n is number of body spheres
        """
        return self.base_fabric_repulsion.collision_status

    def get_palm_pose(self, cspace_position, orientation_convention=None):
        """
        Calculates the position of the palm given joint angles.
        
        Note: orientation_convention is kept for API compatibility but is ignored
        since we only track position now.
        
        Args:
            cspace_position: bxN tensor, joint positions
            orientation_convention: str, ignored (kept for compatibility)
            
        Returns:
            palm_position: bx3 tensor, palm position (x, y, z)
        """
        # Get palm position from taskmap (returns 3D position only)
        palm_position, _ = self.get_taskmap("palm")(cspace_position, None)
        return palm_position

    def set_features(self, finger_forces, palm_position, palm_matrix,
                     batched_cspace_position, batched_cspace_velocity,
                     object_ids, object_indicator,
                     cspace_damping_gain=None, force_scale=None):
        """
        Passes the input features to the various fabric terms.
        
        KEY DESIGN: Uses external_force mechanism for fingertip forces.
        Forces are clamped, scaled, and passed as external_force, which gets
        pulled back via J^T: fπ(a) = γ * J^T(qf) * clamp(a, -1, 1)
        
        Args:
            finger_forces: (b x num_fingers x 3) tensor of RL actions/forces at each fingertip.
                           Will be clamped to [-1, 1] and scaled by force_scale.
            palm_position: (b x 3) tensor, palm position in base/world frame used by the fabric.
            palm_matrix: (b x 3 x 3) tensor, rotation matrix of the palm (row-major).
            batched_cspace_position: (b x num_joints) tensor, current fabric position.
            batched_cspace_velocity: (b x num_joints) tensor, current fabric velocity.
            object_ids: 2D int Warp array referencing object meshes.
            object_indicator: 2D Warp array of type uint64, indicating the presence
                              of a Warp mesh in object_ids at corresponding index
                              0=no mesh, 1=mesh.
            cspace_damping_gain: optional override for damping gain.
            force_scale: optional scaling factor γ (default from config or 1.0).
        """
        # Only set cspace attractor if enabled
        if self.use_cspace_attractor:
            self.fabrics_features["identity"]["cspace_attractor"] = self.default_config
        
        # Only set gripper force features if enabled
        if self.use_gripper_force:
            # Store finger forces
            if finger_forces is not None:
                self._finger_forces.copy_(finger_forces)
        
            gripper_params = self.fabric_params.get("gripper_force", {})
            self._force_scale = gripper_params.get("fingertip_force_scale", 1.0)
        
            # Process forces: clamp to [-1, 1] and scale
            # finger_forces shape: (b x num_fingers x 3)
            forces_clamped = torch.clamp(self._finger_forces, min=-1.0, max=1.0)
            
            # Flatten to fingertip space: (b x num_fingers x 3) -> (b x 9)
            forces_flat = forces_clamped.reshape(self.batch_size, -1)
            
            # Scale by γ
            forces_scaled = self._force_scale * forces_flat
            
            # Set as external_force in fingertip space
            # This will be automatically pulled back via J^T in eval_natural()
            # Since GripperOnlyTaskMap zeros arm columns, only gripper joints are affected
            self.external_forces["fingertips"] = forces_scaled
        else:
            # Clear external_force if gripper force is disabled
            self.external_forces["fingertips"] = None
        
        # Only set palm pose features if enabled
        if self.use_palm_pose:
            assert palm_position is not None and palm_matrix is not None, \
                "palm_position and palm_matrix must both be provided when palm pose control is enabled"
            assert palm_position.shape[1] == 3, \
                "palm_position must have shape (batch_size x 3)"
            assert palm_matrix.shape[1:] == (3, 3), \
                "palm_matrix must have shape (batch_size x 3 x 3)"

            # Store separate palm position and rotation matrix
            self._palm_position.copy_(palm_position)
            self._palm_rotation_matrix.copy_(palm_matrix)
            # Maintain combined pose target tensor for compatibility
            self._palm_pose_target[:, :3] = self._palm_position
            self._palm_pose_target[:, 3:] = self._palm_rotation_matrix.reshape(self.batch_size, 9)
            
            # Convert pose target to collection of points (similar to Kuka demo)
            # Currently returns only origin point, can be extended to multiple points
            palm_points_target = self.convert_transform_to_points()
            
            self.fabrics_features["palm"]["palm_attractor"] = palm_points_target
            self.get_fabric_term("palm", "palm_attractor").damping_position = palm_points_target
        
        # Only set collision avoidance features if enabled
        if self.use_collision_avoidance:
            body_point_pos, jac = self.get_taskmap("body_points")(batched_cspace_position, None)
            body_point_vel = torch.bmm(jac, batched_cspace_velocity.unsqueeze(2)).squeeze(2)

            self.base_fabric_repulsion.calculate_response(body_point_pos,
                                                          body_point_vel,
                                                          object_ids,
                                                          object_indicator)

            self.fabrics_features["body_points"]["repulsion"] = self.base_fabric_repulsion
            self.fabrics_features["body_points"]["geom_repulsion"] = self.base_fabric_repulsion

        if cspace_damping_gain is not None:
            self.fabric_params['cspace_damping']['gain'] = cspace_damping_gain

    def _load_initial_joint_config(self):
        """
        Load initial joint configuration from YAML file.
        Uses the same configuration as Genesis manipulation examples.
        
        Returns:
            torch.Tensor: Initial joint configuration [batch_size, 18]
                         (6 arm joints + 12 gripper joints)
        """
        # Path to the config file in Genesis examples
        genesis_root = Path(__file__).parent.parent.parent.parent.parent.parent
        config_path = genesis_root / "examples" / "manipulation" / "initial_joint_config.yaml"
        
        # Fallback: try relative path if absolute path doesn't work
        if not config_path.exists():
            config_path = Path("examples/manipulation/initial_joint_config.yaml")
        
        if not config_path.exists():
            # If config file not found, use hardcoded values as fallback
            print(f"Warning: Config file not found at {config_path}, using hardcoded defaults")
            default_config = torch.tensor([
                0.0, -1.5707963267948966, -1.5707963267948966, 0.0, 1.5707963267948966, 0.7853981633974483,  # arm
                0.0, 0.0, 0.7853981633974483, 0.5235987755982988,  # finger 1
                -0.4363323129985824, 0.0, 0.7853981633974483, 0.5235987755982988,  # finger 2
                0.4363323129985824, 0.0, 0.7853981633974483, 0.5235987755982988   # finger 3
            ], device=self.device)
            return default_config.unsqueeze(0).repeat(self.batch_size, 1)
        
        # Load YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Parse arm joints in order
        arm_joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        arm_positions = [config["arm_joints"][name] for name in arm_joint_names]
        
        # Parse gripper joints in order
        gripper_joint_names = [
            "F1M1", "F1M2", "F1M3", "F1M4",
            "F2M1", "F2M2", "F2M3", "F2M4",
            "F3M1", "F3M2", "F3M3", "F3M4"
        ]
        gripper_positions = [config["gripper_joints"][name] for name in gripper_joint_names]
        
        # Combine into single tensor
        all_positions = arm_positions + gripper_positions
        # # for cspace test
        # all_positions[0] += 0.2
        # all_positions[1] -= 0.2
        default_config = torch.tensor(all_positions, device=self.device, dtype=torch.float32)
        
        return default_config.unsqueeze(0).repeat(self.batch_size, 1)

