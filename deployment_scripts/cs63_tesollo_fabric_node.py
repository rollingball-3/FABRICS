#!/usr/bin/env python3
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# CS63 (6DoF arm) + Tesollo DG3F (12DoF gripper) Fabrics control node.

from __future__ import annotations

import argparse
import time
from threading import Lock, Thread

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from fabrics_sim.fabrics.cs63_tesollo_fabric import CS63TesolloHand3DPointsAttractorFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.utils.rotation_utils import euler_to_matrix
from fabrics_sim.utils.utils import capture_fabric, initialize_warp
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel


class CS63TesolloFabricNode(Node):
    """
    Runs a CS63+Tesollo Fabrics controller loop.

    Subscribes:
    - arm joint states
    - gripper joint states
    - palm pose commands (xyz+rpy, radians) on pose_cmd_topic
    - tracked points commands (flattened N*3, relative to palm_link) on points_cmd_topic

    Publishes:
    - arm joint commands
    - gripper joint commands
    - fabrics joint_states feedback (q/qd/qdd)
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__("cs63_tesollo_fabric")

        self.device = args.device
        self.rate_hz = float(args.rate_hz)
        self.publish_dt = 1.0 / self.rate_hz
        self.fabric_dt = 1.0 / self.rate_hz
        self.iters_per_cycle = 1 if args.speed_mode == "normal" else 2
        self.heartbeat_time_threshold = float(args.heartbeat_timeout_s)

        initialize_warp(args.warp_device)

        # Joint naming (matches FABRICS cs63_tesollo.urdf by default)
        self.arm_joint_names = args.arm_joint_names or [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.gripper_joint_names = args.gripper_joint_names or [
            "F1M1",
            "F1M2",
            "F1M3",
            "F1M4",
            "F2M1",
            "F2M2",
            "F2M3",
            "F2M4",
            "F3M1",
            "F3M2",
            "F3M3",
            "F3M4",
        ]
        self.num_arm = len(self.arm_joint_names)
        self.num_gripper = len(self.gripper_joint_names)
        self.num_dof = self.num_arm + self.num_gripper

        # Robot feedback
        self._arm_q_lock = Lock()
        self._gripper_q_lock = Lock()
        self._arm_q = None
        self._gripper_q = None
        self.arm_feedback_time = time.time()
        self.gripper_feedback_time = time.time()

        # Targets into the fabric
        self._palm_target_lock = Lock()
        self._points_target_lock = Lock()
        self.palm_pos_target = torch.zeros(1, 3, device=self.device)
        self.palm_R_target = torch.eye(3, device=self.device).unsqueeze(0)
        self.points_target = None  # allocated after fabric init

        # Commands out to robot
        self._arm_cmd_lock = Lock()
        self._gripper_cmd_lock = Lock()
        self._arm_q_cmd = None
        self._gripper_q_cmd = None
        self._arm_qd_cmd = None
        self._gripper_qd_cmd = None

        # Publishers
        self._arm_pub = self.create_publisher(JointState, args.arm_cmd_topic, 1)
        self._gripper_pub = self.create_publisher(JointState, args.gripper_cmd_topic, 1)
        self._arm_timer = self.create_timer(self.publish_dt, self._arm_pub_callback)
        self._gripper_timer = self.create_timer(self.publish_dt, self._gripper_pub_callback)

        # Subscribers (feedback)
        self._arm_sub = self.create_subscription(JointState, args.arm_state_topic, self._arm_sub_callback, 1)
        self._gripper_sub = self.create_subscription(
            JointState, args.gripper_state_topic, self._gripper_sub_callback, 1
        )

        # Subscribers (commands)
        self._pose_cmd_sub = self.create_subscription(
            JointState, args.pose_cmd_topic, self._pose_cmd_callback, 1
        )
        self._points_cmd_sub = self.create_subscription(
            JointState, args.points_cmd_topic, self._points_cmd_callback, 1
        )

        # Fabrics feedback publisher
        self._fabric_states_lock = Lock()
        self.fabric_states_msg = JointState()
        self.fabric_states_msg.name = self.arm_joint_names + self.gripper_joint_names
        self._fabric_pub = self.create_publisher(JointState, args.fabric_state_topic, 1)
        self._fabric_timer = self.create_timer(self.publish_dt, self._fabric_pub_callback)

        self._build_fabric_and_graph(args.fabric_params_filename)
        time.sleep(self.heartbeat_time_threshold + 0.2)

    def _build_fabric_and_graph(self, fabric_params_filename: str):
        batch_size = 1

        world_model = WorldMeshesModel(
            batch_size=batch_size, max_objects_per_env=20, device=self.device, world_filename="floor"
        )
        object_ids, object_indicator = world_model.get_object_ids()

        q_start = torch.zeros(self.num_dof, device=self.device)
        if self.num_arm >= 2:
            q_start[1] = -1.57

        fabric = CS63TesolloHand3DPointsAttractorFabric(
            batch_size,
            device=self.device,
            timestep=self.fabric_dt,
            default_joint_config=q_start,
            num_arm_joints=self.num_arm,
            num_gripper_joints=self.num_gripper,
            fabric_params_filename=fabric_params_filename,
        )
        integrator = DisplacementIntegrator(fabric)

        self._q = torch.zeros(batch_size, self.num_dof, device=self.device)
        self._qd = torch.zeros(batch_size, self.num_dof, device=self.device)
        self._qdd = torch.zeros(batch_size, self.num_dof, device=self.device)

        self.points_target = torch.zeros(batch_size, fabric._num_points, 3, device=self.device)
        self._damping_gain = 10.0 * torch.ones(batch_size, 1, device=self.device)

        inputs = [
            self.points_target,
            self.palm_pos_target,
            self.palm_R_target,
            self._q.detach(),
            self._qd.detach(),
            object_ids,
            object_indicator,
            self._damping_gain,
        ]
        self._graph, self._q_new, self._qd_new, self._qdd_new = capture_fabric(
            fabric, self._q, self._qd, self._qdd, self.fabric_dt, integrator, inputs, self.device
        )

    def _arm_pub_callback(self):
        with self._arm_cmd_lock:
            if self._arm_q_cmd is None or self._arm_qd_cmd is None:
                return
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.arm_joint_names
            msg.position = self._arm_q_cmd
            msg.velocity = self._arm_qd_cmd
            self._arm_pub.publish(msg)

    def _gripper_pub_callback(self):
        with self._gripper_cmd_lock:
            if self._gripper_q_cmd is None or self._gripper_qd_cmd is None:
                return
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.gripper_joint_names
            msg.position = self._gripper_q_cmd
            msg.velocity = self._gripper_qd_cmd
            self._gripper_pub.publish(msg)

    def _fabric_pub_callback(self):
        with self._fabric_states_lock:
            if len(self.fabric_states_msg.position) == 0:
                return
            self.fabric_states_msg.header.stamp = self.get_clock().now().to_msg()
            self._fabric_pub.publish(self.fabric_states_msg)

    def _arm_sub_callback(self, msg: JointState):
        with self._arm_q_lock:
            self.arm_feedback_time = time.time()
            self._arm_q = msg.position
        with self._arm_cmd_lock:
            if self._arm_q_cmd is None:
                self._arm_q_cmd = list(msg.position)
                self._arm_qd_cmd = [0.0] * self.num_arm

    def _gripper_sub_callback(self, msg: JointState):
        with self._gripper_q_lock:
            self.gripper_feedback_time = time.time()
            self._gripper_q = msg.position
        with self._gripper_cmd_lock:
            if self._gripper_q_cmd is None:
                self._gripper_q_cmd = list(msg.position)
                self._gripper_qd_cmd = [0.0] * self.num_gripper

    def _pose_cmd_callback(self, msg: JointState):
        if len(msg.position) < 6:
            return
        pos = torch.tensor([msg.position[:3]], device=self.device, dtype=torch.float32)
        rpy = torch.tensor([msg.position[3:6]], device=self.device, dtype=torch.float32)
        Rm = euler_to_matrix(rpy)[0].unsqueeze(0)
        with self._palm_target_lock:
            self.palm_pos_target.copy_(pos)
            self.palm_R_target.copy_(Rm)

    def _points_cmd_callback(self, msg: JointState):
        if self.points_target is None:
            return
        flat = np.array(msg.position, dtype=np.float32)
        if flat.size != self.points_target.shape[1] * 3:
            return
        tgt = torch.tensor(flat.reshape(1, -1, 3), device=self.device)
        with self._points_target_lock:
            self.points_target.copy_(tgt)

    def _robot_feedback_ok(self) -> bool:
        dt_arm = time.time() - self.arm_feedback_time
        dt_gripper = time.time() - self.gripper_feedback_time
        return max(dt_arm, dt_gripper) < self.heartbeat_time_threshold

    def _set_joint_commands(self, q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor):
        q_np = q.detach().cpu().numpy().astype(float)
        qd_np = qd.detach().cpu().numpy().astype(float)
        qdd_np = qdd.detach().cpu().numpy().astype(float)

        with self._arm_cmd_lock:
            self._arm_q_cmd = list(q_np[0, : self.num_arm])
            self._arm_qd_cmd = list(qd_np[0, : self.num_arm])
        with self._gripper_cmd_lock:
            self._gripper_q_cmd = list(q_np[0, self.num_arm :])
            self._gripper_qd_cmd = list(qd_np[0, self.num_arm :])

        with self._fabric_states_lock:
            self.fabric_states_msg.position = list(q_np[0, :])
            self.fabric_states_msg.velocity = list(qd_np[0, :])
            self.fabric_states_msg.effort = list(qdd_np[0, :])

    def run(self):
        while rclpy.ok():
            with self._arm_q_lock:
                arm_ok = self._arm_q is not None
            with self._gripper_q_lock:
                gripper_ok = self._gripper_q is not None
            if arm_ok and gripper_ok:
                break
            time.sleep(0.01)

        with self._arm_q_lock, self._gripper_q_lock:
            q0 = torch.tensor([list(self._arm_q) + list(self._gripper_q)], device=self.device, dtype=torch.float32)
        self._q.copy_(q0)
        self._qd.zero_()
        self._qdd.zero_()
        self._set_joint_commands(self._q, self._qd, self._qdd)
        time.sleep(0.5)

        while rclpy.ok() and self._robot_feedback_ok():
            start = time.time()
            for _ in range(self.iters_per_cycle):
                self._graph.replay()
                self._q.copy_(self._q_new)
                self._qd.copy_(self._qd_new)
                self._qdd.copy_(self._qdd_new)
            self._set_joint_commands(self._q, self._qd, self._qdd)
            while (time.time() - start) < self.publish_dt:
                time.sleep(0.0002)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed-mode", type=str, default="normal", choices=["normal", "fast"])
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warp-device", type=str, default="0")
    parser.add_argument("--heartbeat-timeout-s", type=float, default=0.1)

    parser.add_argument("--arm-state-topic", type=str, default="/cs63/joint_states")
    parser.add_argument("--arm-cmd-topic", type=str, default="/cs63/joint_commands")
    parser.add_argument("--gripper-state-topic", type=str, default="/tesollo/joint_states")
    parser.add_argument("--gripper-cmd-topic", type=str, default="/tesollo/joint_commands")

    parser.add_argument("--pose-cmd-topic", type=str, default="/cs63_tesollo_fabric/pose_commands")
    parser.add_argument("--points-cmd-topic", type=str, default="/cs63_tesollo_fabric/tracked_points_commands")
    parser.add_argument("--fabric-state-topic", type=str, default="/cs63_tesollo_fabric/joint_states")

    parser.add_argument("--fabric-params-filename", type=str, default="cs63_tesollo_tracked_points_params.yaml")
    parser.add_argument("--arm-joint-names", nargs="*", default=None)
    parser.add_argument("--gripper-joint-names", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = CS63TesolloFabricNode(args)
    spin_thread = Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    time.sleep(0.2)
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

