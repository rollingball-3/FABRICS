#!/usr/bin/env python3
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import time
import argparse
from threading import Lock, Thread

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage


class CS63TesolloStateMachineNode(Node):
    """
    Minimal state machine for CS63+Tesollo Fabrics deployment.

    Intended as an integration test:
    - verify topics wiring
    - verify end-to-end command -> fabric -> joint_commands -> robot motion
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__("cs63_tesollo_state_machine")

        self.publish_rate = float(args.rate_hz)
        self.publish_dt = 1.0 / self.publish_rate
        self.num_points = int(args.num_points)

        self._obj_lock = Lock()
        self.obj_pos = None
        self.obj_feedback_time = time.time()
        self.obj_pos_sub = self.create_subscription(TFMessage, "/tf", self._tf_callback, 10)

        self.pose_pub = self.create_publisher(JointState, "/cs63_tesollo_fabric/pose_commands", 1)
        self.points_pub = self.create_publisher(JointState, "/cs63_tesollo_fabric/tracked_points_commands", 1)

        self._t0 = time.time()
        self.timer = self.create_timer(self.publish_dt, self._tick)

    def _tf_callback(self, msg: TFMessage):
        if len(msg.transforms) == 0:
            return
        for tf in msg.transforms:
            if tf.child_frame_id == "obj_pos":
                with self._obj_lock:
                    self.obj_pos = np.array(
                        [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z],
                        dtype=np.float32,
                    )
                    self.obj_feedback_time = time.time()

    def _tick(self):
        t = time.time() - self._t0

        # Simple two-pose toggle
        if int(t / 2.0) % 2 == 0:
            palm = np.array([-0.40, 0.00, 0.50, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            palm = np.array([-0.45, 0.05, 0.50, 0.0, 0.0, 0.0], dtype=np.float32)

        # Optional object tracking on y
        with self._obj_lock:
            if self.obj_pos is not None and (time.time() - self.obj_feedback_time) < 0.5:
                palm[1] = float(np.clip(self.obj_pos[1], -0.2, 0.2))

        pts = np.zeros((self.num_points, 3), dtype=np.float32)
        pts[:, 0] = 0.02 * np.sin(2.0 * np.pi * 0.5 * t)
        flat = pts.reshape(-1)

        stamp = self.get_clock().now().to_msg()

        pose_msg = JointState()
        pose_msg.header.stamp = stamp
        pose_msg.name = ["x", "y", "z", "roll", "pitch", "yaw"]
        pose_msg.position = palm.tolist()
        self.pose_pub.publish(pose_msg)

        pts_msg = JointState()
        pts_msg.header.stamp = stamp
        pts_msg.name = [f"p{i}_{ax}" for i in range(self.num_points) for ax in ("x", "y", "z")]
        pts_msg.position = flat.tolist()
        self.points_pub.publish(pts_msg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--num-points", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = CS63TesolloStateMachineNode(args)
    spin_thread = Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        while rclpy.ok():
            time.sleep(0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

