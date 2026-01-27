#!/usr/bin/env python3
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import time
import argparse
import threading
from threading import Lock

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class CS63TesolloRandomTargets(Node):
    """Publish simple test targets for CS63+Tesollo Fabrics."""

    def __init__(self, args: argparse.Namespace):
        super().__init__("cs63_tesollo_random_targets")

        self.mutex = Lock()
        self.num_points = int(args.num_points)
        self.rate_hz = float(args.rate_hz)
        self.rate = self.create_rate(frequency=self.rate_hz, clock=self.get_clock())

        self.pose_pub = self.create_publisher(JointState, "/cs63_tesollo_fabric/pose_commands", 1)
        self.points_pub = self.create_publisher(JointState, "/cs63_tesollo_fabric/tracked_points_commands", 1)

        self.nominal_pose = np.array([-0.6868, 0.0320, 0.685, -2.3873, -0.0824, 3.1301], dtype=np.float32)
        self.nominal_points = np.zeros(self.num_points * 3, dtype=np.float32)

    def run(self):
        step = 0
        toggle = 0
        pose = self.nominal_pose.copy()
        pts = self.nominal_points.copy()

        while rclpy.ok():
            if step % 30 == 0:
                pose = self.nominal_pose.copy()
                pose[2] += -0.15 if toggle == 0 else 0.15
                toggle = 1 - toggle
                pts = (np.random.rand(self.num_points * 3).astype(np.float32) - 0.5) * 0.1

            stamp = self.get_clock().now().to_msg()

            pose_msg = JointState()
            pose_msg.header.stamp = stamp
            pose_msg.name = ["x", "y", "z", "roll", "pitch", "yaw"]
            pose_msg.position = pose.tolist()
            self.pose_pub.publish(pose_msg)

            pts_msg = JointState()
            pts_msg.header.stamp = stamp
            pts_msg.name = [f"p{i}_{ax}" for i in range(self.num_points) for ax in ("x", "y", "z")]
            pts_msg.position = pts.tolist()
            self.points_pub.publish(pts_msg)

            self.rate.sleep()
            step += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate-hz", type=float, default=60.0)
    parser.add_argument("--num-points", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = CS63TesolloRandomTargets(args)
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    time.sleep(0.5)
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

