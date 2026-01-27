#!/usr/bin/env python3
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import time
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-frame", type=str, default="robot_base")
    parser.add_argument("--child-frame", type=str, default="camera_depth_optical_frame")
    parser.add_argument("--hz", type=float, default=60.0)
    parser.add_argument("--matrix-file", type=str, default="", help="Optional 4x4 csv file of robot_T_camera")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = rclpy.create_node("cs63_tesollo_camera_tf_publisher")
    broadcaster = TransformBroadcaster(node)

    # Default identity; replace by loading from file.
    T_robot_cam = np.eye(4, dtype=np.float64)
    if args.matrix_file:
        T_robot_cam = np.loadtxt(args.matrix_file, delimiter=",", dtype=float)

    quat_xyzw = R.from_matrix(T_robot_cam[:3, :3]).as_quat()

    dt = 1.0 / float(args.hz)
    while rclpy.ok():
        tfm = TransformStamped()
        tfm.header.stamp = node.get_clock().now().to_msg()
        tfm.header.frame_id = args.parent_frame
        tfm.child_frame_id = args.child_frame
        tfm.transform.translation.x = float(T_robot_cam[0, 3])
        tfm.transform.translation.y = float(T_robot_cam[1, 3])
        tfm.transform.translation.z = float(T_robot_cam[2, 3])
        tfm.transform.rotation.x = float(quat_xyzw[0])
        tfm.transform.rotation.y = float(quat_xyzw[1])
        tfm.transform.rotation.z = float(quat_xyzw[2])
        tfm.transform.rotation.w = float(quat_xyzw[3])
        broadcaster.sendTransform(tfm)
        time.sleep(dt)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

