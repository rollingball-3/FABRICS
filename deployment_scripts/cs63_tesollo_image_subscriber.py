#!/usr/bin/env python3
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


WIDTH = 640
HEIGHT = 480


def object_image_to_tensor(msg: Image):
    """
    Convert uint16 depth image into float32 visualization range.
    """
    img_np = np.frombuffer(msg.data, dtype=np.uint16).reshape(HEIGHT, WIDTH).astype(np.float32)
    img_np = cv2.resize(img_np, (WIDTH // 4, HEIGHT // 4), interpolation=cv2.INTER_LINEAR)
    img_np *= -1e-3
    img_np[img_np < -1.3] = 0
    img_np[img_np > -0.5] = 0
    return img_np


class CS63TesolloImageSubscriber(Node):
    def __init__(self, topic: str):
        super().__init__("cs63_tesollo_image_subscriber")
        self.subscription = self.create_subscription(Image, topic, self.listener_callback, 10)
        self.bridge = CvBridge()

        self.fig = plt.figure()
        x = np.linspace(0, 50.0, num=WIDTH // 4)
        y = np.linspace(0, 50.0, num=HEIGHT // 4)
        X, Y = np.meshgrid(x, y)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.rendered_img = self.ax.imshow(X, vmin=-1.3, vmax=0, cmap="Greys")
        self.fig.canvas.draw()
        plt.title("Depth Input")
        plt.show(block=False)

    def listener_callback(self, msg: Image):
        image = object_image_to_tensor(msg)
        self.rendered_img.set_data(image)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="/camera/aligned_depth_to_color/image_raw")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = CS63TesolloImageSubscriber(args.topic)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

