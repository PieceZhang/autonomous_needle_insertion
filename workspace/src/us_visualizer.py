#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np


def pose_to_dict(msg: PoseStamped):
    return {
        "header": {
            "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
            "frame_id": msg.header.frame_id,
        },
        "position": {
            "x": msg.pose.position.x,
            "y": msg.pose.position.y,
            "z": msg.pose.position.z,
        },
        "orientation": {
            "x": msg.pose.orientation.x,
            "y": msg.pose.orientation.y,
            "z": msg.pose.orientation.z,
            "w": msg.pose.orientation.w,
        },
    }


class USVisualizer(Node):
    def __init__(self):
        super().__init__("us_visualizer")

        # QoS: 图像用 best-effort，姿态用 reliable
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.bridge = CvBridge()

        # 订阅压缩图像
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.on_image,
            qos_profile=image_qos,
        )

        # 订阅探头与针的位姿
        self.probe_sub = self.create_subscription(
            PoseStamped,
            "/ndi/us_probe_pose",
            self.on_probe_pose,
            qos_profile=pose_qos,
        )
        self.needle_sub = self.create_subscription(
            PoseStamped,
            "/ndi/needle_pose",
            self.on_needle_pose,
            qos_profile=pose_qos,
        )

        # 发布解码后的图像
        self.image_pub = self.create_publisher(
            Image,
            "/visualize/us_imaging",
            qos_profile=image_qos,
        )

        self.last_probe_pose = None
        self.last_needle_pose = None

        # 每 5 秒打印一次当前姿态字典（可根据需要调整或去掉）
        self.create_timer(5.0, self.log_latest_poses)

        self.get_logger().info("us_visualizer node started")

    def on_image(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn("Failed to decode compressed image (cv2.imdecode returned None)")
                return
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header = msg.header  # 继承时间戳与 frame_id
            self.image_pub.publish(img_msg)
        except Exception as exc:
            self.get_logger().error(f"Exception decoding/publishing image: {exc}")

    def on_probe_pose(self, msg: PoseStamped):
        self.last_probe_pose = pose_to_dict(msg)

    def on_needle_pose(self, msg: PoseStamped):
        self.last_needle_pose = pose_to_dict(msg)

    def log_latest_poses(self):
        if self.last_probe_pose:
            self.get_logger().debug(f"Probe pose dict: {self.last_probe_pose}")
        if self.last_needle_pose:
            self.get_logger().debug(f"Needle pose dict: {self.last_needle_pose}")


def main():
    rclpy.init()
    node = USVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

