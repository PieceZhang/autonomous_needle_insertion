#!/usr/bin/env python3
import os
import glob
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import xml.etree.ElementTree as ET


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


def parse_matrix_string(matrix_str: str):
    """将多行矩阵字符串解析为 4x4 浮点数组（list[list[float]]）。"""
    # 去除换行和制表符后拆分
    tokens = matrix_str.replace("\n", " ").replace("\t", " ").split()
    vals = [float(t) for t in tokens]
    if len(vals) != 16:
        raise ValueError(f"Matrix expects 16 numbers, got {len(vals)}")
    return [vals[i:i+4] for i in range(0, 16, 4)]


class USVisualizer(Node):
    def __init__(self):
        super().__init__("us_visualizer")

        # 读取校准 XML
        self.stylus_tip_to_stylus = None
        self.image_to_probe = None
        self.load_calibration_xml()

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

        # 每 5 秒打印一次当前姿态字典（debug 级别，可按需调整）
        self.create_timer(5.0, self.log_latest_poses)

        self.get_logger().info("us_visualizer node started")

    def load_calibration_xml(self):
        """从 workspace/calibration 目录读取唯一的 PlusDeviceSet_fCal*.xml，并解析所需矩阵。"""
        ws_dir = os.environ.get("WS_DIR", "/ani_ws")
        calib_dir = os.path.join(ws_dir, "calibration")
        pattern = os.path.join(calib_dir, "PlusDeviceSet_fCal*.xml")
        files = glob.glob(pattern)
        if len(files) == 0:
            msg = f"未找到校准文件 {pattern}"
            self.get_logger().error(msg)
            raise FileNotFoundError(msg)
        if len(files) > 1:
            msg = f"找到多个校准文件 {files}，请确保只有一个以 PlusDeviceSet_fCal 开头的 XML"
            self.get_logger().error(msg)
            raise RuntimeError(msg)

        xml_path = files[0]
        self.get_logger().info(f"读取校准文件: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        def find_transform(frm, to):
            xpath = f".//Transform[@From='{frm}'][@To='{to}']"
            elem = root.find(xpath)
            if elem is None:
                raise RuntimeError(f"在 {xml_path} 中未找到 Transform From='{frm}' To='{to}'")
            matrix_str = elem.get("Matrix")
            if matrix_str is None:
                raise RuntimeError(f"在 {xml_path} 中 Transform From='{frm}' To='{to}' 无 Matrix 属性")
            return parse_matrix_string(matrix_str)

        self.stylus_tip_to_stylus = find_transform("StylusTip", "Stylus")
        self.image_to_probe = find_transform("Image", "Probe")

        self.get_logger().info("已加载 Transform 矩阵：")
        self.get_logger().info(f"StylusTip -> Stylus:\n{self.stylus_tip_to_stylus}")
        self.get_logger().info(f"Image -> Probe:\n{self.image_to_probe}")

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