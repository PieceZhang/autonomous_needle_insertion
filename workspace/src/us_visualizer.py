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

M_TO_MM = 1000.0  # conversion factor


def pose_to_dict(msg: PoseStamped):
    """Return the raw pose dict in meters (as received)."""
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


def quaternion_to_matrix(qx, qy, qz, qw) -> np.ndarray:
    """Convert a unit quaternion to a 3x3 rotation matrix; auto-normalize if not unit."""
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    R = np.array([
        [1 - s*(yy + zz),     s*(xy - wz),       s*(xz + wy)],
        [s*(xy + wz),         1 - s*(xx + zz),   s*(yz - wx)],
        [s*(xz - wy),         s*(yz + wx),       1 - s*(xx + yy)],
    ], dtype=float)
    return R


def pose_to_hmat(msg: PoseStamped) -> np.ndarray:
    """Convert PoseStamped to a 4x4 homogeneous transform (meters in translation)."""
    qx = msg.pose.orientation.x
    qy = msg.pose.orientation.y
    qz = msg.pose.orientation.z
    qw = msg.pose.orientation.w
    R = quaternion_to_matrix(qx, qy, qz, qw)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[0, 3] = msg.pose.position.x
    T[1, 3] = msg.pose.position.y
    T[2, 3] = msg.pose.position.z
    return T


def to_mm(T: np.ndarray) -> np.ndarray:
    """Convert translation of a 4x4 transform from meters to millimeters."""
    T_mm = T.copy()
    T_mm[:3, 3] *= M_TO_MM
    return T_mm


def parse_matrix_string(matrix_str: str) -> np.ndarray:
    """Parse a matrix string into a 4x4 numpy array."""
    tokens = matrix_str.replace("\n", " ").replace("\t", " ").split()
    vals = [float(t) for t in tokens]
    if len(vals) != 16:
        raise ValueError(f"Matrix expects 16 numbers, got {len(vals)}")
    return np.array(vals, dtype=float).reshape(4, 4)


def is_finite_pose(msg: PoseStamped) -> bool:
    vals = [
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
    ]
    return np.all(np.isfinite(vals))


def is_finite_matrix(mat: np.ndarray) -> bool:
    return mat.shape == (4, 4) and np.isfinite(mat).all()


class USVisualizer(Node):
    def __init__(self):
        super().__init__("us_visualizer")

        # Calibration transforms (all in millimeters)
        self.T_stylus_from_tip = None   # StylusTip -> Stylus
        self.T_probe_from_image = None  # Image -> Probe
        self.T_image_from_probe = None  # Probe -> Image (inverse)
        self.load_calibration_xml()

        # Dynamic transforms and availability flags (kept in millimeters)
        self.T_tracker_probe = None     # Tracker -> Probe (mm)
        self.T_tracker_stylus = None    # Tracker -> Stylus (mm)
        self.probe_valid = False
        self.needle_valid = False

        # QoS profiles
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

        # Subscriptions
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/image_raw/compressed",
            self.on_image,
            qos_profile=image_qos,
        )
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

        # Publisher
        self.image_pub = self.create_publisher(
            Image,
            "/visualize/us_imaging",
            qos_profile=image_qos,
        )

        self.last_probe_pose_dict = None  # stored as received (meters)
        self.last_needle_pose_dict = None

        # Periodic debug
        self.create_timer(5.0, self.log_latest_poses)

        self.get_logger().info("us_visualizer node started")

    def load_calibration_xml(self):
        """Load the unique PlusDeviceSet_fCal*.xml and parse required transforms (in mm)."""
        ws_dir = os.environ.get("WS_DIR", "/ani_ws")
        calib_dir = os.path.join(ws_dir, "calibration")
        pattern = os.path.join(calib_dir, "PlusDeviceSet_fCal*.xml")
        files = glob.glob(pattern)
        if len(files) == 0:
            msg = f"Calibration file not found: {pattern}"
            self.get_logger().error(msg)
            raise FileNotFoundError(msg)
        if len(files) > 1:
            msg = f"Multiple calibration files found: {files}. Ensure only one XML starting with PlusDeviceSet_fCal."
            self.get_logger().error(msg)
            raise RuntimeError(msg)

        xml_path = files[0]
        self.get_logger().info(f"Reading calibration file: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        def find_transform(frm, to):
            xpath = f".//Transform[@From='{frm}'][@To='{to}']"
            elem = root.find(xpath)
            if elem is None:
                raise RuntimeError(f"Transform From='{frm}' To='{to}' not found in {xml_path}")
            matrix_str = elem.get("Matrix")
            if matrix_str is None:
                raise RuntimeError(f"Transform From='{frm}' To='{to}' has no Matrix attribute in {xml_path}")
            mat = parse_matrix_string(matrix_str)
            if not is_finite_matrix(mat):
                raise RuntimeError(f"Transform From='{frm}' To='{to}' contains NaN/Inf in {xml_path}")
            return mat

        # StylusTip -> Stylus  (mm)
        self.T_stylus_from_tip = find_transform("StylusTip", "Stylus")
        # Image -> Probe (mm)
        self.T_probe_from_image = find_transform("Image", "Probe")
        # Probe -> Image (inverse) (mm)
        self.T_image_from_probe = np.linalg.inv(self.T_probe_from_image)
        if not is_finite_matrix(self.T_image_from_probe):
            raise RuntimeError("Probe -> Image inverse contains NaN/Inf")

        self.get_logger().info("Loaded transform matrices (units: mm):")
        self.get_logger().info(f"StylusTip -> Stylus:\n{self.T_stylus_from_tip}")
        self.get_logger().info(f"Image -> Probe:\n{self.T_probe_from_image}")
        self.get_logger().info(f"Probe -> Image (computed inverse):\n{self.T_image_from_probe}")

    def on_probe_pose(self, msg: PoseStamped):
        # Tracker -> Probe (incoming meters, convert to mm)
        if not is_finite_pose(msg):
            self.probe_valid = False
            self.T_tracker_probe = None
            self.get_logger().warn("Probe pose contains NaN/Inf, marking as Lost")
            return
        try:
            T_m = pose_to_hmat(msg)        # meters
            T_mm = to_mm(T_m)              # convert translation to mm
            if not is_finite_matrix(T_mm):
                raise ValueError("Probe pose matrix contains NaN/Inf after conversion to mm")
            self.T_tracker_probe = T_mm
            self.last_probe_pose_dict = pose_to_dict(msg)  # stored as received (meters)
            self.probe_valid = True
        except Exception as exc:
            self.get_logger().error(f"Failed to convert probe pose: {exc}")
            self.probe_valid = False
            self.T_tracker_probe = None

    def on_needle_pose(self, msg: PoseStamped):
        # Tracker -> Stylus (incoming meters, convert to mm)
        if not is_finite_pose(msg):
            self.needle_valid = False
            self.T_tracker_stylus = None
            self.get_logger().warn("Needle pose contains NaN/Inf, marking as Lost")
            return
        try:
            T_m = pose_to_hmat(msg)        # meters
            T_mm = to_mm(T_m)              # convert translation to mm
            if not is_finite_matrix(T_mm):
                raise ValueError("Needle pose matrix contains NaN/Inf after conversion to mm")
            self.T_tracker_stylus = T_mm
            self.last_needle_pose_dict = pose_to_dict(msg)  # stored as received (meters)
            self.needle_valid = True
        except Exception as exc:
            self.get_logger().error(f"Failed to convert needle pose: {exc}")
            self.needle_valid = False
            self.T_tracker_stylus = None

    def compute_stylus_and_tip_in_image(self):
        """
        计算 Stylus（针的 marker 原点）和 Needle Tip 在 Image 坐标系下的坐标（单位：mm）。
        返回 (stylus_xyz, tip_xyz)；若无效则返回 None。
        变换链：
          stylus_in_image = T_image_probe * T_probe_tracker * T_tracker_stylus * [0,0,0,1]
          tip_in_image    = T_image_probe * T_probe_tracker * T_tracker_stylus * (T_stylus_from_tip * [0,0,0,1])
        """
        if not (self.probe_valid and self.needle_valid):
            return None
        if self.T_tracker_probe is None or self.T_tracker_stylus is None:
            return None
        try:
            # Probe -> Tracker
            T_probe_from_tracker = np.linalg.inv(self.T_tracker_probe)
            if not is_finite_matrix(T_probe_from_tracker):
                return None

            # Stylus origin (marker) in Stylus frame
            p_stylus_in_stylus = np.array([0, 0, 0, 1.0])
            # StylusTip in Stylus frame
            p_tip_in_stylus = self.T_stylus_from_tip @ np.array([0, 0, 0, 1.0])

            # Stylus origin -> Tracker
            p_stylus_in_tracker = self.T_tracker_stylus @ p_stylus_in_stylus
            # Stylus tip -> Tracker
            p_tip_in_tracker = self.T_tracker_stylus @ p_tip_in_stylus

            # Tracker -> Probe
            p_stylus_in_probe = T_probe_from_tracker @ p_stylus_in_tracker
            p_tip_in_probe = T_probe_from_tracker @ p_tip_in_tracker

            # Probe -> Image
            p_stylus_in_image = self.T_image_from_probe @ p_stylus_in_probe
            p_tip_in_image = self.T_image_from_probe @ p_tip_in_probe

            if not (np.isfinite(p_stylus_in_image).all() and np.isfinite(p_tip_in_image).all()):
                return None

            stylus_xyz = p_stylus_in_image[:3]
            tip_xyz = p_tip_in_image[:3]

            # Log tip coordinates (mm)
            self.get_logger().info(
                f"Needle tip in Image frame (mm): x={tip_xyz[0]:.3f}, y={tip_xyz[1]:.3f}, z={tip_xyz[2]:.3f}"
            )

            return stylus_xyz, tip_xyz
        except Exception as exc:
            self.get_logger().error(f"Failed to compute stylus/tip in image: {exc}")
            return None

    def compute_tip_in_image(self):
        """
        保持原接口：仅返回 needle tip 在 Image 坐标系的坐标（mm）。
        """
        res = self.compute_stylus_and_tip_in_image()
        if res is None:
            return None
        _, tip_xyz = res
        return tip_xyz

    def draw_availability(self, frame):
        """Draw Tracked/Lost status for probe and needle in the top-left corner."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thick = 2
        green = (0, 200, 0)
        red = (0, 0, 255)

        probe_text = "Probe: Tracked" if self.probe_valid else "Probe: Lost"
        probe_color = green if self.probe_valid else red
        needle_text = "Needle: Tracked" if self.needle_valid else "Needle: Lost"
        needle_color = green if self.needle_valid else red

        cv2.putText(frame, probe_text, (10, 30), font, scale, probe_color, thick, cv2.LINE_AA)
        cv2.putText(frame, needle_text, (10, 60), font, scale, needle_color, thick, cv2.LINE_AA)

    @staticmethod
    def draw_dashed_line(frame, pt1, pt2, color, thickness=2, dash_length=10, gap_length=6):
        """
        在 frame 上绘制虚线。
        pt1, pt2: (x, y) 像素坐标（int）
        color: BGR
        """
        x1, y1 = pt1
        x2, y2 = pt2
        dx = x2 - x1
        dy = y2 - y1
        dist = float(np.hypot(dx, dy))
        if dist < 1e-3:
            return
        vx = dx / dist
        vy = dy / dist
        start = 0.0
        while start < dist:
            end = min(start + dash_length, dist)
            sx = int(round(x1 + vx * start))
            sy = int(round(y1 + vy * start))
            ex = int(round(x1 + vx * end))
            ey = int(round(y1 + vy * end))
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
            start += dash_length + gap_length

    def on_image(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn("Failed to decode compressed image (cv2.imdecode returned None)")
                return

            # Draw availability status
            self.draw_availability(frame)

            # Compute stylus origin & tip position, then overlay
            res = self.compute_stylus_and_tip_in_image()
            if res is not None:
                stylus_xyz, tip_xyz = res  # both in mm
                # NOTE: 假设 Image frame 的 x,y 直接对应图像像素坐标（若需 spacing，请先转换）
                u_tip, v_tip = int(round(tip_xyz[0])), int(round(tip_xyz[1]))
                u_sty, v_sty = int(round(stylus_xyz[0])), int(round(stylus_xyz[1]))
                h, w = frame.shape[:2]

                # 画针尖
                if 0 <= u_tip < w and 0 <= v_tip < h:
                    cv2.circle(frame, (u_tip, v_tip), radius=6, color=(0, 0, 255), thickness=2)
                    cv2.putText(frame, "needle tip", (u_tip + 5, v_tip - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    self.get_logger().debug(
                        f"Needle tip projected outside image: (u={u_tip}, v={v_tip}), frame size (w={w}, h={h})"
                    )

                # 画针轴虚线（stylus marker -> needle tip）
                color_axis = (0, 128, 255)  # 橙色系
                if (np.isfinite([u_sty, v_sty, u_tip, v_tip]).all()):
                    self.draw_dashed_line(frame, (u_sty, v_sty), (u_tip, v_tip),
                                          color=color_axis, thickness=2, dash_length=10, gap_length=6)
                else:
                    self.get_logger().debug("Stylus or tip projection not finite; skip axis drawing")
            else:
                self.get_logger().debug("Stylus/tip not computed (missing or invalid transforms)")

            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header = msg.header
            self.image_pub.publish(img_msg)
        except Exception as exc:
            self.get_logger().error(f"Exception decoding/publishing image: {exc}")

    def log_latest_poses(self):
        if self.last_probe_pose_dict:
            self.get_logger().debug(f"Probe pose dict (meters): {self.last_probe_pose_dict}")
        if self.last_needle_pose_dict:
            self.get_logger().debug(f"Needle pose dict (meters): {self.last_needle_pose_dict}")


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