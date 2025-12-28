#!/usr/bin/env python3
import os
import glob
import json
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
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
        self.tip_offset_mm = None         # NeedleBody -> NeedleTip translation (mm)
        self.T_probe_from_image = None    # Image -> Probe
        self.T_image_from_probe = None    # Probe -> Image (inverse)
        self.image_lag_sec = 0.0          # LocalTimeOffsetSec (image lags tracker by this many seconds)
        self.load_calibration_xml()       # still loads Image<->Probe and LocalTimeOffsetSec
        self.load_tip_offset_json()       # load needle tip offset from JSON

        # Dynamic transforms (latest) and availability flags (kept in millimeters)
        self.T_tracker_probe = None          # Tracker -> Probe (mm)
        self.T_tracker_needle_body = None    # Tracker -> NeedleBody (mm)
        self.probe_valid = False
        self.needle_valid = False

        # Pose history buffers for time synchronization
        self.pose_buffer_size = 500  # ~25s at 20 Hz; adjust as needed
        self.probe_buf = deque(maxlen=self.pose_buffer_size)    # list of (t_sec, T_mm)
        self.needle_buf = deque(maxlen=self.pose_buffer_size)   # list of (t_sec, T_mm)

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

        # Publishers (compressed images)
        self.image_pub = self.create_publisher(
            CompressedImage,
            "/visualize/us_imaging/compressed",
            qos_profile=image_qos,
        )
        # New synchronized overlay publisher
        self.image_pub_sync = self.create_publisher(
            CompressedImage,
            "/visualize/us_imaging_sync/compressed",
            qos_profile=image_qos,
        )

        # Publishers: arrays [x, y, z] in mm (Image frame)
        self.tip_pub = self.create_publisher(
            Float32MultiArray,
            "decoded_coor_image/needle_tip",
            qos_profile=pose_qos,
        )
        self.origin_pub = self.create_publisher(
            Float32MultiArray,
            "decoded_coor_image/needle_origin",
            qos_profile=pose_qos,
        )

        # Timer: 20 Hz publishing of decoded coordinates (unsynchronized; unchanged)
        self.create_timer(0.05, self.publish_decoded_coordinates)

        self.last_probe_pose_dict = None  # stored as received (meters)
        self.last_needle_pose_dict = None

        # Periodic debug
        self.create_timer(5.0, self.log_latest_poses)

        self.get_logger().info("us_visualizer node started")

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        """Convert ROS2 builtin_interfaces/Time to float seconds."""
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def load_calibration_xml(self):
        """Load the unique PlusDeviceSet_fCal*.xml and parse required transforms (in mm) and LocalTimeOffsetSec."""
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

        # Image -> Probe
        self.T_probe_from_image = find_transform("Image", "Probe")
        # Probe -> Image (inverse) (mm)
        self.T_image_from_probe = np.linalg.inv(self.T_probe_from_image)
        if not is_finite_matrix(self.T_image_from_probe):
            raise RuntimeError("Probe -> Image inverse contains NaN/Inf")

        # Parse LocalTimeOffsetSec (image lag relative to tracker)
        self.image_lag_sec = 0.0
        tracker_device = root.find(".//Device[@Type='PolarisTracker']") or root.find(".//Device[@Id='TrackerDevice']")
        if tracker_device is not None and tracker_device.get("LocalTimeOffsetSec") is not None:
            try:
                self.image_lag_sec = float(tracker_device.get("LocalTimeOffsetSec"))
            except ValueError:
                self.get_logger().warn(f"Invalid LocalTimeOffsetSec value; defaulting to 0.0")
                self.image_lag_sec = 0.0
        else:
            self.get_logger().warn("LocalTimeOffsetSec not found; defaulting to 0.0")

        self.get_logger().info("Loaded transform matrices (units: mm):")
        self.get_logger().info(f"Image -> Probe:\n{self.T_probe_from_image}")
        self.get_logger().info(f"Probe -> Image (computed inverse):\n{self.T_image_from_probe}")
        self.get_logger().info(f"LocalTimeOffsetSec (image lag vs tracker): {self.image_lag_sec:.6f} s")

    def load_tip_offset_json(self):
        """
        Load tip_offset_mm (length 3, mm) from calibration/needle_1_tip_offset.json,
        representing NeedleBody -> NeedleTip translation (rotation = identity).
        """
        ws_dir = os.environ.get("WS_DIR", "/ani_ws")
        json_path = os.path.join(ws_dir, "calibration", "needle_1_tip_offset.json")
        if not os.path.isfile(json_path):
            msg = f"Tip offset JSON not found: {json_path}"
            self.get_logger().error(msg)
            raise FileNotFoundError(msg)
        with open(json_path, "r") as f:
            data = json.load(f)
        if "tip_offset_mm" not in data:
            raise RuntimeError(f"'tip_offset_mm' not found in {json_path}")
        offset = np.array(data["tip_offset_mm"], dtype=float).reshape(-1)
        if offset.shape[0] != 3 or not np.isfinite(offset).all():
            raise RuntimeError(f"Invalid tip_offset_mm in {json_path}: {offset}")
        self.tip_offset_mm = offset
        self.get_logger().info(f"Loaded tip_offset_mm (NeedleBody -> NeedleTip, mm): {self.tip_offset_mm}")

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
            # Buffer with timestamp (for synchronization)
            t_sec = self._stamp_to_sec(msg.header.stamp)
            self.probe_buf.append((t_sec, T_mm))
        except Exception as exc:
            self.get_logger().error(f"Failed to convert probe pose: {exc}")
            self.probe_valid = False
            self.T_tracker_probe = None

    def on_needle_pose(self, msg: PoseStamped):
        # Tracker -> NeedleBody (incoming meters, convert to mm)
        if not is_finite_pose(msg):
            self.needle_valid = False
            self.T_tracker_needle_body = None
            self.get_logger().warn("Needle pose contains NaN/Inf, marking as Lost")
            return
        try:
            T_m = pose_to_hmat(msg)        # meters
            T_mm = to_mm(T_m)              # convert translation to mm
            if not is_finite_matrix(T_mm):
                raise ValueError("Needle pose matrix contains NaN/Inf after conversion to mm")
            self.T_tracker_needle_body = T_mm
            self.last_needle_pose_dict = pose_to_dict(msg)  # stored as received (meters)
            self.needle_valid = True
            # Buffer with timestamp (for synchronization)
            t_sec = self._stamp_to_sec(msg.header.stamp)
            self.needle_buf.append((t_sec, T_mm))
        except Exception as exc:
            self.get_logger().error(f"Failed to convert needle pose: {exc}")
            self.needle_valid = False
            self.T_tracker_needle_body = None

    def compute_needle_body_and_tip_in_image_from_transforms(self, T_tracker_probe, T_tracker_needle_body, log_tip=True):
        """
        Core computation using provided transforms (Tracker->Probe, Tracker->NeedleBody) in mm.
        Returns (needle_body_xyz, tip_xyz) in Image frame, mm.
        """
        if self.tip_offset_mm is None:
            self.get_logger().error("tip_offset_mm not loaded")
            return None
        try:
            # Probe -> Tracker
            T_probe_from_tracker = np.linalg.inv(T_tracker_probe)
            if not is_finite_matrix(T_probe_from_tracker):
                return None

            # NeedleBody origin in NeedleBody frame
            p_body_in_body = np.array([0, 0, 0, 1.0])
            # NeedleTip in NeedleBody frame (translation only)
            p_tip_in_body = np.array([self.tip_offset_mm[0],
                                      self.tip_offset_mm[1],
                                      self.tip_offset_mm[2],
                                      1.0])

            # NeedleBody origin -> Tracker
            p_body_in_tracker = T_tracker_needle_body @ p_body_in_body
            # Needle tip -> Tracker
            p_tip_in_tracker = T_tracker_needle_body @ p_tip_in_body

            # Tracker -> Probe
            p_body_in_probe = T_probe_from_tracker @ p_body_in_tracker
            p_tip_in_probe = T_probe_from_tracker @ p_tip_in_tracker

            # Probe -> Image
            p_body_in_image = self.T_image_from_probe @ p_body_in_probe
            p_tip_in_image = self.T_image_from_probe @ p_tip_in_probe

            if not (np.isfinite(p_body_in_image).all() and np.isfinite(p_tip_in_image).all()):
                return None

            needle_body_xyz = p_body_in_image[:3]
            tip_xyz = p_tip_in_image[:3]

            if log_tip:
                # Log tip coordinates (mm)
                self.get_logger().info(
                    f"Needle tip in Image frame (mm): x={tip_xyz[0]:.3f}, y={tip_xyz[1]:.3f}, z={tip_xyz[2]:.3f}"
                )

            return needle_body_xyz, tip_xyz
        except Exception as exc:
            self.get_logger().error(f"Failed to compute needle body/tip in image: {exc}")
            return None

    def compute_needle_body_and_tip_in_image(self):
        """
        Compute NeedleBody (marker origin) and NeedleTip in Image frame (mm) using latest transforms.
        """
        if not (self.probe_valid and self.needle_valid):
            return None
        if self.T_tracker_probe is None or self.T_tracker_needle_body is None:
            return None
        return self.compute_needle_body_and_tip_in_image_from_transforms(
            self.T_tracker_probe, self.T_tracker_needle_body, log_tip=True
        )

    def compute_tip_in_image(self):
        """
        Keep original interface: return needle tip in Image frame (mm).
        """
        res = self.compute_needle_body_and_tip_in_image()
        if res is None:
            return None
        _, tip_xyz = res
        return tip_xyz

    def _find_closest_transform(self, buffer_deque, target_time, max_dt=0.5):
        """
        Find the transform in buffer whose timestamp is closest to target_time.
        Returns the transform (4x4) or None if not found or too far.
        """
        if len(buffer_deque) == 0:
            return None
        times = [abs(t - target_time) for (t, _) in buffer_deque]
        idx = int(np.argmin(times))
        if times[idx] > max_dt:
            return None
        return buffer_deque[idx][1]

    def compute_synced_needle_body_and_tip_in_image(self, image_header):
        """
        Compute NeedleBody/Tip in Image frame (mm) using time-synchronized poses.
        Image lags tracker by self.image_lag_sec, so we look for poses at (t_img - lag).
        """
        if image_header is None:
            return None
        t_img = self._stamp_to_sec(image_header.stamp)
        target_time = t_img - self.image_lag_sec

        T_probe_sync = self._find_closest_transform(self.probe_buf, target_time)
        T_needle_sync = self._find_closest_transform(self.needle_buf, target_time)

        if T_probe_sync is None or T_needle_sync is None:
            # Debug only to avoid spamming error logs
            self.get_logger().debug("Synchronized poses not found within tolerance; skip synced overlay")
            return None

        return self.compute_needle_body_and_tip_in_image_from_transforms(
            T_probe_sync, T_needle_sync, log_tip=False
        )

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
        Draw a dashed line on frame.
        pt1, pt2: (x, y) pixel coords (int); color: BGR
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

    @staticmethod
    def draw_text_with_outline(frame, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                               scale=0.55, color=(255, 255, 255), thickness=1):
        """Draw text with outline for readability."""
        # outline
        cv2.putText(frame, text, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # text
        cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)

    def draw_needle_overlay(self, frame, needle_body_xyz, tip_xyz, tag_suffix=""):
        """
        Draw needle tip, body, dashed axis, and coordinates on the given frame.
        tag_suffix (e.g., "[synced]") will be appended to coordinate text lines.
        """
        # NOTE: Assume Image frame x,y directly map to image pixel coordinates (spacing not applied)
        u_tip, v_tip = int(round(tip_xyz[0])), int(round(tip_xyz[1]))
        u_body, v_body = int(round(needle_body_xyz[0])), int(round(needle_body_xyz[1]))
        h, w = frame.shape[:2]

        # Draw needle tip
        if 0 <= u_tip < w and 0 <= v_tip < h:
            cv2.circle(frame, (u_tip, v_tip), radius=6, color=(0, 0, 255), thickness=2)
            cv2.putText(frame, "needle tip", (u_tip + 5, v_tip - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            self.get_logger().debug(
                f"Needle tip projected outside image: (u={u_tip}, v={v_tip}), frame size (w={w}, h={h})"
            )

        # Draw needle axis dashed line
        color_axis = (0, 128, 255)  # orange-ish
        if (np.isfinite([u_body, v_body, u_tip, v_tip]).all()):
            self.draw_dashed_line(frame, (u_body, v_body), (u_tip, v_tip),
                                  color=color_axis, thickness=2, dash_length=10, gap_length=6)
        else:
            self.get_logger().debug("Needle body or tip projection not finite; skip axis drawing")

        # Draw coordinates text (mm)
        tag = f" {tag_suffix}" if tag_suffix else ""
        tip_text = f"Tip (mm): x={tip_xyz[0]:.1f}, y={tip_xyz[1]:.1f}, z={tip_xyz[2]:.1f}{tag}"
        body_text = f"Origin (mm): x={needle_body_xyz[0]:.1f}, y={needle_body_xyz[1]:.1f}, z={needle_body_xyz[2]:.1f}{tag}"
        self.draw_text_with_outline(frame, tip_text, (10, 95))
        self.draw_text_with_outline(frame, body_text, (10, 120))

    def on_image(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            base_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if base_frame is None:
                self.get_logger().warn("Failed to decode compressed image (cv2.imdecode returned None)")
                return

            # -------------------------------
            # Unsynchronized overlay (original behavior) -> /visualize/us_imaging/compressed
            # -------------------------------
            frame_unsynced = base_frame.copy()
            self.draw_availability(frame_unsynced)

            res_unsynced = self.compute_needle_body_and_tip_in_image()
            if res_unsynced is not None:
                needle_body_xyz, tip_xyz = res_unsynced  # both in mm
                self.draw_needle_overlay(frame_unsynced, needle_body_xyz, tip_xyz, tag_suffix="")
            else:
                self.get_logger().debug("NeedleBody/Tip not computed (missing or invalid transforms)")

            img_msg_unsynced = self.bridge.cv2_to_compressed_imgmsg(frame_unsynced, dst_format="jpeg")
            img_msg_unsynced.header = msg.header
            self.image_pub.publish(img_msg_unsynced)

            # -------------------------------
            # Time-synchronized overlay -> /visualize/us_imaging_sync/compressed
            # -------------------------------
            frame_sync = base_frame.copy()
            self.draw_availability(frame_sync)

            res_sync = self.compute_synced_needle_body_and_tip_in_image(msg.header)
            if res_sync is not None:
                needle_body_xyz_sync, tip_xyz_sync = res_sync  # both in mm
                self.draw_needle_overlay(frame_sync, needle_body_xyz_sync, tip_xyz_sync, tag_suffix="[synced]")
            else:
                self.get_logger().debug("Synchronized NeedleBody/Tip not available; skip synced overlay")

            img_msg_sync = self.bridge.cv2_to_compressed_imgmsg(frame_sync, dst_format="jpeg")
            img_msg_sync.header = msg.header
            self.image_pub_sync.publish(img_msg_sync)

        except Exception as exc:
            self.get_logger().error(f"Exception decoding/publishing image: {exc}")

    def publish_decoded_coordinates(self):
        """
        Publish at 20 Hz two topics:
        - decoded_coor_image/needle_tip: Float32MultiArray, data=[x,y,z] (mm)
        - decoded_coor_image/needle_origin: Float32MultiArray, data=[x,y,z] (mm)
        Frame: Image
        Requirement: even if needle is lost, continue publishing with NaN.
        """
        nan = float("nan")
        needle_body_xyz = None
        tip_xyz = None

        res = self.compute_needle_body_and_tip_in_image()
        if res is not None:
            needle_body_xyz, tip_xyz = res

        msg_tip = Float32MultiArray()
        msg_ori = Float32MultiArray()

        if tip_xyz is not None:
            msg_tip.data = [float(tip_xyz[0]), float(tip_xyz[1]), float(tip_xyz[2])]
        else:
            msg_tip.data = [nan, nan, nan]

        if needle_body_xyz is not None:
            msg_ori.data = [float(needle_body_xyz[0]), float(needle_body_xyz[1]), float(needle_body_xyz[2])]
        else:
            msg_ori.data = [nan, nan, nan]

        self.tip_pub.publish(msg_tip)
        self.origin_pub.publish(msg_ori)

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