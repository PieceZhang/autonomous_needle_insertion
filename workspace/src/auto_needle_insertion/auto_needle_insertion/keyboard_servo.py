"""Keyboard servo node for commanding end-effector local motion."""

from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Quaternion, TwistStamped
from moveit_msgs.srv import ServoCommandType
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


def quat_to_rot(q: Quaternion) -> np.ndarray:
    """Convert quaternion to rotation matrix.

    Args:
        q: Quaternion message

    Returns:
        3x3 rotation matrix as numpy array
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])


class MirrorServoNode(Node):
    """Node that maps keyboard input to end-effector local motion."""

    def __init__(self):
        """Initialize the mirror servo node."""
        super().__init__("keyboard_servo_to_ee")

        # --- Parameters (override in your launch if needed)
        self._declare_parameters()
        self._load_parameters()

        # TF setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # IO
        self.pub_twist = self.create_publisher(
            TwistStamped, self.servo_twist_topic, 10
        )
        self.create_subscription(String, self.glyph_topic, self.key_cb, 10)

        # Twist streaming timer
        self.last_linear = np.zeros(3)
        self.last_angular = np.zeros(3)
        self.last_update_rostime: Optional[float] = None
        self.timer = self.create_timer(
            1.0 / max(self.twist_rate, 1.0), self.publish_twist
        )

        # State
        self.last_commanded_local = np.zeros(3)
        self.key_map = {
            "w": np.array([0.0, 1.0, 0.0]),
            "s": np.array([0.0, -1.0, 0.0]),
            "d": np.array([1.0, 0.0, 0.0]),
            "a": np.array([-1.0, 0.0, 0.0]),
            "e": np.array([0.0, 0.0, 1.0]),
            "q": np.array([0.0, 0.0, -1.0]),
        }
        self.rot_key_map = {
            "r": np.array([1.0, 0.0, 0.0]),
            "f": np.array([-1.0, 0.0, 0.0]),
            "t": np.array([0.0, 1.0, 0.0]),
            "g": np.array([0.0, -1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "h": np.array([0.0, 0.0, -1.0]),
        }

        # Ensure Servo is in TWIST mode
        self.switch_servo_to_twist()

        self.get_logger().info(
            f"Keyboard servo ready: [{self.glyph_topic}] drives "
            f"[{self.ee_link}] in [{self.planning_frame}]"
        )
        self.get_logger().info(
            "Key map: W/S => +/-Y, A/D => +/-X, Q/E => +/-Z "
            f"at {self.key_speed:.3f} m/s."
        )
        self.get_logger().info(
            "Rotation map: R/F => +/-roll, T/G => +/-pitch, "
            f"Y/H => +/-yaw at {self.key_rot_speed:.3f} rad/s."
        )

    def _declare_parameters(self) -> None:
        """Declare all ROS parameters with default values."""
        self.declare_parameter(
            "glyphkey_topic", "/keyboard_listener/glyphkey_pressed"
        )
        self.declare_parameter("planning_frame", "base_link")
        self.declare_parameter("ee_link", "tool0")
        self.declare_parameter("servo_node_name", "servo_node")
        self.declare_parameter("servo_pose_topic", "/mirror_servo/pose_cmd")
        self.declare_parameter("servo_twist_topic", "/mirror_servo/twist_cmd")
        self.declare_parameter("twist_publish_rate_hz", 200.0)
        self.declare_parameter("speed_clamp_mps", 0.2)
        self.declare_parameter("rot_clamp_radps", 0.8)
        self.declare_parameter("hold_timeout_sec", 0.02)
        self.declare_parameter("key_speed_mps", 0.03)
        self.declare_parameter("key_rot_speed_radps", 0.3)

    def _load_parameters(self) -> None:
        """Load parameter values from ROS parameter server."""
        self.glyph_topic = (
            self.get_parameter("glyphkey_topic")
            .get_parameter_value().string_value
        )
        self.planning_frame = (
            self.get_parameter("planning_frame")
            .get_parameter_value().string_value
        )
        self.ee_link = (
            self.get_parameter("ee_link")
            .get_parameter_value().string_value
        )
        self.servo_node_name = (
            self.get_parameter("servo_node_name")
            .get_parameter_value().string_value
        )
        self.servo_pose_topic = (
            self.get_parameter("servo_pose_topic")
            .get_parameter_value().string_value
        )
        self.servo_twist_topic = (
            self.get_parameter("servo_twist_topic")
            .get_parameter_value().string_value
        )
        self.twist_rate = (
            self.get_parameter("twist_publish_rate_hz")
            .get_parameter_value().double_value
        )
        self.speed_clamp = (
            self.get_parameter("speed_clamp_mps")
            .get_parameter_value().double_value
        )
        self.rot_clamp = (
            self.get_parameter("rot_clamp_radps")
            .get_parameter_value().double_value
        )
        self.hold_timeout = (
            self.get_parameter("hold_timeout_sec")
            .get_parameter_value().double_value
        )
        self.key_speed = (
            self.get_parameter("key_speed_mps")
            .get_parameter_value().double_value
        )
        self.key_rot_speed = (
            self.get_parameter("key_rot_speed_radps")
            .get_parameter_value().double_value
        )

    def now_stamp(self) -> Time:
        """Get current ROS time as Time message."""
        return self.get_clock().now().to_msg()

    def lookup_transform(self, target_frame: str, source_frame: str):
        """Look up transform between frames."""
        return self.tf_buffer.lookup_transform(
            target_frame, source_frame, rclpy.time.Time(),
            timeout=Duration(seconds=0.2)
        )

    def get_current_ee_rotation(self) -> np.ndarray:
        """Get current end-effector rotation matrix in planning frame."""
        t = self.lookup_transform(self.planning_frame, self.ee_link)
        return quat_to_rot(t.transform.rotation)

    def clamp(self, v: np.ndarray, max_norm: float) -> np.ndarray:
        """Clamp vector magnitude to maximum value."""
        n = np.linalg.norm(v)
        if n <= max_norm or n < 1e-12:
            return v
        return v * (max_norm / n)

    def key_cb(self, msg: String) -> None:
        """Callback for keyboard glyph messages."""
        key = msg.data.strip().lower()
        if not key:
            return

        if key not in self.key_map and key not in self.rot_key_map:
            return

        try:
            R_ee = self.get_current_ee_rotation()
        except Exception as e:
            self.get_logger().warn(f"TF lookup EE failed: {e}")
            return

        v_local = np.zeros(3)
        omega_local = np.zeros(3)
        if key in self.key_map:
            v_local = self.key_map[key] * self.key_speed
        if key in self.rot_key_map:
            omega_local = self.rot_key_map[key] * self.key_rot_speed

        v_world_for_ee = self.clamp(R_ee @ v_local, self.speed_clamp)
        omega_world_for_ee = self.clamp(R_ee @ omega_local, self.rot_clamp)

        self.last_commanded_local = v_local
        self.last_linear = v_world_for_ee
        self.last_angular = omega_world_for_ee
        self.last_update_rostime = (
            self.get_clock().now().nanoseconds * 1e-9
        )

    def publish_twist(self) -> None:
        """Publish twist command to servo."""
        # If we haven't received tool updates recently, publish zero
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        min_hold = 1.0 / max(self.twist_rate, 1.0)
        hold_timeout = max(self.hold_timeout, min_hold)
        should_zero = (
            self.last_update_rostime is None or
            (now_sec - self.last_update_rostime) > hold_timeout
        )

        lin = np.zeros(3) if should_zero else self.last_linear
        ang = np.zeros(3) if should_zero else self.last_angular

        msg = TwistStamped()
        msg.header.stamp = self.now_stamp()
        # IMPORTANT: Twist must be expressed in the planning frame
        msg.header.frame_id = self.planning_frame
        msg.twist.linear.x = float(lin[0])
        msg.twist.linear.y = float(lin[1])
        msg.twist.linear.z = float(lin[2])
        msg.twist.angular.x = float(ang[0])
        msg.twist.angular.y = float(ang[1])
        msg.twist.angular.z = float(ang[2])
        self.pub_twist.publish(msg)

    def switch_servo_to_twist(self) -> None:
        """Switch servo node to TWIST command mode."""
        cli = self.create_client(
            ServoCommandType, f"/{self.servo_node_name}/switch_command_type"
        )
        if not cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                "Servo switch_command_type service not available; "
                "assuming TWIST already."
            )
            return

        req = ServoCommandType.Request()
        req.command_type = ServoCommandType.Request.TWIST
        try:
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
            if fut.result() and fut.result().success:
                self.get_logger().info("Servo set to TWIST input.")
            else:
                self.get_logger().warn("Failed to set Servo input to TWIST.")
        except Exception as e:
            self.get_logger().warn(f"Error switching Servo mode: {e}")


def main():
    """Main entry point."""
    rclpy.init()
    node = MirrorServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
