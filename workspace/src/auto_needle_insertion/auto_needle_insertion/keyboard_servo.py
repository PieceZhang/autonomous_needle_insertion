"""Keyboard servo node for commanding transducer-origin local motion."""

from pathlib import Path
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

from auto_needle_insertion.utils.us_probe import USProbe


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
    """Node that maps keyboard input to transducer-origin local motion."""

    def __init__(self):
        """Initialize the mirror servo node."""
        super().__init__("keyboard_servo_to_ee")

        # --- Parameters (override in your launch if needed)
        self._declare_parameters()
        self._load_parameters()

        self.to_in_ee = np.eye(4)
        if self.rotate_about_transducer_origin:
            self.us_probe = USProbe()
            self.to_in_ee = self._load_transducer_calibration()

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
        self.last_sent_frame_id = self.twist_frame
        self._last_key_event_sec: Optional[float] = None
        self._last_key_dt_sec: Optional[float] = None
        self.key_map = {
            "w": np.array([0.0, 0.0, 1.0]),
            "s": np.array([0.0, 0.0, -1.0]),
            "d": np.array([1.0, 0.0, 0.0]),
            "a": np.array([-1.0, 0.0, 0.0]),
            "e": np.array([0.0, 1.0, 0.0]),
            "q": np.array([0.0, -1.0, 0.0]),
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
            f"[{self.ee_link}] local motion in [{self.twist_frame}]"
        )
        self.get_logger().info(
            "Key map: W/S => +/-Z, A/D => +/-X, Q/E => +/-Y "
            f"at {self.key_speed:.3f} m/s."
        )
        self.get_logger().info(
            "Rotation map: R/F => +/-roll, T/G => +/-pitch, "
            f"Y/H => +/-yaw at {self.key_rot_speed:.3f} rad/s."
        )
        if self.rotate_about_transducer_origin:
            self.get_logger().info(
                "Rotation compensation about transducer origin is enabled."
            )

        if self.enable_axis_debug:
            self.pub_debug_cmd_local = self.create_publisher(
                TwistStamped, f"{self.debug_topic_prefix}/cmd_local", 10
            )
            self.pub_debug_actual_local = self.create_publisher(
                TwistStamped, f"{self.debug_topic_prefix}/actual_local", 10
            )
            self.pub_debug_err_local = self.create_publisher(
                TwistStamped, f"{self.debug_topic_prefix}/error_local", 10
            )
            self._debug_prev_T_base_ee = None
            self._debug_prev_time_sec = None
            self._debug_last_log_sec = -1e9
            self._publish_prev_time_sec = None
            self._publish_active_count = 0
            self._publish_zero_count = 0
            self._debug_last_publish_log_sec = -1e9
            self.debug_timer = self.create_timer(
                1.0 / max(self.debug_rate_hz, 1.0), self.debug_axis_cb
            )
            self.get_logger().info(
                f"Axis debug enabled: publishing local twist diagnostics under [{self.debug_topic_prefix}]"
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
        self.declare_parameter("twist_frame", "ee_link")
        self.declare_parameter("twist_publish_rate_hz", 200.0)
        self.declare_parameter("speed_clamp_mps", 0.2)
        self.declare_parameter("rot_clamp_radps", 0.8)
        self.declare_parameter("hold_timeout_sec", 0.02)
        self.declare_parameter("key_speed_mps", 0.3)
        self.declare_parameter("key_rot_speed_radps", 0.3)
        self.declare_parameter("rotate_about_transducer_origin", False)
        self.declare_parameter("enable_axis_debug", True)
        self.declare_parameter("debug_rate_hz", 30.0)
        self.declare_parameter("debug_log_period_sec", 0.5)
        self.declare_parameter("debug_topic_prefix", "/keyboard_servo/debug")
        self.declare_parameter("debug_motion_epsilon", 1e-4)
        self.declare_parameter(
            "calibration_xml_path",
            "./calibration/PlusDeviceSet_fCal_Wisonic_C5_1_NDIPolaris_2.0_20260111_SRIL.xml",
        )
        self.declare_parameter(
            "hand_eye_json_path",
            "./calibration/hand_eye_20251231_075559.json",
        )

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
        twist_frame_param = (
            self.get_parameter("twist_frame")
            .get_parameter_value().string_value.strip().lower()
        )
        self.twist_frame = (
            self.planning_frame if twist_frame_param == "planning_frame"
            else self.ee_link
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
        self.rotate_about_transducer_origin = (
            self.get_parameter("rotate_about_transducer_origin")
            .get_parameter_value().bool_value
        )
        self.enable_axis_debug = (
            self.get_parameter("enable_axis_debug")
            .get_parameter_value().bool_value
        )
        self.debug_rate_hz = (
            self.get_parameter("debug_rate_hz")
            .get_parameter_value().double_value
        )
        self.debug_log_period_sec = (
            self.get_parameter("debug_log_period_sec")
            .get_parameter_value().double_value
        )
        self.debug_topic_prefix = (
            self.get_parameter("debug_topic_prefix")
            .get_parameter_value().string_value
        )
        self.debug_motion_epsilon = (
            self.get_parameter("debug_motion_epsilon")
            .get_parameter_value().double_value
        )
        self.calibration_xml_path = (
            self.get_parameter("calibration_xml_path")
            .get_parameter_value().string_value
        )
        self.hand_eye_json_path = (
            self.get_parameter("hand_eye_json_path")
            .get_parameter_value().string_value
        )

    def _load_transducer_calibration(self) -> np.ndarray:
        """Load probe calibration and return TO in EE transform."""
        calibration_xml = Path(self.calibration_xml_path)
        hand_eye_json = Path(self.hand_eye_json_path)
        self.us_probe.load_calibrations(calibration_xml, hand_eye_json)
        if self.us_probe.to_in_ee is None:
            raise RuntimeError("US probe calibration failed to compute TO in EE.")
        return self.us_probe.to_in_ee

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

    def get_current_ee_transform(self) -> np.ndarray:
        """Get current end-effector transform in planning frame."""
        t = self.lookup_transform(self.planning_frame, self.ee_link)
        R = quat_to_rot(t.transform.rotation)
        p = np.array([
            t.transform.translation.x,
            t.transform.translation.y,
            t.transform.translation.z,
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def clamp(self, v: np.ndarray, max_norm: float) -> np.ndarray:
        """Clamp vector magnitude to maximum value."""
        n = np.linalg.norm(v)
        if n <= max_norm or n < 1e-12:
            return v
        return v * (max_norm / n)

    def _rotation_matrix_to_rotvec(self, R: np.ndarray) -> np.ndarray:
        """Convert a rotation matrix to a rotation-vector (axis * angle)."""
        tr = float(np.trace(R))
        cos_theta = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
        theta = float(np.arccos(cos_theta))
        if theta < 1e-7:
            return 0.5 * np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ])
        sin_theta = np.sin(theta)
        if abs(sin_theta) < 1e-8:
            # Near pi: fall back to skew approximation for numerical stability.
            return 0.5 * np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ])
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2.0 * sin_theta)
        return axis * theta

    def _publish_debug_twist(self, topic_pub, linear: np.ndarray, angular: np.ndarray) -> None:
        """Publish one debug twist sample in EE-local frame."""
        msg = TwistStamped()
        msg.header.stamp = self.now_stamp()
        msg.header.frame_id = self.ee_link
        msg.twist.linear.x = float(linear[0])
        msg.twist.linear.y = float(linear[1])
        msg.twist.linear.z = float(linear[2])
        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])
        topic_pub.publish(msg)

    def _log_coupling_metrics(self, cmd: np.ndarray, actual: np.ndarray, label: str) -> None:
        """Log primary-axis response and orthogonal leakage."""
        cmd_norm = np.linalg.norm(cmd)
        if cmd_norm < 1e-6:
            return
        i = int(np.argmax(np.abs(cmd)))
        axis_name = ["x", "y", "z"][i]
        primary_cmd = float(cmd[i])
        primary_actual = float(actual[i])
        orth_norm = float(np.linalg.norm(np.delete(actual, i)))
        leakage_ratio = orth_norm / max(abs(primary_cmd), 1e-9)
        self.get_logger().info(
            f"{label}: cmd_{axis_name}={primary_cmd:+.4f}, actual_{axis_name}={primary_actual:+.4f}, "
            f"orth_norm={orth_norm:.4f}, leakage={100.0 * leakage_ratio:.1f}%"
        )

    def _log_cross_domain_coupling(
        self,
        v_local_cmd: np.ndarray,
        omega_local_cmd: np.ndarray,
        v_local_actual: np.ndarray,
        omega_local_actual: np.ndarray,
    ) -> None:
        """Log angular leakage on linear commands and vice versa."""
        eps = self.debug_motion_epsilon
        lin_norm = float(np.linalg.norm(v_local_cmd))
        ang_norm = float(np.linalg.norm(omega_local_cmd))
        actual_lin_norm = float(np.linalg.norm(v_local_actual))
        actual_ang_norm = float(np.linalg.norm(omega_local_actual))

        if lin_norm > eps and ang_norm <= eps:
            rot_leak_pct = 100.0 * actual_ang_norm / max(self.rot_clamp, 1e-9)
            self.get_logger().info(
                f"Cross coupling (linear->angular): actual_omega_norm={actual_ang_norm:.4f} rad/s ({rot_leak_pct:.1f}% of rot clamp)"
            )
        elif ang_norm > eps and lin_norm <= eps:
            lin_leak_pct = 100.0 * actual_lin_norm / max(self.speed_clamp, 1e-9)
            self.get_logger().info(
                f"Cross coupling (angular->linear): actual_v_norm={actual_lin_norm:.4f} m/s ({lin_leak_pct:.1f}% of speed clamp)"
            )

    def debug_axis_cb(self) -> None:
        """Publish and log commanded vs measured EE-local axis motion."""
        try:
            T_base_ee = self.get_current_ee_transform()
        except Exception as e:
            self.get_logger().warn(f"Axis debug TF lookup failed: {e}")
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        R_base_ee = T_base_ee[:3, :3]
        p_base_ee = T_base_ee[:3, 3]

        if self._debug_prev_T_base_ee is None or self._debug_prev_time_sec is None:
            self._debug_prev_T_base_ee = T_base_ee
            self._debug_prev_time_sec = now_sec
            return

        dt = now_sec - self._debug_prev_time_sec
        if dt <= 1e-4:
            return

        R_prev = self._debug_prev_T_base_ee[:3, :3]
        p_prev = self._debug_prev_T_base_ee[:3, 3]
        v_base_actual = (p_base_ee - p_prev) / dt
        R_delta_base = R_base_ee @ R_prev.T
        omega_base_actual = self._rotation_matrix_to_rotvec(R_delta_base) / dt

        v_local_actual = R_base_ee.T @ v_base_actual
        omega_local_actual = R_base_ee.T @ omega_base_actual

        if self.last_sent_frame_id == self.planning_frame:
            v_local_cmd = R_base_ee.T @ self.last_linear
            omega_local_cmd = R_base_ee.T @ self.last_angular
        else:
            v_local_cmd = self.last_linear.copy()
            omega_local_cmd = self.last_angular.copy()

        v_local_err = v_local_actual - v_local_cmd
        omega_local_err = omega_local_actual - omega_local_cmd

        self._publish_debug_twist(self.pub_debug_cmd_local, v_local_cmd, omega_local_cmd)
        self._publish_debug_twist(self.pub_debug_actual_local, v_local_actual, omega_local_actual)
        self._publish_debug_twist(self.pub_debug_err_local, v_local_err, omega_local_err)

        if (now_sec - self._debug_last_log_sec) >= self.debug_log_period_sec:
            self._log_coupling_metrics(v_local_cmd, v_local_actual, "Linear coupling")
            self._log_coupling_metrics(omega_local_cmd, omega_local_actual, "Angular coupling")
            self._log_cross_domain_coupling(
                v_local_cmd,
                omega_local_cmd,
                v_local_actual,
                omega_local_actual,
            )
            self._debug_last_log_sec = now_sec

        self._debug_prev_T_base_ee = T_base_ee
        self._debug_prev_time_sec = now_sec

    def key_cb(self, msg: String) -> None:
        """Callback for keyboard glyph messages."""
        key = msg.data.strip().lower()
        if not key:
            return

        if key not in self.key_map and key not in self.rot_key_map:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self._last_key_event_sec is not None:
            self._last_key_dt_sec = now_sec - self._last_key_event_sec
        self._last_key_event_sec = now_sec

        v_local = np.zeros(3)
        omega_local = np.zeros(3)
        if key in self.key_map:
            v_local = self.key_map[key] * self.key_speed
        if key in self.rot_key_map:
            omega_local = self.rot_key_map[key] * self.key_rot_speed

        v_cmd = v_local.copy()
        omega_cmd = omega_local.copy()

        if self.twist_frame == self.planning_frame:
            try:
                T_base_ee = self.get_current_ee_transform()
            except Exception as e:
                self.get_logger().warn(f"TF lookup EE failed: {e}")
                return
            R_base_ee = T_base_ee[:3, :3]
            v_cmd = R_base_ee @ v_local
            omega_cmd = R_base_ee @ omega_local
            if self.rotate_about_transducer_origin and np.linalg.norm(omega_cmd) > 1e-12:
                r_base = R_base_ee @ self.to_in_ee[:3, 3]
                v_cmd = v_cmd - np.cross(omega_cmd, r_base)
        elif self.rotate_about_transducer_origin and np.linalg.norm(omega_cmd) > 1e-12:
            r_ee = self.to_in_ee[:3, 3]
            v_cmd = v_cmd - np.cross(omega_cmd, r_ee)

        v_world_for_ee = self.clamp(v_cmd, self.speed_clamp)
        omega_world_for_ee = self.clamp(omega_cmd, self.rot_clamp)

        self.last_commanded_local = v_local
        self.last_linear = v_world_for_ee
        self.last_angular = omega_world_for_ee
        self.last_sent_frame_id = self.twist_frame
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
        msg.header.frame_id = self.twist_frame
        msg.twist.linear.x = float(lin[0])
        msg.twist.linear.y = float(lin[1])
        msg.twist.linear.z = float(lin[2])
        msg.twist.angular.x = float(ang[0])
        msg.twist.angular.y = float(ang[1])
        msg.twist.angular.z = float(ang[2])
        self.pub_twist.publish(msg)

        if self.enable_axis_debug:
            if should_zero:
                self._publish_zero_count += 1
            else:
                self._publish_active_count += 1

            if self._publish_prev_time_sec is None:
                pub_dt = 0.0
            else:
                pub_dt = now_sec - self._publish_prev_time_sec
            self._publish_prev_time_sec = now_sec

            if (now_sec - self._debug_last_publish_log_sec) >= self.debug_log_period_sec:
                total = self._publish_active_count + self._publish_zero_count
                active_pct = 100.0 * self._publish_active_count / max(total, 1)
                cmd_age = (
                    -1.0 if self.last_update_rostime is None
                    else now_sec - self.last_update_rostime
                )
                key_dt = -1.0 if self._last_key_dt_sec is None else self._last_key_dt_sec
                self.get_logger().info(
                    "Command stream: "
                    f"active={self._publish_active_count}, zero={self._publish_zero_count}, "
                    f"active_pct={active_pct:.1f}%, hold_timeout={hold_timeout:.3f}s, "
                    f"cmd_age={cmd_age:.3f}s, key_dt={key_dt:.3f}s, pub_dt={pub_dt:.4f}s"
                )
                self._publish_active_count = 0
                self._publish_zero_count = 0
                self._debug_last_publish_log_sec = now_sec

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
