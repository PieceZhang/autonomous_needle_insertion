"""Tool follower node for mirroring tool motion to robot end-effector."""

import math
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Quaternion, TwistStamped
from moveit_msgs.srv import ServoCommandType
from rclpy.duration import Duration
from rclpy.node import Node
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


def rot_to_quat(R: np.ndarray) -> Quaternion:
    """Convert rotation matrix to quaternion.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion message
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    q = Quaternion()
    q.x, q.y, q.z, q.w = x, y, z, w
    return q


def quat_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Multiply two quaternions.

    Args:
        q1: First quaternion
        q2: Second quaternion

    Returns:
        Product quaternion
    """
    x1, y1, z1, w1 = q1.x, q1.y, q1.z, q1.w
    x2, y2, z2, w2 = q2.x, q2.y, q2.z, q2.w

    q = Quaternion()
    q.w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q.x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q.y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q.z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return q


class MirrorServoNode(Node):
    """Node that mirrors tool motion to robot end-effector via MoveIt Servo."""

    def __init__(self):
        """Initialize the mirror servo node."""
        super().__init__("mirror_tool_to_ee")

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
        self.create_subscription(
            PoseStamped, self.tool_topic, self.tool_cb, 10
        )

        # Twist streaming timer
        self.last_linear = np.zeros(3)
        self.last_angular = np.zeros(3)
        self.last_update_rostime: Optional[float] = None
        self.timer = self.create_timer(
            1.0 / max(self.twist_rate, 1.0), self.publish_twist
        )

        # Watchdog: warn if we aren't seeing tool poses
        self.seen_tool_msg = False
        self._watchdog_runs = 0
        self._watchdog = self.create_timer(1.0, self._topic_watchdog)

        # State
        self.prev_tool_pose_world: Optional[PoseStamped] = None

        # Ensure Servo is in TWIST mode
        self.switch_servo_to_twist()

        self.get_logger().info(
            f"Mirroring tool deltas from [{self.tool_topic}] to EE "
            f"[{self.ee_link}] in [{self.planning_frame}]"
        )

    def _declare_parameters(self) -> None:
        """Declare all ROS parameters with default values."""
        self.declare_parameter("tool_pose_topic", "/ndi/probe_pose")
        self.declare_parameter("planning_frame", "base_link")
        self.declare_parameter("ee_link", "tool0")
        self.declare_parameter("servo_node_name", "servo_node")
        self.declare_parameter("servo_pose_topic", "/mirror_servo/pose_cmd")
        self.declare_parameter("scale", 1.5)
        self.declare_parameter("deadband_m", 0.0005)
        self.declare_parameter("max_step_m", 0.01)
        self.declare_parameter("mirror_orientation", False)
        self.declare_parameter("servo_twist_topic", "/mirror_servo/twist_cmd")
        self.declare_parameter("twist_publish_rate_hz", 200.0)
        self.declare_parameter("speed_clamp_mps", 0.2)
        self.declare_parameter("rot_clamp_radps", 0.8)
        self.declare_parameter("hold_timeout_sec", 0.25)
        self.declare_parameter("use_msg_stamp_for_dt", False)
        self.declare_parameter("max_dt_for_velocity", 0.10)
        self.declare_parameter("ema_alpha", 0.6)

    def _load_parameters(self) -> None:
        """Load parameter values from ROS parameter server."""
        self.tool_topic = (
            self.get_parameter("tool_pose_topic")
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
        self.scale = (
            self.get_parameter("scale")
            .get_parameter_value().double_value
        )
        self.deadband = (
            self.get_parameter("deadband_m")
            .get_parameter_value().double_value
        )
        self.max_step = (
            self.get_parameter("max_step_m")
            .get_parameter_value().double_value
        )
        self.mirror_ori = (
            self.get_parameter("mirror_orientation")
            .get_parameter_value().bool_value
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
        self.use_msg_stamp = (
            self.get_parameter("use_msg_stamp_for_dt")
            .get_parameter_value().bool_value
        )
        self.max_dt = (
            self.get_parameter("max_dt_for_velocity")
            .get_parameter_value().double_value
        )
        self.ema_alpha = (
            self.get_parameter("ema_alpha")
            .get_parameter_value().double_value
        )

    def _topic_watchdog(self) -> None:
        """Monitor for incoming tool messages and warn if none received."""
        self._watchdog_runs += 1
        if self.seen_tool_msg:
            try:
                self._watchdog.cancel()
            except Exception:
                pass
            return

        if self._watchdog_runs in (1, 3, 5):
            try:
                topics = dict(self.get_topic_names_and_types())
                pose_topics = [
                    t for t, types in topics.items()
                    if any('geometry_msgs/msg/PoseStamped' in ty for ty in types)
                ]
                ndi_pose_topics = [
                    t for t in pose_topics if t.startswith('/ndi/')
                ]

                if ndi_pose_topics:
                    self.get_logger().warn(
                        f"No messages on '{self.tool_topic}' yet. "
                        f"Found PoseStamped topics under /ndi: {ndi_pose_topics}"
                    )
                else:
                    self.get_logger().warn(
                        f"No messages on '{self.tool_topic}' yet and no "
                        f"PoseStamped topics under /ndi. PoseStamped topics "
                        f"I can see: {pose_topics}"
                    )
            except Exception as e:
                self.get_logger().warn(f"Topic watchdog error: {e}")

        if self._watchdog_runs >= 5:
            try:
                self._watchdog.cancel()
            except Exception:
                pass

    def now_stamp(self) -> Time:
        """Get current ROS time as Time message."""
        return self.get_clock().now().to_msg()

    def lookup_transform(self, target_frame: str, source_frame: str):
        """Look up transform between frames."""
        return self.tf_buffer.lookup_transform(
            target_frame, source_frame, rclpy.time.Time(),
            timeout=Duration(seconds=0.2)
        )

    def transform_pose_to_planning(self, msg: PoseStamped) -> PoseStamped:
        """Transform pose to planning frame."""
        if msg.header.frame_id == self.planning_frame:
            return msg

        # T: source -> planning
        t = self.lookup_transform(self.planning_frame, msg.header.frame_id)
        R = quat_to_rot(t.transform.rotation)
        p = np.array([
            t.transform.translation.x,
            t.transform.translation.y,
            t.transform.translation.z
        ])
        pt = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        Rt = quat_to_rot(msg.pose.orientation)

        pt_out = R @ pt + p
        Rt_out = R @ Rt
        out = PoseStamped()
        out.header.stamp = self.now_stamp()
        out.header.frame_id = self.planning_frame
        out.pose.position.x = pt_out[0]
        out.pose.position.y = pt_out[1]
        out.pose.position.z = pt_out[2]
        out.pose.orientation = rot_to_quat(Rt_out)
        return out

    def get_current_ee_pose(self) -> PoseStamped:
        """Get current end-effector pose in planning frame."""
        t = self.lookup_transform(self.planning_frame, self.ee_link)
        out = PoseStamped()
        out.header.stamp = self.now_stamp()
        out.header.frame_id = self.planning_frame
        out.pose.position.x = t.transform.translation.x
        out.pose.position.y = t.transform.translation.y
        out.pose.position.z = t.transform.translation.z
        out.pose.orientation = t.transform.rotation
        return out

    def clamp(self, v: np.ndarray, max_norm: float) -> np.ndarray:
        """Clamp vector magnitude to maximum value."""
        n = np.linalg.norm(v)
        if n <= max_norm or n < 1e-12:
            return v
        return v * (max_norm / n)

    def tool_cb(self, msg: PoseStamped) -> None:
        """Callback for tool pose messages."""
        try:
            self.seen_tool_msg = True
            tool_world = self.transform_pose_to_planning(msg)
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return

        # Use node clock for dt, optionally use message stamp if requested
        now_t = self.get_clock().now().nanoseconds * 1e-9
        curr_t = now_t
        if self.use_msg_stamp and (msg.header.stamp is not None):
            t_candidate = (
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            )
            # Accept only strictly increasing, non-zero stamps
            if (t_candidate > 0.0 and
                t_candidate > getattr(self, "prev_raw_msg_time", 0.0)):
                curr_t = t_candidate
            self.prev_raw_msg_time = t_candidate

        if self.prev_tool_pose_world is None:
            self.prev_tool_pose_world = tool_world
            self.prev_tool_time = curr_t
            return

        # Δp in world
        p_now = np.array([
            tool_world.pose.position.x,
            tool_world.pose.position.y,
            tool_world.pose.position.z
        ])
        p_prev = np.array([
            self.prev_tool_pose_world.pose.position.x,
            self.prev_tool_pose_world.pose.position.y,
            self.prev_tool_pose_world.pose.position.z
        ])
        dp_world = p_now - p_prev

        if np.linalg.norm(dp_world) < self.deadband:
            self.prev_tool_pose_world = tool_world
            self.prev_tool_time = curr_t
            return

        # Tool-local increment: R_tool_prevᵀ * Δp_world
        R_tool_prev = quat_to_rot(self.prev_tool_pose_world.pose.orientation)
        dp_tool_local = R_tool_prev.T @ dp_world

        # EE current pose (world)
        try:
            ee_now = self.get_current_ee_pose()
        except Exception as e:
            self.get_logger().warn(f"TF lookup EE failed: {e}")
            return

        R_ee = quat_to_rot(ee_now.pose.orientation)

        # Map tool-local delta to EE frame local, then back to world
        dt = max(
            min(curr_t - getattr(self, "prev_tool_time", curr_t), self.max_dt),
            1e-3
        )
        v_world_for_ee = R_ee @ (self.scale * dp_tool_local / dt)

        # Optional angular velocity mirroring
        omega_world_for_ee = np.zeros(3)
        if self.mirror_ori:
            R_tool_now = quat_to_rot(tool_world.pose.orientation)
            dR_tool = R_tool_prev.T @ R_tool_now
            # Convert dR to axis-angle
            angle = math.acos(
                max(min((np.trace(dR_tool) - 1) / 2.0, 1.0), -1.0)
            )
            if angle > 1e-6:
                axis = np.array([
                    dR_tool[2, 1] - dR_tool[1, 2],
                    dR_tool[0, 2] - dR_tool[2, 0],
                    dR_tool[1, 0] - dR_tool[0, 1],
                ]) / (2.0 * math.sin(angle))
                omega_tool_local = axis * (angle / dt)
                omega_world_for_ee = R_ee @ omega_tool_local

        # Exponential moving average for smoothing
        self.last_linear = (
            self.ema_alpha * v_world_for_ee +
            (1.0 - self.ema_alpha) * self.last_linear
        )
        self.last_angular = (
            self.ema_alpha * omega_world_for_ee +
            (1.0 - self.ema_alpha) * self.last_angular
        )
        v_world_for_ee = self.last_linear
        omega_world_for_ee = self.last_angular

        # Clamp speeds
        v_world_for_ee = self.clamp(v_world_for_ee, self.speed_clamp)
        omega_world_for_ee = self.clamp(omega_world_for_ee, self.rot_clamp)

        # Deadband on linear speed
        if np.linalg.norm(v_world_for_ee) < self.deadband:
            v_world_for_ee[:] = 0.0
            omega_world_for_ee[:] = 0.0

        # Update the last commanded twist
        self.last_linear = v_world_for_ee
        self.last_angular = omega_world_for_ee
        self.last_update_rostime = curr_t

        self.prev_tool_time = curr_t
        self.prev_tool_pose_world = tool_world

    def publish_twist(self) -> None:
        """Publish twist command to servo."""
        # If we haven't received tool updates recently, publish zero
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        should_zero = (
            self.last_update_rostime is None or
            (now_sec - self.last_update_rostime) > (2.0 * self.hold_timeout)
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