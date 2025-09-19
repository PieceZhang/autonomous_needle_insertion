import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# Optional: read from TF instead of a topic
try:
    import tf2_ros
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

class NDIPrintPose(Node):
    def __init__(self):
        super().__init__("ndi_print_pose")
        self.pose_topic = None

        # Parameters:
        #  - pose_topic: a PoseStamped topic to subscribe to
        #  - tf_child_frame: if set, read pose from TF instead
        #  - tf_parent_frame: TF parent (default: 'world')
        self.declare_parameter("pose_topic", "")
        self.declare_parameter("tf_child_frame", "")
        self.declare_parameter("tf_parent_frame", "world")

        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        tf_child = self.get_parameter("tf_child_frame").get_parameter_value().string_value

        if pose_topic:
            self._subscribe_pose(pose_topic)
        elif tf_child:
            self._listen_tf(
                self.get_parameter("tf_parent_frame").get_parameter_value().string_value,
                tf_child
            )
        else:
            # Auto-detect a PoseStamped topic (prefer `/ndi/...`)
            autodetected = self._auto_detect_pose_topic()
            self._subscribe_pose(autodetected)

    def _subscribe_pose(self, topic: str):
        self.pose_topic = topic
        self.subscription = self.create_subscription(
            PoseStamped, topic, self._on_pose, 10
        )
        self.get_logger().info(f"Subscribing to PoseStamped: {topic}")

    def _listen_tf(self, parent: str, child: str):
        if not TF_AVAILABLE:
            raise RuntimeError("tf2_ros not available. Install and add exec_depend tf2_ros.")
        self.get_logger().info(f"Listening TF: {parent} -> {child}")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        # Poll TF at 20 Hz
        self.timer = self.create_timer(0.05, lambda: self._on_tf(parent, child))

    def _auto_detect_pose_topic(self) -> str:
        # Give discovery a moment
        for _ in range(10):
            names_types = self.get_topic_names_and_types()
            candidates = [n for n, types in names_types if "geometry_msgs/msg/PoseStamped" in types]
            ndi_candidates = [n for n in candidates if n.startswith("/ndi")]
            if ndi_candidates:
                return sorted(ndi_candidates)[0]
            if candidates:
                return sorted(candidates)[0]
            time.sleep(0.1)
        raise RuntimeError("No PoseStamped topic found. Set 'pose_topic' or 'tf_child_frame'.")

    def _on_pose(self, msg: PoseStamped):
        p = msg.pose.position
        topic_str = self.pose_topic if self.pose_topic else "<unknown>"
        self.get_logger().info(
            f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} "
            f"[topic: {topic_str}] [{msg.header.frame_id}] "
            f"pos = ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})"
        )

    def _on_tf(self, parent: str, child: str):
        try:
            t = self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time())
            tr = t.transform.translation
            self.get_logger().info(
                f"{t.header.stamp.sec}.{t.header.stamp.nanosec:09d} "
                f"{parent}->{child} pos = ({tr.x:.4f}, {tr.y:.4f}, {tr.z:.4f})"
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")

def main():
    rclpy.init()
    node = NDIPrintPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()