#!/usr/bin/env python3
import os
import signal
import subprocess
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


TOPIC_NAME = "rosbag_control"
RECORD_SCRIPT = "/opt/rosbag_control/run_openh_rosbag_record.sh"

# 可按需改成你的环境；留空则不额外 source
ROS_SETUP = "/opt/ros/jazzy/setup.bash"
# WS_SETUP = "/root/ros2_ws/install/local_setup.bash"


class RosbagControlService(Node):
    def __init__(self):
        super().__init__("rosbag_control_service")

        self._proc: Optional[subprocess.Popen] = None

        self.create_subscription(String, TOPIC_NAME, self._on_msg, 10)
        self.get_logger().info(f"Listening on /{TOPIC_NAME} (std_msgs/String).")
        self.get_logger().info("Commands: 'start' -> start recording, 'stop' -> SIGINT (Ctrl+C) recording.")

    def _on_msg(self, msg: String):
        cmd = (msg.data or "").strip().lower()
        if cmd == "start":
            self._start_recording()
        elif cmd == "stop":
            self._stop_recording()
        else:
            self.get_logger().warn(f"Unknown command: {msg.data!r} (expected 'start' or 'stop')")

    def _start_recording(self):
        if self._proc is not None and self._proc.poll() is None:
            self.get_logger().info("Recording is already running; ignoring 'start'.")
            return

        if not os.path.exists(RECORD_SCRIPT):
            self.get_logger().error(f"Recording script not found: {RECORD_SCRIPT}")
            return

        # 用 bash -lc 确保能 source 环境（也便于脚本里用 ROS2 命令）
        source_parts = []
        if os.path.exists(ROS_SETUP):
            source_parts.append(f"source {ROS_SETUP}")
        # if os.path.exists(WS_SETUP):
        #     source_parts.append(f"source {WS_SETUP}")

        cmd = " && ".join(source_parts + [f"exec bash {RECORD_SCRIPT}"]) if source_parts else f"exec bash {RECORD_SCRIPT}"

        self.get_logger().info(f"Starting recording: {RECORD_SCRIPT}")
        self._proc = subprocess.Popen(
            ["bash", "-lc", cmd],
            preexec_fn=os.setsid,   # 让子进程成为新进程组，便于发送 Ctrl+C 给整组
            stdout=None,
            stderr=None,
        )

    def _stop_recording(self):
        if self._proc is None or self._proc.poll() is not None:
            self.get_logger().info("Recording is not running; ignoring 'stop'.")
            return

        try:
            pgid = os.getpgid(self._proc.pid)
            self.get_logger().info(f"Stopping recording (SIGINT/Ctrl+C) pgid={pgid} ...")
            os.killpg(pgid, signal.SIGINT)

            try:
                self._proc.wait(timeout=15)
                self.get_logger().info(f"Recording stopped (exit code {self._proc.returncode}).")
            except subprocess.TimeoutExpired:
                self.get_logger().warn("Recording did not exit after SIGINT; sending SIGTERM ...")
                os.killpg(pgid, signal.SIGTERM)
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.get_logger().error("Still not exited; sending SIGKILL ...")
                    os.killpg(pgid, signal.SIGKILL)
                    self._proc.wait(timeout=5)

        finally:
            self._proc = None

    def shutdown(self):
        # 容器/service 停止时，尽量优雅停掉录包
        self._stop_recording()


def main():
    rclpy.init()
    node = RosbagControlService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
