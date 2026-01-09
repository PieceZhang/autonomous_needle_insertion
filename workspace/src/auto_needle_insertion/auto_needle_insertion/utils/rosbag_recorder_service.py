#!/usr/bin/env python3
import os
import shlex
import signal
import subprocess

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class RosbagRecorder(Node):
    def __init__(self) -> None:
        super().__init__("rosbag_recorder")
        self._record_script = os.getenv(
            "ROSBAG_RECORD_SCRIPT", "/ani_ws/scripts/run_openh_rosbag_record.sh"
        )
        self._process: subprocess.Popen | None = None
        self._start_service = self.create_service(Trigger, "rosbag/start", self._start_cb)
        self._stop_service = self.create_service(Trigger, "rosbag/stop", self._stop_cb)
        self.get_logger().info(
            "Rosbag recorder ready. Call /rosbag/start and /rosbag/stop from the dev container."
        )

    def _start_cb(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self._process and self._process.poll() is None:
            response.success = False
            response.message = f"Recording already running (pid {self._process.pid})."
            return response
        cmd = shlex.split(self._record_script)
        if not cmd:
            response.success = False
            response.message = "ROSBAG_RECORD_SCRIPT is empty."
            return response
        self.get_logger().info(f"Starting rosbag recorder script: {cmd}")
        self._process = subprocess.Popen(cmd)
        response.success = True
        response.message = "Recording started."
        return response

    def _stop_cb(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._process or self._process.poll() is not None:
            response.success = False
            response.message = "No active recording to stop."
            return response
        self.get_logger().info(f"Stopping ros2 bag record (pid {self._process.pid}).")
        self._process.send_signal(signal.SIGINT)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.get_logger().warning("Recorder did not exit in time; killing process.")
            self._process.kill()
            self._process.wait(timeout=5)
        self._process = None
        response.success = True
        response.message = "Recording stopped."
        return response

    def stop_if_running(self) -> None:
        if self._process and self._process.poll() is None:
            self.get_logger().info("Stopping ros2 bag record on shutdown.")
            self._process.send_signal(signal.SIGINT)
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.get_logger().warning("Recorder did not exit in time; killing process.")
                self._process.kill()
                self._process.wait(timeout=5)
        self._process = None


def main() -> None:
    rclpy.init()
    node = RosbagRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.stop_if_running()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
