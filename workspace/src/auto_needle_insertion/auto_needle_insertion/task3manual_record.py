import json
import logging
import os
import threading
import time
from typing import Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String

from auto_needle_insertion.rosbag_recorder_control import (
    RosbagController,
    TaskInfoPublisher,
    KeystrokeTopicInput,
    sleep_with_spin,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NODE_NAME = "task3manual_record"
TASK_NAME = "task3manual_record"
DELAY_START_ROSBAG_S = 1.5
DELAY_STOP_ROSBAG_S = 1.0
PATH_RECORD_DURATION_S = 3.0


def _auto_continue_enabled() -> bool:
    return os.getenv("TASK3_AUTO_CONTINUE", "").strip().lower() in ("1", "true", "yes")


def _is_enter_key(token: Optional[str]) -> bool:
    if token is None:
        return False
    t = token.strip().lower()
    return t in ("", "enter", "return", "\n", "\r", "<enter>", "<return>", "<vk_13>", "65293", "<vk_65293>")


def _is_cancel_key(token: Optional[str]) -> bool:
    if token is None:
        return False
    return token.strip().lower() == "c"


def _wait_for_enter(key_input: KeystrokeTopicInput, prompt: str, allow_cancel: bool = True) -> bool:
    print(prompt, flush=True)
    if _auto_continue_enabled():
        return True
    while rclpy.ok():
        token = key_input.get_key()
        if _is_enter_key(token):
            return True
        if allow_cancel and _is_cancel_key(token):
            return False
        time.sleep(0.05)
    return False


def _cancel_requested(key_input: KeystrokeTopicInput) -> bool:
    while True:
        token = key_input.get_key()
        if token is None:
            return False
        if _is_cancel_key(token):
            return True


class TaskProcedurePublisher(rclpy.node.Node):
    """Publish step changes on 'task_procedure'."""
    def __init__(self, topic_name: str = "task_procedure") -> None:
        super().__init__("task3_task_procedure_pub")
        self._pub = self.create_publisher(String, topic_name, 10)
        self._last: Optional[str] = None

    def publish_step(self, step: str) -> None:
        if step != self._last:
            self._last = step
            msg = String()
            msg.data = step
            self._pub.publish(msg)


class TaskInfoParamsPublisher(rclpy.node.Node):
    """Publish task parameters periodically (JSON string)."""
    def __init__(self, *, topic_name: str = "/task_info", hz: float = 1.0, payload: Optional[dict] = None) -> None:
        super().__init__("task3_params_publisher")
        self._pub = self.create_publisher(String, topic_name, 10)
        self._payload = payload or {}
        self._timer = self.create_timer(1.0 / hz, self._timer_cb)

    def update(self, key: str, value) -> None:
        self._payload[key] = value

    def _timer_cb(self) -> None:
        base_payload = {
            "TASK_NAME": TASK_NAME,
            "PHASES": [
                "place_probe_tumor",
                "plane_recording",
                "path_recording",
            ],
            "DELAY_START_ROSBAG_S": DELAY_START_ROSBAG_S,
            "DELAY_STOP_ROSBAG_S": DELAY_STOP_ROSBAG_S,
            "PATH_RECORD_DURATION_S": PATH_RECORD_DURATION_S,
        }
        payload = {**base_payload, **self._payload}
        msg = String()
        msg.data = json.dumps(payload)
        self._pub.publish(msg)


class ManualRecordTask:
    def __init__(self) -> None:
        rclpy.init()
        self.executor = SingleThreadedExecutor()
        self._spin_running = True

        self.task_info_pub = TaskInfoPublisher(topic_name="task_info_collection_states")
        self.task_proc_pub = TaskProcedurePublisher(topic_name="task_procedure")
        self.params_pub = TaskInfoParamsPublisher(topic_name="/task_info", hz=1.0)

        self.key_input = KeystrokeTopicInput(
            glyph_topic="/keyboard_listener/glyphkey_pressed",
            keycode_topic="/keyboard_listener/key_pressed",
        )

        self.executor.add_node(self.task_info_pub)
        self.executor.add_node(self.task_proc_pub)
        self.executor.add_node(self.params_pub)
        self.executor.add_node(self.key_input)

        self._spin_thread = threading.Thread(target=self._spin_executor, daemon=True)
        self._spin_thread.start()

        self.rosbag = RosbagController()

    def _spin_executor(self) -> None:
        while rclpy.ok() and self._spin_running:
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except Exception:
                break

    def run_cycle(self) -> bool:
        if _cancel_requested(self.key_input):
            return False

        # Step 3: place probe
        self.task_proc_pub.publish_step("place_probe_tumor")
        # make sure initial label is cleared or neutral if desired
        if not _wait_for_enter(
            self.key_input,
            "Place the probe around the tumor location, then press Enter (or 'c' to cancel)...",
        ):
            return False

        # Step 4: start rosbag for insertion plane
        self.task_proc_pub.publish_step("find_plane_start")
        # update task label for plane phase
        self.params_pub.update("task_label_FORCE", "Task 3 Manual plane")
        self.task_info_pub.set_state("started")
        print("Starting rosbag: find optimal insertion plane.", flush=True)
        self.rosbag.start_recording()
        sleep_with_spin(self.executor, DELAY_START_ROSBAG_S)

        # Step 5: stop rosbag and prompt for path determination
        self.task_proc_pub.publish_step("find_plane_stop")
        if not _wait_for_enter(
            self.key_input,
            "Press Enter to start recording when insertion path found. "
            "Probe should remain still while determining path (or 'c' to cancel)...",
        ):
            return False
        self.task_info_pub.set_state("stopped_success")
        print("Stopping rosbag (plane).", flush=True)
        self.rosbag.stop_recording("Success")
        sleep_with_spin(self.executor, DELAY_STOP_ROSBAG_S)

        if not _wait_for_enter(
            self.key_input,
            "Press Enter to stop recording (plane found). "
            "Probe should remain still while determining path (or 'c' to cancel)...",
        ):
            return False
        # Step 6: start rosbag for insertion path (auto stop after duration)
        self.task_proc_pub.publish_step("find_insertion_path_start")
        # update task label for path phase
        self.params_pub.update("task_label_FORCE", "Task 3 Manual path")
        self.task_info_pub.set_state("started")
        print("Starting rosbag: determine insertion path (probe should stay fixed).", flush=True)
        self.rosbag.start_recording()
        sleep_with_spin(self.executor, PATH_RECORD_DURATION_S)

        # Step 7: stop rosbag after duration
        self.task_proc_pub.publish_step("find_insertion_path_stop")
        self.task_info_pub.set_state("stopped_success")
        print("Stopping rosbag (path).", flush=True)
        self.rosbag.stop_recording("Success")
        sleep_with_spin(self.executor, DELAY_STOP_ROSBAG_S)

        return True

    def close(self) -> None:
        self._spin_running = False
        try:
            self.executor.remove_node(self.task_info_pub)
            self.executor.remove_node(self.task_proc_pub)
            self.executor.remove_node(self.params_pub)
            self.executor.remove_node(self.key_input)
        except Exception:
            pass
        try:
            self.executor.shutdown()
        except Exception:
            pass
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        try:
            self.task_info_pub.destroy_node()
            self.task_proc_pub.destroy_node()
            self.params_pub.destroy_node()
            self.key_input.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    task = ManualRecordTask()
    try:
        print("Starting task3 manual record. Steps 3-6 will loop until you press 'c'.", flush=True)
        while rclpy.ok():
            if not task.run_cycle():
                break
            print("Completed one cycle.", flush=True)
        print("Exiting task3 manual record.", flush=True)
    finally:
        task.close()


if __name__ == "__main__":
    main()

