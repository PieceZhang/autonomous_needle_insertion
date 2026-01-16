import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

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

NODE_NAME = "task2manual_record"
TASK_NAME = "task2manual_record"
DELAY_START_ROSBAG_S = 0.5
DELAY_STOP_ROSBAG_S = 0.5
ACTIONS = ["Action1", "Action2", "Action3"]


def _auto_continue_enabled() -> bool:
    return os.getenv("TASK2_AUTO_CONTINUE", "").strip().lower() in ("1", "true", "yes")


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
        super().__init__("task2_manual_task_procedure_pub")
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
        super().__init__("task2_manual_params_publisher")
        self._pub = self.create_publisher(String, topic_name, 10)
        self._payload = payload or {}
        self._timer = self.create_timer(1.0 / hz, self._timer_cb)

    def update(self, key: str, value) -> None:
        self._payload[key] = value

    def _timer_cb(self) -> None:
        base_payload = {
            "TASK_NAME": TASK_NAME,
            "ACTIONS": ACTIONS,
            "DELAY_START_ROSBAG_S": DELAY_START_ROSBAG_S,
            "DELAY_STOP_ROSBAG_S": DELAY_STOP_ROSBAG_S,
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
        for idx, action_name in enumerate(ACTIONS, start=1):
            if _cancel_requested(self.key_input):
                return False
            # Prompt to place probe
            self.task_proc_pub.publish_step(f"place_{action_name}")
            if not _wait_for_enter(
                self.key_input,
                f"Place probe to starting position of {action_name}, then press Enter (or 'c' to cancel)...",
            ):
                return False

            # Start rosbag
            self.task_proc_pub.publish_step(f"start_{action_name}")
            self.params_pub.update("task_label_FORCE", f"Task 2 Manual Action {idx}")
            self.task_info_pub.set_state("started")
            print(f"Starting rosbag for {action_name}.", flush=True)
            self.rosbag.start_recording()
            sleep_with_spin(self.executor, DELAY_START_ROSBAG_S)

            # Stop rosbag
            self.task_proc_pub.publish_step(f"stop_{action_name}")
            if not _wait_for_enter(
                self.key_input,
                f"Press Enter to stop recording for {action_name} (or 'c' to cancel)...",
            ):
                return False
            self.task_info_pub.set_state("stopped_success")
            print(f"Stopping rosbag for {action_name}.", flush=True)
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
        print("Starting task2 manual record. Steps 3-9 will loop (Action1-3) until you press 'c'.", flush=True)
        while rclpy.ok():
            if not task.run_cycle():
                break
            print("Completed one Action1-3 cycle.", flush=True)
        print("Exiting task2 manual record.", flush=True)
    finally:
        task.close()


if __name__ == "__main__":
    main()

