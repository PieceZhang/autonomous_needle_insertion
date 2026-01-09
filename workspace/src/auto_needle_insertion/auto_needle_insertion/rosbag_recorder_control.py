#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subscribe to keyboard listener topic and control rosbag recording.
- v: Start recording（调用 /ani_ws/scripts/run_openh_rosbag_record.sh），并将 task_info_collection_states 置为 started
- b/n/m: Wait for 200ms then stop recording（status code: Success / Failure / Recovery），并将 task_info_collection_states 分别置为 stopped_success / stopped_failure / stopped_recovery
- q: Quit
"""

import os
import sys
import time
import signal
import logging
import queue
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String, UInt32

# ----------------- 日志配置 -----------------
def configure_run_logging(log_dir: str = "/ani_ws/log") -> str:
    # 若不可写则回退到 /tmp
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        testfile = Path(log_dir) / ".wtest"
        with open(testfile, "w") as f:
            f.write("ok")
        testfile.unlink(missing_ok=True)
        chosen_dir = log_dir
    except Exception:
        chosen_dir = "/tmp"
        Path(chosen_dir).mkdir(parents=True, exist_ok=True)

    logfile = str(Path(chosen_dir) / "rosbag_recorder_control.log")
    # 追加模式，不新建新文件名
    logging.basicConfig(
        level=logging.INFO,
        filename=logfile,
        filemode="a",
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logfile

logfile = configure_run_logging(log_dir="/ani_ws/log")
logger = logging.getLogger("rosbag_recorder_control")
logger.info("Logging to %s", logfile)


# ----------------- 键盘订阅 -----------------
KEY_UP = "<UP>"
KEY_DOWN = "<DOWN>"
KEY_LEFT = "<LEFT>"
KEY_RIGHT = "<RIGHT>"
KEY_SPACE = "<SPACE>"

VK_MAP = {
    0x25: KEY_LEFT,
    0x26: KEY_UP,
    0x27: KEY_RIGHT,
    0x28: KEY_DOWN,
    0x20: KEY_SPACE,
    0x1B: "<ESC>",
}
VK_MAP.update(
    {
        65361: KEY_LEFT,
        65362: KEY_UP,
        65363: KEY_RIGHT,
        65364: KEY_DOWN,
    }
)

class KeystrokeTopicInput(Node):
    """
    订阅键盘话题，返回：
      - 单字符 glyph（如 'v','b','n','m'）
      - 特殊 token（如 '<UP>'）
    """
    def __init__(
        self,
        glyph_topic: str = "/keyboard_listener/glyphkey_pressed",
        keycode_topic: str = "/keyboard_listener/key_pressed",
        queue_depth: int = 200,
    ) -> None:
        super().__init__("rosbag_recorder_key_sub")
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=queue_depth)
        self._sub_glyph = self.create_subscription(String, glyph_topic, self._on_glyph, 10)
        self._sub_code = self.create_subscription(UInt32, keycode_topic, self._on_code, 10)

    def _push(self, token: str) -> None:
        try:
            self._q.put_nowait(token)
        except queue.Full:
            pass  # 忽略过载

    def _on_glyph(self, msg: String) -> None:
        if msg.data:
            self._push(msg.data)

    def _on_code(self, msg: UInt32) -> None:
        code = int(msg.data)
        token = VK_MAP.get(code, f"<VK_{code}>")
        self._push(token)

    def get_key(self) -> Optional[str]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None


# ----------------- 录制控制 -----------------
# Previously: SCRIPT_PATH = "/ani_ws/scripts/run_openh_rosbag_record.sh"
# Use send_rosbag_start_command.sh for both start and stop as requested
START_SCRIPT = "/ani_ws/scripts/send_rosbag_start_command.sh"
STOP_SCRIPT = "/ani_ws/scripts/send_rosbag_stop_command.sh"
STOP_WAIT_SEC = 0.2  # 200 ms

class RosbagController:
    """
    Spawn start/stop scripts asynchronously (do not wait for exit).
    Keep a small list of spawned processes for cleanup.
    """
    def __init__(self):
        self._is_recording = False
        self._procs: List[subprocess.Popen] = []

    def _cleanup_finished_procs(self) -> None:
        # drop processes that have already exited
        alive: List[subprocess.Popen] = []
        for p in self._procs:
            try:
                if p.poll() is None:
                    alive.append(p)
                else:
                    # ensure file descriptors closed if we opened them via Popen
                    try:
                        p.stdout and p.stdout.close()
                    except Exception:
                        pass
                    try:
                        p.stderr and p.stderr.close()
                    except Exception:
                        pass
            except Exception:
                pass
        self._procs = alive

    def _spawn_script_async(self, script_path: str) -> Optional[subprocess.Popen]:
        if not os.path.isfile(script_path):
            msg = f"The script does not exist: {script_path}"
            logger.error(msg)
            print(msg, flush=True)
            return None
        if not os.access(script_path, os.X_OK):
            msg = f"The script is not executable, run: chmod +x {script_path}"
            logger.error(msg)
            print(msg, flush=True)
            return None
        try:
            # Append stdout/stderr to the same logfile configured at module load
            # Open the logfile in append mode so child processes can write there.
            logf = open(logfile, "a")
            proc = subprocess.Popen(
                [script_path],
                stdout=logf,
                stderr=logf,
                start_new_session=True,
            )
            self._procs.append(proc)
            msg = f"Spawned script asynchronously: {script_path} (pid={proc.pid})"
            logger.info(msg)
            print(msg, flush=True)
            return proc
        except Exception as e:
            msg = f"Failed to spawn script {script_path}: {e}"
            logger.error(msg)
            print(msg, flush=True)
            return None

    def start_recording(self) -> None:
        # cleanup any finished child procs
        self._cleanup_finished_procs()

        if self._is_recording:
            msg = "Recording already in progress, ignored new start command."
            logger.warning(msg)
            print(msg, flush=True)
            return

        msg = f"Invoking start script asynchronously: {START_SCRIPT}"
        logger.info(msg)
        print(msg, flush=True)
        proc = self._spawn_script_async(START_SCRIPT)
        if proc:
            # consider start successful if script was spawned
            self._is_recording = True
            msg2 = f"Start script spawned (pid={proc.pid}), recording marked as started."
            logger.info(msg2)
            print(msg2, flush=True)
        else:
            msg2 = "Failed to spawn start script; recording not started."
            logger.error(msg2)
            print(msg2, flush=True)

    def stop_recording(self, reason: str) -> None:
        # cleanup any finished child procs
        self._cleanup_finished_procs()

        if not self._is_recording:
            msg = f"Recording not in progress, ignored stop command (reason: {reason})."
            logger.info(msg)
            print(msg, flush=True)
            return

        msg = f"Invoking stop script asynchronously (reason: {reason}): {STOP_SCRIPT}"
        logger.info(msg)
        print(msg, flush=True)
        proc = self._spawn_script_async(STOP_SCRIPT)
        if proc:
            msg2 = f"Stop script spawned (pid={proc.pid}), recording marked as stopped."
            logger.info(msg2)
            print(msg2, flush=True)
        else:
            msg2 = "Failed to spawn stop script; recording state will be cleared anyway."
            logger.warning(msg2)
            print(msg2, flush=True)

        # Clear state immediately (we do not wait for the stop script to finish)
        self._is_recording = False


# ----------------- 状态发布（10 Hz） -----------------
class TaskInfoPublisher(Node):
    """
    以 10 Hz 发布 task_info_collection_states（std_msgs/String）
    """
    def __init__(self, topic_name: str = "task_info_collection_states") -> None:
        super().__init__("task_info_publisher")
        self._pub = self.create_publisher(String, topic_name, 10)
        self._state = "idle"
        self._timer = self.create_timer(0.1, self._on_timer)  # 10 Hz

    def set_state(self, state: str) -> None:
        self._state = state

    def _on_timer(self) -> None:
        msg = String()
        msg.data = self._state
        self._pub.publish(msg)


# ----------------- 辅助：带 spin 的等待 -----------------
def sleep_with_spin(exec_: SingleThreadedExecutor, duration: float, step: float = 0.01) -> None:
    """
    在等待 duration 秒期间，持续 spin_once，确保计时器/回调不被阻塞。
    """
    end = time.monotonic() + duration
    while time.monotonic() < end:
        exec_.spin_once(timeout_sec=0.0)
        time.sleep(step)


# ----------------- 主循环 -----------------
def main():
    rclpy.init()
    key_in = KeystrokeTopicInput(
        glyph_topic="/keyboard_listener/glyphkey_pressed",
        keycode_topic="/keyboard_listener/key_pressed",
    )
    task_pub = TaskInfoPublisher(topic_name="task_info_collection_states")
    exec_ = SingleThreadedExecutor()
    exec_.add_node(key_in)
    exec_.add_node(task_pub)

    controller = RosbagController()

    try:
        print(
            "rosbag recording control node has started: "
            "Press v to start recording, b/n/m to stop recording, q to quit recording",
            flush=True,
        )
        while rclpy.ok():
            exec_.spin_once(timeout_sec=0.0)
            key = key_in.get_key()
            if key is None:
                time.sleep(0.02)
                continue

            key_norm = key.lower() if len(key) == 1 else key

            if key_norm == "q":
                msg = "Receiving q, stopping node."
                logger.info(msg)
                print(msg, flush=True)
                break

            if key_norm == "v":
                task_pub.set_state("started")
                controller.start_recording()
                continue

            if key_norm in ("b", "n", "m"):
                reason_map = {
                    "b": ("Success", "stopped_success"),
                    "n": ("Failure", "stopped_failure"),
                    "m": ("Recovery", "stopped_recovery"),
                }
                reason, state = reason_map[key_norm]
                task_pub.set_state(state)
                # 在停止前等待 STOP_WAIT_SEC，同时保持 spin，确保状态能被及时发布
                sleep_with_spin(exec_, STOP_WAIT_SEC)
                controller.stop_recording(reason)
                continue

            # 其他按键忽略
    finally:
        try:
            exec_.remove_node(key_in)
        except Exception:
            pass
        try:
            exec_.remove_node(task_pub)
        except Exception:
            pass
        try:
            key_in.destroy_node()
        except Exception:
            pass
        try:
            task_pub.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == "__main__":
    main()