#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于键盘话题的 rosbag 录制控制：
- V：开始录制（调用 /ani_ws/scripts/run_openh_rosbag_record.sh）
- B/N/M：等待 200ms 后停止录制（分别代表 Success / Failure / Recovery）
- Q：退出
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
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String, UInt32

# ----------------- 日志配置 -----------------
def configure_run_logging(log_dir: str = "/tmp") -> str:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = str(Path(log_dir) / f"rosbag_recorder_control_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        filename=logfile,
        filemode="w",
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logfile

logfile = configure_run_logging(log_dir="../log")
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
SCRIPT_PATH = "/ani_ws/scripts/run_openh_rosbag_record.sh"
STOP_WAIT_SEC = 0.2  # 200 ms

class RosbagController:
    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None

    def start_recording(self) -> None:
        if self._proc and self._proc.poll() is None:
            msg = "录制已在进行中，忽略新的开始命令。"
            logger.warning(msg)
            print(msg, flush=True)
            return

        if not os.path.isfile(SCRIPT_PATH):
            msg = f"录制脚本不存在: {SCRIPT_PATH}"
            logger.error(msg)
            print(msg, flush=True)
            return
        if not os.access(SCRIPT_PATH, os.X_OK):
            msg = f"录制脚本不可执行，请 chmod +x {SCRIPT_PATH}"
            logger.error(msg)
            print(msg, flush=True)
            return

        try:
            # 使用新的进程组，便于后续发送 Ctrl+C (SIGINT) 终止
            self._proc = subprocess.Popen(
                [SCRIPT_PATH],
                stdout=sys.stdout,
                stderr=sys.stderr,
                preexec_fn=os.setsid,  # 启动新进程组
            )
            msg = f"开始录制，启动脚本: {SCRIPT_PATH} (pid={self._proc.pid})"
            logger.info(msg)
            print(msg, flush=True)
        except Exception as e:
            msg = f"启动录制失败: {e}"
            logger.error(msg)
            print(msg, flush=True)

    def stop_recording(self, reason: str) -> None:
        if not self._proc or self._proc.poll() is not None:
            msg = f"未在录制，忽略停止命令（原因: {reason}）。"
            logger.info(msg)
            print(msg, flush=True)
            return

        # 按需求等待 200 ms 再停止
        time.sleep(STOP_WAIT_SEC)

        try:
            pgid = os.getpgid(self._proc.pid)
        except Exception:
            pgid = None

        msg = f"停止录制（原因: {reason}），发送 SIGINT..."
        logger.info(msg)
        print(msg, flush=True)

        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGINT)
            else:
                self._proc.send_signal(signal.SIGINT)
        except Exception as e:
            logger.warning(f"发送 SIGINT 失败: {e}")

        try:
            ret = self._proc.wait(timeout=5.0)
            msg = f"录制脚本已退出，退出码: {ret}"
            logger.info(msg)
            print(msg, flush=True)
        except subprocess.TimeoutExpired:
            msg = "录制脚本未及时退出，发送 SIGTERM..."
            logger.warning(msg)
            print(msg, flush=True)
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    self._proc.terminate()
                ret = self._proc.wait(timeout=3.0)
                msg2 = f"录制脚本已退出，退出码: {ret}"
                logger.info(msg2)
                print(msg2, flush=True)
            except subprocess.TimeoutExpired:
                msg3 = "录制脚本仍未退出，发送 SIGKILL..."
                logger.error(msg3)
                print(msg3, flush=True)
                try:
                    if pgid is not None:
                        os.killpg(pgid, signal.SIGKILL)
                    else:
                        self._proc.kill()
                except Exception as e:
                    logger.error(f"发送 SIGKILL 失败: {e}")
        finally:
            self._proc = None


# ----------------- 主循环 -----------------
def main():
    rclpy.init()
    key_in = KeystrokeTopicInput(
        glyph_topic="/keyboard_listener/glyphkey_pressed",
        keycode_topic="/keyboard_listener/key_pressed",
    )
    exec_ = SingleThreadedExecutor()
    exec_.add_node(key_in)

    controller = RosbagController()

    try:
        print("Rosbag 录制控制节点已启动：按 V 开始录制；按 B/N/M 停止录制；按 Q 退出。", flush=True)
        while rclpy.ok():
            exec_.spin_once(timeout_sec=0.0)
            key = key_in.get_key()
            if key is None:
                time.sleep(0.02)
                continue

            key_norm = key.lower() if len(key) == 1 else key

            if key_norm == "q":
                msg = "收到 Q，退出节点。"
                logger.info(msg)
                print(msg, flush=True)
                break

            if key_norm == "v":
                controller.start_recording()
                continue

            if key_norm in ("b", "n", "m"):
                reason_map = {"b": "Success", "n": "Failure", "m": "Recovery"}
                controller.stop_recording(reason_map[key_norm])
                continue

            # 其他按键忽略
    finally:
        try:
            exec_.remove_node(key_in)
        except Exception:
            pass
        try:
            key_in.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == "__main__":
    main()

