import json
import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit.core.kinematic_constraints import construct_link_constraint
from moveit.planning import MoveItPy, PlanRequestParameters
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String

from auto_needle_insertion.utils.us_probe import USProbe
from auto_needle_insertion.utils.optical_tracking import read_instrument_pose
from auto_needle_insertion.utils.transducer_motions import apply_random_small_perturbation
from auto_needle_insertion.rosbag_recorder_control import (
    RosbagController,
    TaskInfoPublisher,
    KeystrokeTopicInput,
)

# ----------------- Publishers -----------------
class TaskProcedurePublisher(rclpy.node.Node):
    """Publish step changes on 'task_procedure'."""
    def __init__(self, topic_name: str = "task_procedure") -> None:
        super().__init__("task2_refine_task_procedure_pub")
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
        super().__init__("task2_refine_params_publisher")
        self._pub = self.create_publisher(String, topic_name, 10)
        self._payload: dict = payload or {}
        self._timer = self.create_timer(1.0 / hz, self._timer_cb)

    def update(self, key: str, value) -> None:
        self._payload[key] = value

    def remove(self, key: str) -> None:
        self._payload.pop(key, None)

    def _timer_cb(self) -> None:
        base_payload = {
            "TASK_NAME": "task2robot_exe_points_refine",
            "TARGET_CHOICES": TARGET_CHOICES,
            "PERTURB_ROT_DEG": PERTURB_ROT_DEG,
            "PERTURB_SWEEP_MM": PERTURB_SWEEP_MM,
            "PERTURB_SLIDE_MM": PERTURB_SLIDE_MM,
            "MAX_VELOCITY_SCALING": MAX_VELOCITY_SCALING,
            "MAX_ACCELERATION_SCALING": MAX_ACCELERATION_SCALING,
            "CONTROLLER_NAMES": CONTROLLER_NAMES,
            "PREFERRED_TIP_LINKS": PREFERRED_TIP_LINKS,
            "DELAY_START_ROSBAG_S": DELAY_START_ROSBAG_S,
            "DELAY_STOP_ROSBAG_S": DELAY_STOP_ROSBAG_S,
        }
        payload = {**base_payload, **self._payload}
        msg = String()
        msg.data = json.dumps(payload)
        self._pub.publish(msg)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Parameters -----------------
NODE_NAME = "task2_exe_points_refine"
PLANNING_SCENE_SYNC_DELAY = 0.5
MAX_VELOCITY_SCALING = 0.2
MAX_ACCELERATION_SCALING = 0.1
PLANNING_TIME = 10.0
PLANNING_ATTEMPTS = 5
GOAL_POSITION_TOLERANCE = 5e-4
GOAL_ORIENTATION_TOLERANCE = 5e-3
CONTROLLER_NAMES = ["scaled_joint_trajectory_controller", "joint_trajectory_controller", ""]
PREFERRED_TIP_LINKS = ["tool0", "ee_link"]
MAXIMUM_TRACKER_LOST = 5
DEFAULT_TARGET_IDS = ["1", "2", "3"]

def _resolve_target_choices() -> List[str]:
    env_val = os.getenv("TARGET_P", "").strip()
    if not env_val:
        return DEFAULT_TARGET_IDS.copy()
    choices = [c.strip() for c in env_val.split(",") if c.strip()]
    return choices or DEFAULT_TARGET_IDS.copy()

TARGET_CHOICES = _resolve_target_choices()

# Perturbation ranges (deg, mm) similar to task4
PERTURB_ROT_DEG = (-8.0, 8.0)
PERTURB_SWEEP_MM = (-10.0, 10.0)
PERTURB_SLIDE_MM = (-10.0, 10.0)
DELAY_START_ROSBAG_S = 1.5
DELAY_STOP_ROSBAG_S = 1.0

# ----------------- Helpers -----------------
def _controller_order_from_env() -> List[str]:
    env_val = os.getenv("TASK2_CONTROLLER_ORDER", "").strip()
    if not env_val:
        return CONTROLLER_NAMES
    order = [c.strip() for c in env_val.split(",") if c.strip()]
    return order or CONTROLLER_NAMES


def _auto_continue_enabled() -> bool:
    auto_env = os.getenv("TASK2_AUTO_CONTINUE", "").strip().lower()
    return auto_env in ("1", "true", "yes")


def _is_enter_key(token: Optional[str]) -> bool:
    if token is None:
        return False
    t = token.strip().lower()
    return t in ("", "enter", "return", "\n", "\r", "<enter>", "<return>", "<vk_13>", "65293", "<vk_65293>")


def _is_cancel_key(token: Optional[str]) -> bool:
    if token is None:
        return False
    return token.strip().lower() == "c"


def _check_hmat(T: np.ndarray, name: str = "T") -> None:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise RuntimeError(f"{name} must be (4,4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise RuntimeError(f"{name} contains NaN/Inf")
    if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise RuntimeError(f"{name} last row must be [0 0 0 1], got {T[3, :]}")


def quat_to_T(quat: Tuple[float, float, float, float, float, float, float]) -> np.ndarray:
    px, py, pz, qx, qy, qz, qw = quat
    vals = np.array([px, py, pz, qx, qy, qz, qw], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError(f"Pose contains NaN/Inf: {vals.tolist()}")
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )
    T[:3, 3] = [px, py, pz]
    return T


def _rotmat_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    R = np.asarray(R, dtype=float)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= math.sqrt(float(np.dot(q, q)))
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def homogeneous_to_pose_msg(T: np.ndarray) -> PoseStamped:
    _check_hmat(T, "T")
    R = T[:3, :3]
    p = T[:3, 3]
    qx, qy, qz, qw = _rotmat_to_quat_xyzw(R)
    ps = PoseStamped()
    ps.header.frame_id = ""
    ps.pose.position.x = float(p[0])
    ps.pose.position.y = float(p[1])
    ps.pose.position.z = float(p[2])
    ps.pose.orientation.x = qx
    ps.pose.orientation.y = qy
    ps.pose.orientation.z = qz
    ps.pose.orientation.w = qw
    return ps


# ----------------- Data classes -----------------
TargetSet = Dict[str, Optional[np.ndarray]]


# ----------------- Core logic -----------------
class ExecPointsRefine:
    def __init__(self, target_name: str, targets: TargetSet) -> None:
        rclpy.init()
        self.executor = SingleThreadedExecutor()
        self._spin_running = True

        self.task_info_pub = TaskInfoPublisher(topic_name="task_info_collection_states")
        self.task_proc_pub = TaskProcedurePublisher(topic_name="task_procedure")
        self.params_pub = TaskInfoParamsPublisher(topic_name="/task_info", hz=1.0)
        self.params_pub.update("TARGET", target_name)

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

        self.target_name = target_name
        self.targets = targets

        self.robot = None
        self.arm = None
        self.plan_params = None
        self.planning_frame = None
        self.tip_link = None
        self.arm_group_name = None
        self.controller_order = list(dict.fromkeys(_controller_order_from_env()))

        self.us_probe = USProbe()
        calib_root = Path(__file__).resolve().parents[3] / "calibration"
        xml_path = calib_root / "PlusDeviceSet_fCal_Wisonic_C5_1_NDIPolaris_2.0_20260111_SRIL.xml"
        hand_eye_path = calib_root / "hand_eye_20251231_075559.json"
        self.us_probe.load_calibrations(xml_path, hand_eye_path)
        if self.us_probe.to_in_probe is None or self.us_probe.to_in_ee is None:
            raise RuntimeError("US probe calibration failed (need to_in_probe and to_in_ee).")

        self.rosbag = RosbagController()
        self._last_probe_pose: Optional[Tuple[float, ...]] = None
        self._last_probe_pose_time: Optional[float] = None
        self._tracker_in_base_cached: Optional[np.ndarray] = None

    # --------------- ROS2 spin ---------------
    def _spin_executor(self) -> None:
        while rclpy.ok() and self._spin_running:
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except Exception:
                break

    # --------------- MoveIt init ---------------
    def initialize_moveit(self) -> None:
        if self.robot is not None:
            return
        print("Initializing MoveIt (requires URCap running)...", flush=True)
        self.robot = MoveItPy(node_name=NODE_NAME)
        time.sleep(PLANNING_SCENE_SYNC_DELAY)
        psm = self.robot.get_planning_scene_monitor()
        with psm.read_write() as scene:
            scene.current_state.update()
        with psm.read_only() as scene_ro:
            self.planning_frame = scene_ro.planning_frame
        self.arm_group_name = self._get_planning_group_name()
        self.tip_link = self._get_tip_link_name()
        self.arm = self.robot.get_planning_component(self.arm_group_name)
        self.plan_params = PlanRequestParameters(self.robot, "")
        self.plan_params.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
        self.plan_params.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING
        self.plan_params.planning_time = PLANNING_TIME
        self.plan_params.planning_attempts = PLANNING_ATTEMPTS
        print(f"Planning frame: {self.planning_frame}", flush=True)
        print(f"Using group: {self.arm_group_name}, tip link: {self.tip_link}", flush=True)

    def _get_planning_group_name(self) -> str:
        group_names = self.robot.get_robot_model().joint_model_group_names
        if not group_names:
            raise RuntimeError("No planning groups available")
        for g in group_names:
            if "manipulator" in g or "ur" in g:
                return g
        return group_names[0]

    def _get_tip_link_name(self) -> str:
        group = self.robot.get_robot_model().get_joint_model_group(self.arm_group_name)
        link_names = list(group.link_model_names)
        for preferred in PREFERRED_TIP_LINKS:
            if preferred in link_names:
                return preferred
        return link_names[-1]

    # --------------- Tracker helpers ---------------
    def _validate_probe_pose(self, pose: Tuple[float, ...]) -> Tuple[float, ...]:
        arr = np.asarray(pose, dtype=float).reshape(-1)
        if arr.size != 7 or not np.all(np.isfinite(arr)):
            raise RuntimeError(f"Tracker pose invalid: {arr.tolist()}")
        return pose

    def _safe_probe_pose(self, timeout_sec: float = 2.0) -> Tuple[float, ...]:
        now = time.monotonic()
        try:
            pose = self.us_probe.report_pose(timeout_sec=timeout_sec)
            pose = self._validate_probe_pose(pose)
            self._last_probe_pose = pose
            self._last_probe_pose_time = now
            return pose
        except Exception as exc:
            if (
                self._last_probe_pose is not None
                and self._last_probe_pose_time is not None
                and (now - self._last_probe_pose_time) <= MAXIMUM_TRACKER_LOST
            ):
                print("Tracker pose unavailable; using cached pose", flush=True)
                return self._last_probe_pose
            raise RuntimeError("Tracker pose unavailable and cache expired") from exc

    def _capture_to_in_tracker(self) -> np.ndarray:
        probe_pose = self._safe_probe_pose(timeout_sec=2.0)
        to_in_tracker = quat_to_T(probe_pose) @ self.us_probe.to_in_probe
        _check_hmat(to_in_tracker, "to_in_tracker")
        return to_in_tracker

    def _tracker_in_base(self) -> np.ndarray:
        if self._tracker_in_base_cached is None:
            current_ee_transform = self._current_ee_transform()
            probe_pose = self._safe_probe_pose(timeout_sec=2.0)
            self._tracker_in_base_cached = current_ee_transform @ self.us_probe.probe_in_ee @ np.linalg.inv(quat_to_T(probe_pose))
            _check_hmat(self._tracker_in_base_cached, "tracker_in_base_cached")
            print("Computed tracker_in_base (cached).", flush=True)
        return self._tracker_in_base_cached

    def _current_ee_transform(self) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("MoveIt not initialized")
        with self.robot.get_planning_scene_monitor().read_only() as scene:
            scene.current_state.update()
            return scene.current_state.get_global_link_transform(self.tip_link)

    # --------------- Planning/execution ---------------
    def _plan_and_execute_to_to_frame(self, to_in_tracker: np.ndarray, label: str = "") -> None:
        self.initialize_moveit()
        _check_hmat(to_in_tracker, "to_in_tracker")
        tracker_in_base = self._tracker_in_base()
        ee_target = tracker_in_base @ to_in_tracker @ np.linalg.inv(self.us_probe.to_in_ee)
        pose_goal = homogeneous_to_pose_msg(ee_target)
        pose_goal.header.frame_id = self.planning_frame
        self.arm.set_start_state_to_current_state()
        pos = pose_goal.pose.position
        ori = pose_goal.pose.orientation
        print(
            f"Planning for {label or 'target'} | pos=({pos.x:.4f},{pos.y:.4f},{pos.z:.4f}) "
            f"ori=({ori.x:.4f},{ori.y:.4f},{ori.z:.4f},{ori.w:.4f})",
            flush=True,
        )
        goal_c = construct_link_constraint(
            link_name=self.tip_link,
            source_frame=self.planning_frame,
            cartesian_position=[pos.x, pos.y, pos.z],
            cartesian_position_tolerance=GOAL_POSITION_TOLERANCE,
            orientation=[ori.x, ori.y, ori.z, ori.w],
            orientation_tolerance=GOAL_ORIENTATION_TOLERANCE,
        )
        self.arm.set_goal_state(motion_plan_constraints=[goal_c])
        plan_result = self.arm.plan(single_plan_parameters=self.plan_params)
        if not plan_result:
            raise RuntimeError(f"Planning failed for {label or 'target'}")
        if not self._execute_trajectory(plan_result.trajectory):
            raise RuntimeError(f"Execution failed for {label or 'target'}")

    def _execute_trajectory(self, trajectory) -> bool:
        print(f"Attempting execution with controllers: {self.controller_order}", flush=True)
        for controller in self.controller_order:
            if controller is None:
                continue
            try:
                print(f"Sending trajectory to controller '{controller or 'default'}'", flush=True)
                if controller:
                    self.robot.execute(trajectory, controllers=[controller])
                else:
                    self.robot.execute(trajectory)
                print(f"Controller '{controller or 'default'}' accepted trajectory", flush=True)
                return True
            except Exception as e:
                logger.warning(f"Controller '{controller or 'default'}' failed: {e}")
        logger.error("All controllers failed")
        return False

    # --------------- Perturbation ---------------
    def _apply_perturbation(self, to_in_tracker: np.ndarray) -> List[np.ndarray]:
        seq, _ = apply_random_small_perturbation(
            to_in_tracker,
            rot_range_deg=PERTURB_ROT_DEG,
            sweep_range_mm=PERTURB_SWEEP_MM,
            slide_range_mm=PERTURB_SLIDE_MM,
            rng=np.random.default_rng(),
        )
        return seq or [to_in_tracker]

    # --------------- Loop steps ---------------
    def loop_once(self, target_to_in_tracker: np.ndarray) -> bool:
        if _cancel_requested(self):
            return False

        # Step 3: move to target
        self.task_proc_pub.publish_step("3")
        print(f"Moving to target {self.target_name} (from recorded JSON)", flush=True)
        self._plan_and_execute_to_to_frame(target_to_in_tracker, label=self.target_name)

        # Step 4: apply perturbation sequence
        self.task_proc_pub.publish_step("4")
        perturb_seq = self._apply_perturbation(target_to_in_tracker)
        if len(perturb_seq) <= 1:
            print("Perturbation sequence trivial; skipping motion.", flush=True)
        else:
            print(f"Executing perturbation sequence of length {len(perturb_seq)}", flush=True)
            for idx, pose in enumerate(perturb_seq[1:], start=1):
                if _cancel_requested(self):
                    return False
                self._plan_and_execute_to_to_frame(pose, label=f"perturb_{idx}")

        # Step 5: start rosbag
        self.task_proc_pub.publish_step("5")
        print("Starting rosbag recording.", flush=True)
        self.task_info_pub.set_state("started")
        recording_started = False
        try:
            label = f"Task 2 Robot motion P{self.target_name}"
            self.params_pub.update("task_label_FORCE", label)
            self.rosbag.start_recording()
            recording_started = True
            time.sleep(DELAY_START_ROSBAG_S)
            # Step 6: move back to target
            self.task_proc_pub.publish_step("6")
            print(f"Returning to target {self.target_name}", flush=True)
            self._plan_and_execute_to_to_frame(target_to_in_tracker, label=f"return_{self.target_name}")

            # Step 7: stop rosbag
            self.task_proc_pub.publish_step("7")
            print("Stopping rosbag recording.", flush=True)
            self.task_info_pub.set_state("stopped_success")
            self.rosbag.stop_recording("Success")
            time.sleep(DELAY_STOP_ROSBAG_S)
            return True
        finally:
            if recording_started:
                self.params_pub.remove("task_label_FORCE")

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


# ----------------- Key wait helpers -----------------
def _wait_for_enter(task: ExecPointsRefine, prompt: str, allow_cancel: bool = True) -> bool:
    print(prompt, flush=True)
    if _auto_continue_enabled():
        return True
    while rclpy.ok():
        token = task.key_input.get_key()
        if _is_enter_key(token):
            return True
        if allow_cancel and _is_cancel_key(token):
            return False
        time.sleep(0.05)
    return False


def _cancel_requested(task: ExecPointsRefine) -> bool:
    while True:
        token = task.key_input.get_key()
        if token is None:
            return False
        if _is_cancel_key(token):
            return True


# ----------------- File loading -----------------
def _load_latest_targets(base_dir: Path) -> TargetSet:
    candidates = sorted(base_dir.glob("task2_pose_*.json"))
    if not candidates:
        raise FileNotFoundError("No task2_pose_*.json files found")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    data = json.loads(latest.read_text(encoding="utf-8"))
    pts = data.get("points", {}) if isinstance(data, dict) else {}

    def to_mat(key: str) -> Optional[np.ndarray]:
        val = pts.get(key)
        if val is None:
            return None
        arr = np.asarray(val, dtype=float)
        if arr.shape != (4, 4):
            raise RuntimeError(f"Point {key} is not 4x4 in {latest}")
        _check_hmat(arr, key)
        return arr

    print(f"Loaded targets from {latest}", flush=True)
    return dict(P1=to_mat("P1"), P2=to_mat("P2"), P3=to_mat("P3"))


# ----------------- CLI -----------------
def _get_target_from_env() -> str:
    if not TARGET_CHOICES:
        raise RuntimeError("No TARGET_CHOICES resolved; set TARGET_P or use defaults 1,2,3")
    env_val = os.getenv("TARGET_P", "").strip()
    if not env_val or "," in env_val:
        return TARGET_CHOICES[0]
    if env_val not in TARGET_CHOICES:
        raise RuntimeError(f"TARGET_P must be one of {TARGET_CHOICES}, got '{env_val}'")
    return env_val


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    targets = _load_latest_targets(base_dir)
    target_name = _get_target_from_env()
    target_mat = targets.get('P' + target_name)
    if target_mat is None:
        raise RuntimeError(f"Target {target_name} not found in latest JSON")

    task = ExecPointsRefine(target_name, targets)
    try:
        print("Looping steps 3-7; press 'c' to cancel at any prompt.", flush=True)
        while rclpy.ok():
            if _cancel_requested(task):
                break
            try:
                if not task.loop_once(target_mat):
                    break
                print("Completed one cycle.", flush=True)
            except Exception as exc:
                print(f"Error during cycle: {exc}", flush=True)
                break
        print("Exiting task2 refine loop.", flush=True)
    finally:
        task.close()


if __name__ == "__main__":
    main()

