import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Pose
from moveit.core.kinematic_constraints import construct_link_constraint
from moveit.planning import MoveItPy, PlanRequestParameters
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String

from auto_needle_insertion.utils.needle import Needle
from auto_needle_insertion.utils.us_probe import USProbe
from auto_needle_insertion.utils.optical_tracking import read_instrument_pose
from auto_needle_insertion.utils.transducer_motions import compose_transducer_motions, transducer_motions
from auto_needle_insertion.rosbag_recorder_control import (
    RosbagController,
    TaskInfoPublisher,
    sleep_with_spin,
)

# ----------------- Parameters -----------------
NODE_NAME = "task1_probe_placement"
PLANNING_SCENE_SYNC_DELAY = 0.5
MAX_VELOCITY_SCALING = 0.2
MAX_ACCELERATION_SCALING = 0.2
CONTROLLER_NAMES = ["scaled_joint_trajectory_controller", "", "joint_trajectory_controller"]
PREFERRED_TIP_LINKS = ["tool0", "ee_link"]

# Random ranges (editable):
RAND_ROT_DEG = 8.0              # rotation jitter for P2 (roll/pitch/yaw, deg)
STANDARD_ROT_Y_MAX_DEG = 6.0    # rotation motion amplitude about +/−Y
STANDARD_ROCK_Z_MAX_DEG = 6.0   # rock motion amplitude about +/−Z
STANDARD_TILT_X_MAX_DEG = 6.0   # tilt motion amplitude about +/−X

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------- Math helpers -----------------
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
    R = Needle._quat_xyzw_to_rotmat(qx, qy, qz, qw)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = [px, py, pz]
    return T


def _rotmat_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise RuntimeError(f"Rotation matrix must be (3,3), got {R.shape}")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise RuntimeError("Rotation matrix is not orthonormal")
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


def homogeneous_to_pose_msg(T: np.ndarray) -> Pose:
    _check_hmat(T, "T")
    R = T[:3, :3]
    p = T[:3, 3]
    qx, qy, qz, qw = _rotmat_to_quat_xyzw(R)
    pose = Pose()
    pose.position.x = float(p[0])
    pose.position.y = float(p[1])
    pose.position.z = float(p[2])
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def homogeneous_to_pose_stamped(T: np.ndarray, frame_id: str) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.pose = homogeneous_to_pose_msg(T)
    return ps


# ----------------- Publishers -----------------
class TaskProcedurePublisher(rclpy.node.Node):
    """Publish step changes on 'task_procedure'."""
    def __init__(self, topic_name: str = "task_procedure") -> None:
        super().__init__("task_procedure_publisher")
        self._pub = self.create_publisher(String, topic_name, 10)
        self._last: Optional[str] = None

    def publish_step(self, step: str) -> None:
        if step != self._last:
            self._last = step
            msg = String()
            msg.data = step
            self._pub.publish(msg)


# ----------------- Planning helpers -----------------
def get_planning_group_name(robot: MoveItPy) -> str:
    group_names = robot.get_robot_model().joint_model_group_names
    if not group_names:
        raise RuntimeError("No planning groups available")
    for group_name in group_names:
        if "manipulator" in group_name or "ur" in group_name:
            return group_name
    return group_names[0]


def get_tip_link_name(robot: MoveItPy, group_name: str) -> str:
    group = robot.get_robot_model().get_joint_model_group(group_name)
    if not group:
        raise RuntimeError(f"Planning group '{group_name}' not found")
    link_names = list(group.link_model_names)
    if not link_names:
        raise RuntimeError(f"No links found in planning group '{group_name}'")
    for preferred_link in PREFERRED_TIP_LINKS:
        if preferred_link in link_names:
            return preferred_link
    return link_names[-1]


def execute_trajectory_with_fallback(robot: MoveItPy, trajectory, controllers: List[str] = CONTROLLER_NAMES) -> bool:
    for controller in controllers:
        try:
            if controller:
                robot.execute(trajectory, controllers=[controller])
            else:
                robot.execute(trajectory)
            return True
        except Exception as e:
            logger.warning(f"Controller '{controller}' failed: {e}")
    logger.error("All controllers failed")
    return False


# ----------------- Data classes -----------------
@dataclass
class CaptureState:
    gt_to_in_tracker: Optional[np.ndarray] = None
    corners: List[np.ndarray] = None

    def __post_init__(self):
        if self.corners is None:
            self.corners = []


# ----------------- Core logic -----------------
class ProbePlacementTask:
    def __init__(self) -> None:
        rclpy.init()
        self.executor = SingleThreadedExecutor()
        self._spin_running = True

        self.task_info_pub = TaskInfoPublisher(topic_name="task_info_collection_states")
        self.task_proc_pub = TaskProcedurePublisher(topic_name="task_procedure")
        self.executor.add_node(self.task_info_pub)
        self.executor.add_node(self.task_proc_pub)

        self._spin_thread = threading.Thread(target=self._spin_executor, daemon=True)
        self._spin_thread.start()

        # MoveIt objects are initialized lazily after user starts URCap
        self.robot = None
        self.arm = None
        self.plan_params = None
        self.planning_frame = None
        self.tip_link = None
        self.arm_group_name = None

        # Calibration
        self.us_probe = USProbe()
        self.us_probe.load_calibrations(
            "./calibration/PlusDeviceSet_fCal_Wisonic_C5_1_NDIPolaris_2.0_20251230_SRIL.xml",
            "./calibration/hand_eye_20251231_075559.json",
        )
        if self.us_probe.to_in_probe is None or self.us_probe.to_in_ee is None:
            raise RuntimeError("US probe calibration failed.")
        self.needle = Needle()
        self.needle.load_tip_offset("./calibration/needle_1_tip_offset.json")

        self.capture_state = CaptureState()
        self.rosbag = RosbagController()

    def initialize_moveit(self) -> None:
        if self.robot is not None:
            return

        logger.info("Initializing MoveIt (requires URCap running)...")
        self.robot = MoveItPy(node_name=NODE_NAME)
        time.sleep(PLANNING_SCENE_SYNC_DELAY)

        psm = self.robot.get_planning_scene_monitor()
        with psm.read_write() as scene:
            scene.current_state.update()
        with psm.read_only() as scene_ro:
            self.planning_frame = scene_ro.planning_frame

        self.arm_group_name = get_planning_group_name(self.robot)
        self.tip_link = get_tip_link_name(self.robot, self.arm_group_name)
        self.arm = self.robot.get_planning_component(self.arm_group_name)

        logger.info(f"Planning frame: {self.planning_frame}")
        logger.info(f"Using group: {self.arm_group_name}, tip link: {self.tip_link}")

        self.plan_params = PlanRequestParameters(self.robot, "")
        self.plan_params.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
        self.plan_params.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING

    def _spin_executor(self) -> None:
        while rclpy.ok() and self._spin_running:
            try:
                self.executor.spin_once(timeout_sec=0.05)
            except Exception:
                break

    def _current_ee_transform(self) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("MoveIt is not initialized; ensure URCap is running and call initialize_moveit().")
        with self.robot.get_planning_scene_monitor().read_only() as scene:
            scene.current_state.update()
            return scene.current_state.get_global_link_transform(self.tip_link)

    def _tracker_in_base(self) -> np.ndarray:
        current_ee_transform = self._current_ee_transform()
        probe_pose = self.us_probe.report_pose(timeout_sec=2.0)
        tracker_in_base = current_ee_transform @ self.us_probe.probe_in_ee @ np.linalg.inv(quat_to_T(probe_pose))
        return tracker_in_base

    def _plan_and_execute_to_to_frame(self, to_in_tracker: np.ndarray, label: str = "") -> None:
        self.initialize_moveit()
        _check_hmat(to_in_tracker, "to_in_tracker")
        tracker_in_base = self._tracker_in_base()
        ee_target = tracker_in_base @ to_in_tracker @ np.linalg.inv(self.us_probe.to_in_ee)
        pose_goal = homogeneous_to_pose_stamped(ee_target, self.planning_frame)
        self.arm.set_start_state_to_current_state()
        pos = pose_goal.pose.position
        ori = pose_goal.pose.orientation
        goal_c = construct_link_constraint(
            link_name=self.tip_link,
            source_frame=self.planning_frame,
            cartesian_position=[pos.x, pos.y, pos.z],
            cartesian_position_tolerance=1e-4,
            orientation=[ori.x, ori.y, ori.z, ori.w],
            orientation_tolerance=1e-4,
        )
        self.arm.set_goal_state(motion_plan_constraints=[goal_c])
        plan_result = self.arm.plan(single_plan_parameters=self.plan_params)
        if not plan_result:
            raise RuntimeError(f"Planning failed for {label or 'target'}")
        if not execute_trajectory_with_fallback(self.robot, plan_result.trajectory):
            raise RuntimeError(f"Execution failed for {label or 'target'}")

    def _capture_to_in_tracker(self) -> np.ndarray:
        probe_pose = self.us_probe.report_pose(timeout_sec=2.0)
        to_in_tracker = quat_to_T(probe_pose) @ self.us_probe.to_in_probe
        _check_hmat(to_in_tracker, "to_in_tracker")
        return to_in_tracker

    def capture_gt(self) -> None:
        self.task_proc_pub.publish_step("1")
        self.capture_state.gt_to_in_tracker = self._capture_to_in_tracker()
        logger.info("Captured GT point P1 (to_in_tracker).")

    def capture_corner(self) -> None:
        if self.capture_state.gt_to_in_tracker is None:
            raise RuntimeError("Capture GT (step 1) first.")
        self.task_proc_pub.publish_step("2")
        T_corner = self._capture_to_in_tracker()
        self.capture_state.corners.append(T_corner)
        logger.info(f"Captured corner C{len(self.capture_state.corners)}.")

    def _sample_random_pose_in_area(self) -> np.ndarray:
        if self.capture_state.gt_to_in_tracker is None or len(self.capture_state.corners) < 4:
            raise RuntimeError("Need GT and 4 corners captured before sampling P2.")
        corners = self.capture_state.corners[:4]
        weights = np.random.dirichlet([1.0, 1.0, 1.0, 1.0])
        positions = np.stack([c[:3, 3] for c in corners], axis=0)
        p = weights @ positions
        R_gt = self.capture_state.gt_to_in_tracker[:3, :3]
        jitter_seq = [
            ("rx", random.uniform(-RAND_ROT_DEG, RAND_ROT_DEG)),
            ("ry", random.uniform(-RAND_ROT_DEG, RAND_ROT_DEG)),
            ("rz", random.uniform(-RAND_ROT_DEG, RAND_ROT_DEG)),
        ]
        R_delta = compose_transducer_motions(jitter_seq)[:3, :3]
        R = R_gt @ R_delta
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = p
        _check_hmat(T, "P2")
        return T

    def move_to_home_pose(self, T_home: np.ndarray) -> None:
        self.initialize_moveit()
        self.task_proc_pub.publish_step("3")
        self._plan_and_execute_to_to_frame(T_home, label="P2 home")

    def start_recording(self) -> None:
        self.task_proc_pub.publish_step("4")
        self.task_info_pub.set_state("started")
        self.rosbag.start_recording()

    def stop_recording(self) -> None:
        self.task_proc_pub.publish_step("7")
        self.task_info_pub.set_state("stopped_success")
        sleep_with_spin(self.executor, 0.2)
        self.rosbag.stop_recording("Success")

    def _standard_motion_sequence(self, max_deg: float) -> List[float]:
        amp = random.uniform(0.2 * max_deg, max_deg)
        return [0.0, +amp, -amp, 0.0]

    def perform_standard_action(self, T_home: np.ndarray) -> None:
        self.initialize_moveit()
        self.task_proc_pub.publish_step("5")
        sequences = [
            ("rotation", self._standard_motion_sequence(STANDARD_ROT_Y_MAX_DEG)),
            ("rock", self._standard_motion_sequence(STANDARD_ROCK_Z_MAX_DEG)),
            ("tilt", self._standard_motion_sequence(STANDARD_TILT_X_MAX_DEG)),
        ]
        for motion, values in sequences:
            for val in values:
                T_delta = transducer_motions(motion, val)
                T_target = T_home @ T_delta
                self._plan_and_execute_to_to_frame(T_target, label=f"{motion}:{val:.2f}")

    def move_to_gt(self) -> None:
        self.initialize_moveit()
        self.task_proc_pub.publish_step("6")
        if self.capture_state.gt_to_in_tracker is None:
            raise RuntimeError("GT not captured")
        self._plan_and_execute_to_to_frame(self.capture_state.gt_to_in_tracker, label="GT P1")

    def close(self) -> None:
        self._spin_running = False
        try:
            self.executor.shutdown()
        except Exception:
            pass
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        rclpy.shutdown()


# ----------------- CLI Loop -----------------
def main() -> None:
    print("Before continuing, PAUSE/STOP the URCap program on the UR control panel, then press Enter.", flush=True)
    input("Confirm URCap is paused/stopped and press Enter to continue...")

    task = ProbePlacementTask()
    print(
        "Task 1 probe placement: press Enter to capture GT (step1), then Enter 4x to capture corners (step2).\n"
        "Afterward, press Enter to run steps 3-7 (home->record->standard action->GT->stop). Press 'c' then Enter to exit.",
        flush=True,
    )
    try:
        # Step 1: GT capture
        input("[Step 1] Press Enter to capture GT point P1...")
        task.capture_gt()

        # Step 2: Corners
        for i in range(4):
            input(f"[Step 2] Press Enter to capture corner C{i+1}...")
            task.capture_corner()
        print("GT and 4 corners captured.", flush=True)
        input("Now START the URCap program on the UR control panel, then press Enter to continue to motions...")
        print("Repeated trials: press Enter to run steps 3-7, or 'c'+Enter to stop.", flush=True)

        while True:
            user_in = input("Press Enter to run sequence, or 'c'+Enter to stop: ")
            if user_in.strip().lower() == "c":
                break
            # Steps 3-7
            T_home = task._sample_random_pose_in_area()
            task.move_to_home_pose(T_home)
            task.start_recording()
            task.perform_standard_action(T_home)
            task.move_to_gt()
            task.stop_recording()

        print("Exiting task.", flush=True)
    except KeyboardInterrupt:
        print("Interrupted by user.", flush=True)
    finally:
        task.close()


if __name__ == "__main__":
    main()

'''
# run with:
source ./install/setup.bash 
ros2 launch auto_needle_insertion dataset.launch.py mode:=task1_probe_placement
'''

'''
prompt:
@project Write code in task1_probe_placement.py following similar structure like find_needle_static.py. 
Read key stroke from console to control the step. 
Steps: 1. record GT point P1 on skin 2. manualy set 4 corner points C1-C4 for initialization area A. This area should not be too large. 
3. reset to a random home posture P2 (6-dof) inside A, rotational dof should not be too large 4. start rosbag recording 
5. perform a defined standard action, control the end effector to rotate in place at a small angle. After performing standard action, the end effector should stop at P2. 
6. auto manipulate the probe from P2 to GT point P1. 7. stop rosbag recording 
Press 'Enter' to start from step 1 (read current point, and save the to_in_tracker position as GT point). 
Then by sequentially press 'Enter' 4 times, read 4 to_in_tracker corner points C1-C4 respectively to set A (step 2). 
After step 1&2, by pressing 'Enter' once, step 3-7 can be performed repeatly until pressing 'c' to stop All involved random range should be set at the begining of the code. 
To control rosbag recording in step 4&7, follow rosbag_recorder_control.py. 
For the standard action in step 5, following transducer_motions.py to do a sequence of motion with random range: 
'rotation' from 0 to +y to -y to 0, then ''rock' from 0 to +z to -z to 0, then 'tilt' from 0 to +x to -x to 0. 
The code should publish 2 topics: 1. task_info_collection_states, following rosbag_recorder_control.py, always end at 'Success'. 
2. task_procedure, publishing current task step, only publish message when step changed, e.g. 1, 2, 3...
'''