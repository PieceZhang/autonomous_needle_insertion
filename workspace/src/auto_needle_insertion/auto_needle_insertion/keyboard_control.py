"""
Keyboard teleoperation for end-effector incremental motions in its local frame
using MoveItPy. Each key press triggers a small point-to-point plan and execution
with conservative speed scaling.

This version SUBSCRIBES to RoverRobotics 'keystroke' topics:
  - /keyboard_listener/glyphkey_pressed (std_msgs/String): letters/numbers/symbols
  - /keyboard_listener/key_pressed      (std_msgs/UInt32): arrows/space/ctrl/F-keys, etc.

Controls (default):
  - Arrow Right/Left:  +X / -X  (right/left when facing the flange)
  - Arrow Up/Down:     +Y / -Y  (flange face pointing outward is +Y)
  - W / S:             +Z / -Z
  - + (or =) / -:      increase/decrease step (xy & z together)
  - R / E:             +roll / -roll   (local X axis)
  - P / O:             +pitch / -pitch (local Y axis)
  - Y / T:             +yaw / -yaw     (local Z axis)
  - I / D:             increase/decrease rotation step (degrees)
  - H:                 show help
  - Space:             no-op (useful for refresh)
  - Q:                 quit
"""

import time
import logging
import queue
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

from std_msgs.msg import String, UInt32
from geometry_msgs.msg import PoseStamped, Quaternion
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.kinematic_constraints import construct_link_constraint


def configure_run_logging(log_dir: str = "/tmp") -> str:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # local time
    logfile = str(Path(log_dir) / f"keyboard_control_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        filename=logfile,
        filemode="w",  # one fresh file per run
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logfile


logfile = configure_run_logging(log_dir="../log")
logger = logging.getLogger("auto_needle_insertion.keyboard_control")
logger.info("Logging to %s", logfile)

NODE_NAME = "auto_needle_insertion"

# Translation steps
DEFAULT_STEP_XY = 0.01  # m
DEFAULT_STEP_Z = 0.01   # m
STEP_MIN = 0.001
STEP_MAX = 0.10
STEP_SCALE_UP = 1.5
STEP_SCALE_DOWN = 1.0 / STEP_SCALE_UP

MAX_VELOCITY_SCALING = 0.1
MAX_ACCELERATION_SCALING = 0.1

# Rotation steps (degrees)
DEFAULT_ANGLE_DEG = 2.0
ANGLE_MIN_DEG = 0.2
ANGLE_MAX_DEG = 15
ANGLE_SCALE_UP = 1.5
ANGLE_SCALE_DOWN = 1.0 / ANGLE_SCALE_UP

CONTROLLER_NAMES = [
    "scaled_joint_trajectory_controller",
    "",
    "joint_trajectory_controller",
]

PREFERRED_TIP_LINKS = ["tool0", "ee_link"]


def get_planning_group_name(robot: MoveItPy) -> str:
    names = robot.get_robot_model().joint_model_group_names
    if not names:
        raise RuntimeError("No planning groups available")
    for n in names:
        if "manipulator" in n or "ur" in n:
            return n
    return names[0]


def get_tip_link_name(robot: MoveItPy, group_name: str) -> str:
    group = robot.get_robot_model().get_joint_model_group(group_name)
    if not group:
        raise RuntimeError(f"Planning group '{group_name}' not found.")
    links = list(group.link_model_names)
    if not links:
        raise RuntimeError(f"No links in planning group '{group_name}'.")
    for pref in PREFERRED_TIP_LINKS:
        if pref in links:
            return pref
    return links[-1]


def execute_trajectory_with_fallback(robot: MoveItPy, trajectory, controllers: List[str]) -> bool:
    for c in controllers:
        try:
            if c:
                robot.execute(trajectory, controllers=[c])
            else:
                robot.execute(trajectory)
            return True
        except Exception as e:
            logger.warning(f"Controller '{c}' failed: {e}")
    logger.error("All controllers failed.")
    return False


def quaternion_from_xyzw(q: tuple[float, float, float, float]) -> Quaternion:
    quat = Quaternion()
    quat.x = float(q[0])
    quat.y = float(q[1])
    quat.z = float(q[2])
    quat.w = float(q[3])
    return quat


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=float)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-16:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def axis_angle_to_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-16:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    axis = axis / n
    s = np.sin(angle_rad / 2.0)
    c = np.cos(angle_rad / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=float)


def create_pose_stamped_local_increment(
    robot: MoveItPy,
    tip_link: str,
    planning_frame: str,
    dx: float,
    dy: float,
    dz: float,
    droll_rad: float,
    dpitch_rad: float,
    dyaw_rad: float,
) -> PoseStamped:
    """
    Read current transform, apply local-frame translation (dx,dy,dz).
    Rotations are applied in local frame by composing current orientation with incremental quats:
      q_target = q_curr * (Rx(droll) * Ry(dpitch) * Rz(dyaw))
    """
    psm = robot.get_planning_scene_monitor()
    with psm.read_only() as scene:
        scene.current_state.update()
        T = scene.current_state.get_global_link_transform(tip_link)
        current_pose = scene.current_state.get_pose(tip_link)

    R = T[:3, :3]
    origin = T[:3, 3]
    x_axis = R[:, 0] / (np.linalg.norm(R[:, 0]) + 1e-12)
    y_axis = R[:, 1] / (np.linalg.norm(R[:, 1]) + 1e-12)
    z_axis = R[:, 2] / (np.linalg.norm(R[:, 2]) + 1e-12)

    target = origin + dx * x_axis + dy * y_axis + dz * z_axis

    q_curr = np.array(
        [
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w,
        ],
        dtype=float,
    )

    q_rx = axis_angle_to_quat(np.array([1, 0, 0]), droll_rad) if abs(droll_rad) > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
    q_ry = axis_angle_to_quat(np.array([0, 1, 0]), dpitch_rad) if abs(dpitch_rad) > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
    q_rz = axis_angle_to_quat(np.array([0, 0, 1]), dyaw_rad) if abs(dyaw_rad) > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])

    q_inc = quat_multiply(quat_multiply(q_rx, q_ry), q_rz)
    q_target = quat_normalize(quat_multiply(q_curr, q_inc))

    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = planning_frame
    pose_stamped.pose.position.x = float(target[0])
    pose_stamped.pose.position.y = float(target[1])
    pose_stamped.pose.position.z = float(target[2])
    pose_stamped.pose.orientation = quaternion_from_xyzw(tuple(q_target.tolist()))
    return pose_stamped


def _help_text(step_xy: float, step_z: float, angle_deg: float) -> None:
    logger.info(
        "[Teleop Help]\n"
        " Arrows: X/Y moves (Right=+X, Left=-X, Up=+Y, Down=-Y)\n"
        " W/S:    Z move (+Z/-Z)\n"
        " +(=)/-: Inc./Dec. translation (current: xy=%.3f m, z=%.3f m)\n"
        " R/P/Y:  Rotate counterclockwise (+)\n"
        " E/O/T:  Rotate clockwise (-)\n"
        " I / D:  Inc./Dec. rotation (current: %.2f deg)\n"
        " Space:  no-op\n"
        " H:      help\n"
        " Q:      quit\n"
        % (step_xy, step_z, angle_deg)
    )


class SimpleUI:
    """
    Lightweight replacement for the old curses UI.
    Keeps your write_log/write_line call sites intact.
    """

    def write_log(self, text: str) -> None:
        print(text, flush=True)

    def write_line(self, text: str) -> None:
        print(text, flush=True)


# ---------------- Topic-driven key input ----------------

KEY_UP = "<UP>"
KEY_DOWN = "<DOWN>"
KEY_LEFT = "<LEFT>"
KEY_RIGHT = "<RIGHT>"
KEY_SPACE = "<SPACE>"

# Win32 VK codes (commonly produced by pynput on Windows; also used by some cross-platform mappings):
# VK_LEFT=0x25(37), VK_UP=0x26(38), VK_RIGHT=0x27(39), VK_DOWN=0x28(40), VK_SPACE=0x20(32)
# See Microsoft Virtual-Key Codes documentation. :contentReference[oaicite:2]{index=2}
VK_MAP = {
    0x25: KEY_LEFT,
    0x26: KEY_UP,
    0x27: KEY_RIGHT,
    0x28: KEY_DOWN,
    0x20: KEY_SPACE,  # space
    0x1B: "<ESC>",    # esc (optional)
}

# Common X11 keysyms for arrows (pynput on Linux/X11 often reports these):
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
    Subscribes to:
      - /keyboard_listener/glyphkey_pressed (String)
      - /keyboard_listener/key_pressed      (UInt32)
    and exposes a non-blocking get_key() that returns either:
      - a single-character glyph (e.g., 'w', 'S', '+')
      - a token like '<UP>', '<SPACE>', or '<VK_###>' for unknown codes
    """

    def __init__(
        self,
        glyph_topic: str = "/keyboard_listener/glyphkey_pressed",
        keycode_topic: str = "/keyboard_listener/key_pressed",
        queue_depth: int = 200,
    ) -> None:
        super().__init__("ee_moveit_keyboard_key_sub")
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=queue_depth)

        self._sub_glyph = self.create_subscription(String, glyph_topic, self._on_glyph, 10)
        self._sub_code = self.create_subscription(UInt32, keycode_topic, self._on_code, 10)

    def _push(self, token: str) -> None:
        try:
            self._q.put_nowait(token)
        except queue.Full:
            # If planning/execution is slow, we may accumulate keystrokes.
            # Dropping newest is acceptable for teleop safety.
            pass

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


def teleop_loop(
    robot: MoveItPy,
    arm_group_name: str,
    tip_link: str,
    initial_step_xy: float,
    initial_step_z: float,
    initial_angle_deg: float,
):
    """
    Main teleop loop: incremental local-frame moves and rotations.
    Key input is read from ROS 2 topics via KeystrokeTopicInput.
    """
    psm = robot.get_planning_scene_monitor()
    with psm.read_only() as scene_ro:
        planning_frame = scene_ro.planning_frame

    arm = robot.get_planning_component(arm_group_name)
    step_xy = float(initial_step_xy)
    step_z = float(initial_step_z)
    angle_deg = float(initial_angle_deg)

    logger.info(f"[Teleop] Planning group: {arm_group_name}")
    logger.info(f"[Teleop] Tip link: {tip_link}")
    logger.info(f"[Teleop] Planning frame: {planning_frame}")
    logger.info(f"[Teleop] Initial step length: XY={step_xy:.3f} m, Z={step_z:.3f} m")
    _help_text(step_xy, step_z, angle_deg)

    ui = SimpleUI()

    key_in = KeystrokeTopicInput(
        glyph_topic="/keyboard_listener/glyphkey_pressed",
        keycode_topic="/keyboard_listener/key_pressed",
    )
    exec_ = SingleThreadedExecutor()
    exec_.add_node(key_in)

    try:
        while rclpy.ok():
            # Process subscription callbacks
            exec_.spin_once(timeout_sec=0.0)

            key = key_in.get_key()
            if key is None:
                time.sleep(0.02)
                continue

            # Normalize single-character glyph keys to lower-case for matching commands.
            key_norm = key.lower() if len(key) == 1 else key

            # Exit
            if key_norm == "q":
                logger.info("[Teleop] Exit")
                break

            # Help
            if key_norm == "h":
                _help_text(step_xy, step_z, angle_deg)
                continue

            # Translation step length adjustment
            if key in ("+", "="):
                step_xy = min(STEP_MAX, step_xy * STEP_SCALE_UP)
                step_z = min(STEP_MAX, step_z * STEP_SCALE_UP)
                msg = f"[Teleop] Increase step length: XY={step_xy:.3f} m, Z={step_z:.3f} m"
                logger.info(msg)
                ui.write_log(msg)
                continue

            if key == "-":
                step_xy = max(STEP_MIN, step_xy * STEP_SCALE_DOWN)
                step_z = max(STEP_MIN, step_z * STEP_SCALE_DOWN)
                msg = f"[Teleop] Decrease step length: XY={step_xy:.3f} m, Z={step_z:.3f} m"
                logger.info(msg)
                ui.write_log(msg)
                continue

            # Rotation step length adjustment
            if key_norm == "i":
                angle_deg = min(ANGLE_MAX_DEG, angle_deg * ANGLE_SCALE_UP)
                msg = f"[Teleop] Increase rotation step: Angle={angle_deg:.2f} deg"
                logger.info(msg)
                ui.write_log(msg)
                continue

            if key_norm == "d":
                angle_deg = max(ANGLE_MIN_DEG, angle_deg * ANGLE_SCALE_DOWN)
                msg = f"[Teleop] Decrease rotation step: Angle={angle_deg:.2f} deg"
                logger.info(msg)
                ui.write_log(msg)
                continue

            # Move and rotate (single DoF per key)
            dx = dy = dz = 0.0
            droll = dpitch = dyaw = 0.0

            if key == KEY_RIGHT:
                dx = +step_xy
            elif key == KEY_LEFT:
                dx = -step_xy
            elif key == KEY_UP:
                dy = +step_xy
            elif key == KEY_DOWN:
                dy = -step_xy
            elif key_norm == "w":
                dz = +step_z
            elif key_norm == "s":
                dz = -step_z
            elif key_norm == "r":
                droll = np.radians(+angle_deg)
            elif key_norm == "e":
                droll = np.radians(-angle_deg)
            elif key_norm == "p":
                dpitch = np.radians(+angle_deg)
            elif key_norm == "o":
                dpitch = np.radians(-angle_deg)
            elif key_norm == "y":
                dyaw = np.radians(+angle_deg)
            elif key_norm == "t":
                dyaw = np.radians(-angle_deg)
            elif key == KEY_SPACE:
                ui.write_line("Space: no-op")
                continue
            else:
                # Ignore other keys (including unknown <VK_...>)
                continue

            waypoint_pose = create_pose_stamped_local_increment(
                robot=robot,
                tip_link=tip_link,
                planning_frame=planning_frame,
                dx=dx,
                dy=dy,
                dz=dz,
                droll_rad=droll,
                dpitch_rad=dpitch,
                dyaw_rad=dyaw,
            )

            try:
                arm.set_start_state_to_current_state()

                pos = waypoint_pose.pose.position
                ori = waypoint_pose.pose.orientation

                goal_c = construct_link_constraint(
                    link_name=tip_link,
                    source_frame=planning_frame,
                    cartesian_position=[pos.x, pos.y, pos.z],
                    cartesian_position_tolerance=1e-4,  # meters
                    orientation=[ori.x, ori.y, ori.z, ori.w],
                    orientation_tolerance=1e-4,  # radians
                )
                arm.set_goal_state(motion_plan_constraints=[goal_c])

                plan_params = PlanRequestParameters(robot, "")
                plan_params.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
                plan_params.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING

                plan_result = arm.plan(single_plan_parameters=plan_params)

                ok = False
                if plan_result is not None:
                    traj = getattr(plan_result, "trajectory", None)
                    ok = traj is not None

                if not ok:
                    msg = "[Teleop] Planning failed (this increment)"
                    logger.warning(msg)
                    ui.write_log(msg)
                    continue

                if not execute_trajectory_with_fallback(robot, plan_result.trajectory, CONTROLLER_NAMES):
                    msg = "[Teleop] Execute failed (this increment)"
                    logger.warning(msg)
                    ui.write_log(msg)
                    continue

                # Full-detail message for file log
                full_msg = (
                    f"[Teleop] Move: "
                    f"dX={dx:.3f}, dY={dy:.3f}, dZ={dz:.3f}, "
                    f"dR={np.degrees(droll):.2f}, dP={np.degrees(dpitch):.2f}, dYaw={np.degrees(dyaw):.2f}"
                )
                logger.info(full_msg)

                # Compact live message: only the single active DoF
                eps_pos = 1e-9  # m
                eps_ang = 1e-9  # rad

                if abs(dx) > eps_pos:
                    live_part = f"dX={dx:.3f}"
                elif abs(dy) > eps_pos:
                    live_part = f"dY={dy:.3f}"
                elif abs(dz) > eps_pos:
                    live_part = f"dZ={dz:.3f}"
                elif abs(droll) > eps_ang:
                    live_part = f"dR={np.degrees(droll):.2f}"
                elif abs(dpitch) > eps_ang:
                    live_part = f"dP={np.degrees(dpitch):.2f}"
                elif abs(dyaw) > eps_ang:
                    live_part = f"dYaw={np.degrees(dyaw):.2f}"
                else:
                    live_part = "no-op"

                ui.write_log(f"Move: {live_part}")

            except Exception as e:
                msg = f"[Teleop] Planning/Execution error: {e}"
                logger.error(msg)
                ui.write_log(msg)
                continue

    finally:
        try:
            exec_.remove_node(key_in)
        except Exception:
            pass
        try:
            key_in.destroy_node()
        except Exception:
            pass


def main():
    rclpy.init()
    try:
        robot = MoveItPy(node_name=NODE_NAME)

        time.sleep(0.2)
        psm = robot.get_planning_scene_monitor()
        with psm.read_write() as scene:
            scene.current_state.update()

        arm_group_name = get_planning_group_name(robot)
        tip_link = get_tip_link_name(robot, arm_group_name)

        teleop_loop(
            robot=robot,
            arm_group_name=arm_group_name,
            tip_link=tip_link,
            initial_step_xy=DEFAULT_STEP_XY,
            initial_step_z=DEFAULT_STEP_Z,
            initial_angle_deg=DEFAULT_ANGLE_DEG,
        )

    except Exception as e:
        logger.error(f"ee_moveit_keyboard failed: {e}")
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
