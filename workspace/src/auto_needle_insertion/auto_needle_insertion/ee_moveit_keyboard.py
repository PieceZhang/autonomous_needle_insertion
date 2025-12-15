"""
Keyboard teleoperation for end-effector incremental motions in its local frame
using MoveItPy. Eeach key press triggers a small point-to-point plan and execution 
with conservative speed scaling.

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

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import curses
import tty
import termios
import fcntl

import numpy as np
import rclpy
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
logger = logging.getLogger("auto_needle_insertion.ee_moveit_keyboard")
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


def quaternion_from_xyzw(
    q: tuple[float, float, float, float]
) -> Quaternion:
    """
        Create Quaternion message from (x,y,z,w) tuple.
    """
    quat = Quaternion()
    quat.x = float(q[0])
    quat.y = float(q[1])
    quat.z = float(q[2])
    quat.w = float(q[3])
    return quat


def quat_multiply(
    q1: np.ndarray, 
    q2: np.ndarray
) -> np.ndarray:
    """
        Multiply two quaternions: q = q1 * q2
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=float)


def quat_normalize(
    q: np.ndarray
) -> np.ndarray:
    """
        Normalize quaternion to unit length.
    """
    n = np.linalg.norm(q)
    if n < 1e-16:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def axis_angle_to_quat(
    axis: np.ndarray, 
    angle_rad: float
) -> np.ndarray:
    """
        Convert axis-angle to quaternion.
    """
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-16:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    axis = axis / n
    s = np.sin(angle_rad / 2.0)
    c = np.cos(angle_rad / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=float)


# -------------------- Debug related -------------------------- #
def quat_to_rotmat(
    q: np.ndarray
) -> np.ndarray:
    """
        Construct a rotation matrix from quaterion
    """
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [2*(w*w+xx)-1, 2*(xy-wz),    2*(xz+wy)],
        [2*(xy+wz),   2*(w*w+yy)-1,  2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),     2*(w*w+zz)-1]
    ])
    return R

def euler_x_to_rotmat(
    theta: np.ndarray
) -> np.ndarray:
    """
        Construct a rotation matrix from euler angle (x)
    """
    R = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return R
# -------------------- Debug related -------------------------- #


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

    # Current orientation -> [x, y, z, w]
    q_curr = np.array(
        [current_pose.orientation.x, current_pose.orientation.y,
         current_pose.orientation.z, current_pose.orientation.w],
        dtype=float
    )

    q_rx = axis_angle_to_quat(np.array([1,0,0]), droll_rad) if abs(droll_rad) > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
    q_ry = axis_angle_to_quat(np.array([0,1,0]), dpitch_rad) if abs(dpitch_rad) > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
    q_rz = axis_angle_to_quat(np.array([0,0,1]), dyaw_rad) if abs(dyaw_rad) > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])

    q_inc = quat_multiply(quat_multiply(q_rx, q_ry), q_rz)
    q_target = quat_normalize(quat_multiply(q_curr, q_inc))

    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = planning_frame
    pose_stamped.pose.position.x = float(target[0])
    pose_stamped.pose.position.y = float(target[1])
    pose_stamped.pose.position.z = float(target[2])
    # pose_stamped.pose.orientation = current_pose.orientation  # lock orientation
    pose_stamped.pose.orientation = quaternion_from_xyzw(tuple(q_target.tolist()))
    return pose_stamped


def _help_text(
    step_xy: float, 
    step_z: float, 
    angle_deg: float,
) -> None:
    logger.info(
        "[Teleop Help]\n"
        " Arrows: X/Y moves (Right=+X, Left=-X, Up=+Y, Down=-Y)\n"
        " W/S:    Z move (+Z/-Z)\n"
        " +(=)/-: Inc./Dec. translation (current: xy=%.3f m, z=%.3f m)\n"
        " R/P/Y:  Rotate counterclockwise (+)\n"
        " E/O/T:  Rotate clockwise (-)"
        " I / D:  Inc./Dec. rotation (current: %.2f deg)"
        " Space:  no-op\n"
        " H:      help\n"
        " Q:      quit\n" % (step_xy, step_z, angle_deg)
    )

class TTYInput:
    """
        Capture key from /dev/tty. Using curses to enter raw mode without blocking reads 
        and restore terminal state upon destruction.
    """
    def __init__(self):
        # Open /dev/tty. If failed, back to stdin
        self.tty_path = "/dev/tty"
        self.tty_file = None
        self.fd = None
        self._curses_screen = None
        self._orig_fl = None  # non-blocking flag 
        self._init_ok = False

        try:
            self.tty_file = open(self.tty_path, "rb+", buffering=0)
            self.fd = self.tty_file.fileno()
        except Exception as e:
            logger.warning(f"Cannot open {self.tty_path}: {e}, try to use stdin.")
            self.fd = sys.stdin.fileno()

        self._saved_stdin = os.dup(0)
        self._saved_stdout = os.dup(1)
        os.dup2(self.fd, 0)
        os.dup2(self.fd, 1)

        try:
            self._orig_fl = fcntl.fcntl(self.fd, fcntl.F_GETFL)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, self._orig_fl | os.O_NONBLOCK)

            self._curses_screen = curses.initscr()
            lines, cols = self._curses_screen.getmaxyx()
            lines = max(lines, 2)  # need at least 2 rows: log + status
            self._log_win = curses.newwin(lines - 1, cols, 0, 0)
            self._log_win.scrollok(True)
            self._status_win = curses.newwin(1, cols, lines - 1, 0)

            curses.noecho()
            curses.cbreak()
            self._curses_screen.keypad(True)
            self._curses_screen.nodelay(True)  # 非阻塞 getch
            self._init_ok = True
        except Exception as e:
            logger.error(f"Initialize TTY/curses: {e} failed")
            self.close()
            raise

    def get_key(self) -> Optional[int]:
        """
            Non-blocking read of a key code; returns None if no key is pressed.
        """
        if not self._init_ok:
            return None
        try:
            ch = self._curses_screen.getch()
            if ch == -1:
                return None
            return ch
        except Exception:
            return None

    def write_log(self, text: str):
        """
            Write a status line at the botton of the terminal screen
        """
        if not self._init_ok:
            return
        try:
            _, cols = self._log_win.getmaxyx()
            self._log_win.addnstr(text, cols - 1)
            self._log_win.addstr("\n")
            self._log_win.refresh()
        except Exception:
            pass

    def write_line(self, text: str):
        """
            Write a status line at the botton of the terminal screen
        """
        if not self._init_ok:
            return
        try:
            _, cols = self._status_win.getmaxyx()
            self._status_win.erase()
            self._status_win.addnstr(0, 0, text, cols - 1)
            self._status_win.refresh()
        except Exception:
            pass

    def close(self):
        # recover curses
        try:
            if self._curses_screen is not None:
                self._curses_screen.keypad(False)
                curses.nocbreak()
                curses.echo()
                curses.endwin()
        except Exception:
            pass

        try:
            if self.fd is not None and self._orig_fl is not None:
                fcntl.fcntl(self.fd, fcntl.F_SETFL, self._orig_fl)
        except Exception:
            pass

        try:
            os.dup2(self._saved_stdin, 0)
            os.dup2(self._saved_stdout, 1)
            os.close(self._saved_stdin)
            os.close(self._saved_stdout)
        except Exception:
            pass

        # close /dev/tty file
        try:
            if self.tty_file is not None:
                self.tty_file.close()
        except Exception:
            pass

        self._init_ok = False


# Key mapping
# ---------- transaltion ------------ # 
KEY_UP = curses.KEY_UP
KEY_DOWN = curses.KEY_DOWN
KEY_LEFT = curses.KEY_LEFT
KEY_RIGHT = curses.KEY_RIGHT
ORD_w = ord('w')
ORD_W = ord('W')
ORD_s = ord('s')
ORD_S = ord('S')

# ------------ quit & help ----------- #
ORD_q = ord('q')
ORD_Q = ord('Q')
ORD_h = ord('h')
ORD_H = ord('H')
ORD_SPACE = ord(' ')

# ------------ Adjust translation step length ------------ #
ORD_PLUS = ord('+')
ORD_EQ = ord('=')
ORD_MINUS = ord('-')

# ------------- Orientation ------------ #
ORD_roll = ord('r')     # x-axis  counterclockwise (+)
ORD_ROLL = ord('R')
ORD_pitch = ord('p')    # y-axis  counterclockwise (+)
ORD_PITCH = ord('P')
ORD_yaw = ord('y')      # z-axis  counterclockwise  (+)
ORD_YAW = ord('Y') 
ORD_n_roll = ord('e')   # x-axis  clockwise (-)
ORD_N_ROLL = ord('E')
ORD_n_pitch = ord('o')  # x-axis  clockwise (-)
ORD_N_PITCH = ord('O')
ORD_n_yaw = ord('t')    # x-axis  clockwise (-)
ORD_N_YAW = ord('T')

# --------------- Adjust rotation step length -------- #
ORD_ORI_PLUS = ord('I')
ORD_ori_plus = ord('i')
ORD_ORI_DEC = ord('D')
ORD_ori_dec = ord('d')


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

    tty_in = TTYInput()
    # tty_in.write_line("\nTeleop: Arrows/W/S--move, +/- set step llength, H-help, Q-Exit\n")

    try:
        while rclpy.ok():
            ch = tty_in.get_key()
            if ch is None:
                time.sleep(0.02)
                continue

            # Exit
            if ch in (ORD_q, ORD_Q):
                logger.info("\n [Teleop] Exit\n")
                break

            # Help
            if ch in (ORD_h, ORD_H):
                _help_text(step_xy, step_z, angle_deg)
                # tty_in.write_line("\n Show the help\n")
                continue

            # Translation step length adjustment
            if ch in (ORD_PLUS, ORD_EQ):
                step_xy = min(STEP_MAX, step_xy * STEP_SCALE_UP)
                step_z = min(STEP_MAX, step_z * STEP_SCALE_UP)
                msg = f"\n [Teleop] increase step length: XY={step_xy:.3f} m, Z={step_z:.3f} m\n"
                logger.info(msg)
                tty_in.write_log(msg)
                continue

            if ch == ORD_MINUS:
                step_xy = max(STEP_MIN, step_xy * STEP_SCALE_DOWN)
                step_z = max(STEP_MIN, step_z * STEP_SCALE_DOWN)
                msg = f"\n [Teleop] decrease step length: XY={step_xy:.3f} m, Z={step_z:.3f} m\n"
                logger.info(msg)
                tty_in.write_log(msg)
                continue

            # Rotation step length adjustment
            if ch in (ORD_ORI_PLUS, ORD_ori_plus):
                angle_deg = min(ANGLE_MAX_DEG, angle_deg * ANGLE_SCALE_UP)
                msg = f"[Teleop] Increase rotation step: Angle={angle_deg:.2f} deg"
                logger.info(msg)
                tty_in.write_log(msg)
                continue

            if ch in (ORD_ORI_DEC, ORD_ori_dec):
                angle_deg = max(ANGLE_MIN_DEG, angle_deg * ANGLE_SCALE_DOWN)
                msg = f"[Teleop] Decrease rotation step: Angle={angle_deg:.2f} deg"
                logger.info(msg)
                tty_in.write_log(msg)
                continue

            # Move and rotate
            dx = dy = dz = 0.0
            droll = dpitch = dyaw = 0.0

            if ch == KEY_RIGHT:
                dx = +step_xy
            elif ch == KEY_LEFT:
                dx = -step_xy
            elif ch == KEY_UP:
                dy = +step_xy
            elif ch == KEY_DOWN:
                dy = -step_xy
            elif ch in (ORD_w, ORD_W):
                dz = +step_z
            elif ch in (ORD_s, ORD_S):
                dz = -step_z
            elif ch in (ORD_roll, ORD_ROLL):
                droll = np.radians(+angle_deg)
            elif ch in (ORD_n_roll, ORD_N_ROLL):
                droll = np.radians(-angle_deg)
            elif ch in (ORD_pitch, ORD_PITCH):
                dpitch = np.radians(+angle_deg)
            elif ch in (ORD_n_pitch, ORD_N_PITCH):
                dpitch = np.radians(-angle_deg)
            elif ch in (ORD_yaw, ORD_YAW):
                dyaw = np.radians(+angle_deg)  
            elif ch in (ORD_n_yaw, ORD_N_YAW):
                dyaw = np.radians(-angle_deg)
            elif ch == ORD_SPACE:
                tty_in.write_line("\n Space: No-operation\n")
                continue
            else:
                # ignore other keys
                continue

            waypoint_pose = create_pose_stamped_local_increment(
                robot=robot,
                tip_link=tip_link,
                planning_frame=planning_frame,
                dx=dx, dy=dy, dz=dz,
                droll_rad=droll, dpitch_rad=dpitch, dyaw_rad=dyaw,
            )

            try:
                arm.set_start_state_to_current_state()
                pos = waypoint_pose.pose.position
                ori = waypoint_pose.pose.orientation
                goal_c = construct_link_constraint(
                    link_name=tip_link,
                    source_frame=planning_frame,
                    cartesian_position=[pos.x, pos.y, pos.z],
                    cartesian_position_tolerance=1e-4,  # meters (start here)
                    orientation=[ori.x, ori.y, ori.z, ori.w],
                    orientation_tolerance=1e-4,  # radians (~0.057°) (start here)
                )
                arm.set_goal_state(motion_plan_constraints=[goal_c])
                # arm.set_goal_state(pose_stamped_msg=waypoint_pose, pose_link=tip_link)

                plan_params = PlanRequestParameters(robot, "")
                plan_params.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
                plan_params.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING

                plan_result = arm.plan(single_plan_parameters=plan_params)

                ok = False
                if plan_result is not None:
                    traj = getattr(plan_result, "trajectory", None)
                    ok = traj is not None

                if not ok:
                    msg = "\n [Teleop] Planning failed (this increment)\n"
                    logger.warning(msg)
                    tty_in.write_log(msg)
                    continue

                if not execute_trajectory_with_fallback(robot, plan_result.trajectory, CONTROLLER_NAMES):
                    msg = "\n [Teleop] Excute failed (this increment)\n"
                    logger.warning(msg)
                    tty_in.write_log(msg)
                    continue

                msg = f"[Teleop] Move: dX={dx:.3f}, dY={dy:.3f}, dZ={dz:.3f}, dR={np.degrees(droll):.2f}, dP={np.degrees(dpitch):.2f}, dY={np.degrees(dyaw):.2f}"
                logger.info(msg)
                tty_in.write_log(msg)
            except Exception as e:
                msg = f"\n [Teleop] Planning/Execution error: {e}\n"
                logger.error(msg)
                tty_in.write_log(msg)
                continue
    finally:
        tty_in.close()


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