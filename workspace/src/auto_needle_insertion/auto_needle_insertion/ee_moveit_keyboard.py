"""
Keyboard teleoperation for end-effector incremental motions in its local frame
using MoveItPy. Orientation is kept constant; each key press triggers a small
point-to-point plan and execution with conservative speed scaling.

Controls (default):
  - Arrow Right/Left:  +X / -X  (right/left when facing the flange)
  - Arrow Up/Down:     +Y / -Y  (flange face pointing outward is +Y)
  - W / S:             +Z / -Z
  - + / -:             increase/decrease step (xy & z together)
  - H:                 show help
  - Space:             no-op (useful for refresh)
  - Q:                 quit
"""

import sys
import time
import logging
import select
from typing import List, Optional, Tuple, Union

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit.planning import MoveItPy, PlanRequestParameters

# Module constants
NODE_NAME = "auto_needle_insertion"
PLANNING_SCENE_SYNC_DELAY = 0.5  # seconds for initial joint state sync

DEFAULT_STEP_XY = 0.01  # 1 cm per key press in X/Y
DEFAULT_STEP_Z = 0.01   # 1 cm per key press in Z
STEP_MIN = 0.001        # 1 mm
STEP_MAX = 0.10         # 10 cm
STEP_SCALE_UP = 1.5
STEP_SCALE_DOWN = 1.0 / STEP_SCALE_UP

MAX_VELOCITY_SCALING = 0.2
MAX_ACCELERATION_SCALING = 0.2

# Controller fallback order (hardware/sim)
CONTROLLER_NAMES = [
    "scaled_joint_trajectory_controller",  # typical for UR hardware
    "",                                    # default
    "joint_trajectory_controller"          # sim/common
]

# Preferred tip links to choose from a planning group
PREFERRED_TIP_LINKS = ["tool0", "ee_link"]

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_planning_group_name(robot: MoveItPy) -> str:
    """
        Select planning group using heuristics.
    """
    group_names = robot.get_robot_model().joint_model_group_names
    if not group_names:
        raise RuntimeError("No planning groups available")
    logger.info(f"Available planning groups: {group_names}")
    for name in group_names:
        if "manipulator" in name or "ur" in name:
            return name
    return group_names[0]


def get_tip_link_name(robot: MoveItPy, group_name: str) -> str:
    """
        Resolve appropriate tip link for the planning group.
    """
    group = robot.get_robot_model().get_joint_model_group(group_name)
    if not group:
        raise RuntimeError(f"Planning group '{group_name}' not found.")
    link_names = list(group.link_model_names)
    if not link_names:
        raise RuntimeError(f"No links in planning group '{group_name}'.")
    for preferred in PREFERRED_TIP_LINKS:
        if preferred in link_names:
            return preferred
    return link_names[-1]


def execute_trajectory_with_fallback(
    robot: MoveItPy, 
    trajectory, 
    controllers: List[str] = CONTROLLER_NAMES
) -> bool:
    """
        Execute trajectory with controller fallback strategy.
    """
    for controller in controllers:
        try:
            if controller:
                robot.execute(trajectory, controllers=[controller])
            else:
                robot.execute(trajectory)
            return True
        except Exception as e:
            logger.warning(f"Controller '{controller}' failed: {e}")
            continue
    logger.error("All controllers failed.")
    return False


def create_pose_stamped_local_increment(
    robot: MoveItPy,
    tip_link: str,
    planning_frame: str,
    dx: float,
    dy: float,
    dz: float = 0.0
) -> PoseStamped:
    """
        Read current EE transform, apply local-frame translation (dx,dy,dz).
    """
    with robot.get_planning_scene_monitor().read_only() as scene:
        scene.current_state.update()
        T = scene.current_state.get_global_link_transform(tip_link)
        current_pose = scene.current_state.get_pose(tip_link)

    R = T[:3, :3]
    origin = T[:3, 3]
    x_axis = R[:, 0] / (np.linalg.norm(R[:, 0]) + 1e-12)
    y_axis = R[:, 1] / (np.linalg.norm(R[:, 1]) + 1e-12)
    z_axis = R[:, 2] / (np.linalg.norm(R[:, 2]) + 1e-12)

    target = origin + dx * x_axis + dy * y_axis + dz * z_axis

    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = planning_frame
    pose_stamped.pose.position.x = float(target[0])
    pose_stamped.pose.position.y = float(target[1])
    pose_stamped.pose.position.z = float(target[2])
    pose_stamped.pose.orientation = current_pose.orientation  # lock orientation
    return pose_stamped


def _getch_nonblocking() -> Optional[Union[str, bytes]]:
    """Non-blocking single keystroke reader.
    - Windows: uses msvcrt
    - POSIX: uses termios/tty/select; parses arrow escape sequences
    Returns:
      - bytes for Windows special keys (e.g., b'\xe0M' for Right)
      - str for POSIX keys (e.g., '\x1b[C' for Right)
      - None if no key available
    """
    try:
        # Windows
        import msvcrt
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b'\x00', b'\xe0'):  # special keys: next byte is code
                ch2 = msvcrt.getch()
                return ch + ch2
            return ch
        return None
    except ImportError:
        # POSIX
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            dr, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not dr:
                return None
            ch = sys.stdin.read(1)
            # Arrow keys are ESC [ A/B/C/D
            if ch == '\x1b':
                # Try to read the rest quickly
                dr, _, _ = select.select([sys.stdin], [], [], 0.01)
                if dr:
                    ch2 = sys.stdin.read(1)
                    dr, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if dr:
                        ch3 = sys.stdin.read(1)
                        return ch + ch2 + ch3
                return ch
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _map_key_to_command(
    key: Union[str, bytes],
    step_xy: float,
    step_z: float
) -> Optional[Tuple[str, Tuple[float, float, float]]]:
    """Map a key to a command.
    Returns:
      ('MOVE', (dx, dy, dz)) or ('QUIT', (0,0,0)) or ('HELP', ...) or ('NOP', ...)
      None if unrecognized
    """
    # Windows virtual keys (bytes)
    if isinstance(key, bytes):
        # Arrows: b'\xe0M'(Right) b'\xe0K'(Left) b'\xe0H'(Up) b'\xe0P'(Down)
        if key == b'\xe0M':
            return ('MOVE', ( step_xy, 0.0, 0.0))
        if key == b'\xe0K':
            return ('MOVE', (-step_xy, 0.0, 0.0))
        if key == b'\xe0H':
            return ('MOVE', (0.0,  step_xy, 0.0))
        if key == b'\xe0P':
            return ('MOVE', (0.0, -step_xy, 0.0))
        # PageUp/PageDown (optional Z) b'\xe0I' / b'\xe0Q'
        if key == b'\xe0I':
            return ('MOVE', (0.0, 0.0,  step_z))
        if key == b'\xe0Q':
            return ('MOVE', (0.0, 0.0, -step_z))
        return None

    # POSIX strings
    if key == '\x1b[C':  # Right
        return ('MOVE', ( step_xy, 0.0, 0.0))
    if key == '\x1b[D':  # Left
        return ('MOVE', (-step_xy, 0.0, 0.0))
    if key == '\x1b[A':  # Up
        return ('MOVE', (0.0,  step_xy, 0.0))
    if key == '\x1b[B':  # Down
        return ('MOVE', (0.0, -step_xy, 0.0))
    # Single chars
    if key in ('w', 'W'):
        return ('MOVE', (0.0, 0.0,  step_z))
    if key in ('s', 'S'):
        return ('MOVE', (0.0, 0.0, -step_z))
    if key in ('h', 'H'):
        return ('HELP', (0.0, 0.0, 0.0))
    if key in (' ',):
        return ('NOP', (0.0, 0.0, 0.0))
    if key in ('q', 'Q'):
        return ('QUIT', (0.0, 0.0, 0.0))
    if key in ('+', '='):
        return ('STEP_UP', (0.0, 0.0, 0.0))
    if key == '-':
        return ('STEP_DOWN', (0.0, 0.0, 0.0))
    return None


def _print_help(step_xy: float, step_z: float) -> None:
    logger.info(
        "\n[Teleop Help]\n"
        "  Arrows: X/Y moves (Right=+X, Left=-X, Up=+Y, Down=-Y)\n"
        "  W/S:    Z move (+Z/-Z)\n"
        "  + / - : Increase / Decrease step (current: xy=%.3f m, z=%.3f m)\n"
        "  Space:  no-op\n"
        "  H:      help\n"
        "  Q:      quit\n" % (step_xy, step_z)
    )


def keyboard_teleop_end_effector(
    robot: MoveItPy,
    arm_group_name: Optional[str] = None,
    tip_link: Optional[str] = None,
    initial_step_xy: float = DEFAULT_STEP_XY,
    initial_step_z: float = DEFAULT_STEP_Z
) -> None:
    """
        Main teleop loop: incremental local-frame moves keeping orientation.
    """
    psm = robot.get_planning_scene_monitor()
    with psm.read_only() as scene_ro:
        planning_frame = scene_ro.planning_frame

    if arm_group_name is None:
        arm_group_name = get_planning_group_name(robot)
    if tip_link is None:
        tip_link = get_tip_link_name(robot, arm_group_name)

    arm = robot.get_planning_component(arm_group_name)
    step_xy = float(initial_step_xy)
    step_z = float(initial_step_z)

    logger.info(f"[Teleop] Planning group: {arm_group_name}")
    logger.info(f"[Teleop] Tip link: {tip_link}")
    logger.info(f"[Teleop] Planning frame: {planning_frame}")
    logger.info(f"[Teleop] Initial step: XY={step_xy:.3f} m, Z={step_z:.3f} m")
    _print_help(step_xy, step_z)

    # Main loop
    while rclpy.ok():
        key = _getch_nonblocking()
        if key is None:
            time.sleep(0.01)
            continue

        mapped = _map_key_to_command(key, step_xy, step_z)
        if mapped is None:
            continue

        cmd, data = mapped

        if cmd == 'QUIT':
            logger.info("[Teleop] Quit requested.")
            break
        elif cmd == 'HELP':
            _print_help(step_xy, step_z)
            continue
        elif cmd == 'NOP':
            logger.info("[Teleop] No-op.")
            continue
        elif cmd == 'STEP_UP':
            step_xy = min(STEP_MAX, step_xy * STEP_SCALE_UP)
            step_z = min(STEP_MAX, step_z * STEP_SCALE_UP)
            logger.info(f"[Teleop] Step increased: XY={step_xy:.3f} m, Z={step_z:.3f} m")
            continue
        elif cmd == 'STEP_DOWN':
            step_xy = max(STEP_MIN, step_xy * STEP_SCALE_DOWN)
            step_z = max(STEP_MIN, step_z * STEP_SCALE_DOWN)
            logger.info(f"[Teleop] Step decreased: XY={step_xy:.3f} m, Z={step_z:.3f} m")
            continue

        # MOVE command
        dx, dy, dz = data

        # Compose goal pose from current EE pose + local increment
        waypoint_pose = create_pose_stamped_local_increment(
            robot=robot,
            tip_link=tip_link,
            planning_frame=planning_frame,
            dx=dx, dy=dy, dz=dz
        )

        # Plan and execute
        try:
            arm.set_start_state_to_current_state()
            arm.set_goal_state(pose_stamped_msg=waypoint_pose, pose_link=tip_link)

            plan_params = PlanRequestParameters(robot, "")
            plan_params.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
            plan_params.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING
            # If supported in your MoveItPy version:
            # plan_params.planning_time = 5.0

            plan_result = arm.plan(single_plan_parameters=plan_params)

            # Robustness check across MoveItPy versions
            ok = False
            if plan_result is None:
                ok = False
            else:
                traj = getattr(plan_result, "trajectory", None)
                ok = traj is not None

            if not ok:
                logger.warning("[Teleop] Planning failed for this step.")
                continue

            if not execute_trajectory_with_fallback(robot, plan_result.trajectory):
                logger.warning("[Teleop] Execution failed for this step.")
                continue

            logger.info(f"[Teleop] Moved: dX={dx:.3f}, dY={dy:.3f}, dZ={dz:.3f}")
        except Exception as e:
            logger.error(f"[Teleop] Error during plan/execute: {e}")

    logger.info("[Teleop] Exiting teleop loop.")


def main() -> None:
    rclpy.init()
    try:
        robot = MoveItPy(node_name=NODE_NAME)

        # Allow planning scene to sync with current joint states
        time.sleep(PLANNING_SCENE_SYNC_DELAY)

        # Force an initial state update
        psm = robot.get_planning_scene_monitor()
        with psm.read_write() as scene:
            scene.current_state.update()

        # Determine group and tip link automatically
        arm_group_name = get_planning_group_name(robot)
        tip_link = get_tip_link_name(robot, arm_group_name)

        # Enter keyboard teleoperation
        keyboard_teleop_end_effector(
            robot=robot,
            arm_group_name=arm_group_name,
            tip_link=tip_link,
            initial_step_xy=DEFAULT_STEP_XY,
            initial_step_z=DEFAULT_STEP_Z
        )

    except Exception as e:
        logger.error(f"ee_moveit_keyboard failed: {e}")
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()