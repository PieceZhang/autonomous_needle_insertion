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
from typing import List, Optional, Tuple, Union
from queue import Queue, Empty
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from moveit.planning import MoveItPy, PlanRequestParameters
from pynput import keyboard

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
    psm = robot.get_planning_scene_monitor()
    # Use read_only context for thread safety
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

    pose_stamped = PoseStamped()
    pose_stamped.header.frame_id = planning_frame
    pose_stamped.pose.position.x = float(target[0])
    pose_stamped.pose.position.y = float(target[1])
    pose_stamped.pose.position.z = float(target[2])
    pose_stamped.pose.orientation = current_pose.orientation  # lock orientation
    return pose_stamped


# Commands emitted by keyboard
# ('MOVE', (dx, dy, dz)), ('QUIT', ...), ('HELP', ...), ('NOP', ...),
# ('STEP_UP', ...), ('STEP_DOWN', ...)
Command = Tuple[str, Tuple[float, float, float]]


def _map_key_to_command_pynput(
    key: Union[keyboard.Key, keyboard.KeyCode],
    step_xy: float,
    step_z: float
) -> Optional[Command]:
    """
    Map pynput key event to command tuple.
    """
    try:
        if key == keyboard.Key.right:
            return ('MOVE', ( step_xy, 0.0, 0.0))
        if key == keyboard.Key.left:
            return ('MOVE', (-step_xy, 0.0, 0.0))
        if key == keyboard.Key.up:
            return ('MOVE', (0.0,  step_xy, 0.0))
        if key == keyboard.Key.down:
            return ('MOVE', (0.0, -step_xy, 0.0))
        if key == keyboard.Key.space:
            return ('NOP', (0.0, 0.0, 0.0))
        if key == keyboard.Key.esc:
            # Not used; keep for completeness
            return None

        if isinstance(key, keyboard.KeyCode):
            ch = key.char
            if ch is None:
                return None
            if ch in ('w', 'W'):
                return ('MOVE', (0.0, 0.0,  step_z))
            if ch in ('s', 'S'):
                return ('MOVE', (0.0, 0.0, -step_z))
            if ch in ('h', 'H'):
                return ('HELP', (0.0, 0.0, 0.0))
            if ch in ('q', 'Q'):
                return ('QUIT', (0.0, 0.0, 0.0))
            if ch in ('+', '='):
                return ('STEP_UP', (0.0, 0.0, 0.0))
            if ch == '-':
                return ('STEP_DOWN', (0.0, 0.0, 0.0))
    except Exception:
        pass
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
        Uses pynput for keyboard events.
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

    # Thread-safe queue for key commands
    cmd_queue: Queue[Command] = Queue()

    quit_flag = threading.Event()

    def on_press(key):
        nonlocal step_xy, step_z
        # Map to command with current steps
        cmd = _map_key_to_command_pynput(key, step_xy, step_z)
        if cmd is None:
            return
        # Update step immediately for step events to keep UI responsive
        ctype, _ = cmd
        if ctype == 'STEP_UP':
            step_xy = min(STEP_MAX, step_xy * STEP_SCALE_UP)
            step_z = min(STEP_MAX, step_z * STEP_SCALE_UP)
            logger.info(f"[Teleop] Step increased: XY={step_xy:.3f} m, Z={step_z:.3f} m")
            return
        if ctype == 'STEP_DOWN':
            step_xy = max(STEP_MIN, step_xy * STEP_SCALE_DOWN)
            step_z = max(STEP_MIN, step_z * STEP_SCALE_DOWN)
            logger.info(f"[Teleop] Step decreased: XY={step_xy:.3f} m, Z={step_z:.3f} m")
            return

        # Enqueue other commands
        cmd_queue.put(cmd)

        if ctype == 'QUIT':
            quit_flag.set()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    logger.info("[Teleop] Keyboard listener started (pynput).")

    try:
        # Main loop: poll queue and execute one command at a time
        while rclpy.ok() and not quit_flag.is_set():
            try:
                cmd, data = cmd_queue.get(timeout=0.05)
            except Empty:
                continue

            if cmd == 'QUIT':
                logger.info("[Teleop] Quit requested.")
                break
            elif cmd == 'HELP':
                _print_help(step_xy, step_z)
                continue
            elif cmd == 'NOP':
                logger.info("[Teleop] No-op.")
                continue
            elif cmd == 'MOVE':
                dx, dy, dz = data
                waypoint_pose = create_pose_stamped_local_increment(
                    robot=robot,
                    tip_link=tip_link,
                    planning_frame=planning_frame,
                    dx=dx, dy=dy, dz=dz
                )

                try:
                    arm.set_start_state_to_current_state()
                    arm.set_goal_state(pose_stamped_msg=waypoint_pose, pose_link=tip_link)

                    plan_params = PlanRequestParameters(robot, "")
                    plan_params.max_velocity_scaling_factor = MAX_VELOCITY_SCALING
                    plan_params.max_acceleration_scaling_factor = MAX_ACCELERATION_SCALING
                    # plan_params.planning_time = 5.0  # if supported

                    plan_result = arm.plan(single_plan_parameters=plan_params)

                    ok = False
                    if plan_result is not None:
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
            else:
                # Unknown command type (should not happen)
                continue

    finally:
        try:
            listener.stop()
        except Exception:
            pass
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
        logger.error(f"ee_moveit_keyboard (pynput) failed: {e}")
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()