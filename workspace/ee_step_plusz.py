#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit.planning import MoveItPy
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
import tf2_ros
import time

def quat_to_rot(qx, qy, qz, qw):
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy+zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),   1 - 2*(xx+zz),   2*(yz - wx)],
        [2*(xz - wy),   2*(yz + wx),     1 - 2*(xx+yy)],
    ])

def main():
    rclpy.init()
    node = Node('ee_step_plusz')

    # UR robots expose both `base_link` and `base`. Prefer base_link, fallback to base.
    candidate_bases = ['base_link', 'base']
    ee_link = 'tool0'               # default UR tool frame
    planning_group = 'ur_manipulator'

    # Start TF listener; spin in a background thread AND give it a moment to receive /tf
    tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
    tf_listener = tf2_ros.TransformListener(tf_buffer, node, spin_thread=True)
    t_deadline = time.time() + 2.0
    while time.time() < t_deadline:
        rclpy.spin_once(node, timeout_sec=0.05)  # ensure callbacks are serviced

    # Pick the first base frame that is actually available
    base_frame = None
    for bf in candidate_bases:
        if tf_buffer.can_transform(bf, ee_link, Time()):
            base_frame = bf
            break
    if base_frame is None:
        node.get_logger().error("No TF from tool0 to any of ['base_link','base']. "
                                "Is robot_state_publisher/UR driver running?")
        rclpy.shutdown()
        return

    # Lookup the current EE pose
    t = tf_buffer.lookup_transform(base_frame, ee_link, Time())
    px = t.transform.translation.x
    py = t.transform.translation.y
    pz = t.transform.translation.z
    qx = t.transform.rotation.x
    qy = t.transform.rotation.y
    qz = t.transform.rotation.z
    qw = t.transform.rotation.w

    # 10 mm along the tool's +Z, expressed in world
    R = quat_to_rot(qx, qy, qz, qw)
    step_world = R @ np.array([0.0, 0.0, 0.01])

    target = PoseStamped()
    target.header.frame_id = base_frame
    target.pose.position.x = px + float(step_world[0])
    target.pose.position.y = py + float(step_world[1])
    target.pose.position.z = pz + float(step_world[2])
    target.pose.orientation.x = qx
    target.pose.orientation.y = qy
    target.pose.orientation.z = qz
    target.pose.orientation.w = qw

    # Load only what this node still needs: kinematics + planning pipelines + controllers
    ur_moveit = Path(get_package_share_directory("ur_moveit_config"))
    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur5e", package_name="ur_moveit_config")
        .robot_description_kinematics(file_path=str(ur_moveit / "config" / "kinematics.yaml"))
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"])  # use OMPL
        .trajectory_execution(file_path=str(ur_moveit / "config" / "controllers.yaml"))
        .to_moveit_configs()
    ).to_dict()

    robot = MoveItPy(node_name='moveit_py_ee_step', config_dict=moveit_config)
    arm = robot.get_planning_component(planning_group)
    arm.set_start_state_to_current_state()
    arm.set_goal_state(pose_stamped_msg=target, pose_link=ee_link)

    plan_result = arm.plan()
    if not plan_result:
        node.get_logger().error('Planning failed')
    else:
        robot.execute(plan_result.trajectory, controllers=[])
        node.get_logger().info(f'Executed +10 mm along tool Z in {base_frame}.')

    rclpy.shutdown()

if __name__ == '__main__':
    main()