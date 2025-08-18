#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit.planning import MoveItPy
import tf2_ros

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

    base_frame = 'base_link'
    ee_link = 'tool0'
    planning_group = 'ur_manipulator'   # sometimes 'manipulator' depending on your SRDF

    node = Node('ee_step_plusz')
    tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
    tf_listener = tf2_ros.TransformListener(tf_buffer, node)

    if not tf_buffer.can_transform(base_frame, ee_link, rclpy.time.Time(),
                                   timeout=Duration(seconds=2.0)):
        node.get_logger().error(f'Cannot transform {ee_link} -> {base_frame}')
        rclpy.shutdown()
        return

    t = tf_buffer.lookup_transform(base_frame, ee_link, rclpy.time.Time())
    px, py, pz = (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z)
    qx, qy, qz, qw = (t.transform.rotation.x, t.transform.rotation.y,
                      t.transform.rotation.z, t.transform.rotation.w)

    # 10 mm step along the tool Z, expressed in the base frame
    R = quat_to_rot(qx, qy, qz, qw)
    step_world = R @ np.array([0.0, 0.0, 0.01])

    target = PoseStamped()
    target.header.frame_id = base_frame
    target.pose.position.x = float(px + step_world[0])
    target.pose.position.y = float(py + step_world[1])
    target.pose.position.z = float(pz + step_world[2])
    target.pose.orientation.x = qx
    target.pose.orientation.y = qy
    target.pose.orientation.z = qz
    target.pose.orientation.w = qw

    robot = MoveItPy(node_name='moveit_py_ee_step')
    arm = robot.get_planning_component(planning_group)

    arm.set_start_state_to_current_state()
    arm.set_goal_state(pose_stamped_msg=target, pose_link=ee_link)

    plan_result = arm.plan()
    if not plan_result:
        node.get_logger().error('Planning failed')
    else:
        robot.execute(plan_result.trajectory, controllers=[])
        node.get_logger().info('Executed +10 mm along tool Z.')

    rclpy.shutdown()

if __name__ == '__main__':
    main()
