import rclpy
from geometry_msgs.msg import PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
import time

def main():
    rclpy.init()
    node_name = "ani_moveit_py"
    robot = MoveItPy(node_name=node_name)

    # Allow time for /joint_states to arrive and populate the planning scene
    time.sleep(0.5)
    psm = robot.get_planning_scene_monitor()
    # Ensure the planning scene reflects the latest robot state
    with psm.read_write() as scene:
        scene.current_state.update()

    # --- discover planning groups (so we don't guess the name) ---
    group_names = robot.get_robot_model().joint_model_group_names
    print("Available planning groups:", group_names)
    # Prefer a UR arm group; fall back to first group if needed
    arm_group_name = next((g for g in group_names if "manipulator" in g or "ur" in g), group_names[0])
    arm = robot.get_planning_component(arm_group_name)
    # Determine a valid tip link for the chosen group (UR uses 'tool0' or 'ee_link')
    group = robot.get_robot_model().get_joint_model_group(arm_group_name)
    link_names = list(group.link_model_names) if group else []
    candidate_links = ["tool0", "ee_link"]
    tip_link = next((l for l in candidate_links if l in link_names), link_names[-1])
    print("Using planning group:", arm_group_name)
    print("Using tip link:", tip_link)

    # --- Plan to a simple joint-space goal ---
    # arm.set_start_state_to_current_state()
    # with robot.get_planning_scene_monitor().read_only() as scene:
    #     rs = RobotState(robot.get_robot_model())
    #     rs.set_to_default_values()
    #     current = scene.current_state.get_joint_group_positions(arm_group_name)
    #     if current is not None and len(current) > 0:
    #         current = list(current)  # ensure mutability
    #         current[0] += 0.2
    #         rs.set_joint_group_positions(arm_group_name, current)
    # arm.set_goal_state(robot_state=rs)
    # plan_result = arm.plan()
    # if plan_result:
    #     robot.execute(plan_result.trajectory, controllers=["joint_trajectory_controller"])

    # --- Plan to a pose goal (adjust frame & link if needed) ---
    arm.set_start_state_to_current_state()
    pose = PoseStamped()
    pose.header.frame_id = "base_link"   # typical base frame; change if your config differs
    pose.pose.orientation.w = 0.5
    pose.pose.position.x = 0.3
    pose.pose.position.y = 0.3
    pose.pose.position.z = 0.3
    arm.set_goal_state(pose_stamped_msg=pose, pose_link=tip_link)  # UR end-effector link
    plan_result = arm.plan()
    if plan_result:
        robot.execute(plan_result.trajectory, controllers=["joint_trajectory_controller"])

    rclpy.shutdown()

if __name__ == "__main__":
    main()