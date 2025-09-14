import rclpy
from geometry_msgs.msg import PoseStamped
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters
import numpy as np
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

    # Determine the planning frame from the PlanningScene (do not hard-code)
    with psm.read_only() as scene_ro:
        planning_frame = scene_ro.planning_frame
    print("Planning frame:", planning_frame)

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

    # --- Draw a 10 mm square in the EE's local Oxy plane (keep orientation fixed) ---
    arm.set_start_state_to_current_state()

    # Read the latest state and compute current EE transform
    with robot.get_planning_scene_monitor().read_only() as scene:
        scene.current_state.update()
        T = scene.current_state.get_global_link_transform(tip_link)  # 4x4 numpy array
        current_pose = scene.current_state.get_pose(tip_link)  # geometry_msgs/Pose

    # Local axes of the EE expressed in the planning frame
    R = T[:3, :3]
    x_axis = R[:, 0] / np.linalg.norm(R[:, 0])
    y_axis = R[:, 1] / np.linalg.norm(R[:, 1])
    p0 = T[:3, 3]

    # Define a square of edge 10 mm centered at the current EE position in its local Oxy plane
    edge = 0.2  # in m
    h = edge / 2.0
    # Waypoints (dx, dy) in the EE local frame; start at bottom-left, go CCW, return, then to center
    local_corners = [
        (-h, -h),
        ( h, -h),
        ( h,  h),
        (-h,  h),
        (-h, -h),
        ( 0.0, 0.0),  # return to the square center (original start)
    ]

    def make_pose(dx: float, dy: float) -> PoseStamped:
        """Create a PoseStamped at p0 + dx*x + dy*y with current orientation."""
        p_new = p0 + dx * x_axis + dy * y_axis
        ps = PoseStamped()
        ps.header.frame_id = planning_frame
        ps.pose.position.x = float(p_new[0])
        ps.pose.position.y = float(p_new[1])
        ps.pose.position.z = float(p_new[2])
        ps.pose.orientation = current_pose.orientation  # lock orientation
        return ps

    waypoint_poses = [make_pose(dx, dy) for (dx, dy) in local_corners]

    # Plan and execute segment-by-segment to each corner
    for i, wp in enumerate(waypoint_poses):
        arm.set_start_state_to_current_state()
        arm.set_goal_state(pose_stamped_msg=wp, pose_link=tip_link)
        plan_params = PlanRequestParameters(robot, "")
        plan_params.max_velocity_scaling_factor = 0.2
        plan_params.max_acceleration_scaling_factor = 0.2
        plan_result = arm.plan(single_plan_parameters=plan_params)
        if not plan_result:
            print(f"[Square] Planning to corner {i} failed; aborting the shape.")
            break
        try:
            robot.execute(plan_result.trajectory, controllers=["scaled_joint_trajectory_controller"])  # UR real HW
        except Exception:
            try:
                robot.execute(plan_result.trajectory)
            except Exception:
                robot.execute(plan_result.trajectory, controllers=["joint_trajectory_controller"])  # sim/mock
        print(f"[Square] Reached corner {i} / {len(waypoint_poses)-1}")

    rclpy.shutdown()

if __name__ == "__main__":
    main()