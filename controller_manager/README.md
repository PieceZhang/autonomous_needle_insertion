# controller_manager (vendored)

This is a vendored copy of the `controller_manager` package from
[ros2_control](https://github.com/ros-controls/ros2_control), pinned at
**version 4.39.2**.

## Why this copy exists

The upstream `ros2_control` repository in this workspace ships
`controller_manager` **4.42.0**, which introduced a version mismatch with the
`gz_ros2_control` package. That mismatch caused the Gazebo simulation stack
(`ros_gz_sim` / `gz_ros2_control`) to fail at build time.

To unblock the build, `controller_manager` **4.39.2** was copied here so that
it is resolved before the upstream version during the `colcon` overlay build.
This gives `gz_ros2_control` (and the rest of the Franka simulation pipeline) a
compatible `controller_manager` without having to modify the upstream
`ros2_control` source.
