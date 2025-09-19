#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class URScriptMover(Node):
    def __init__(self, dz=0.01, speed=0.05, accel=0.5):
        super().__init__('urscript_relative_move')
        # UR driver exposes this topic by default
        self.pub = self.create_publisher(String, '/urscript_interface/script_command', 10)
        self.dz = dz       # meters (0.01 = 10 mm)
        self.speed = speed # m/s for moveL
        self.accel = accel # m/s^2 for moveL

    def send(self):
        # Move 10 mm along the TCP's local +Z axis from the current pose
        script = f'''
def ee_offset_move():
  pose_current = get_actual_tcp_pose()
  pose_target  = pose_trans(pose_current, p[0.0, 0.0, {self.dz:.6f}, 0.0, 0.0, 0.0])
  movel(pose_target, a={self.accel:.3f}, v={self.speed:.3f}, r=0.0)
end
ee_offset_move()
'''
        self.get_logger().info('Publishing URScript to /urscript_interface/script_command')
        msg = String(); msg.data = script
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = URScriptMover(dz=0.02)
    node.send()
    # Give the middleware a short moment to actually send the message
    rclpy.spin_once(node, timeout_sec=0.25)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()