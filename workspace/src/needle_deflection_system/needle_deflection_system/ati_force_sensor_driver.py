import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import socket
import struct
import threading
import time
import xml.etree.ElementTree as ET
import urllib.request


class ATIForceSensorDriver(Node):
    def __init__(self):
        super().__init__('ati_force_sensor_driver')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('sensor_ip', '192.168.56.15'),
                ('rdt_port', 49152),
                ('max_retries', 3),
                ('timeout_seconds', 2.0),
                ('queue_size', 1000),
                ('publish_rate', 100.0)
            ]
        )

        self.sensor_ip = self.get_parameter('sensor_ip').value
        self.rdt_port = self.get_parameter('rdt_port').value
        self.publish_rate = self.get_parameter('publish_rate').value

        self.publisher = self.create_publisher(
            WrenchStamped,
            'force_torque',
            self.get_parameter('queue_size').value
        )

        self.hdr = 0x1234
        self.cmd_stop = 0x0000
        self.cmd_start_realtime = 0x0002

        self.socket = None
        self.running = False
        self.receiver_thread = None

        self.cfgcpf = 100000
        self.cfgcpt = 100000

        self.force_unit = "N"
        self.torque_unit = "Nm"

        self.lock = threading.Lock()
        self.last_data = None

        self.get_logger().info(f"ATI Force Sensor Driver starting with IP: {self.sensor_ip}")

        self.initialize_sensor()

        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_data)

    def initialize_sensor(self):
        try:
            self.fetch_counts_factors()

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(("0.0.0.0", 0))
            self.socket.settimeout(1.0)

            self.running = True
            self.receiver_thread = threading.Thread(target=self.receive_data)
            self.receiver_thread.start()

            self.get_logger().info("ATI Force Sensor initialized successfully")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize sensor: {str(e)}")
            raise

    def fetch_counts_factors(self):
        try:
            url = f"http://{self.sensor_ip}/netftapi2.xml"
            with urllib.request.urlopen(url, timeout=2.0) as response:
                xml_bytes = response.read()

            root = ET.fromstring(xml_bytes)
            self.cfgcpf = int(root.findtext("cfgcpf"))
            self.cfgcpt = int(root.findtext("cfgcpt"))
            self.force_unit = root.findtext("scfgfu")
            self.torque_unit = root.findtext("scfgtu")

            self.get_logger().info(f"Force unit: {self.force_unit}, Torque unit: {self.torque_unit}")
            self.get_logger().info(f"Counts per force: {self.cfgcpf}, Counts per torque: {self.cfgcpt}")

        except Exception as e:
            self.get_logger().warning(f"Could not fetch sensor config from {str(url)}, using defaults: {str(e)}")

    def pack_rdt_request(self, cmd, sample_count=0):
        return struct.pack("!HHI", self.hdr, cmd, sample_count)

    def unpack_rdt_record(self, payload):
        if len(payload) != 36:
            raise ValueError(f"Unexpected payload length {len(payload)}")
        rdt_seq, ft_seq, status, fx, fy, fz, tx, ty, tz = struct.unpack("!IIIiiiiii", payload)
        return rdt_seq, ft_seq, status, (fx, fy, fz, tx, ty, tz)

    def receive_data(self):
        try:
            self.socket.sendto(self.pack_rdt_request(self.cmd_start_realtime),
                               (self.sensor_ip, self.rdt_port))

            self.get_logger().info("Started RDT data stream")

            while self.running:
                try:
                    data, _ = self.socket.recvfrom(4096)

                    if len(data) % 36 != 0:
                        continue

                    rec = data[-36:]
                    rdt_seq, ft_seq, status, counts = self.unpack_rdt_record(rec)

                    fx_c, fy_c, fz_c, tx_c, ty_c, tz_c = counts

                    Fx = fx_c / self.cfgcpf
                    Fy = fy_c / self.cfgcpf
                    Fz = fz_c / self.cfgcpf
                    Tx = tx_c / self.cfgcpt
                    Ty = ty_c / self.cfgcpt
                    Tz = tz_c / self.cfgcpt

                    with self.lock:
                        self.last_data = {
                            'timestamp': time.time(),
                            'rdt_seq': rdt_seq,
                            'ft_seq': ft_seq,
                            'status': status,
                            'fx': Fx,
                            'fy': Fy,
                            'fz': Fz,
                            'tx': Tx,
                            'ty': Ty,
                            'tz': Tz
                        }

                except socket.timeout:
                    continue
                except Exception as e:
                    self.get_logger().error(f"Error receiving data: {str(e)}")
                    time.sleep(0.1)

        except Exception as e:
            self.get_logger().error(f"Receiver thread error: {str(e)}")
        finally:
            if self.socket:
                try:
                    self.socket.sendto(self.pack_rdt_request(self.cmd_stop),
                                       (self.sensor_ip, self.rdt_port))
                except:
                    pass

    def publish_data(self):
        with self.lock:
            if self.last_data is None:
                return

            msg = WrenchStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "ati_sensor"

            msg.wrench.force.x = self.last_data['fx']
            msg.wrench.force.y = self.last_data['fy']
            msg.wrench.force.z = self.last_data['fz']
            msg.wrench.torque.x = self.last_data['tx']
            msg.wrench.torque.y = self.last_data['ty']
            msg.wrench.torque.z = self.last_data['tz']

            self.publisher.publish(msg)

    def destroy_node(self):
        self.running = False

        if self.receiver_thread and self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=2.0)

        if self.socket:
            self.socket.close()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ATIForceSensorDriver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()