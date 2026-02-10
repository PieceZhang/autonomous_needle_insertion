import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, PoseStamped
import numpy as np
# import matplotlib.pyplot as plt
from collections import deque
import threading
import time


class NeedleDeflectionModel:
    def __init__(self, calibration_factor=1.0, needle_length=0.2,
                 needle_diameter=0.00127, E=200e9, G=80e9, mu=0.28):
        self.needle_length = needle_length
        self.needle_diameter = needle_diameter
        self.E = E
        self.G = G
        self.mu = mu
        self.A = np.pi * (needle_diameter / 2) ** 2
        self.I = np.pi * needle_diameter ** 4 / 64
        self.J = 2 * self.I
        self.calibration_factor = calibration_factor

        self.H1 = np.array([
            [12.0, -6.0, 0.0, 0.0],
            [-6.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 12.0, 6.0],
            [0.0, 0.0, 6.0, 4.0]
        ], dtype=float)

        self.H2 = np.array([
            [-3.0 / 5, 1.0 / 20, 0.0, 0.0],
            [1.0 / 20, -1.0 / 15, 0.0, 0.0],
            [0.0, 0.0, -3.0 / 5, -1.0 / 20],
            [0.0, 0.0, -1.0 / 20, -1.0 / 15]
        ], dtype=float)

        self.H3 = np.array([
            [0.0, 0.0, 0.0, -1.0 / 2],
            [0.0, 0.0, -1.0 / 2, -1.0 / 4],
            [0.0, -1.0 / 2, 0., 0.0],
            [-1.0 / 2, -1.0 / 4, 0.0, 0.0]
        ], dtype=float)

        self.H4 = np.array([
            [1.0 / 700, -1.0 / 1400, 0.0, 0.0],
            [-1.0 / 1400, 11.0 / 6300, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 700, 1.0 / 1400],
            [0.0, 0.0, 1.0 / 1400, 11.0 / 6300]
        ], dtype=float)

        self.H5 = np.array([
            [0.0, 0.0, 0.0, 1.0 / 60],
            [0.0, 0.0, 1.0 / 60, 0.0],
            [0.0, 1.0 / 60, 0.0, 0.0],
            [1.0 / 60, 0.0, 0.0, 0.0]
        ], dtype=float)

        self.H6 = np.array([
            [1.0 / 5, -1.0 / 10, 0.0, 0.0],
            [-1.0 / 10, 1.0 / 20, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 5, 1.0 / 10],
            [0.0, 0.0, 1.0 / 10, 1.0 / 20]
        ], dtype=float)

        self.H7 = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ], dtype=float)

    def set_calibration_factor(self, factor):
        self.calibration_factor = factor

    def normalize_parameters(self, F_x, F_y, F_z, M_x, M_y, M_z, L, n=1):
        f_x = F_x * L ** 2 / (n ** 2 * self.E * self.I)
        f_y = F_y * L ** 2 / (n ** 2 * self.E * self.I)
        f_z = F_z * L ** 2 / (n ** 2 * self.E * self.I)
        m_x = M_x * L / (n * self.E * self.I)
        m_y = M_y * L / (n * self.E * self.I)
        m_z = M_z * L / (n * self.E * self.I)
        return f_x, f_y, f_z, m_x, m_y, m_z

    def denormalize_displacement(self, u_y, theta_z, u_z, theta_y, L, n=1):
        U_y = u_y * L / n
        U_z = u_z * L / n
        Theta_y = theta_y
        Theta_z = theta_z
        return U_y, Theta_z, U_z, Theta_y

    def calculate_residual(self, v, f_x, f_y, f_z, m_x, m_y, m_z):
        u_y, theta_z, u_z, theta_y = v
        m_xd = m_x + theta_z * m_y + theta_y * m_z
        g = np.array([f_y, m_z, f_z, m_y])
        term1 = self.H1 @ v
        term2 = (2 * f_x * self.H2 + m_xd * (2 * self.H3 + self.H7)) @ v
        term3 = (f_x ** 2 * self.H4 + m_xd * f_z * self.H5 + m_xd ** 2 * self.H6) @ v
        return g - (term1 - term2 - term3)

    def calculate_jacobian(self, v, f_x, f_y, f_z, m_x, m_y, m_z):
        u_y, theta_z, u_z, theta_y = v
        m_xd = m_x + theta_z * m_y + theta_y * m_z
        J = -self.H1.copy()
        J += (2 * f_x * self.H2 + m_xd * (2 * self.H3 + self.H7))
        J += (f_x ** 2 * self.H4 + m_xd * f_z * self.H5 + m_xd ** 2 * self.H6)
        dm_xd_dv = np.array([0, m_y, 0, m_z])
        term_mxd_linear = (2 * self.H3 + self.H7) @ v
        term_mxd_quadratic = (f_z * self.H5 + 2 * m_xd * self.H6) @ v
        J += np.outer(term_mxd_linear, dm_xd_dv)
        J += np.outer(term_mxd_quadratic, dm_xd_dv)
        return J

    def calculate_tip_deflection_newton(self, F_x, F_y, F_z, M_x, M_y, M_z, L):
        try:
            if (abs(F_x) > 100 or abs(F_y) > 100 or abs(F_z) > 100 or
                    abs(M_x) > 10 or abs(M_y) > 10 or abs(M_z) > 10):
                return 0, 0, 0, 0, 0

            f_x, f_y, f_z, m_x, m_y, m_z = self.normalize_parameters(F_x, F_y, F_z, M_x, M_y, M_z, L)

            try:
                v_linear = np.linalg.solve(self.H1, np.array([f_y, m_z, f_z, m_y]))
                v = v_linear * 0.5
            except Exception:
                v = np.array([0.001, 0.001, 0.001, 0.001])

            max_iterations = 20
            tolerance = 1e-8
            for _ in range(max_iterations):
                F_v = self.calculate_residual(v, f_x, f_y, f_z, m_x, m_y, m_z)
                J = self.calculate_jacobian(v, f_x, f_y, f_z, m_x, m_y, m_z)
                if np.linalg.norm(F_v) < tolerance:
                    break

                try:
                    delta_v = np.linalg.solve(J, -F_v)
                except np.linalg.LinAlgError:
                    delta_v = np.linalg.lstsq(J, -F_v, rcond=None)[0]

                v_old = v.copy()
                v = v + delta_v

                if np.linalg.norm(delta_v) > 10.0:
                    v = v_old + 0.5 * delta_v

                if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                    v = np.array([0.0001, 0.0001, 0.0001, 0.0001])

            u_y, theta_z, u_z, theta_y = v
            U_y, Theta_z, U_z, Theta_y = self.denormalize_displacement(u_y, theta_z, u_z, theta_y, L)
            U_x = (F_x * L) / (self.E * self.A)

            U_x = U_x * self.calibration_factor
            U_y = U_y * self.calibration_factor
            U_z = U_z * self.calibration_factor
            Theta_y = Theta_y * self.calibration_factor
            Theta_z = Theta_z * self.calibration_factor

            return U_x, U_y, U_z, Theta_y, Theta_z

        except Exception as e:
            return 0, 0, 0, 0, 0


class NeedleDeflectionCalculator(Node):
    def __init__(self):
        super().__init__('needle_deflection_calculator')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('calibration_factor', 0.69),
                ('needle_length', 0.2),
                ('needle_diameter', 0.00127),
                ('youngs_modulus', 200e9),
                ('shear_modulus', 80e9),
                ('poisson_ratio', 0.28),
                ('filter_window', 5),
                ('history_seconds', 10.0),
                ('update_period', 0.05),
                ('enable_visualization', True),
                ('publish_rate', 100.0)
            ]
        )

        self.calibration_factor = self.get_parameter('calibration_factor').value
        self.needle_length = self.get_parameter('needle_length').value
        self.filter_window = self.get_parameter('filter_window').value
        self.history_seconds = self.get_parameter('history_seconds').value
        self.update_period = self.get_parameter('update_period').value
        self.enable_visualization = self.get_parameter('enable_visualization').value

        self.get_logger().info(
            f"Parameters -> calibration_factor={self.calibration_factor}, needle_length={self.needle_length}m, "
            f"filter_window={self.filter_window}, history={self.history_seconds}s, update_period={self.update_period}s"
        )

        self.model = NeedleDeflectionModel(
            calibration_factor=self.calibration_factor,
            needle_length=self.needle_length,
            needle_diameter=self.get_parameter('needle_diameter').value,
            E=self.get_parameter('youngs_modulus').value,
            G=self.get_parameter('shear_modulus').value,
            mu=self.get_parameter('poisson_ratio').value
        )

        self.subscription = self.create_subscription(
            WrenchStamped,
            'force_torque',
            self.force_callback,
            10
        )

        self.publisher = self.create_publisher(
            PoseStamped,
            'needle_deflection',
            10
        )

        self.ts = []
        self.ux = []
        self.uy = []
        self.uz = []
        self.thy = []
        self.thz = []
        self.utotal = []
        self.utotal_filtered = []

        self.force_buffer = deque(maxlen=self.filter_window)
        self.moment_buffer = deque(maxlen=self.filter_window)

        self.start_time = time.time()
        self.last_update = 0

        self.lock = threading.Lock()

        # if self.enable_visualization:
        #     self.init_visualization()
        #     self.viz_thread = threading.Thread(target=self.visualization_loop)
        #     self.viz_thread.start()
        if self.enable_visualization:
            self.get_logger().warning(
                "Visualization requested but disabled because plotting code is commented out."
            )

        self.get_logger().info(
            f"Needle Deflection Calculator started with calibration factor: {self.calibration_factor}"
        )

    def apply_filter(self, force_data):
        self.force_buffer.append(force_data[:3])
        self.moment_buffer.append(force_data[3:])

        if len(self.force_buffer) == 0:
            return force_data

        if len(self.force_buffer) < self.filter_window:
            self.get_logger().debug(
                f"Filter warm-up: {len(self.force_buffer)}/{self.filter_window} samples"
            )

        filtered_force = np.mean(self.force_buffer, axis=0)
        filtered_moment = np.mean(self.moment_buffer, axis=0)

        return np.concatenate([filtered_force, filtered_moment])

    def force_callback(self, msg):
        current_time = time.time()

        if current_time - self.last_update < self.update_period:
            self.get_logger().debug("Update skipped due to rate limit.")
            return

        self.last_update = current_time

        Fx_s, Fy_s, Fz_s = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        Mx_s, My_s, Mz_s = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z

        force_data = np.array([Fx_s, Fy_s, Fz_s, Mx_s, My_s, Mz_s])
        filtered_data = self.apply_filter(force_data)

        Fx_s_filt, Fy_s_filt, Fz_s_filt, Mx_s_filt, My_s_filt, Mz_s_filt = filtered_data

        Fx_m = Fz_s_filt
        Fy_m = Fx_s_filt
        Fz_m = Fy_s_filt
        Mx_m = Mz_s_filt
        My_m = Mx_s_filt
        Mz_m = My_s_filt

        Ux, Uy, Uz, Thy, Thz = self.model.calculate_tip_deflection_newton(
            Fx_m, Fy_m, Fz_m, Mx_m, My_m, Mz_m, self.needle_length
        )

        Uy_mm = Uy * 1000
        Uz_mm = Uz * 1000
        U_total = np.sqrt(Uy_mm ** 2 + Uz_mm ** 2)

        t_rel = current_time - self.start_time

        with self.lock:
            self.ts.append(t_rel)
            self.ux.append(Ux * 1000)
            self.uy.append(Uy_mm)
            self.uz.append(Uz_mm)
            self.thy.append(Thy)
            self.thz.append(Thz)
            self.utotal.append(U_total)

            if len(self.utotal) >= self.filter_window:
                filtered_utotal = np.mean(self.utotal[-self.filter_window:])
                self.utotal_filtered.append(filtered_utotal)

            while self.ts and (self.ts[-1] - self.ts[0] > self.history_seconds):
                self.ts.pop(0)
                self.ux.pop(0)
                self.uy.pop(0)
                self.uz.pop(0)
                self.thy.pop(0)
                self.thz.pop(0)
                self.utotal.pop(0)
                if self.utotal_filtered:
                    self.utotal_filtered.pop(0)

        deflection_msg = PoseStamped()
        deflection_msg.header.stamp = self.get_clock().now().to_msg()
        deflection_msg.header.frame_id = "needle_tip"

        deflection_msg.pose.position.x = Ux * 1000
        deflection_msg.pose.position.y = Uy_mm
        deflection_msg.pose.position.z = Uz_mm

        qx, qy, qz, qw = self._angles_to_quaternion(Thy, Thz)
        deflection_msg.pose.orientation.x = qx
        deflection_msg.pose.orientation.y = qy
        deflection_msg.pose.orientation.z = qz
        deflection_msg.pose.orientation.w = qw

        self.publisher.publish(deflection_msg)
        self.get_logger().info(
            f"[topic: needle_deflection] t={t_rel:.2f}s -> "
            f"Ux={Ux * 1000:.3f}mm, Uy={Uy_mm:.3f}mm, Uz={Uz_mm:.3f}mm, "
            f"θy={Thy:.6f}rad, θz={Thz:.6f}rad, U_total={U_total:.3f}mm"
        )

    def init_visualization(self):
        self.get_logger().debug("Visualization disabled; init_visualization skipped.")
        # plt.ion()
        # self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        # self.fig.suptitle(f"Needle Deflection Calculator [Calibration Factor: {self.calibration_factor}]")
        #
        # (self.l_ux,) = self.ax1.plot([], [], label="Ux (mm)", color='blue', linewidth=1.5)
        # (self.l_uy,) = self.ax1.plot([], [], label="Uy (mm)", color='green', linewidth=1.5)
        # (self.l_uz,) = self.ax1.plot([], [], label="Uz (mm)", color='red', linewidth=1.5)
        # (self.l_utotal,) = self.ax1.plot([], [], label="U_total (mm)", color='purple', linewidth=2, linestyle='--')
        # self.ax1.set_ylabel("Displacement (mm)")
        # self.ax1.set_title("Displacement Components")
        # self.ax1.grid(True, alpha=0.3)
        # self.ax1.legend(loc="upper left", fontsize=9)
        #
        # (self.l_thy,) = self.ax2.plot([], [], label="θy (rad)", color='orange', linewidth=1.5)
        # (self.l_thz,) = self.ax2.plot([], [], label="θz (rad)", color='brown', linewidth=1.5)
        # self.ax2.set_ylabel("Angle (rad)")
        # self.ax2.set_title("Angular Displacement")
        # self.ax2.grid(True, alpha=0.3)
        # self.ax2.legend(loc="upper left", fontsize=9)
        #
        # (self.l_utotal_alone,) = self.ax3.plot([], [], label=f"U_total (CF={self.calibration_factor})",
        #                                        color='darkviolet', linewidth=2.5)
        # (self.l_utotal_filtered_line,) = self.ax3.plot([], [], label=f"Filtered (window={self.filter_window})",
        #                                                color='cyan', linewidth=1.5, alpha=0.7)
        # self.ax3.set_xlabel("Time (s)")
        # self.ax3.set_ylabel("Total Displacement (mm)")
        # self.ax3.set_title("Total Needle Deflection")
        # self.ax3.grid(True, alpha=0.3)
        # self.ax3.legend(loc="upper left", fontsize=10)
        #
        # self.force_lines = []
        # colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        # labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        # for i in range(6):
        #     line, = self.ax4.plot([], [], label=labels[i], color=colors[i], linewidth=1.2)
        #     self.force_lines.append(line)
        # self.ax4.set_xlabel("Time (s)")
        # self.ax4.set_ylabel("Force (N) / Torque (Nm)")
        # self.ax4.set_title("Raw Sensor Data")
        # self.ax4.grid(True, alpha=0.3)
        # self.ax4.legend(loc="upper left", fontsize=8, ncol=2)
        #
        # plt.tight_layout()

    def update_visualization(self):
        self.get_logger().debug("Visualization disabled; update_visualization skipped.")
        # with self.lock:
        #     if not self.ts:
        #         return
        #
        #     self.l_ux.set_data(self.ts, self.ux)
        #     self.l_uy.set_data(self.ts, self.uy)
        #     self.l_uz.set_data(self.ts, self.uz)
        #     self.l_utotal.set_data(self.ts, self.utotal)
        #     self.ax1.relim()
        #     self.ax1.autoscale_view()
        #
        #     self.l_thy.set_data(self.ts, self.thy)
        #     self.l_thz.set_data(self.ts, self.thz)
        #     self.ax2.relim()
        #     self.ax2.autoscale_view()
        #
        #     self.l_utotal_alone.set_data(self.ts, self.utotal)
        #     if self.utotal_filtered and len(self.utotal_filtered) == len(self.ts):
        #         self.l_utotal_filtered_line.set_data(self.ts, self.utotal_filtered)
        #     self.ax3.relim()
        #     self.ax3.autoscale_view()
        #
        #     if self.ts and self.utotal:
        #         current_utotal = self.utotal[-1]
        #         if self.utotal_filtered:
        #             current_filtered = self.utotal_filtered[-1]
        #             self.ax3.set_title(
        #                 f"Total Needle Deflection [CF={self.calibration_factor}]\nRaw: {current_utotal:.3f}mm, Filtered: {current_filtered:.3f}mm")
        #         else:
        #             self.ax3.set_title(
        #                 f"Total Needle Deflection [CF={self.calibration_factor}]\nCurrent: {current_utotal:.3f} mm")
        #
        # plt.pause(0.001)

    def visualization_loop(self):
        self.get_logger().debug("Visualization disabled; visualization_loop skipped.")
        # while rclpy.ok() and self.enable_visualization:
        #     self.update_visualization()
        #     time.sleep(0.05)

    def destroy_node(self):
        # if hasattr(self, 'viz_thread') and self.viz_thread.is_alive():
        #     plt.close('all')
        self.get_logger().info("Needle Deflection Calculator shutting down.")
        super().destroy_node()

    def _angles_to_quaternion(self, theta_y, theta_z):
        half_y = theta_y * 0.5
        half_z = theta_z * 0.5
        cy, sy = np.cos(half_z), np.sin(half_z)
        cp, sp = np.cos(half_y), np.sin(half_y)
        return (-sp * sy, sp * cy, cp * sy, cp * cy)


def main(args=None):
    rclpy.init(args=args)
    node = NeedleDeflectionCalculator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

