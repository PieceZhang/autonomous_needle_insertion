import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rqt_gui_py.plugin import Plugin
from python_qt_binding import QtWidgets, QtCore
from auto_needle_insertion.rosbag_recorder_control import RosbagController


class TaskPanel(Plugin):
    """
    RQT 插件：提供表单输入，并以 1 Hz 频率将 JSON 负载发布到 /task_info。
    同时订阅 task_info_collection_states，更新界面状态显示。
    """

    def __init__(self, context):
        super().__init__(context)
        self.setObjectName('TaskPanel')

        # Init ROS if needed
        if not rclpy.ok():
            rclpy.init(args=None)

        self.node: Node = rclpy.create_node('rqt_task_interface')
        self.pub = self.node.create_publisher(String, '/task_info', 10)
        self.sub_state = self.node.create_subscription(
            String,
            'task_info_collection_states',
            self.on_collection_state,
            10
        )
        self.rosbag_controller = RosbagController()
        self.is_recording = False
        self.pending_start_timer: QtCore.QTimer | None = None

        # Build UI
        self.widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        # Task Label
        self.task_combo = QtWidgets.QComboBox()
        self.task_combo.setEditable(True)
        self.task_combo.addItems(['Task 1',
                                  'Task 2', 'Task 2 Manual',
                                  'Task 3', 'Task 3 Manual',
                                  'Task 4.1', 'Task 4.2'])
        self.task_combo.setCurrentIndex(-1)
        self.task_combo.lineEdit().setPlaceholderText('Select or type...')
        form.addRow('Task Label:', self.task_combo)

        # Operator Name
        self.operator_edit = QtWidgets.QLineEdit()
        self.operator_edit.setPlaceholderText('Enter operator name')
        form.addRow('Operator Name:', self.operator_edit)

        # Patient Name
        self.patient_edit = QtWidgets.QLineEdit()
        self.patient_edit.setPlaceholderText('Enter patient name')
        form.addRow('Patient Name:', self.patient_edit)

        # Operator Skill Level
        self.skill_combo = QtWidgets.QComboBox()
        self.skill_combo.setEditable(True)
        self.skill_combo.addItems(['Expert', 'Trainee', 'Novice'])
        self.skill_combo.setCurrentIndex(-1)
        self.skill_combo.lineEdit().setPlaceholderText('Select or type...')
        form.addRow('Operator Skill Level:', self.skill_combo)

        # Phantom Info
        self.phantom_combo = QtWidgets.QComboBox()
        self.phantom_combo.setEditable(True)
        self.phantom_combo.addItems(['Abdominal Phantom', 'Lambar Phantom', 'Silicone Phantom', 'Pig Liver', 'Water'])
        self.phantom_combo.setCurrentIndex(-1)
        self.phantom_combo.lineEdit().setPlaceholderText('Select or type...')
        form.addRow('Phantom Info:', self.phantom_combo)

        # Probe Type
        self.probe_type_combo = QtWidgets.QComboBox()
        self.probe_type_combo.setEditable(True)
        probe_type_options = ['Wisonic_Clover60_C5-1_convex', 'Wisonic_Clover60_L15-4_linear']
        self.probe_type_combo.addItems(probe_type_options)
        self.probe_type_combo.setCurrentText('Wisonic_Clover60_C5-1_convex')
        self.probe_type_combo.lineEdit().setPlaceholderText('Select or type...')
        form.addRow('Probe Type:', self.probe_type_combo)

        # Probe Setup
        self.probe_setup_combo = QtWidgets.QComboBox()
        self.probe_setup_combo.setEditable(True)
        probe_setup_options = ['Free-hand', 'Robotic', 'Static']
        self.probe_setup_combo.addItems(probe_setup_options)
        self.probe_setup_combo.setCurrentText('Robotic')
        self.probe_setup_combo.lineEdit().setPlaceholderText('Select or type...')
        form.addRow('Probe Setup:', self.probe_setup_combo)

        # Needle Setup
        self.needle_setup_combo = QtWidgets.QComboBox()
        self.needle_setup_combo.setEditable(True)
        needle_setup_options = ['Free-hand', 'Robotic', 'Static', 'NA']
        self.needle_setup_combo.addItems(needle_setup_options)
        self.needle_setup_combo.setCurrentText('Free-hand')
        self.needle_setup_combo.lineEdit().setPlaceholderText('Select or type...')
        form.addRow('Needle Setup:', self.needle_setup_combo)

        # Needle Gauge
        self.needle_gauge_combo = QtWidgets.QComboBox()
        self.needle_gauge_combo.setEditable(True)
        needle_gauge_options = ['14G', '16G', '18G', '20G', '22G']
        self.needle_gauge_combo.addItems(needle_gauge_options)
        self.needle_gauge_combo.setCurrentText('18G')
        self.needle_gauge_combo.lineEdit().setPlaceholderText('Select or type...')
        form.addRow('Needle Gauge:', self.needle_gauge_combo)

        # Speed (NEW)
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.setEditable(True)
        speed_options = ['NA', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
        self.speed_combo.addItems(speed_options)
        self.speed_combo.setCurrentText('50')
        self.speed_combo.lineEdit().setPlaceholderText('Select or type...')
        form.addRow('Speed:', self.speed_combo)

        # Comments
        self.comments_edit = QtWidgets.QTextEdit()
        self.comments_edit.setPlaceholderText('Enter comments')
        self.comments_edit.setFixedHeight(80)
        form.addRow('Comments:', self.comments_edit)
        self.start_delay_spin = QtWidgets.QDoubleSpinBox()
        self.start_delay_spin.setRange(0.0, 60.0)
        self.start_delay_spin.setDecimals(1)
        self.start_delay_spin.setSingleStep(0.5)
        self.start_delay_spin.setSuffix(' s')
        self.start_delay_spin.setValue(0.0)
        form.addRow('Start Delay (s):', self.start_delay_spin)

        # Rosbag 状态显示
        self.status_label = QtWidgets.QLabel('Rosbag Recording Stopped')
        self.status_label.setWordWrap(True)
        form.addRow('Rosbag Status:', self.status_label)
        self.record_button = QtWidgets.QPushButton('Start')
        self.record_button.clicked.connect(self.on_record_button_clicked)
        form.addRow('Rosbag Control:', self.record_button)

        self.widget.setLayout(form)
        if context.serial_number() > 1:
            self.widget.setWindowTitle(f'Task Panel ({context.serial_number()})')
        else:
            self.widget.setWindowTitle('Task Panel')

        context.add_widget(self.widget)

        # 1 Hz timer to publish
        self.timer = QtCore.QTimer(self.widget)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.publish_state)
        self.timer.start()

        # 定期 spin rclpy 以处理订阅回调
        self.spin_timer = QtCore.QTimer(self.widget)
        self.spin_timer.setInterval(50)  # 20 Hz 足够响应
        self.spin_timer.timeout.connect(self.spin_once)
        self.spin_timer.start()

    def set_rosbag_status(self, text: str, recording_state: bool | None = None):
        self.status_label.setText(text)
        if recording_state is not None:
            self.is_recording = recording_state
        self.update_record_button()

    def on_collection_state(self, msg: String):
        state = msg.data.strip().lower()
        if state == 'started':
            text = 'Rosbag Recording Started'
            self.is_recording = True
            self.clear_pending_start_timer()
        elif state == 'stopped_success':
            text = 'Rosbag Recording Stopped with Status: Success'
            self.is_recording = False
        elif state == 'stopped_failure':
            text = 'Rosbag Recording Stopped with Status: Failure'
            self.is_recording = False
        elif state == 'stopped_recovery':
            text = 'Rosbag Recording Stopped with Status: Recovery'
            self.is_recording = False
        else:
            # 未知状态保持原文，或可选择保持不变
            text = f'Rosbag Recording State: {msg.data.strip()}'
        self.status_label.setText(text)
        self.update_record_button()

    def spin_once(self):
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception:
            # 安全忽略单次 spin 中的异常，避免阻塞 UI
            pass

    def publish_state(self):
        data = {
            'task_label': self.task_combo.currentText().strip(),
            'operator_name': self.operator_edit.text().strip(),
            'patient_name': self.patient_edit.text().strip(),
            'operator_skill_level': self.skill_combo.currentText().strip(),
            'phantom_info': self.phantom_combo.currentText().strip(),
            'probe_type': self.probe_type_combo.currentText().strip(),
            'probe_setup': self.probe_setup_combo.currentText().strip(),
            'needle_setup': self.needle_setup_combo.currentText().strip(),
            'needle_gauge': self.needle_gauge_combo.currentText().strip(),
            'speed': self.speed_combo.currentText().strip(),  # NEW
            'comments': self.comments_edit.toPlainText().strip(),
            'timestamp_iso': datetime.utcnow().isoformat() + 'Z',
        }
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.pub.publish(msg)

    def update_record_button(self):
        if self.pending_start_timer is not None:
            self.record_button.setText('Cancel Start')
        else:
            self.record_button.setText('Stop' if self.is_recording else 'Start')

    def on_record_button_clicked(self):
        if self.pending_start_timer is not None:
            self.cancel_pending_start()
            return
        if self.is_recording:
            self.set_rosbag_status('Sending rosbag stop command...', recording_state=False)
            try:
                self.rosbag_controller.stop_recording('TaskPanelButton')
            except Exception as exc:
                self.node.get_logger().error(f'Failed to stop rosbag recording: {exc}')
                self.set_rosbag_status('Rosbag stop command failed, see logs.', recording_state=True)
        else:
            delay_s = self.start_delay_spin.value()
            if delay_s > 0.0:
                self.schedule_start_recording(delay_s)
            else:
                self.send_start_command()

    def send_start_command(self):
        self.set_rosbag_status('Sending rosbag start command...', recording_state=True)
        try:
            self.rosbag_controller.start_recording()
        except Exception as exc:
            self.node.get_logger().error(f'Failed to start rosbag recording: {exc}')
            self.set_rosbag_status('Rosbag start command failed, see logs.', recording_state=False)

    def schedule_start_recording(self, delay_s: float):
        self.clear_pending_start_timer()
        self.pending_start_timer = QtCore.QTimer(self.widget)
        self.pending_start_timer.setSingleShot(True)
        self.pending_start_timer.timeout.connect(self._on_pending_start_timeout)
        self.pending_start_timer.start(int(delay_s * 1000))
        self.set_rosbag_status(f'Rosbag start scheduled in {delay_s:.1f} s...', recording_state=False)

    def _on_pending_start_timeout(self):
        timer = self.pending_start_timer
        self.pending_start_timer = None
        if timer is not None:
            timer.deleteLater()
        self.send_start_command()

    def cancel_pending_start(self, notify: bool = True):
        if self.pending_start_timer is None:
            return
        self.pending_start_timer.stop()
        self.pending_start_timer.deleteLater()
        self.pending_start_timer = None
        if notify:
            self.set_rosbag_status('Scheduled rosbag start canceled.', recording_state=False)
        else:
            self.update_record_button()

    def clear_pending_start_timer(self):
        if self.pending_start_timer is not None:
            self.pending_start_timer.stop()
            self.pending_start_timer.deleteLater()
            self.pending_start_timer = None
            self.update_record_button()

    def shutdown_plugin(self):
        self.cancel_pending_start(notify=False)
        self.timer.stop()
        self.spin_timer.stop()
        if getattr(self, 'rosbag_controller', None) is not None and self.is_recording:
            try:
                self.rosbag_controller.stop_recording('TaskPanelShutdown')
            except Exception:
                pass
        if self.node is not None:
            try:
                self.node.destroy_subscription(self.sub_state)
            except Exception:
                pass
            try:
                self.node.destroy_publisher(self.pub)
            except Exception:
                pass
            try:
                self.node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()

    def save_settings(self, plugin_settings, instance_settings):
        instance_settings.set_value('task_label', self.task_combo.currentText())
        instance_settings.set_value('operator_name', self.operator_edit.text())
        instance_settings.set_value('patient_name', self.patient_edit.text())
        instance_settings.set_value('operator_skill_level', self.skill_combo.currentText())
        instance_settings.set_value('phantom_info', self.phantom_combo.currentText())
        instance_settings.set_value('probe_type', self.probe_type_combo.currentText())
        instance_settings.set_value('probe_setup', self.probe_setup_combo.currentText())
        instance_settings.set_value('needle_setup', self.needle_setup_combo.currentText())
        instance_settings.set_value('needle_gauge', self.needle_gauge_combo.currentText())
        instance_settings.set_value('speed', self.speed_combo.currentText())  # NEW
        instance_settings.set_value('comments', self.comments_edit.toPlainText())
        instance_settings.set_value('start_delay', self.start_delay_spin.value())

    def restore_settings(self, plugin_settings, instance_settings):
        self.task_combo.setCurrentText(instance_settings.value('task_label', 'Task 4.1'))
        self.operator_edit.setText(instance_settings.value('operator_name', 'Yuelin Zhang'))
        self.patient_edit.setText(instance_settings.value('patient_name', ''))
        self.skill_combo.setCurrentText(instance_settings.value('operator_skill_level', ''))
        self.phantom_combo.setCurrentText(instance_settings.value('phantom_info', 'Pig Liver'))
        self.probe_type_combo.setCurrentText(instance_settings.value('probe_type', 'Wisonic_Clover60_C5-1_convex'))
        self.probe_setup_combo.setCurrentText(instance_settings.value('probe_setup', 'Robotic'))
        self.needle_setup_combo.setCurrentText(instance_settings.value('needle_setup', 'Static'))
        self.needle_gauge_combo.setCurrentText(instance_settings.value('needle_gauge', '18G'))
        self.speed_combo.setCurrentText(instance_settings.value('speed', '50'))  # NEW
        self.comments_edit.setPlainText(instance_settings.value('comments', ''))
        delay_value = instance_settings.value('start_delay', 0.0)
        try:
            delay_value = float(delay_value)
        except (TypeError, ValueError):
            delay_value = 0.0
        self.start_delay_spin.setValue(delay_value)

