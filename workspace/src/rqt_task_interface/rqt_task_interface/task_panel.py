import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rqt_gui_py.plugin import Plugin
from python_qt_binding import QtWidgets, QtCore


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

        # Build UI
        self.widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        # Task Label
        self.task_combo = QtWidgets.QComboBox()
        self.task_combo.setEditable(True)
        self.task_combo.addItems(['Task 1', 'Task 2', 'Task 3', 'Task 4.1', 'Task 4.2'])
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
        self.phantom_combo.addItems(['Abdominal Phantom', 'Lambar Phantom', 'Silicone Phantom', 'Pork'])
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

        # Comments
        self.comments_edit = QtWidgets.QTextEdit()
        self.comments_edit.setPlaceholderText('Enter comments')
        self.comments_edit.setFixedHeight(80)
        form.addRow('Comments:', self.comments_edit)

        # Rosbag 状态显示
        self.status_label = QtWidgets.QLabel('Rosbag Recording Stopped')
        self.status_label.setWordWrap(True)
        form.addRow('Rosbag Status:', self.status_label)

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

    def on_collection_state(self, msg: String):
        state = msg.data.strip().lower()
        if state == 'started':
            text = 'Rosbag Recording Started'
        elif state == 'stopped_success':
            text = 'Rosbag Recording Stopped with Status: Success'
        elif state == 'stopped_failure':
            text = 'Rosbag Recording Stopped with Status: Failure'
        elif state == 'stopped_recovery':
            text = 'Rosbag Recording Stopped with Status: Recovery'
        else:
            # 未知状态保持原文，或可选择保持不变
            text = f'Rosbag Recording State: {msg.data.strip()}'
        self.status_label.setText(text)

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
            'comments': self.comments_edit.toPlainText().strip(),
            'timestamp_iso': datetime.utcnow().isoformat() + 'Z',
        }
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.pub.publish(msg)

    def shutdown_plugin(self):
        self.timer.stop()
        self.spin_timer.stop()
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
        instance_settings.set_value('comments', self.comments_edit.toPlainText())

    def restore_settings(self, plugin_settings, instance_settings):
        self.task_combo.setCurrentText(instance_settings.value('task_label', ''))
        self.operator_edit.setText(instance_settings.value('operator_name', ''))
        self.patient_edit.setText(instance_settings.value('patient_name', ''))
        self.skill_combo.setCurrentText(instance_settings.value('operator_skill_level', ''))
        self.phantom_combo.setCurrentText(instance_settings.value('phantom_info', ''))
        self.probe_type_combo.setCurrentText(instance_settings.value('probe_type', 'Wisonic_Clover60_C5-1_convex'))
        self.probe_setup_combo.setCurrentText(instance_settings.value('probe_setup', 'Robotic'))
        self.needle_setup_combo.setCurrentText(instance_settings.value('needle_setup', 'Free-hand'))
        self.needle_gauge_combo.setCurrentText(instance_settings.value('needle_gauge', ''))
        self.comments_edit.setPlainText(instance_settings.value('comments', ''))