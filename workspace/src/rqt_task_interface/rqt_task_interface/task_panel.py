import json
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from rqt_gui_py.plugin import Plugin
from python_qt_binding import QtWidgets, QtCore


class TaskPanel(Plugin):
    """
    RQT plugin that provides simple form inputs and publishes them at 1 Hz
    on /task_info as std_msgs/String (JSON payload).
    """

    def __init__(self, context):
        super().__init__(context)
        self.setObjectName('TaskPanel')

        # Init ROS if needed
        if not rclpy.ok():
            rclpy.init(args=None)

        self.node: Node = rclpy.create_node('rqt_task_interface')
        self.pub = self.node.create_publisher(String, '/task_info', 10)

        # Build UI
        self.widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        # Task Label
        self.task_combo = QtWidgets.QComboBox()
        self.task_combo.setEditable(True)
        self.task_combo.addItems(['Task 1', 'Task 2', 'Task 3'])
        self.task_combo.setCurrentIndex(-1)
        self.task_combo.lineEdit().setPlaceholderText('Select or type...')
        form.addRow('Task Label:', self.task_combo)

        # Operator Name
        self.operator_edit = QtWidgets.QLineEdit()
        self.operator_edit.setPlaceholderText('Enter operator name')
        form.addRow('Operator Name:', self.operator_edit)

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
        self.phantom_combo.addItems(['Abdominal Phantom', 'Lambar Phantom', 'Prok'])
        self.phantom_combo.setCurrentIndex(-1)
        self.phantom_combo.lineEdit().setPlaceholderText('Select or type...')
        form.addRow('Phantom Info:', self.phantom_combo)

        # Comments
        self.comments_edit = QtWidgets.QTextEdit()
        self.comments_edit.setPlaceholderText('Enter comments')
        self.comments_edit.setFixedHeight(80)
        form.addRow('Comments:', self.comments_edit)

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

    def publish_state(self):
        data = {
            'task_label': self.task_combo.currentText().strip(),
            'operator_name': self.operator_edit.text().strip(),
            'operator_skill_level': self.skill_combo.currentText().strip(),
            'phantom_info': self.phantom_combo.currentText().strip(),
            'comments': self.comments_edit.toPlainText().strip(),
            'timestamp_iso': datetime.utcnow().isoformat() + 'Z',
        }
        msg = String()
        msg.data = json.dumps(data, ensure_ascii=False)
        self.pub.publish(msg)

    def shutdown_plugin(self):
        self.timer.stop()
        if self.node is not None:
            self.node.destroy_publisher(self.pub)
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def save_settings(self, plugin_settings, instance_settings):
        instance_settings.set_value('task_label', self.task_combo.currentText())
        instance_settings.set_value('operator_name', self.operator_edit.text())
        instance_settings.set_value('operator_skill_level', self.skill_combo.currentText())
        instance_settings.set_value('phantom_info', self.phantom_combo.currentText())
        instance_settings.set_value('comments', self.comments_edit.toPlainText())

    def restore_settings(self, plugin_settings, instance_settings):
        self.task_combo.setCurrentText(instance_settings.value('task_label', ''))
        self.operator_edit.setText(instance_settings.value('operator_name', ''))
        self.skill_combo.setCurrentText(instance_settings.value('operator_skill_level', ''))
        self.phantom_combo.setCurrentText(instance_settings.value('phantom_info', ''))
        self.comments_edit.setPlainText(instance_settings.value('comments', ''))

