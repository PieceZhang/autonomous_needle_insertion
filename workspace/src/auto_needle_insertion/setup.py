from setuptools import setup, find_packages
from glob import glob
import os

package_name = "auto_needle_insertion"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [os.path.join("resource", package_name)]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
        ("share/" + package_name + "/config", glob(os.path.join("config", "*.yaml"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer='Quinn Ding',
    maintainer_email='qp.ding@link.cuhk.edu.hk',
    description="Robot arm and needle control for ultrasound-guided autonomous needle insertion",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "keyboard_control = auto_needle_insertion.keyboard_control:main",
            "ee_moveit_keyboard = auto_needle_insertion.ee_moveit_keyboard:main",
            "ee_moveit_square = auto_needle_insertion.ee_moveit_square:main",
            "ee_pose_logger = auto_needle_insertion.ee_pose_logger:main",
            "hand_eye_calib = auto_needle_insertion.hand_eye_calib:main",
            "tool_reporter = auto_needle_insertion.tool_reporter:main",
            "tool_follower = auto_needle_insertion.tool_follower:main",
        ],
    },
)
