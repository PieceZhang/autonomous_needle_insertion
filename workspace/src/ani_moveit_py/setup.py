from setuptools import setup, find_packages
from glob import glob
import os

package_name = "ani_moveit_py"

setup(
    name=package_name,
    version="0.1",
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
    description="Robot control for autonomous needle insertion",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ee_moveit_square = ani_moveit_py.ee_moveit_square:main",
            "tool_reporter = ani_moveit_py.tool_reporter:main",
            "tool_follower = ani_moveit_py.tool_follower:main",
        ],
    },
)
