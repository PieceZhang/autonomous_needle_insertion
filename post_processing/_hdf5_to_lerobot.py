#!/usr/bin/env python
"""
A script to convert robotics data from HDF5 files into the LeRobot format (v2.1)
with an efficient MP4 video backend.

This script processes a directory of HDF5 files, where each file represents a
single episode. It extracts observations, actions, and state information, and
packages them into a LeRobotDataset with visual data stored as compressed MP4
videos, then optionally pushes the result to the Hugging Face Hub.

Expected HDF5 File Structure:
------------------------------
The script assumes a directory with zero-indexed HDF5 files (e.g., `data_0.hdf5`).
Each file should contain the following structure:

/data/demo_0/
    ├── action                (Dataset): Actions taken at each step.
    ├── observations/
    │   └── rgb               (Dataset): RGB image observations.
    ├── abs_joint_pos         (Dataset): Absolute joint positions.
    └── timestep              (Dataset): Timestamps for each data point.

Usage:
------
    python convert_data_to_lerobot_video.py --data-dir /path/to/your/hdf5/files --repo-id your-username/your-dataset-name

To also push to the Hub:
    python convert_data_to_lerobot_video.py --data-dir /path/to/your/hdf5/files --repo-id your-username/your-dataset-name --push-to-hub
"""

import glob
import os
import shutil
from pathlib import Path

import h5py
import tqdm
import tyro

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.constants import HF_LEROBOT_HOME


def convert_data_to_lerobot(data_dir: Path, repo_id: str, *, push_to_hub: bool = False):
    """
    Converts a directory of HDF5 files to a LeRobotDataset with a video backend.

    Args:
        data_dir: The path to the directory containing the HDF5 files.
        repo_id: The repository ID for the dataset on the Hugging Face Hub.
        push_to_hub: Whether to push the dataset to the Hub after conversion.
    """
    final_output_path = os.path.join(HF_LEROBOT_HOME, repo_id)
    if final_output_path.exists():
        print(f"Removing existing dataset at {final_output_path}")
        shutil.rmtree(final_output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=data_dir,
        robot_type='panda',
        fps=30,
        use_videos=True,  # use video by default
        features={
            "observation.images.ultrasound": {  # ultrasound images: /image_raw/compressed
                "dtype": "video",
                "shape": (1920, 1080, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.room_rgb_camera": {  # room RGB camera: /vega_vt/image_raw
                "dtype": "video",
                "shape": (1024, 768, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist_camera_depth": {  # wrist RGBD camera depth: /camera/camera/depth/...
                "dtype": "video",
                "shape": (848, 480, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.wrist_camera_rgb": {  # wrist RGBD camera rgb: /camera/camera/color/...
                "dtype": "video",
                "shape": (1280, 720, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.meta.force_torque": {  # Force sensor: /ati_ft_broadcaster/wrench
                "dtype": "float32",
                "shape": (6,),
                "names": ["fx", "fy", "fz", "tx", "ty", "tz"],
                "info": {
                    "sensor": "wrist_ft",
                    "units": "N + N m"
                }
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (27,),
                "names": ["elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint",
                          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                          # 0~5: UR5e joint-space state: /joint_states
                          "probe_ur_x", "probe_ur_y", "probe_ur_z",
                          "probe_ur_ux", "probe_ur_uy", "probe_ur_uz", "probe_ur_w",
                          # 6~12: UR5e task-space (probe TCP) state: /tcp_pose_broadcaster/pose
                          "probe_ndi_x", "probe_ndi_y", "probe_ndi_z",
                          "probe_ndi_ux", "probe_ndi_uy", "probe_ndi_uz", "probe_ndi_w",
                          # 13~19: Probe task-space (NDI Polaris marker) state: /ndi/us_probe_pose
                          "needle_ndi_x", "needle_ndi_y", "needle_ndi_z",
                          "needle_ndi_ux", "needle_ndi_uy", "needle_ndi_uz", "needle_ndi_w",
                          # 20~26: Needle task-space (NDI Polaris marker) state: /ndi/needle_pose
                          # "needle_tip_x1", "needle_tip_y1", "needle_tip_x2", "needle_tip_y2",  # 27~30: needle tip bbox in US image coords TODO
                          # "needle_insert_length",  # 31: needle insertion length in mm  TODO
                         ],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["probe_delta_x", "probe_delta_y", "probe_delta_z",
                          "probe_delta_ux", "probe_delta_uy", "probe_delta_uz", "probe_delta_w",  # 0~6: delta_pose for probe
                          "needle_delta_x", "needle_delta_y", "needle_delta_z",
                          "needle_delta_ux", "needle_delta_uy", "needle_delta_uz", "needle_delta_w",  # 7~13: delta_pose for needle
                          ],
            },
            # "keystroke": {
            #     "dtype": "float32",
            #     "shape": (6,),
            #     "names": ["x", "y", "z", "roll", "pitch", "yaw"],  # TODO
            # },
            # "procedure_info": {
            #     "dtype": "string",
            #     "shape": (5,),
            #     "names": ["task_label",  # task 1~3
            #               "completion_status",  # success / fail / recovered
            #               "operator_name", "operator_skill_level",
            #               "phantom_info"],  # TODO
            # },

            """ Below are static metadata """
            
            # US acquisition settings
            "observation.meta.probe_type": {
                "dtype": "string",
                "shape": (1,),
                "names": ["company_model_endwith'linear'_or_'convex'"],
                # "Wisonic_Clover60_C51_convex"
            },
            # set "linear_fov_mm" to be 0, if using convex probe
            # set "convex_radius_mm" and "convex_fov_deg" to be 0, if using linear probe
            "observation.meta.probe_acquisition_param": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["center_frequency_mhz", "num_elements", "imaging_depth_cm",
                          "linear_fov_mm", "convex_radius_mm", "convex_fov_deg"],
            },
            # For room RGB camera: extrinsic calibration matrix from tracker to color
            "observation.meta.roomcam_cali_mtx_tracker_to_color": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["tx_mm", "ty_mm", "tz_mm", "qx", "qy", "qz", "qw"],
                # [-14.8879, 34.6886, -65.7274, 0.709725, 0.704474, -0.001833, 0.002027],
            },
            # For wrist RGBD camera: extrinsic calibration matrix with respect to the end effector (hand-eye)
            "observation.meta.wristcam_cali_mtx": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"],
            },
            # For wrist RGBD camera: extrinsic calibration matrix from depth to color
            "observation.meta.wristcam_cali_mtx_depth_to_color": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"],
            },
            # For ultrasound probe: spatial calibration matrix, ultrasound image with respect to the probe NDI marker
            "observation.meta.prob_cali_mtx": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["tx_m", "ty_m", "tz_m", "qx", "qy", "qz", "qw"],
            }
        },
        image_writer_processes=16,
        image_writer_threads=20,
        tolerance_s=0.1,
    )

    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))

    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}. Exiting.")
        return

    print(f"Found {len(hdf5_files)} episodes to convert.")

    task_description = "Conduct a liver ultrasound scan"

    for hdf5_path in tqdm.tqdm(hdf5_files, desc="Converting Episodes"):
        try:
            with h5py.File(hdf5_path, "r") as f:
                root_name = "data/demo_0"
                if root_name not in f:
                    print(f"Warning: Skipping {hdf5_path} because '{root_name}' group was not found.")
                    continue

                num_steps = len(f[f"{root_name}/action"])

                # Add each frame from the episode to the internal buffer.
                for step in range(num_steps):
                    frame_data = {
                        "observation.image": f[f"{root_name}/observations/rgb"][step],
                        "observation.wrist_image": f[f"{root_name}/observations/rgb"][step],
                        "observation.state": f[f"{root_name}/abs_joint_pos"][step],
                        "action": f[f"{root_name}/action"][step],
                    }
                    timestamp = f[f"{root_name}/timestep"][step]
                    dataset.add_frame(frame_data, task=task_description, timestamp=timestamp)

            # After processing all frames for an HDF5 file, save the buffered
            # data as a completed episode. This will trigger the video encoding
            # for the 'image' and 'wrist_image' frames collected.
            dataset.save_episode()

        except Exception as e:
            print(f"Error processing {hdf5_path}: {e}")
            # It's good practice to clear the buffer on error to prevent
            # a failed episode from contaminating the next one.
            dataset.clear_episode_buffer()

    print(f"Dataset conversion complete. Saved to {final_output_path}")

    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
        dataset.push_to_hub()
        print("Push complete.")


def main(
    data_dir: Path = Path("path/to/your/data"),
    repo_id: str = "your-username/your-dataset-name",
    *,
    push_to_hub: bool = False,
):
    """
    Main entry point for the conversion script.

    Args:
        data_dir: The directory containing HDF5 episode files.
        repo_id: The desired Hugging Face Hub repository ID.
        push_to_hub: If True, uploads the dataset to the Hub.
    """
    if not data_dir.is_dir():
        print(f"Error: The provided data directory does not exist: {data_dir}")
        return

    if repo_id == "your-username/your-dataset-name":
        print("Warning: Using the default repo_id. Please specify your own with --repo-id.")

    convert_data_to_lerobot(data_dir, repo_id, push_to_hub=push_to_hub)


if __name__ == "__main__":
    tyro.cli(main)