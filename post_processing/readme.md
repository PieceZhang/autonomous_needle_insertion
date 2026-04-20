# LeRobot Dataset Conversion for Autonomous Needle Insertion

This repository provides scripts to convert decoded ROS2 / MCAP data into the **LeRobot dataset format**, and to inspect the converted datasets using a unified visualization tool.

The pipeline is organized around **multiple surgical robotics tasks**, while sharing common abstractions for:
- sensor synchronization
- calibration handling
- LeRobot episode construction
- visualization

---

## Supported Tasks

| Task ID | Task Name | Conversion Script | Visualization |
|-------|----------|-------------------|---------------|
| Task 1 | Probe Placement | `task1_to_lerobot.py` | shared |
| Task 4.1 | Needle Retrieval | `task4_to_lerobot.py` | shared |
| Task 4.2 | Needle Retrieval | `task4_to_lerobot.py` | shared |

> **Note**  
> Task 4.1 and Task 4.2 differ only in raw data content and task semantics.  
> They are processed using the **same conversion script**.

---

## Repository Structure

```text
.
├── task1_to_lerobot.py              # Task 1 conversion
├── task4_to_lerobot.py              # Task 4.1 / 4.2 conversion
├── visualize_lerobot_gui.py         # Shared visualization tool
├── task1_to_lerobot.sh
├── task4_to_lerobot.sh
├── README.md
└── calibration/
    └── PlusDeviceSet_*.xml      # Probe / tracker / camera calibration
    └── hand_eye_20251231_075559.json
    └── hand_eye_20260112_071955.json
````

---

## Environment Setup

The conversion and visualization scripts are tested with the following environment:

- **Python**: 3.10.19  
- **LeRobot**: 0.3.3 

```bash
conda activate lerobot
```

Make sure `lerobot` and common dependencies (NumPy, OpenCV, tqdm, etc.) are installed.

---

## Task 1 — Probe Placement

### Convert Raw Data to LeRobot Format

```bash
python task1_to_lerobot.py \
  --raw_root <RAW_DATA_ROOT>/task1 \
  --out_root <OUTPUT_ROOT>/task1_output \
  --workspace_root <WORKSPACE_ROOT> \
  --calib_xml <WORKSPACE_ROOT>/calibration/PlusDeviceSet_fCal_Wisonic_C5_1_NDIPolaris_2.0_20260111_SRIL.xml
```

---

## Task 4.1 / Task 4.2 — Needle Retrieval

### Convert Raw Data to LeRobot Format

Both Task 4.1 and Task 4.2 use the **same conversion script**.

```bash
python task4_to_lerobot.py \
  --raw_root <RAW_DATA_ROOT>/task41 \
  --out_root <OUTPUT_ROOT>/task41_output \
  --workspace_root <WORKSPACE_ROOT> \
  --calib_xml <WORKSPACE_ROOT>/calibration/PlusDeviceSet_fCal_Wisonic_C5_1_NDIPolaris_2.0_20260111_SRIL.xml
```

> To process Task 4.2, simply change `--raw_root` and `--out_root` accordingly.

---

## Shared Visualization from Lerobot Format(All Tasks)

All tasks use the **same visualization script** to inspect converted LeRobot datasets.

```bash
python task_viz_lerobot_gui.py \
  --root <OUTPUT_ROOT>/<TASK_OUTPUT_DIR>
```

This viewer supports:

* per-episode video playback
* multi-stream inspection
* numeric state / action verification

---

## Shared Visualization from Rosbag_decode(All Tasks)

```bash
python task_viz.py \
  --raw_root <ROSBAG_DECODE_ROOT>/<TASK_OUTPUT_DIR>
  --task task4 || task1
```

## Shell Script Shortcuts

For convenience:

```bash
bash task1_to_lerobot.sh
bash task4_to_lerobot.sh
```

Each script wraps the corresponding Python command with editable local paths.

---

## Design Notes

* Multi-FPS streams are temporally aligned at the **episode level**
* Numeric states and actions are stored independently from video streams
* Visualization is shared and task-agnostic
* Task-specific logic is isolated in conversion scripts only

---

## Extending to New Tasks

To add a new task:

1. Implement a new `taskX_to_lerobot.py` (or reuse an existing one)
2. Follow the same raw / output directory structure
3. Use the shared visualization tool without modification

---

## Contact

For questions or extensions, please open an issue or contact the maintainer.
