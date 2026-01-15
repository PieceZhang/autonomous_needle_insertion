# LeRobot Dataset Conversion for Autonomous Needle Insertion

This repository provides scripts to convert decoded ROS2 / MCAP data into the **LeRobot dataset format**, and to inspect the converted dataset using visualization tools.

The project is organized around **multiple surgical robotics tasks**, currently including:

- **Task 1**: Probe Placement  
- **Task 4.1 / 4.2**: Needle Retrieval *(planned / under extension)*

---

## Repository Structure

```text
.
├── task1_to_lerobot.py              # Convert raw Task 1 data to LeRobot format
├── task1_viz_lerobot_gui.py         # Visualize converted LeRobot episodes
├── task1_to_lerobot.sh              # Convenience script for Task 1 conversion
├── task1_viz_lerobot_gui.sh         # Convenience script for visualization
├── README.md
└── workspace/
    └── calibration/
        └── PlusDeviceSet_*.xml      # Hand-eye & probe calibration files
        └── hand_eye_20251231_075559.json
        └── hand_eye_20260112_071955.json
````

---

## Tasks Overview

### **Task 1 — Probe Placement**

**Key Outputs**

* RGB / depth / ultrasound video streams
* Robot poses and actions
* Synchronized multi-sensor episodes

---

### **Task 4.1 / Task 4.2 — Needle Retrieval**

**Status**
Planned extension.
The same dataset abstraction and conversion pipeline will be reused, with task-specific action/state definitions.

---

## Environment Setup

Make sure you have:

* Python ≥ 3.9
* `lerobot` installed and available in your environment, **recommend 0.4.2**
* OpenCV, NumPy, tqdm, etc.

Example:

```bash
conda activate lerobot
```

---

## Task 1: Convert Raw Data to LeRobot Format

### 1. Data Conversion

Use `task1_to_lerobot.py` to convert decoded raw data into a LeRobot dataset.

```bash
python task1_to_lerobot.py \
  --raw_root <RAW_DATA_ROOT>/task1 \
  --out_root <OUTPUT_ROOT>/task1_output \
  --workspace_root <WORKSPACE_ROOT> \
  --calib_xml <WORKSPACE_ROOT>/calibration/PlusDeviceSet_fCal_Wisonic_C5_1_NDIPolaris_2.0_20260111_SRIL.xml
```

#### Arguments

| Argument           | Description                                            |
| ------------------ | ------------------------------------------------------ |
| `--raw_root`       | Root directory of decoded Task 1 raw data              |
| `--out_root`       | Output directory for LeRobot-formatted dataset         |
| `--workspace_root` | Workspace directory containing calibration and configs |
| `--calib_xml`      | PLUS calibration XML (probe ↔ tracker ↔ camera)        |

---

### 2. Visualization

After conversion, visualize the dataset using the GUI viewer:

```bash
python task1_viz_lerobot_gui.py \
  --root <OUTPUT_ROOT>/task1_output
```

This tool allows you to:

* Inspect per-episode videos
* Verify sensor synchronization
* Check robot state / action consistency

---

## Shell Script Shortcuts

For convenience, the following shell scripts wrap the commands above:

```bash
bash task1_to_lerobot.sh
bash task1_viz_lerobot_gui.sh
```

You may edit these scripts to point to your local paths.

---

## Notes & Design Choices

* **Multiple FPS streams** are temporally aligned at the episode level.
* Numeric states and actions are stored in LeRobot-compatible arrays for downstream learning.
* Video visualization is decoupled from numeric inspection to avoid performance bottlenecks.
* The pipeline is designed to be **task-agnostic**, enabling easy extension to Task 4.x.

---

## Planned Extensions

* [ ] Task 4.1 / 4.2 conversion scripts
* [ ] Unified task configuration (`task_spec.json`)

---

## Contact

For questions or extensions, please refer to the project maintainer or open an issue.

