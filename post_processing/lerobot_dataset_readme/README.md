# A Multimodal Dataset for Autonomous Probe Placement and Needle Retrieval in Ultrasound-Guided Liver Biopsy (US-PPNR) - README

---

## 📋 At a Glance

Multimodal, time-synchronized demonstrations of ultrasound-guided needle biopsy on clinically realistic phantoms/tissue, including ultrasound imaging, multi-view video, robot kinematics, force/torque, and optical tracking.

---

## 📖 Dataset Overview

This dataset supports training and evaluating vision-language-action (VLA) and imitation-learning policies for **ultrasound (US)-guided needle biopsy**. It covers the procedures including: scanning and probe placement, needle tracking, and probe adjustments to maintain needle visualization. Each trajectory provides synchronized **US B-mode**, **RGB room video**, **RGB-D wrist video**, **robot joint/task-space states**, **force/torque**, and **NDI optical tracking** for needle/probe/camera poses, plus time-aligned narration/metadata.

|                        |                                                                                                        |
|:-----------------------|:-------------------------------------------------------------------------------------------------------|
| **Total Trajectories** | 2,546                                                                                                  |
| **Total Hours**        | 10:51:09                                                                                               |
| **Data Type**          | \[✅] Clinical  \[✅] Ex-Vivo  \[✅] Table-Top Phantom  \[ ] Digital Simulation  \[✅] Physical Simulation |
| **License**            | CC BY 4.0                                                                                              |
| **Version**            | 1.0                                                                                                    |

---

## 🎯 Tasks & Domain

### Domain

- [ ] **Surgical Robotics**
- [x] **Ultrasound Robotics**
- [ ] **Other Healthcare Robotics** (Please specify: \[Your Domain\])

### Demonstrated Skills

- Probe Placement (**PP.tar.gz**):
  - Autonomous probe placement for liver scanning
- Needle Retrieval (**NR.tar.gz**):
  - Robust needle tracking during insertion
  - Autonomous needle retrieval from phantom/tissue under US guidance
  - Autonomous probe scanning (sweeps, plane finding)
  - Adaptive probe manipulation to maintain visualization

---

## 🔬 Data Collection Details

### Collection Method

- [x] **Human Teleoperation**
- [x] **Programmatic/State-Machine**
- [ ] **AI Policy / Autonomous**
- [ ] **Other** (Please specify: \[Your Method\])


### Operator Details

|                          | Description                                                                                                                                                              |
|:-------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Operator Count**       | 6                                                                                                                                                                        |
| **Operator Skill Level** | \[x] Expert (e.g., Sonographer/Clinician) <br> \[x] Intermediate (e.g., Trained Researcher) <br> \[x] Novice (e.g., ML Researcher with minimal experience) <br> \[ ] N/A |
| **Collection Period**    | From 2025-12-20 to 2026-01-18                                                                                                                                            |

### Recovery Demonstrations

- [ ] **Yes**
- [x] **No**

---

## 💡 Diversity Dimensions

- [x] **Camera Position / Angle**
- [x] **Lighting Conditions**
- [x] **Target Object** (e.g., different phantom models, tissue types, lesion locations)
- [x] **Spatial Layout** (e.g., varying lesion position and insertion approach)
- [ ] **Robot Embodiment** (if multiple robots were used)
- [x] **Task Execution** (e.g., different scanning/insertion techniques)
- [x] **Background / Scene**
- [ ] **Other** (Please specify: tissue properties / anatomical scenario difficulty)

**Elaboration:**

Data were collected across both commercial and custom-built phantoms to vary tissue appearance, acoustic properties, and lesion presentation. 
Trials span a range of target locations, scanning planes, insertion angles, and operator strategies (expert vs sub-expert), yielding diversity in both perception (US appearance, needle visibility) and control (probe/needle trajectories, contact forces).

---

## 🛠️ Equipment & Setup

### Robotic Platform(s)

- UR5e robotic arm for **probe manipulation** (end-effector-mounted force/torque sensing; tracked probe marker)


### Sensors & Cameras

| Type                       | Model/Details                                                                                             |
|:---------------------------|:----------------------------------------------------------------------------------------------------------|
| **Medical Imager**         | Ultrasound B-mode stream — **Wisonic Clover 60** + **Wisonic C5-1 convex transducer**, 1920×1080 @ 30 FPS |
| **Room/3rd Person Camera** | **NDI Polaris Vega VT RGB camera**, 1024×768 @ 30 FPS                                                     |
| **RGB-D Wrist Camera**     | **ZED 2**, RGB + Depth, 640×480 @ 30 FPS (mounted on probe end-effector)                                  |
| **Force/Torque Sensor**    | **ATI Axia80-M8**, 30 HZ, End-effector F/T sensor                                                         |
| **Optical Tracker**        | **NDI Polaris Vega VT**, 30 Hz, tracked poses for needle/probe/cameras/components                         |

---

## 🎯 Action & State Space Representation

This dataset follows the **LeRobot** format and provides synchronized actions, robot state, imaging, and tracking.

### Action Space Representation

**Primary Action Representation:**
- [x] **Absolute Cartesian** (position/orientation relative to robot base)
- [x] **Relative Cartesian** (delta position/orientation from current pose) *(typical for teleop step commands)*
- [x] **Joint Space** (direct joint commands recorded from robot interface)
- [ ] **Other** (Please specify: discrete teleop inputs / insertion depth commands)

**Orientation Representation:**
- [x] **Quaternions** (x, y, z, w)
- [ ] **Euler Angles** (roll, pitch, yaw)
- [ ] **Axis-Angle** (rotation vector)
- [ ] **Rotation Matrix** (3x3 matrix)
- [ ] **Other** (Please specify: \[Your Representation\])

**Reference Frame:**
- [x] **Robot Base Frame**
- [x] **Tool/End-Effector Frame**
- [x] **World/Global Frame** *(via optical tracker frame)*
- [ ] **Camera Frame**
- [x] **Other** (calibrated US image frame)

**Action Dimensions:**

Actions are provided per-controlled subsystem (probe manipulation and needle insertion).

**Example:**
```text
action.probe_delta: "probe_delta_x", "probe_delta_y", "probe_delta_z", "probe_delta_ux", "probe_delta_uy", "probe_delta_uz", "probe_delta_w",
- "probe_delta_x", "probe_delta_y", "probe_delta_z": position in UR5e robot base frame (meters)
- "probe_delta_ux", "probe_delta_uy", "probe_delta_uz", "probe_delta_w": orientation as quaternion

action.needle_delta: "needle_delta_x", "needle_delta_y", "needle_delta_z", "needle_delta_ux", "needle_delta_uy", "needle_delta_uz", "needle_delta_w"
- "needle_delta_x", "needle_delta_y", "needle_delta_z": position in NDI optical tracker frame (meters)
- "needle_delta_ux", "needle_delta_uy", "needle_delta_uz", "needle_delta_w": orientation as quaternion
```

### State Space Representation

**State Information Included:**
- [x] **Joint Positions** (all articulated joints)
- [ ] **Joint Velocities**
- [x] **End-Effector Pose** (Cartesian position/orientation)
- [x] **Force/Torque Readings**
- [ ] **Gripper State**
- [x] **Other** (optical tracker poses for needle/probe; calibrated transforms)

**State Dimensions:**
```text
observation.state.joint_positions: "elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
observation.state.ee_pose: "probe_ur_x", "probe_ur_y", "probe_ur_z", "probe_ur_ux", "probe_ur_uy", "probe_ur_uz", "probe_ur_w",
observation.state.ee_pose_ndi: "probe_ndi_x", "probe_ndi_y", "probe_ndi_z", "probe_ndi_ux", "probe_ndi_uy", "probe_ndi_uz", "probe_ndi_w",
observation.state.needle_pose_ndi: "needle_ndi_x", "needle_ndi_y", "needle_ndi_z", "needle_ndi_ux", "needle_ndi_uy", "needle_ndi_uz", "needle_ndi_w",
observation.meta.force_torque: "fx", "fy", "fz", "tx", "ty", "tz"
```

---

## ⏱️ Data Synchronization Approach

All modalities are synchronized to a common time base and exported with corrected timestamps.

- **Streams & rates:** robot kinematics and force/torque at **500 Hz**; NDI optical tracking at **60 Hz**; RGB/RGB-D/US at **30 FPS**.
- **Delay measurement:** the robot executes controlled sinusoidal motions while each sensor stream observes the motion (needle/probe/camera movement). Phase differences between each sensor’s measured sinusoid and the robot reference are used to estimate per-stream delays. The delay was calculated using [PLUS Toolkit](https://pmc.ncbi.nlm.nih.gov/articles/PMC4437531/).
- **Alignment:** timestamps are corrected using measured delays, and asynchronous streams are aligned via interpolation/resampling to produce unified, time-aligned trajectories.
- **Recording:** all raw streams and teleop inputs are recorded in ROS (rosbag).

---

## 👥 Attribution & Contact

|                            |                                                                                                                                                                                                                                                                                                                                                                                                                             |
|:---------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Dataset Lead**           | Shing Shin Cheng; Wei Wang; Qingpeng Ding; Yuelin Zhang; Zhouyang Hong; Luoyao Kang; Wenxuan Xie                                                                                                                                                                                                                                                                                                                            |
| **Institution**            | The Chinese University of Hong Kong (CUHK); First Affiliated Hospital, Sun Yat-Sen University                                                                                                                                                                                                                                                                                                                               |
| **Contact Email**          | sscheng@cuhk.edu.hk                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Citation (BibTeX)**      | <pre><code>@misc{openh_ausnb_2026,<br>  author = {Cheng, Shing Shin and Wang, Wei and Ding, Qingpeng and Zhang, Yuelin and Hong, Zhouyang and Kang, Luoyao and Xie, Wenxuan},<br>  title = {A Multimodal Dataset for Autonomous Probe Placement and Needle Retrieval in Ultrasound-Guided Liver Biopsy (US-PPNR)},<br>  year = {2026},<br>  publisher = {Open-H-Embodiment},<br>  license = {CC BY 4.0}, <br>}</code></pre> |
---