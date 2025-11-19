import logging
import os
import sys
import time
import math
import numpy as np
import qt, ctk, slicer
import itertools
from slicer.ScriptedLoadableModule import *

# ----------------------------------------------------------
# 加载你的 meta_marker 包路径（按你的实际路径写）
# ----------------------------------------------------------
PROJECT_ROOT = "/Users/leo17/Desktop/surgical_robotics/equipment/autonomous_needle_insertion/"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ndicapy
from meta_marker.calib_test import set_up_tool
from meta_marker.read_stray_markers import connect_to, parse_tx1000_reply

ROM_PATH = PROJECT_ROOT + "meta_marker/8700340.rom"


# -------------------------------------------------------------------------
# Slicer module
# -------------------------------------------------------------------------

class RegisterOpticalMarkers(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "Register Optical Markers"
        parent.categories = ["Registration"]
        parent.helpText = "Acquire Polaris passive stray markers using TX 1000."
        parent.acknowledgementText = "Part of autonomous needle insertion project."


# -------------------------------------------------------------------------
# GUI
# -------------------------------------------------------------------------

class RegisterOpticalMarkersWidget(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logic = RegisterOpticalMarkersLogic()

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        layout = self.layout

        panel = ctk.ctkCollapsibleButton()
        panel.text = "Polaris Controls"
        layout.addWidget(panel)

        form = qt.QFormLayout(panel)

        # --- IP / Port ---
        self.ipEdit = qt.QLineEdit()
        self.ipEdit.setText("192.168.56.5")
        form.addRow("IP", self.ipEdit)

        self.portEdit = qt.QLineEdit()
        self.portEdit.setText("8765")
        form.addRow("Port", self.portEdit)

        # --- Buttons ---
        self.connectButton = qt.QPushButton("Connect")
        form.addRow(self.connectButton)

        self.startButton = qt.QPushButton("Start Tracking")
        form.addRow(self.startButton)

        self.stopButton = qt.QPushButton("Stop Tracking")
        form.addRow(self.stopButton)

        self.acquireOnceButton = qt.QPushButton("Acquire Once (TX1000)")
        form.addRow(self.acquireOnceButton)

        self.acquireMeanButton = qt.QPushButton("Acquire 100 frames (Mean)")
        form.addRow(self.acquireMeanButton)

        self.computeCTErrorButton = qt.QPushButton("Compute Error vs CT")
        form.addRow(self.computeCTErrorButton)


        # --- Markups Node Selector ---
        self.markupsSelector = slicer.qMRMLNodeComboBox()
        self.markupsSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.markupsSelector.addEnabled = True
        self.markupsSelector.selectNodeUponCreation = True
        self.markupsSelector.noneEnabled = False
        self.markupsSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Output Markups", self.markupsSelector)

        # 必须：保证默认有一个标注节点
        if not self.markupsSelector.currentNode():
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "StrayMarkers"
            )
            self.markupsSelector.setCurrentNode(node)

        # 信号绑定
        self.connectButton.connect("clicked()", self.onConnect)
        self.startButton.connect("clicked()", self.onStart)
        self.stopButton.connect("clicked()", self.onStop)
        self.acquireOnceButton.connect("clicked()", self.onAcquireOnce)
        self.acquireMeanButton.connect("clicked()", self.onAcquireMean)
        self.computeCTErrorButton.connect("clicked()", self.onComputeCTError)


        layout.addStretch(1)

    # ---------------------------------------------------------
    # 回调函数
    # ---------------------------------------------------------

    def _getMarkups(self):
        markups = self.markupsSelector.currentNode()
        if markups is None:
            markups = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLMarkupsFiducialNode", "StrayMarkers"
            )
            self.markupsSelector.setCurrentNode(markups)
        return markups

    def onConnect(self):
        ip = self.ipEdit.text.strip()
        port = int(self.portEdit.text.strip())

        try:
            self.logic.connect(ip, port)
            slicer.util.infoDisplay("Connected to Polaris and ROM loaded.")
        except Exception as e:
            slicer.util.errorDisplay(f"Connect failed:\n{e}")

    def onStart(self):
        try:
            self.logic.startTracking()
            slicer.util.infoDisplay("Tracking started.")
        except Exception as e:
            slicer.util.errorDisplay(f"TSTART failed:\n{e}")

    def onStop(self):
        try:
            self.logic.stopTracking()
            slicer.util.infoDisplay("Tracking stopped.")
        except Exception as e:
            slicer.util.errorDisplay(f"TSTOP failed:\n{e}")

    def onAcquireOnce(self):
        try:
            pts = self.logic.acquireOnce()
        except Exception as e:
            slicer.util.errorDisplay(f"Acquire once failed:\n{e}")
            return

        markups = self._getMarkups()
        markups.RemoveAllControlPoints()

        import vtk
        for i, p in enumerate(pts):
            markups.AddControlPoint(vtk.vtkVector3d(*p))
            markups.SetNthControlPointLabel(i, f"M{i+1}")

    def onAcquireMean(self):
        try:
            pts = self.logic.acquireMean(100)
        except Exception as e:
            slicer.util.errorDisplay(f"Acquire mean failed:\n{e}")
            return

        markups = self._getMarkups()
        markups.RemoveAllControlPoints()

        import vtk
        for i, p in enumerate(pts):
            markups.AddControlPoint(vtk.vtkVector3d(*p))
            markups.SetNthControlPointLabel(i, f"M{i+1}")

    def onComputeCTError(self):
        """
        从当前 Markups 中读取 9 个实时 marker（例如 mean 100 帧后得到的点），
        与 CT 文件中的 18 个 segment 质心（两两取中点）做刚体注册并计算误差。
        结果会打印在 Python console。
        """
        # 1. CT 文件路径
        ct_path = "/Users/leo17/Desktop/surgical_robotics/equipment/autonomous_needle_insertion/data/registration/preop/lumbar_MRI.tsv"

        # 2. 从 Markups 读取当前 live marker
        markups = self._getMarkups()
        N = markups.GetNumberOfControlPoints()
        if N == 0:
            slicer.util.errorDisplay("No points in Markups. Please acquire markers first (e.g., mean 100 frames).")
            return

        live_pts = np.zeros((N, 3), dtype=float)
        for i in range(N):
            p = [0.0, 0.0, 0.0]
            markups.GetNthControlPointPosition(i, p)
            live_pts[i] = p

        try:
            rmse_val, residuals, perm, R, t = self.logic.computeErrorVsCT(ct_path, live_pts)
        except Exception as e:
            slicer.util.errorDisplay(f"Compute error failed:\n{e}")
            return

        slicer.util.infoDisplay(
            f"Registration to lumbar_CT finished.\nRMSE = {rmse_val:.4f} mm\n"
            f"详见 Python console 中的详细输出。"
        )



# -------------------------------------------------------------------------
# Logic (后台负责连接 / Tracking / TX1000)
# -------------------------------------------------------------------------

class RegisterOpticalMarkersLogic(ScriptedLoadableModuleLogic):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dev = None
        self.handle = None

    # --------------------- CONNECT -------------------------
    def connect(self, ip, port):
        logging.info(f"Connecting to Polaris at {ip}:{port}")

        self.dev = connect_to(ip, port)  # INIT
        logging.info("INIT OK")

        logging.info("Loading ROM...")
        self.handle = set_up_tool(self.dev, ROM_PATH)
        logging.info("ROM loaded.")

    # --------------------- TRACKING -------------------------
    def startTracking(self):
        if self.dev is None:
            raise RuntimeError("Not connected.")

        reply = ndicapy.ndiCommand(self.dev, "TSTART ")
        if not reply.startswith("OKAY"):
            raise RuntimeError(f"TSTART failed: {reply}")

    def stopTracking(self):
        if self.dev:
            ndicapy.ndiCommand(self.dev, "TSTOP ")

    # --------------------- ACQUIRE -------------------------

    def acquireOnce(self):
        print("AcquireOnce clicked")
        if self.dev is None:
            raise RuntimeError("Not connected.")
        
        print("Sending TX 1000...")
        raw = ndicapy.ndiCommand(self.dev, "TX 1000")
        print("TX1000 reply:", raw)
        markers = parse_tx1000_reply(raw)
        print("Parsed markers:", markers)
        return np.array(markers, dtype=float)

    def acquireMean(self, N=100):
        """
        连续采集 N 帧 TX 1000，并打印每一帧的 markers。
        最后对所有有效帧求均值，返回 (min_n, 3) 的 numpy 数组。
        """
        if self.dev is None:
            raise RuntimeError("Not connected.")

        frames = []

        print(f"\n[Polaris] Start acquiring {N} frames for mean computation...")
        for i in range(N):
            raw = ndicapy.ndiCommand(self.dev, "TX 1000").strip()
            markers = parse_tx1000_reply(raw)

            print(f"Frame {i:03d}: got {len(markers)} marker(s).")
            if len(markers) > 0:
                arr = np.array(markers, dtype=float)
                print(arr)  # 显示本帧的坐标
                frames.append(arr)

            slicer.app.processEvents()
            time.sleep(0.01)

        if len(frames) == 0:
            raise RuntimeError("No valid frames (no markers detected in any frame).")

        # 统一长度（有的帧可能少几个点）
        min_n = min(f.shape[0] for f in frames)
        print(f"\n[Polaris] Valid frames: {len(frames)} / {N}, "
              f"using first {min_n} markers of each frame for mean.")

        clipped = np.stack([f[:min_n] for f in frames], axis=0)  # (F, min_n, 3)
        mean_pts = clipped.mean(axis=0)  # (min_n, 3)

        print("\n[Polaris] Mean of 100 frames (each row = one marker, columns = X Y Z in mm):")
        print(mean_pts)

        return mean_pts

    def computeErrorVsCT(self, ct_path: str, live_pts: np.ndarray):
        """
        ct_path: lumbar_CT.tsv 文件路径（18 行, Centroid_r/a/s）
        live_pts: 实时 marker 的坐标 (N x 3)，这里期望 N=9（mean 之后的 9 个 meta-fid 中心）
        """

        import pandas as pd

        # ---- 1. 读取 CT 18 个 segment 的质心 ----
        df = pd.read_csv(ct_path, sep="\t")
        cols = ["Centroid_r", "Centroid_a", "Centroid_s"]
        for c in cols:
            if c not in df.columns:
                raise RuntimeError(f"CT file missing column: {c}")
        ct18 = df[cols].to_numpy(dtype=float)   # 18 x 3

        if ct18.shape[0] != 18:
            raise RuntimeError(f"Expect 18 CT segments, got {ct18.shape[0]}")

        # ---- 2. 每两个 segment 取中点，得到 9 个 meta-fid center ----
        # 对应关系: (1,2), (3,4), ..., (17,18)
        ct9 = 0.5 * (ct18[0::2, :] + ct18[1::2, :])   # shape (9, 3)

        # ---- 3. 检查 live_pts 维度 ----
        if live_pts.shape[0] != 9:
            raise RuntimeError(f"Live markers must be 9, got {live_pts.shape[0]}")

        # ---- 4. 用距离签名做无序匹配（live -> CT）----
        perm = match_unlabeled_bruteforce(ct9, live_pts)
        live_matched = live_pts[perm]

        # ---- 5. 刚体配准（传感器 -> CT）----
        R, t = rigid_transform_row(live_matched, ct9)
        aligned = live_matched @ R + t

        # ---- 6. 计算误差 ----
        residuals = np.linalg.norm(ct9 - aligned, axis=1)
        e_rmse = rmse(ct9, aligned)

        # ---- 7. 打印详细信息 ----
        np.set_printoptions(precision=4, suppress=True)
        print("\n===== Registration to lumbar_CT =====")
        print("CT meta-fid centers (9 x 3):")
        print(ct9)
        print("\nLive markers (after permutation):")
        print(live_matched)
        print("\nPermutation (live index -> CT meta index):")
        print(perm)
        print("\nRotation R:\n", R)
        print("Translation t:\n", t)
        print(f"\nRMSE: {e_rmse:.4f}")
        print("Residuals per marker:", residuals)
        print("=====================================\n")

        return e_rmse, residuals, perm, R, t



import itertools  # 新增

# ---------- Registration utilities (pure NumPy, no SciPy) ----------

def pairwise_distances(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

def distance_signatures(D: np.ndarray) -> np.ndarray:
    """
    For each point i, take distances to others, sort them;
    used as a permutation-invariant signature.
    """
    N = D.shape[0]
    S = np.empty((N, N - 1), dtype=float)
    for i in range(N):
        row = np.delete(D[i], i)
        row.sort()
        S[i] = row
    return S

def match_unlabeled_bruteforce(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Match B to A by brute-force Hungarian-like search on distance-signature L2 cost.
    N<=9 so brute-force permutations (9! ~= 3.6e5) 是可以接受的。
    返回 perm，使得 B[perm[i]] 与 A[i] 对应。
    """
    N = A.shape[0]
    Da, Db = pairwise_distances(A), pairwise_distances(B)
    Sa, Sb = distance_signatures(Da), distance_signatures(Db)

    best_perm = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(N)):
        perm = list(perm)
        # cost = sum over i of || Sa[i] - Sb[perm[i]] ||
        diff = Sa - Sb[perm, :]
        cost = np.linalg.norm(diff)
        if cost < best_cost:
            best_cost = cost
            best_perm = np.array(perm, dtype=int)
    return best_perm

def rigid_transform_row(src: np.ndarray, dst: np.ndarray):
    """
    Solve R,t for dst ≈ src @ R + t (刚体配准，Kabsch).
    """
    c_src, c_dst = src.mean(axis=0), dst.mean(axis=0)
    X, Y = src - c_src, dst - c_dst
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = c_dst - c_src @ R
    return R, t

def rmse(A: np.ndarray, B: np.ndarray) -> float:
    return np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1)))
