import qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging, re, time
import numpy as np

# External dep
# import ndicapy  # provided by ndicapi wheel

#
# RegisterOpticalMarkers
#
class RegisterOpticalMarkers(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "Register Polaris Optical Markers"
        self.parent.categories = ["Registration"]
        self.parent.contributors = ["Your Name"]
        self.parent.helpText = (
            "Connect directly to NDI Polaris (Vega/Polaris) via TCP and acquire stray marker positions.\n"
            "Uses TX with reply option 0x1000 (passive) or 0x0004 (active)."
        )
        self.parent.acknowledgementText = "Uses ndicapi/ndicapy."

#
# RegisterOpticalMarkersWidget
#
class RegisterOpticalMarkersWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = RegisterOpticalMarkersLogic()

        #
        # UI
        #
        form = qt.QFormLayout(self.parent)

        # IP / Port
        self.ipEdit = qt.QLineEdit(); self.ipEdit.placeholderText = "Tracker IP (e.g. 192.168.0.10)"
        self.portEdit = qt.QLineEdit(); self.portEdit.setText("")  # leave blank if using device default
        self.portEdit.setValidator(qt.QIntValidator(1, 65535))
        form.addRow("IP", self.ipEdit)
        form.addRow("Port", self.portEdit)

        # Stray type
        self.strayType = qt.QComboBox()
        self.strayType.addItem("Passive stray (0x1000)")
        self.strayType.addItem("Active stray (0x0004)")
        form.addRow("Stray type", self.strayType)

        # Markups node selector
        self.markupsSelector = slicer.qMRMLNodeComboBox()
        self.markupsSelector.objectName = "markupsSelector"
        self.markupsSelector.toolTip = "Choose/Make a Markups Fiducial to display points"
        self.markupsSelector.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
        self.markupsSelector.addEnabled = True
        self.markupsSelector.removeEnabled = False
        self.markupsSelector.noneEnabled = False
        self.markupsSelector.selectNodeUponCreation = True
        form.addRow("Output markups", self.markupsSelector)

        # Buttons
        btnRow = qt.QHBoxLayout()
        self.connectBtn = qt.QPushButton("Connect")
        self.disconnectBtn = qt.QPushButton("Disconnect"); self.disconnectBtn.enabled = False
        self.initBtn = qt.QPushButton("INIT & TSTART"); self.initBtn.enabled = False
        btnRow.addWidget(self.connectBtn); btnRow.addWidget(self.disconnectBtn); btnRow.addWidget(self.initBtn)
        form.addRow(btnRow)

        btnRow2 = qt.QHBoxLayout()
        self.getOnceBtn = qt.QPushButton("Get stray once"); self.getOnceBtn.enabled = False
        self.streamToggle = qt.QPushButton("Start streaming"); self.streamToggle.checkable = True; self.streamToggle.enabled = False
        self.clearBtn = qt.QPushButton("Clear markups")
        btnRow2.addWidget(self.getOnceBtn); btnRow2.addWidget(self.streamToggle); btnRow2.addWidget(self.clearBtn)
        form.addRow(btnRow2)

        # Raw reply (for debugging)
        self.rawReply = qt.QPlainTextEdit(); self.rawReply.setReadOnly(True); self.rawReply.setMaximumBlockCount(200)
        form.addRow("Raw TX reply", self.rawReply)

        # Timer for streaming
        self.timer = qt.QTimer(); self.timer.setInterval(100)  # 10 Hz default
        self.timer.timeout.connect(self.onGetOnce)

        # Signals
        self.connectBtn.clicked.connect(self.onConnect)
        self.disconnectBtn.clicked.connect(self.onDisconnect)
        self.initBtn.clicked.connect(self.onInit)
        self.getOnceBtn.clicked.connect(self.onGetOnce)
        self.streamToggle.toggled.connect(self.onStreamToggle)
        self.clearBtn.clicked.connect(self.onClear)

        self.parent.layout().addLayout(form)

        # Ensure an output node exists
        if not self.markupsSelector.currentNode():
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "StrayMarkers")
            self.markupsSelector.setCurrentNode(node)

    # --- UI handlers ---

    def onConnect(self):
        ip = self.ipEdit.text.strip()
        portText = self.portEdit.text.strip()
        port = int(portText) if len(portText) else None

        try:
            self.logic.connect(ip, port)
            self.connectBtn.enabled = False
            self.disconnectBtn.enabled = True
            self.initBtn.enabled = True
            self.getOnceBtn.enabled = True
            self.streamToggle.enabled = True
            slicer.util.infoDisplay("Connected to NDI device.")
        except Exception as e:
            slicer.util.errorDisplay(f"Connect failed: {e}")

    def onDisconnect(self):
        self.timer.stop()
        try:
            self.logic.disconnect()
        except Exception as e:
            logging.warning(f"During disconnect: {e}")
        self.connectBtn.enabled = True
        self.disconnectBtn.enabled = False
        self.initBtn.enabled = False
        self.getOnceBtn.enabled = False
        self.streamToggle.setChecked(False); self.streamToggle.enabled = False
        slicer.util.infoDisplay("Disconnected.")

    def onInit(self):
        try:
            self.logic.initAndStartTracking()
            slicer.util.infoDisplay("Tracker initialized and started (INIT/TSTART).")
        except Exception as e:
            slicer.util.errorDisplay(f"INIT/TSTART failed: {e}")

    def onGetOnce(self):
        node = self.markupsSelector.currentNode()
        if node is None:
            slicer.util.errorDisplay("Please create/select a Markups Fiducial node.")
            return
        active = (self.strayType.currentIndex == 1)
        try:
            points, raw = self.logic.getStrayOnce(active=active)
            self.rawReply.setPlainText(raw)
            # Append points to Markups
            for p in points:
                # p is (x,y,z) in tracker coordinates, units mm
                node.AddFiducialFromArray(np.array(p, dtype=float), f"stray-{time.time():.3f}")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to get stray markers: {e}")

    def onStreamToggle(self, checked):
        if checked:
            self.streamToggle.setText("Stop streaming")
            self.timer.start()
        else:
            self.streamToggle.setText("Start streaming")
            self.timer.stop()

    def onClear(self):
        node = self.markupsSelector.currentNode()
        if node:
            node.RemoveAllControlPoints()

#
# RegisterOpticalMarkersLogic
#
class RegisterOpticalMarkersLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self.device = None

    def connect(self, ip, port=None):
        if not ip:
            raise ValueError("IP address is required.")
        # If port is None, ndicapy uses its internal default for the device.
        if port is None:
            self.device = ndicapy.ndiOpenNetwork(ip)
        else:
            self.device = ndicapy.ndiOpenNetwork(ip, int(port))
        if not self.device:
            raise RuntimeError("ndiOpenNetwork returned null handle")

    def initAndStartTracking(self):
        self._ensureConnected()
        # Initialize device and start tracking
        self._send("INIT:")
        self._send("TSTART:")

    def disconnect(self):
        if self.device:
            try:
                self._send("TSTOP:")
            except Exception:
                pass
            ndicapy.ndiClose(self.device)
            self.device = None

    def getStrayOnce(self, active=False):
        """Return ([(x,y,z), ...], raw_reply_text)"""
        self._ensureConnected()
        # Reply option: passive stray 0x1000, active stray 0x0004
        reply_opt = 0x0004 if active else 0x1000
        # Use TX (ASCII) so we can parse text robustly
        cmd = f"TX:{reply_opt:04X}"
        self._send(cmd)
        # Try generic reply getter (works in many ndicapy builds)
        raw = ""
        try:
            raw = ndicapy.ndiGetReply(self.device)  # returns bytes or str depending on build
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode(errors="ignore")
        except AttributeError:
            # Fallback: some ndicapy versions auto-return from ndiCommand
            # We re-send TX and rely on any bound helper (rare); otherwise raise
            raise RuntimeError("Your ndicapy build lacks ndiGetReply(); consider upgrading ndicapi.")

        pts = self._parse_tx_stray_ascii(raw)
        return pts, raw

    # --- helpers ---

    def _send(self, s):
        r = ndicapy.ndiCommand(self.device, s)
        if r != ndicapy.NDI_OKAY:
            raise RuntimeError(f"NDI command failed ({s}): code {r}")

    def _ensureConnected(self):
        if not self.device:
            raise RuntimeError("Not connected")

    @staticmethod
    def _parse_tx_stray_ascii(replyText):
        """
        Heuristic ASCII parser for stray-marker triples in TX reply.
        It tries to extract sensible (x,y,z) triples in millimetres.
        Filters out absurd values and deduplicates nearby points.
        """
        # Extract all numbers
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", replyText)
        vals = [float(n) for n in nums]
        # Group into triples and filter unrealistic magnitudes
        pts = []
        for i in range(0, len(vals) - 2, 3):
            x, y, z = vals[i], vals[i + 1], vals[i + 2]
            if all(abs(v) < 5000 for v in (x, y, z)):  # ±5 m sanity window
                pts.append((x, y, z))
        # Deduplicate near-identical points
        dedup = []
        def near(a, b): return sum((a[k]-b[k])**2 for k in range(3)) < 0.1**2
        for p in pts:
            if not any(near(p, q) for q in dedup):
                dedup.append(p)
        return dedup