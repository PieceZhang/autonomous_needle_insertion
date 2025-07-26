"""
Pivot calibration for an NDI wireless tool using ndicapy.
Collects N ≥ 30 frames while the user pivots the probe tip in place,
then computes the tool-tip offset in the tool frame (mm).
"""
import ndicapy
from six import int2byte
import numpy as np
import threading, time, pathlib, textwrap

# ----------------------------------------------------------------------
# --- 1.  Tracker connection & tool enable (reuse your helpers) --------
# ----------------------------------------------------------------------
from calib_test import connect_to, set_up_tool   # or copy the functions here

IP_ADDR = "192.168.33.190"     # adjust as needed
PORT    = 8765

dev = connect_to(IP_ADDR, PORT)
handle_hex = set_up_tool(dev, "8700340.rom")     # your ROM file
handle     = int(handle_hex, 16)
port_byte  = int2byte(handle)

# put tracker into Tracking mode
if not ndicapy.ndiCommand(dev, "TSTART ").startswith("OKAY"):
    raise RuntimeError("TSTART failed")

# ----------------------------------------------------------------------
# --- 2.  Streaming & sample collection --------------------------------
# ----------------------------------------------------------------------
samples_R, samples_p = [], []
streaming = False
thread    = None

def euler_to_matrix(quat):
    """Convert [qw,qx,qy,qz] from ndicapy to 3×3 rotation matrix."""
    qw, qx, qy, qz = quat
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx**2+qy**2)]
    ])
    return R

def frame_worker():
    global streaming
    while streaming:
        # ask for a single BX2/BX frame
        ndicapy.ndiCommand(dev, "BX 0001")
        data = ndicapy.ndiGetBXTransform(dev, port_byte)
        if data:
            quat = data[:4]
            xyz  = np.array(data[4:7])
            R    = euler_to_matrix(quat)
            samples_R.append(R)
            samples_p.append(xyz)
            print(f"\rCollected {len(samples_p)} frames", end='')
        time.sleep(0.05)   # 20 Hz


if __name__ == "__main__":
    print("\n=== Pivot Calibration ===")
    print("Hold the probe by the handle and rotate/pivot the TIP")
    print("freely but keeping it approximately fixed in space.")
    input("Press <Enter> to start collecting frames…")

    streaming = True
    thread = threading.Thread(target=frame_worker, daemon=True)
    thread.start()

    input("\nPress <Enter> again to STOP collection…")
    streaming = False
    thread.join()
    print(f"\nTotal frames collected: {len(samples_p)}")

    if len(samples_p) < 30:
        print("Need at least 30 frames for a robust fit – collect more next time.")

    # ----------------------------------------------------------------------
    # --- 3.  Solve least-squares pivot problem ----------------------------
    # ----------------------------------------------------------------------
    # Stack A = [R_i | -I],   b = -p_i
    A_blocks = []
    b_blocks = []
    I = -np.eye(3)
    for R, p in zip(samples_R, samples_p):
        A_blocks.append(np.hstack((R, I)))
        b_blocks.append(-p.reshape(3,1))
    A = np.vstack(A_blocks)
    b = np.vstack(b_blocks)

    x, *_ = np.linalg.lstsq(A, b, rcond=None)   # x = [t ; p]
    tip_offset = x[:3,0]   # (mm) expressed in TOOL frame
    pivot_world = x[3:,0]  # (mm) in tracker/world frame

    print("\n=== Calibration Result ===")
    print(f"Tip offset in TOOL frame (mm):  "
        f"X={tip_offset[0]:.2f}  Y={tip_offset[1]:.2f}  Z={tip_offset[2]:.2f}")
    print(f"Pivot point in WORLD frame (mm):"
        f"X={pivot_world[0]:.2f}  Y={pivot_world[1]:.2f}  Z={pivot_world[2]:.2f}")

    # ----------------------------------------------------------------------
    # --- 4.  Validate tip offset by live pivot ----------------------------
    # ----------------------------------------------------------------------
    def tip_worker():
        """Stream tip position using calibrated tip_offset."""
        local_stream = True
        first_tip = None
        tip_samples = []
        while local_stream:
            ndicapy.ndiCommand(dev, "BX 0001")
            data = ndicapy.ndiGetBXTransform(dev, port_byte)
            if data:
                quat = data[:4]
                xyz  = np.array(data[4:7])
                R    = euler_to_matrix(quat)
                tip_world = R @ tip_offset + xyz
                if first_tip is None:
                    first_tip = tip_world.copy()
                delta = np.linalg.norm(tip_world - first_tip)
                print(f"\rTip @ (X={tip_world[0]:.1f}, Y={tip_world[1]:.1f}, "
                      f"Z={tip_world[2]:.1f})  Δ={delta:.2f} mm", end='')
                tip_samples.append(tip_world)
            time.sleep(0.05)
            # read the global flag
            local_stream = streaming_test

        return np.array(tip_samples)

    print("\nPress <Enter> to START live tip validation…")
    input()
    streaming_test = True
    tip_thread = threading.Thread(target=tip_worker, daemon=True)
    tip_thread.start()

    input("\nPress <Enter> again to STOP validation…")
    streaming_test = False
    tip_thread.join()

    # analyse collected samples
    tips = tip_thread._return if hasattr(tip_thread, "_return") else []
    if len(tips):
        centre = np.mean(tips, axis=0)
        deviations = np.linalg.norm(tips - centre, axis=1)
        print(f"\nCollected {len(tips)} tip samples.")
        print(f"Mean tip (mm): X={centre[0]:.2f}  Y={centre[1]:.2f}  Z={centre[2]:.2f}")
        print(f"Std dev: {np.std(deviations):.2f} mm  |  Max dev: {np.max(deviations):.2f} mm")
    else:
        print("No tip samples captured.")

    # ----------------------------------------------------------------------
    # --- 5.  Persist tip offset to disk ------------------------------------
    # ----------------------------------------------------------------------
    import json, os
    offset_path = pathlib.Path(__file__).with_name("tip_offset.json")
    try:
        with offset_path.open("w") as fp:
            json.dump({"tip_offset_mm": tip_offset.tolist()}, fp, indent=2)
        print(f"\nTip offset saved ➜ {offset_path}")
    except Exception as e:
        print(f"⚠️  Could not save tip offset: {e}")

    # ----------------------------------------------------------------------
    # --- 6.  Clean‑up ------------------------------------------------------
    # ----------------------------------------------------------------------
    ndicapy.ndiCommand(dev, "TSTOP ")
    ndicapy.ndiCloseNetwork(dev)
    print("Tracker closed – done.")