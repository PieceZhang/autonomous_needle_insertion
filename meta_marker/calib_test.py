import ndicapy
from six import int2byte
import numpy as np
import time
import sys
import textwrap
import pathlib
import math
import threading
import csv
import datetime
from collections import deque

# ---------------------------------------------------------------
# Load previously calibrated tip offset if available
# ---------------------------------------------------------------
import json
TIP_OFFSET = np.zeros(3)
_offset_file = pathlib.Path(__file__).with_name("tip_offset.json")
if _offset_file.exists():
    try:
        with _offset_file.open() as fp:
            data = json.load(fp)
        TIP_OFFSET = np.array(data.get("tip_offset_mm", [0, 0, 0]), dtype=float)
        print(f"Loaded tip offset (mm): {TIP_OFFSET}")
    except Exception as e:
        print(f"⚠️  Could not read tip_offset.json – using zero offset ({e})")
else:
    print("No tip_offset.json found – using zero offset.")

# Ring buffer for last N tip measurements during streaming
TIP_BUFFER_SIZE = 10
tip_buffer = deque(maxlen=TIP_BUFFER_SIZE)


def connect_to(ip_addr: str="192.168.33.190", port: int=8765):
    dev = ndicapy.ndiOpenNetwork(ip_addr, port)
    if not dev:
        raise IOError("Cannot open network device")
    reply = ndicapy.ndiCommand(dev, 'INIT ')      # Set the system to Setup mode 
    if not reply.startswith('OKAY'):
        raise RuntimeError(f'INIT failed: {reply}')
    else:
        print('Initialization success')

    return dev


def set_up_tool(dev, rom_path: str="8700340.rom"):
    # Resolve the ROM path relative to this script, so it works
    # no matter what the current working directory is.
    rom_path = (pathlib.Path(__file__).resolve().parent / rom_path).resolve()
    # Request a handle for a wireless tool
    #    Syntax: PHRQ <HardwareDevice><SystemType><ToolType><PortNumber><DummyTool>
    #    Use wildcards (*) for everything except ToolType=1 (wireless) and Dummy=**
    reply = ndicapy.ndiCommand(dev, 'PHRQ *********1****')
    if reply.startswith('ERROR'):
        raise RuntimeError(f'PHRQ failed when requesting a tool handle: {reply}')
    # Extract the first two hex chars as the tool handle
    handle_hex = reply[:2]
    print(f'Acquired port handle as {handle_hex}')
    # handle = int(handle_hex, 16)

    # Load the wireless tool ROM into that handle
    load_rom(dev, handle_hex, str(rom_path))
    print(f'Loaded tool ROM from {rom_path}')

    # Initialize the handle—PENA will implicitly init if needed
    reply = ndicapy.ndiCommand(dev, f'PENA {handle_hex}D')   # enable & init the tool as Dynamic
    if not reply.startswith('OKAY'):
        raise RuntimeError(f'PENA failed: {reply}')
    else:
        print(f'Port {handle_hex} enabled')
    # ndicapy.ndiCommand(dev, f'PENA 01D')  # “D” = dynamic tool; also initializes 

    return handle_hex


def load_rom(device, handle_hex: str, rom_path: str):
    """Upload a wireless-tool ROM in 64‑byte chunks with PVWR. `rom_path` must be an absolute path."""
    data_hex = pathlib.Path(rom_path).read_bytes().hex().upper()
    # split into 128-hex-char chunks (=64 bytes)
    chunks = textwrap.wrap(data_hex, 128)
    # pad last chunk if shorter
    if len(chunks[-1]) < 128:
        chunks[-1] = chunks[-1].ljust(128, '0')

    for idx, chunk in enumerate(chunks):
        addr = f'{idx*0x40:04X}'          # 0x0000, 0x0040, 0x0080, …
        cmd  = f'PVWR {handle_hex}{addr}{chunk}'
        reply = ndicapy.ndiCommand(device, cmd)
        if not reply.startswith('OKAY'):
            raise RuntimeError(f'PVWR failed at {addr}: {reply}')


def get_tool_pose(dev, port_byte):
    """
    Query the tracker for the current pose of the given handle.
    Returns a 4×4 homogeneous matrix **or** None if the tool is not visible
    (the Combined‑API reports MISSING / DISABLED).
    """
    # Issue BX command for this single port
    ndicapy.ndiCommand(dev, "BX 0001")
    q = ndicapy.ndiGetBXTransform(dev, port_byte)

    # When the tool is out of view, ndicapy returns an empty list or raises,
    # depending on firmware––handle both cases.
    if not q or len(q) < 8:
        return None

    # Convert quaternion+XYZ to 4×4
    mat_flat = ndicapy.ndiTransformToMatrixd(q)
    pose = np.transpose(np.reshape(mat_flat, (4, 4)))

    # Detect 'MISSING' flag (quality == 0) even if numbers were returned
    if isinstance(q[-1], (int, float)) and q[-1] == 0:
        return None

    return pose


def apply_tip_offset(pose4x4: np.ndarray) -> np.ndarray:
    """
    Apply the calibrated TIP_OFFSET (in the tool frame) to the 4×4 pose matrix,
    returning the tool-tip position in world coordinates.
    """
    R = pose4x4[:3, :3]
    p = pose4x4[:3, 3]
    return R @ TIP_OFFSET + p


def stream_positions():
    global streaming
    while streaming:
        pose = get_tool_pose(dev, port_byte)
        if pose is None:
            print("\rTool missing...                           ", end='')
        else:
            tip = apply_tip_offset(pose)
            tip_buffer.append(tip)
            print(f"\rStreaming Tip: X={tip[0]:.2f} Y={tip[1]:.2f} Z={tip[2]:.2f}", end='')
        time.sleep(0.1)


def parse_pos_field(field: str) -> float:
    """
    Convert a 7-char string like '+012345' into a float:
      +012345 → +123.45 mm
      -001234 → -12.34 mm
    """
    sign = 1 if field[0] == '+' else -1
    # take the next six digits as an integer, then divide by 100
    value = int(field[1:]) / 100.0
    return sign * value


def parse_tx1000_reply(reply: str):
    """
    Parse the ASCII reply from 'TX 1000' into a list of (x, y, z) floats.

    reply format on success:
      <NumMarkers><OutOfVolHex..><Txn><Tyn><Tzn>...[SystemStatus][CRC16]\r

    We:
      • read NumMarkers (2 hex chars),
      • skip the Out-of-Volume flags (1 hex char per 4 markers),
      • then for each marker pull out 3×7 characters and convert via parse_pos_field().
    """
    # Strip trailing CRC16 (4-char CRC) after cleaning newlines
    header, _lf, option1000 = reply.rpartition('\n')  # works even if there is only one LF
    core = option1000.rstrip('\r')
    n_markers = int(core[0:2], 16)

    # Length of the “Out of Volume” block, in hex chars
    ov_len = math.ceil(n_markers / 4)
    # Skip past NumMarkers + OOV flags
    data_start = 2 + ov_len
    data = core[data_start:]

    # Parse each marker: 21 chars = 3 fields × 7 chars each
    poses = []
    for i in range(n_markers):
        chunk = data[i * 21 : (i + 1) * 21]
        tx = parse_pos_field(chunk[0:7])
        ty = parse_pos_field(chunk[7:14])
        tz = parse_pos_field(chunk[14:21])
        poses.append((tx, ty, tz))

    return poses


# ---------------------------------------------------------------
# Reusable two-caveat measurement for one marker
# ---------------------------------------------------------------
def run_two_caveat_measurement_for_marker(marker_id: int):
    """Runs the interactive two-caveat workflow once and returns
    (tip1, tip2, midpoint) as three np.ndarray(3,).
    Uses global: dev, port_byte, stream_positions, TIP_BUFFER_SIZE, tip_buffer
    """
    global streaming, thread

    def prompt_phase_local(p):
        print(f"\n[Marker {marker_id}] Place the probe in caveat {p}.")
        print("Streaming started automatically. Press 'r' to record, 'q' to abort this marker.\n")

    streaming = False
    thread = None
    recorded_positions_local = []
    phase = 1

    prompt_phase_local(phase)

    # auto-start streaming
    streaming = True
    tip_buffer.clear()
    thread = threading.Thread(target=stream_positions, daemon=True)
    thread.start()

    while True:
        cmd = input(f"[Marker {marker_id}] Command [r=record, q=quit]: ").strip().lower()
        if cmd == 'r':
            streaming = False
            thread.join()

            if len(tip_buffer) == 0:
                print("⚠️  No samples in buffer – cannot record. Streaming resumes. Try again.")
                streaming = True
                thread = threading.Thread(target=stream_positions, daemon=True)
                thread.start()
                continue

            tip = np.mean(np.array(tip_buffer), axis=0)
            recorded_positions_local.append(tip)
            tip_buffer.clear()
            print(f"Recorded Tip #{len(recorded_positions_local)} (Caveat {phase}): "
                  f"X={tip[0]:.2f} Y={tip[1]:.2f} Z={tip[2]:.2f}")

            if phase == 1:
                phase = 2
                prompt_phase_local(phase)
                streaming = True
                thread = threading.Thread(target=stream_positions, daemon=True)
                thread.start()
                print("Streaming resumed. Press 'r' to record, 'q' to abort this marker.")
            else:
                # done for this marker
                if len(recorded_positions_local) >= 2:
                    mid = (recorded_positions_local[0] + recorded_positions_local[1]) / 2.0
                    return recorded_positions_local[0], recorded_positions_local[1], mid
                else:
                    return None, None, None
        elif cmd == 'q':
            print(f"[Marker {marker_id}] aborted by user.")
            return None, None, None
        else:
            print("Unknown command. Use 'r' (record) or 'q' (quit).")


def measure_error_for_single_visible_marker(mid: np.ndarray):
    """Reads one TX 1000 frame (or several) and returns (error_vec, error_norm, measured_xyz).
    Assumes only ONE marker is visible. Returns None if none is found."""
    raw = ndicapy.ndiCommand(dev, 'TX 1000').strip()
    poses = parse_tx1000_reply(raw)
    if len(poses) == 0:
        return None, None, None
    # take the first pose in the frame (the only marker in range)
    x, y, z = poses[0]
    m = np.array([x, y, z])
    error_vec = mid - m
    error_norm = np.linalg.norm(error_vec)
    return error_vec, float(error_norm), m


if __name__ == "__main__":
    # Connect & INIT
    dev = connect_to("192.168.33.190", 8765)
    # ndicapy.ndiCommand(dev, 'BEEP 1')

    tool_handle = set_up_tool(dev)
    # Get tool position
    # Convert hex handle to a single byte
    handle = int(tool_handle, 16)
    port_byte = int2byte(handle)
    time.sleep(0.1)

    # Start tracking for stray markers and read a pose
    reply = ndicapy.ndiCommand(dev, 'TSTART ')            # enter Tracking mode 
    if not reply.startswith('OKAY'):
        raise RuntimeError(f'TSTART failed: {reply}')
    else:
        print('Start tracking')

    # ---------------------------------------------------------------
    # Repeat two-caveat measurement for 10 markers (sequentially)
    # ---------------------------------------------------------------
    all_rows = []  # (marker_id, tip1_x,y,z, tip2_x,y,z, mid_x,y,z, err_x,y,z, err_norm)

    for marker_id in range(1, 11):
        tip1, tip2, mid = run_two_caveat_measurement_for_marker(marker_id)
        if tip1 is None:
            print(f"Marker {marker_id}: no data recorded, skipping.")
            continue

        # attempt to measure the error for this marker: only one marker should be visible now
        # You can repeat this a few times and average; here we do it once for brevity
        error_vec, error_norm, measured_xyz = measure_error_for_single_visible_marker(mid)
        if error_vec is None:
            print(f"Marker {marker_id}: could not obtain a TX1000 pose for error calculation.")
            error_vec = np.array([np.nan, np.nan, np.nan])
            error_norm = np.nan
            measured_xyz = np.array([np.nan, np.nan, np.nan])
        print(f"Error vector: {error_vec}, Error norm: {error_norm:.2f}")

        all_rows.append([
            marker_id,
            *tip1.tolist(), *tip2.tolist(), *mid.tolist(),
            *error_vec.tolist(), error_norm,
            *measured_xyz.tolist()
        ])

    # ---------------------------------------------------------------
    # Save per-marker results
    # ---------------------------------------------------------------
    csv_path = pathlib.Path(__file__).with_name("marker_errors.csv")
    try:
        with csv_path.open('w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "marker_id",
                "tip1_x", "tip1_y", "tip1_z",
                "tip2_x", "tip2_y", "tip2_z",
                "mid_x", "mid_y", "mid_z",
                "err_x", "err_y", "err_z", "err_norm",
                "meas_x", "meas_y", "meas_z"
            ])
            writer.writerows(all_rows)
        print(f"\nSaved per-marker results to {csv_path}")
    except Exception as e:
        print(f"\n⚠️  Failed to write per-marker CSV: {e}")

    # Cleanup
    ndicapy.ndiCommand(dev, 'TSTOP ')             # stop tracking 
    ndicapy.ndiCloseNetwork(dev)                  # close device 
    print('\nDevice closed')
