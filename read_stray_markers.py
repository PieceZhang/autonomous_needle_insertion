import ndicapy
import numpy as np
import time
import math
import argparse
from typing import List, Tuple
from meta_marker.calib_test import connect_to, set_up_tool

def connect_to(ip_addr: str, port: int):
    dev = ndicapy.ndiOpenNetwork(ip_addr, port)
    if not dev:
        raise IOError("Cannot open network device")
    reply = ndicapy.ndiCommand(dev, 'INIT ')
    if not reply.startswith('OKAY'):
        raise RuntimeError(f'INIT failed: {reply}')
    return dev

def parse_pos_field(field: str) -> float:
    """'+012345' -> +123.45 mm ; '-001234' -> -12.34 mm"""
    sign = 1.0 if field[0] == '+' else -1.0
    return sign * (int(field[1:]) / 100.0)

def parse_tx1000_reply(reply: str) -> List[Tuple[float, float, float]]:
    """
    Parse ASCII reply of 'TX 1000' into a list of (x,y,z) in mm.
    Format (simplified):
      <NumMarkers(hex)><OOV flags> <marker triplets of 3x7 chars> ...
    """
    # reply may contain \r\n; keep the last line (the TX 1000 payload)
    header, _lf, option1000 = reply.rpartition('\n')  # works even if there is only one LF
    core = option1000.rstrip('\r')
    # core = reply.strip().splitlines()[-1]
    n_markers = int(core[0:2], 16)

    ov_len = math.ceil(n_markers / 4)  # out-of-volume bitfield length in hex chars
    data_start = 2 + ov_len
    data = core[data_start:]

    poses = []
    for i in range(n_markers):
        chunk = data[i * 21 : (i + 1) * 21]
        if len(chunk) < 21:
            break
        x = parse_pos_field(chunk[0:7])
        y = parse_pos_field(chunk[7:14])
        z = parse_pos_field(chunk[14:21])
        poses.append((x, y, z))
    return poses

def main():
    ap = argparse.ArgumentParser(
        description="Read and print positions of up to 9 stray markers via TX 1000.")
    ap.add_argument("--ip", default="192.168.33.190", help="Tracker IP")
    ap.add_argument("--port", type=int, default=8765, help="Tracker port")
    ap.add_argument("--rate", type=float, default=5.0,
                    help="Print rate in Hz (default 5)")
    ap.add_argument("--frames", type=int, default=0,
                    help="Number of frames to read (0 = infinite)")
    args = ap.parse_args()

    dev = connect_to(args.ip, args.port)
    handle_hex = set_up_tool(dev, "8700340.rom")
    try:
        # Enter tracking mode
        reply = ndicapy.ndiCommand(dev, 'TSTART ')
        if not reply.startswith('OKAY'):
            raise RuntimeError(f'TSTART failed: {reply}')
            
        ndicapy.ndiCommand(dev, 'BEEP 1')

        print("Streaming stray markers (TX 1000). Press Ctrl+C to stop.")
        period = 1.0 / args.rate
        count = 0

        while True:
            t0 = time.time()
            raw = ndicapy.ndiCommand(dev, 'TX 1000').strip()
            markers = parse_tx1000_reply(raw)

            # keep only the first 9
            markers = markers[:9]
            if markers:
                print(f"Frame {count:05d} | {len(markers)} marker(s):")
                for i, (x, y, z) in enumerate(markers, start=1):
                    print(f"  M{i:02d}: X={x:8.2f}  Y={y:8.2f}  Z={z:8.2f} mm")
            else:
                print(f"Frame {count:05d} | no markers")

            count += 1
            if args.frames and count >= args.frames:
                break

            # simple rate control
            dt = time.time() - t0
            to_sleep = period - dt
            if to_sleep > 0:
                time.sleep(to_sleep)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        ndicapy.ndiCommand(dev, 'TSTOP ')
        ndicapy.ndiCloseNetwork(dev)
        print("Device closed.")

if __name__ == "__main__":
    main()