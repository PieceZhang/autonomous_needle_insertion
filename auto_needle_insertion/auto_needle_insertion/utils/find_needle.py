"""Helpers for needle alignment and centering in image planes."""

from __future__ import annotations

import numpy as np


def align_image_to_needle_axis(
    T_image_in_tracker: np.ndarray,
    needle_marker_origin_in_tracker: np.ndarray,
    needle_tip_pos_in_tracker: np.ndarray,
    position_unit: str = "m",
) -> np.ndarray:
    """Compute the target pose in tracker after aligning the image plane to the needle axis."""
    T = np.asarray(T_image_in_tracker, dtype=float)
    if T.shape != (4, 4):
        raise RuntimeError(f"T_image_in_tracker must be shape (4,4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise RuntimeError("T_image_in_tracker contains NaN/Inf")

    if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=1e-9):
        raise RuntimeError(f"T_image_in_tracker last row must be [0 0 0 1], got {T[3, :]}")

    R_cur = T[:3, :3]

    y_keep = R_cur[:, 1]
    y_norm = float(np.linalg.norm(y_keep))
    if y_norm < 1e-12 or not np.isfinite(y_norm):
        raise RuntimeError("Image plane current Y axis is invalid")
    y_keep = y_keep / y_norm

    P0 = np.asarray(needle_marker_origin_in_tracker, dtype=float).reshape(-1)
    P1 = np.asarray(needle_tip_pos_in_tracker, dtype=float).reshape(-1)
    if P0.shape != (3,) or P1.shape != (3,):
        raise RuntimeError("needle_marker_origin_in_tracker and needle_tip_pos_in_tracker must be shape (3,)")
    if not (np.all(np.isfinite(P0)) and np.all(np.isfinite(P1))):
        raise RuntimeError("Needle points contain NaN/Inf")

    if position_unit not in ("m", "mm"):
        raise ValueError("position_unit must be 'm' or 'mm'")

    v = P1 - P0
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-9 or not np.isfinite(v_norm):
        raise RuntimeError("Needle marker origin and tip are too close (degenerate line)")

    v_proj = v - float(np.dot(v, y_keep)) * y_keep
    proj_norm = float(np.linalg.norm(v_proj))

    x_cur = R_cur[:, 0]

    if proj_norm < 1e-9:
        x_target = x_cur - float(np.dot(x_cur, y_keep)) * y_keep
        x_n = float(np.linalg.norm(x_target))
        if x_n < 1e-9:
            raise RuntimeError("Cannot construct X axis: current X is parallel to Y")
        x_target = x_target / x_n
    else:
        x_target = v_proj / proj_norm
        if float(np.dot(x_target, x_cur)) < 0.0:
            x_target = -x_target

    z_target = np.cross(x_target, y_keep)
    z_n = float(np.linalg.norm(z_target))
    if z_n < 1e-9 or not np.isfinite(z_n):
        raise RuntimeError("Cannot construct Z axis (degenerate cross product)")
    z_target = z_target / z_n

    x_target = np.cross(y_keep, z_target)
    x_n2 = float(np.linalg.norm(x_target))
    if x_n2 < 1e-9 or not np.isfinite(x_n2):
        raise RuntimeError("Cannot re-orthonormalize X axis")
    x_target = x_target / x_n2

    R_tgt = np.column_stack((x_target, y_keep, z_target))

    p_tgt = P0

    T_tgt = np.eye(4, dtype=float)
    T_tgt[:3, :3] = R_tgt
    T_tgt[:3, 3] = p_tgt

    if not np.all(np.isfinite(T_tgt)):
        raise RuntimeError("Computed target pose contains NaN/Inf")

    return T_tgt


def center_needle_in_image(
    T_image_in_tracker: np.ndarray,
    needle_marker_origin_in_tracker: np.ndarray,
    needle_tip_pos_in_tracker: np.ndarray,
    x_center_in_plane: float = 0.0,
    y_target_in_plane: float = 0.0,
    reference: str = "tip",
    position_unit: str = "m",
) -> np.ndarray:
    """Translate the image plane (no rotation) so the needle tip lands at a desired (x,y) in the plane frame."""
    if position_unit not in ("m", "mm"):
        raise ValueError("position_unit must be 'm' or 'mm'")

    T_cur = np.asarray(T_image_in_tracker, dtype=float)
    if T_cur.shape != (4, 4):
        raise RuntimeError(f"T_image_in_tracker must be shape (4,4), got {T_cur.shape}")
    if not np.all(np.isfinite(T_cur)):
        raise RuntimeError("T_image_in_tracker contains NaN/Inf")
    if not np.allclose(T_cur[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise RuntimeError("T_image_in_tracker last row must be [0, 0, 0, 1]")

    R_cur = T_cur[:3, :3]
    p_cur = T_cur[:3, 3]

    if not np.allclose(R_cur.T @ R_cur, np.eye(3), atol=1e-6):
        raise RuntimeError("Rotation part of T_image_in_tracker is not orthonormal")

    P0 = np.asarray(needle_marker_origin_in_tracker, dtype=float).reshape(-1)
    P1 = np.asarray(needle_tip_pos_in_tracker, dtype=float).reshape(-1)
    if P0.shape != (3,) or P1.shape != (3,):
        raise RuntimeError("Needle points must be shape (3,)")
    if not (np.all(np.isfinite(P0)) and np.all(np.isfinite(P1))):
        raise RuntimeError("Needle points contain NaN/Inf")

    ref = reference.strip().lower()
    if ref not in ("tip", "needle_tip"):
        raise ValueError("center_needle_in_image is intended to center the needle tip; set reference='tip'.")
    P_ref = P1

    p_ref_plane = R_cur.T @ (P_ref - p_cur)
    if not np.all(np.isfinite(p_ref_plane)):
        raise RuntimeError("Reference point projection produced NaN/Inf")

    tol = 2e-3 if position_unit == "m" else 2.0
    if abs(float(p_ref_plane[2])) > tol:
        raise RuntimeError(
            f"Reference point is not on the image plane (z={p_ref_plane[2]:.6g} {position_unit}). "
            "Make sure you have already applied the plane alignment motion."
        )

    x_ref = float(p_ref_plane[0])
    y_ref = float(p_ref_plane[1])

    dx = x_ref - float(x_center_in_plane)
    dy = y_ref - float(y_target_in_plane)

    x_axis_tracker = R_cur[:, 0]
    y_axis_tracker = R_cur[:, 1]
    p_tgt = p_cur + dx * x_axis_tracker + dy * y_axis_tracker

    T_tgt = np.eye(4, dtype=float)
    T_tgt[:3, :3] = R_cur
    T_tgt[:3, 3] = p_tgt

    if not np.all(np.isfinite(T_tgt)):
        raise RuntimeError("Computed target pose contains NaN/Inf")

    return T_tgt


def needle_segment_in_image(
    T_image_in_tracker: np.ndarray,
    needle_marker_origin_in_tracker: np.ndarray,
    needle_tip_pos_in_tracker: np.ndarray,
    image_width: float,
    image_height: float,
    position_unit: str = "m",
    plane_tolerance: float | None = None,
    margin: float = 0.0,
) -> bool:
    """Check if any portion of the needle segment intersects the image plane bounds.

    Args:
        T_image_in_tracker: 4x4 pose of the image plane in tracker frame.
        needle_marker_origin_in_tracker: 3D point on the needle axis (tracker frame).
        needle_tip_pos_in_tracker: 3D needle tip position (tracker frame).
        image_width: Image width in the plane frame (same units as points).
        image_height: Image height in the plane frame (same units as points).
        position_unit: "m" or "mm", used only for default plane tolerance.
        plane_tolerance: Allowed distance from plane (same unit as points). Defaults
            to 2e-3 m or 2.0 mm, matching center_needle_in_image.
        margin: Optional margin to shrink the visible bounds (same unit as points).

    Returns:
        True if any part of the needle segment lies within the image bounds on
        the plane, otherwise False.
    """
    if position_unit not in ("m", "mm"):
        raise ValueError("position_unit must be 'm' or 'mm'")
    if image_width <= 0.0 or image_height <= 0.0:
        raise ValueError("image_width and image_height must be > 0")
    if margin < 0.0:
        raise ValueError("margin must be >= 0")

    T = np.asarray(T_image_in_tracker, dtype=float)
    if T.shape != (4, 4):
        raise RuntimeError(f"T_image_in_tracker must be shape (4,4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise RuntimeError("T_image_in_tracker contains NaN/Inf")
    if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise RuntimeError("T_image_in_tracker last row must be [0, 0, 0, 1]")

    R = T[:3, :3]
    p = T[:3, 3]

    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise RuntimeError("Rotation part of T_image_in_tracker is not orthonormal")

    P0 = np.asarray(needle_marker_origin_in_tracker, dtype=float).reshape(-1)
    P1 = np.asarray(needle_tip_pos_in_tracker, dtype=float).reshape(-1)
    if P0.shape != (3,) or P1.shape != (3,):
        raise RuntimeError("Needle points must be shape (3,)")
    if not (np.all(np.isfinite(P0)) and np.all(np.isfinite(P1))):
        raise RuntimeError("Needle points contain NaN/Inf")

    tol = plane_tolerance
    if tol is None:
        tol = 2e-3 if position_unit == "m" else 2.0
    tol = float(tol)
    if tol < 0.0:
        raise ValueError("plane_tolerance must be >= 0")

    P0_plane = R.T @ (P0 - p)
    P1_plane = R.T @ (P1 - p)

    def _within_bounds(x: float, y: float) -> bool:
        half_w = image_width * 0.5 - margin
        half_h = image_height * 0.5 - margin
        if half_w < 0.0 or half_h < 0.0:
            return False
        return (-half_w <= x <= half_w) and (-half_h <= y <= half_h)

    z0 = float(P0_plane[2])
    z1 = float(P1_plane[2])

    # Endpoint visibility (near plane and within bounds).
    if abs(z0) <= tol and _within_bounds(float(P0_plane[0]), float(P0_plane[1])):
        return True
    if abs(z1) <= tol and _within_bounds(float(P1_plane[0]), float(P1_plane[1])):
        return True

    # If the segment crosses the plane (z=0), check intersection point.
    if (z0 == 0.0 and z1 == 0.0) or (z0 * z1 < 0.0):
        denom = z0 - z1
        if abs(denom) > 1e-12:
            t = z0 / (z0 - z1)
            x_int = float(P0_plane[0] + t * (P1_plane[0] - P0_plane[0]))
            y_int = float(P0_plane[1] + t * (P1_plane[1] - P0_plane[1]))
            if _within_bounds(x_int, y_int):
                return True

    return False
