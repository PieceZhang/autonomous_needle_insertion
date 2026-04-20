"""Pose representation utilities and conversions."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node


def quat_xyzw_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a ROS quaternion (x,y,z,w) to a 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=float)
    if not np.all(np.isfinite(q)):
        raise RuntimeError("Quaternion contains NaN/Inf")

    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise RuntimeError("Quaternion has zero norm")
    q = q / norm
    qx, qy, qz, qw = q

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def rotmat_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a unit quaternion (x, y, z, w) in ROS ordering."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise RuntimeError(f"Rotation matrix must be (3,3), got {R.shape}")
    if not np.all(np.isfinite(R)):
        raise RuntimeError("Rotation matrix contains NaN/Inf")

    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        raise RuntimeError("Rotation matrix is not orthonormal")

    tr = float(R[0, 0] + R[1, 1] + R[2, 2])

    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=float)
    if not np.all(np.isfinite(q)):
        raise RuntimeError("Quaternion contains NaN/Inf")

    n = float(np.dot(q, q))
    if n < 1e-12 or not np.isfinite(n):
        raise RuntimeError(f"Quaternion norm too small: {n}")

    q /= np.sqrt(n)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def quat_to_T(quat: Tuple[float, float, float, float, float, float, float]) -> np.ndarray:
    """Convert a pose (px,py,pz,qx,qy,qz,qw) to a 4x4 homogeneous transform."""
    px, py, pz, qx, qy, qz, qw = quat
    vals = np.array([px, py, pz, qx, qy, qz, qw], dtype=float)
    if not np.all(np.isfinite(vals)):
        raise RuntimeError(f"Pose contains NaN/Inf: {vals.tolist()}")

    R = quat_xyzw_to_rotmat(qx, qy, qz, qw)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = [px, py, pz]

    if not np.all(np.isfinite(T)):
        raise RuntimeError("Computed transform contains NaN/Inf")

    return T


def homogeneous_to_pose_msg(T: np.ndarray) -> Pose:
    """Convert a 4x4 homogeneous transform into a geometry_msgs/Pose message."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise RuntimeError(f"Transform must be (4,4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise RuntimeError("Transform contains NaN/Inf")
    if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=1e-8):
        raise RuntimeError(f"Transform last row must be [0 0 0 1], got {T[3, :]}")

    R = T[:3, :3]
    p = T[:3, 3]

    qx, qy, qz, qw = rotmat_to_quat_xyzw(R)

    pose = Pose()
    pose.position.x = float(p[0])
    pose.position.y = float(p[1])
    pose.position.z = float(p[2])
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def homogeneous_to_pose_stamped(
    T: np.ndarray,
    frame_id: str,
    node: Optional[Node] = None,
) -> PoseStamped:
    """Convert a 4x4 homogeneous transform into a geometry_msgs/PoseStamped."""
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    if node is not None:
        ps.header.stamp = node.get_clock().now().to_msg()
    else:
        ps.header.stamp = rclpy.clock.Clock().now().to_msg()

    ps.pose = homogeneous_to_pose_msg(T)
    return ps


def translation_scale(translation_in: str, translation_out: str) -> float:
    """Return the scale factor to convert translation units."""
    if translation_in == "mm" and translation_out == "m":
        return 1e-3
    if translation_in == "m" and translation_out == "mm":
        return 1e3
    if translation_in == translation_out:
        return 1.0
    raise ValueError(f"Unsupported translation conversion: {translation_in} -> {translation_out}")


def check_hmat(T: np.ndarray, name: str) -> None:
    """Validate a homogeneous transform matrix."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise RuntimeError(f"{name} must be shape (4,4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise RuntimeError(f"{name} contains NaN/Inf")
    if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=1e-8):
        raise RuntimeError(f"{name} last row must be [0 0 0 1], got {T[3, :]}")


def closest_rotation_svd(R: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix onto the closest rotation matrix."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise RuntimeError(f"Rotation matrix must be (3,3), got {R.shape}")
    if not np.all(np.isfinite(R)):
        raise RuntimeError("Rotation matrix contains NaN/Inf")

    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt

    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt

    return R_proj


def project_to_se3(T: np.ndarray) -> np.ndarray:
    """Project a homogeneous transform onto SE(3) by orthonormalizing rotation."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise RuntimeError(f"Transform must be (4,4), got {T.shape}")
    if not np.all(np.isfinite(T)):
        raise RuntimeError("Transform contains NaN/Inf")

    R_proj = closest_rotation_svd(T[:3, :3])
    t = T[:3, 3].copy()

    Tout = np.eye(4)
    Tout[:3, :3] = R_proj
    Tout[:3, 3] = t
    return Tout


def probe_from_transducer_origin_pose(
    T_probe_from_image: np.ndarray,
    T_top_from_image: np.ndarray,
    T_to_from_top: np.ndarray,
    *,
    translation_in: str = "mm",
    translation_out: str = "m",
    orthonormalize_rotation: bool = False,
):
    """Compute the transducer origin pose expressed in the probe frame."""
    check_hmat(T_probe_from_image, "T_probe_from_image")
    check_hmat(T_top_from_image, "T_top_from_image")
    check_hmat(T_to_from_top, "T_to_from_top")

    T_probe_from_image = np.asarray(T_probe_from_image, dtype=float)
    T_top_from_image = np.asarray(T_top_from_image, dtype=float)
    T_to_from_top = np.asarray(T_to_from_top, dtype=float)

    T_to_from_image = T_to_from_top @ T_top_from_image
    T_probe_from_to = T_probe_from_image @ np.linalg.inv(T_to_from_image)

    R = T_probe_from_to[:3, :3].copy()
    t = T_probe_from_to[:3, 3].copy()

    if orthonormalize_rotation:
        R = closest_rotation_svd(R)
        T_probe_from_to[:3, :3] = R

    q_xyzw = rotmat_to_quat_xyzw(R)

    scale = translation_scale(translation_in, translation_out)

    pose = {
        "position": {"x": float(t[0] * scale), "y": float(t[1] * scale), "z": float(t[2] * scale)},
        "orientation": {"x": float(q_xyzw[0]), "y": float(q_xyzw[1]), "z": float(q_xyzw[2]), "w": float(q_xyzw[3])},
    }

    return T_probe_from_to, pose

