import math
import numpy as np

_MOTION_MAP = {
    "slide": "x", "x": "x",
    "compression": "y", "y": "y",
    "sweep": "z", "z": "z",
    "rock": "rz", "rz": "rz",
    "fan": "rx", "tilt": "rx", "rx": "rx",
    "rotation": "ry", "ry": "ry",
}

def _rot_matrix(axis: str, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    if axis == "rx":
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, c, -s],
                         [0.0, s,  c]], dtype=float)
    if axis == "ry":
        return np.array([[ c, 0.0, s],
                         [0.0, 1.0, 0.0],
                         [-s, 0.0, c]], dtype=float)
    # axis == "rz"
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def transducer_motions(motion: str, value: float) -> np.ndarray:
    """Return a 4x4 transform for the given motion.

    Args:
        motion: motion type or alias ('slide', 'sweep', 'compression', 'rock', 'fan', 'tilt', 'rotation', or axis tags).
        value: translation distance in millimeters (for x/y/z) or rotation angle in degrees (for rx/ry/rz).

    Returns:
        4x4 numpy.ndarray homogeneous transform.

    Raises:
        ValueError: on unknown motion type or non-finite value.
    """
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    key = motion.strip().lower()
    if key not in _MOTION_MAP:
        raise ValueError(f"Unsupported motion type: {motion}")
    kind = _MOTION_MAP[key]

    T = np.eye(4, dtype=float)
    if kind in ("x", "y", "z"):
        dist_m = value / 1000.0  # mm -> m
        idx = {"x": 0, "y": 1, "z": 2}[kind]
        T[idx, 3] = dist_m
    else:  # rotations rx/ry/rz
        angle_rad = math.radians(value)
        T[:3, :3] = _rot_matrix(kind, angle_rad)
    return T

def compose_transducer_motions(sequence) -> np.ndarray:
    """Apply an ordered sequence of (motion, value) pairs and return the combined 4x4 transform.

    Args:
        sequence: iterable of (motion, value) where motion is any alias accepted by
                  ``transducer_motions`` and value is distance (mm) for translations
                  or angle (deg) for rotations.

    Returns:
        4x4 numpy.ndarray homogeneous transform representing all steps applied in order.

    Raises:
        ValueError: if any element is not a 2-tuple/list or motion is unsupported.
    """
    T = np.eye(4, dtype=float)
    try:
        iterator = iter(sequence)
    except Exception as e:
        raise ValueError("sequence must be iterable of (motion, value)") from e

    for idx, item in enumerate(iterator):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError(f"sequence element {idx} must be a (motion, value) pair")
        motion, value = item
        T_step = transducer_motions(motion, value)
        T = T @ T_step
    return T
