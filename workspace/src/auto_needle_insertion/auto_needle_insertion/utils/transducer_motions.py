import math
from dataclasses import dataclass
from typing import List, Tuple

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


def random_small_perturbation_sequence(
    rot_range_deg: Tuple[float, float] = (-3.0, 3.0),
    sweep_range_mm: Tuple[float, float] = (-4.0, 4.0),
    slide_range_mm: Tuple[float, float] = (-4.0, 4.0),
    rng: np.random.Generator | None = None,
) -> List[Tuple[str, float]]:
    """Return a small perturbation sequence: rotation -> sweep -> slide.

    Defaults are intentionally conservative so the needle remains in view.

    Args:
        rot_range_deg: Inclusive min/max rotation about probe Y (degrees).
        sweep_range_mm: Inclusive min/max sweep translation along Z (mm).
        slide_range_mm: Inclusive min/max slide translation along X (mm).
        rng: Optional NumPy random generator for reproducible sampling.

    Returns:
        List of (motion, value) pairs ordered as rotation, sweep, slide.
    """
    rng = np.random.default_rng() if rng is None else rng
    rot = float(rng.uniform(*rot_range_deg))
    sweep = float(rng.uniform(*sweep_range_mm))
    slide = float(rng.uniform(*slide_range_mm))
    return [("rotation", rot), ("sweep", sweep), ("slide", slide)]


def apply_random_small_perturbation(
    T_probe: np.ndarray,
    rot_range_deg: Tuple[float, float] = (-3.0, 3.0),
    sweep_range_mm: Tuple[float, float] = (-4.0, 4.0),
    slide_range_mm: Tuple[float, float] = (-4.0, 4.0),
    rng: np.random.Generator | None = None,
) -> Tuple[List[np.ndarray], List[Tuple[str, float]]]:
    """Apply a small random rotation->sweep->slide perturbation to a probe pose.

    Args:
        T_probe: 4x4 homogeneous pose of the probe.
        rot_range_deg: Inclusive min/max rotation about probe Y (degrees).
        sweep_range_mm: Inclusive min/max sweep translation along Z (mm).
        slide_range_mm: Inclusive min/max slide translation along X (mm).
        rng: Optional NumPy random generator for reproducible sampling.

    Returns:
        (poses, sequence) where poses are the incremental probe poses after each
        step in the sampled motion list.
    """
    if T_probe.shape != (4, 4):
        raise ValueError("T_probe must be a 4x4 homogeneous matrix")
    seq = random_small_perturbation_sequence(
        rot_range_deg=rot_range_deg,
        sweep_range_mm=sweep_range_mm,
        slide_range_mm=slide_range_mm,
        rng=rng,
    )
    poses: List[np.ndarray] = []
    T_running = np.array(T_probe, dtype=float, copy=True)
    for motion, value in seq:
        T_step = transducer_motions(motion, value)
        T_running = T_running @ T_step
        poses.append(T_running)
    return poses, seq


@dataclass
class LocalDelta:
    """Pose delta in the *current EE local frame*."""
    dx: float; dy: float; dz: float      # meters
    droll: float; dpitch: float; dyaw: float  # radians
