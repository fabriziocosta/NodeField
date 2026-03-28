"""Interpolation helpers used by graph-conditioning workflows."""

from __future__ import annotations

import numpy as np


def interpolate_integer_series(start, end, ts, minimum):
    values = np.rint([(1.0 - t) * start + t * end for t in ts]).astype(np.int64)
    return np.maximum(values, np.int64(minimum))


def scaled_slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between vectors on the hypersphere while blending magnitudes linearly."""
    mag0 = np.linalg.norm(v0)
    mag1 = np.linalg.norm(v1)

    v0_unit = v0 / mag0 if mag0 != 0 else v0
    v1_unit = v1 / mag1 if mag1 != 0 else v1

    dot = np.clip(np.dot(v0_unit, v1_unit), -1.0, 1.0)
    theta = np.arccos(dot)

    if theta < 1e-6:
        direction = (1 - t) * v0_unit + t * v1_unit
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm != 0 else direction
    else:
        sin_theta = np.sin(theta)
        direction = (
            np.sin((1 - t) * theta) * v0_unit +
            np.sin(t * theta) * v1_unit
        ) / sin_theta

    mag = (1 - t) * mag0 + t * mag1
    return direction * mag


def scaled_slerp_average(vectors: np.ndarray) -> np.ndarray:
    """Compute a magnitude-aware mean direction for a batch of vectors."""
    vs = np.asarray(vectors, dtype=float)
    mags = np.linalg.norm(vs, axis=1)
    unit_vs = np.zeros_like(vs)
    nonzero = mags > 0
    unit_vs[nonzero] = vs[nonzero] / mags[nonzero, None]

    avg_dir = unit_vs.sum(axis=0)
    norm = np.linalg.norm(avg_dir)
    if norm > 0:
        avg_dir /= norm

    avg_mag = mags.mean()
    return avg_dir * avg_mag
