"""Pipeline-local point-cloud and compact-mask serialization helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def pack_exclusion_masks(masks: np.ndarray) -> np.ndarray:
    """Pack boolean F,H,W masks into one bit per pixel."""

    if masks.ndim != 3:
        raise ValueError("Exclusion masks must have shape F,H,W")
    return np.packbits(masks.reshape(len(masks), -1), axis=1)


def write_pointcloud_ply(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    *,
    track_ids: np.ndarray | None = None,
) -> None:
    """Write the binary PLY schema used by diagnostic dynamic point clouds."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = len(points)
    if colors.shape != (count, 3) or confidence.shape != (count,):
        raise ValueError("Inconsistent point-cloud field sizes")
    dtype: list[tuple[str, str]] = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("confidence", "f4"),
    ]
    if track_ids is not None:
        dtype.append(("track_id", "i4"))
    vertices = np.empty(count, dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    vertices["confidence"] = confidence
    if track_ids is not None:
        vertices["track_id"] = track_ids
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(output))

