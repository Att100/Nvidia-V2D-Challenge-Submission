"""Self-contained geometry and I/O helpers for hand surface alignment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


HAND_LABELS = ("left", "right")


def rigidify_camera_to_world(camera_to_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove uniform Sim(3) scale from camera-to-world pose matrices."""

    poses = np.asarray(camera_to_world, dtype=np.float64)
    linear = poses[..., :3, :3]
    u, singular_values, vh = np.linalg.svd(linear)
    rotation = u @ vh
    negative = np.linalg.det(rotation) < 0
    if np.any(negative):
        u = u.copy()
        u[..., :, -1] = np.where(negative[..., None], -u[..., :, -1], u[..., :, -1])
        rotation = u @ vh
    rigid = poses.copy()
    rigid[..., :3, :3] = rotation
    scales = np.prod(singular_values, axis=-1) ** (1.0 / 3.0)
    return rigid, scales


def world_pointmap_to_camera(
    world_points: np.ndarray, camera_to_world: np.ndarray
) -> np.ndarray:
    """Transform a world point map to fixed-scale OpenCV camera coordinates."""

    rigid_pose, _ = rigidify_camera_to_world(camera_to_world)
    world_to_camera = np.linalg.inv(rigid_pose)
    return np.asarray(world_points) @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]


def resize_instance_maps(instance_map: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize integer instance maps with nearest-neighbor interpolation."""

    return np.stack(
        [
            cv2.resize(frame.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
            for frame in instance_map
        ],
        axis=0,
    ).astype(np.int32, copy=False)


def project_camera_points(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project OpenCV camera-space points to pixels."""

    values = np.asarray(points, dtype=np.float64)
    projected = values @ np.asarray(intrinsics, dtype=np.float64).T
    pixels = np.full(values.shape[:-1] + (2,), np.nan, dtype=np.float64)
    valid = np.isfinite(projected).all(axis=-1) & (values[..., 2] > 1e-8)
    pixels[valid] = projected[valid, :2] / projected[valid, 2:3]
    return pixels.astype(np.float32)


def camera_points_to_world(
    points_camera: np.ndarray, camera_to_world: np.ndarray
) -> np.ndarray:
    """Transform OpenCV camera points into the rigid shared world frame."""

    rigid_pose, _ = rigidify_camera_to_world(camera_to_world)
    points = np.asarray(points_camera, dtype=np.float64)
    return (points @ rigid_pose[:3, :3].T + rigid_pose[:3, 3]).astype(np.float32)


def write_stage_manifest(
    output_dir: Path,
    *,
    stage: str,
    inputs: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    parameters: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    """Write the compatible stage-manifest schema without external utilities."""

    def normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        return value

    path = output_dir / "stage_manifest.json"
    path.write_text(
        json.dumps(
            {
                "stage": stage,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": normalize(inputs),
                "artifacts": normalize(artifacts),
                "parameters": normalize(parameters),
                "metrics": normalize(metrics),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path

