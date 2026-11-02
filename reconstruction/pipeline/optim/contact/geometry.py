"""Geometry utilities shared by contact optimization modules and tools."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _skew_torch(vectors: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vectors[..., 0])
    x, y, z = vectors.unbind(dim=-1)
    return torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1).reshape(
        vectors.shape[:-1] + (3, 3)
    )


def axis_angle_to_matrix(vectors: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues conversion with stable small-angle coefficients."""
    squared = (vectors * vectors).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(squared.clamp_min(1e-16))
    coefficient_a = torch.where(
        squared < 1e-8,
        1.0 - squared / 6.0 + squared * squared / 120.0,
        torch.sin(theta) / theta.clamp_min(1e-8),
    )
    coefficient_b = torch.where(
        squared < 1e-8,
        0.5 - squared / 24.0 + squared * squared / 720.0,
        (1.0 - torch.cos(theta)) / squared.clamp_min(1e-8),
    )
    skew = _skew_torch(vectors)
    identity = torch.eye(3, dtype=vectors.dtype, device=vectors.device)
    identity = identity.expand(vectors.shape[:-1] + (3, 3))
    return (
        identity
        + coefficient_a[..., None] * skew
        + coefficient_b[..., None] * (skew @ skew)
    )


@dataclass(frozen=True)
class PenetrationMetrics:
    """Approximate hand/object penetration measured in object coordinates."""

    loss: float
    maximum_depth: float
    penetrating_fraction: float


@dataclass(frozen=True)
class VoxelCollisionField:
    """Filled voxel proxy for collision queries on a non-watertight mesh."""

    voxel_grid: trimesh.voxel.VoxelGrid
    surface_tree: cKDTree
    pitch: float

    @classmethod
    def from_mesh(
        cls,
        mesh: trimesh.Trimesh,
        surface_samples: np.ndarray,
        *,
        pitch: float = 0.003,
    ) -> "VoxelCollisionField":
        if pitch <= 0:
            raise ValueError("Voxel pitch must be positive")
        voxel_grid = mesh.voxelized(pitch).fill()
        return cls(
            voxel_grid=voxel_grid,
            surface_tree=cKDTree(np.asarray(surface_samples, dtype=np.float64)),
            pitch=float(pitch),
        )

    def penetration_metrics(
        self,
        points_object: np.ndarray,
        *,
        tolerance: float = 0.003,
    ) -> PenetrationMetrics:
        """Return a robust loss for vertices deeper than the surface tolerance."""

        points = np.asarray(points_object, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_object must have shape N,3")
        if tolerance < 0:
            raise ValueError("Penetration tolerance must be non-negative")
        if not len(points):
            return PenetrationMetrics(0.0, 0.0, 0.0)
        inside = np.asarray(self.voxel_grid.is_filled(points), dtype=bool)
        depths = np.zeros(len(points), dtype=np.float64)
        if np.any(inside):
            depths[inside] = self.surface_tree.query(points[inside], workers=-1)[0]
        violation = np.maximum(depths - tolerance, 0.0)
        positive = violation[violation > 0]
        if not len(positive):
            return PenetrationMetrics(
                loss=0.0,
                maximum_depth=float(depths.max(initial=0.0)),
                penetrating_fraction=0.0,
            )
        # A broad upper-tail statistic prevents a single vertex from dominating,
        # while still penalizing collision clusters smaller than 10% of the hand.
        loss = float(np.quantile(positive, 0.75) + 0.25 * np.quantile(positive, 0.95))
        return PenetrationMetrics(
            loss=loss,
            maximum_depth=float(depths.max(initial=0.0)),
            penetrating_fraction=float(np.mean(violation > 0)),
        )


FINGER_JOINTS = {
    "thumb": (2, 3, 4),
    "index": (6, 7, 8),
    "middle": (10, 11, 12),
    "ring": (14, 15, 16),
    "little": (18, 19, 20),
    "pinky": (18, 19, 20),
}


PALM_CENTER_JOINTS = (0, 5, 9, 13, 17)


def _aggregate(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool).reshape(-1)]
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "median": None, "p90": None, "mean": None, "max": None}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _select_finger_vertices(
    vertices: np.ndarray,
    joints: np.ndarray,
    joint_ids: tuple[int, ...],
    count: int,
) -> np.ndarray:
    finger_joints = joints[np.asarray(joint_ids, dtype=np.int64)]
    distance = np.linalg.norm(
        vertices[:, None, :] - finger_joints[None, :, :], axis=-1
    ).min(axis=1)
    return np.argsort(distance)[: min(max(1, int(count)), len(vertices))]


def _select_palm_vertices(
    vertices: np.ndarray,
    joints: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select a compact MANO surface patch around the anatomical palm center."""

    palm_center = np.mean(
        joints[np.asarray(PALM_CENTER_JOINTS, dtype=np.int64)], axis=0
    )
    distance = np.linalg.norm(vertices - palm_center[None], axis=1)
    return np.argsort(distance)[: min(max(1, int(count)), len(vertices))]


def _trajectory_smoothness(poses: np.ndarray) -> dict[str, Any]:
    """Measure translational and angular acceleration of an SE(3) sequence."""

    translation = np.asarray(poses[:, :3, 3], dtype=np.float64)
    translation_acceleration = np.linalg.norm(
        translation[2:] - 2.0 * translation[1:-1] + translation[:-2], axis=1
    )
    rotation = np.asarray(poses[:, :3, :3], dtype=np.float64)
    relative = np.einsum("fij,fkj->fik", rotation[1:], rotation[:-1])
    angular_velocity = Rotation.from_matrix(relative).as_rotvec()
    angular_acceleration = np.linalg.norm(
        angular_velocity[1:] - angular_velocity[:-1], axis=1
    )
    return {
        "translation_acceleration_meters_per_frame2": _aggregate(
            translation_acceleration
        ),
        "angular_acceleration_degrees_per_frame2": _aggregate(
            np.rad2deg(angular_acceleration)
        ),
    }
