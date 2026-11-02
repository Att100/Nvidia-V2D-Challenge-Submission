"""Fit canonical SAM 3D meshes to metric masked stereo point clouds.

This is the FoundationPose bootstrap scale stage.  It deliberately estimates
only the metric mesh dimensions; the accompanying diagnostic pose is not used
as a FoundationPose pose anchor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import trimesh

@dataclass(frozen=True)
class RobustFrame:
    """Robust PCA frame and dimensions for a masked metric point cloud."""

    center: np.ndarray
    axes: np.ndarray
    extents: np.ndarray
    inliers: np.ndarray


def _statistical_inliers(points: np.ndarray, neighbors: int = 16) -> np.ndarray:
    """Reject isolated samples using a median absolute-deviation threshold."""

    finite = np.isfinite(points).all(axis=1)
    ids = np.flatnonzero(finite)
    if len(ids) < 5:
        return finite
    k = min(max(2, neighbors + 1), len(ids))
    distances, _ = cKDTree(points[ids]).query(points[ids], k=k, workers=-1)
    spacing = distances[:, 1:].mean(axis=1)
    median = float(np.median(spacing))
    mad = float(np.median(np.abs(spacing - median)))
    sigma = max(1.4826 * mad, np.finfo(np.float64).eps)
    keep = spacing <= median + 4.5 * sigma
    result = np.zeros(len(points), dtype=bool)
    result[ids[keep]] = True
    return result


def _robust_frame(
    points: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_quantile: float,
    extent_quantile: float,
) -> RobustFrame:
    """Estimate a trimmed PCA frame and bias-corrected metric extents."""

    if len(points) < 20:
        raise ValueError(f"Only {len(points)} masked point samples are available")
    keep = _statistical_inliers(points)
    finite_confidence = np.isfinite(confidence)
    if finite_confidence.any() and confidence_quantile > 0:
        threshold = float(np.quantile(confidence[finite_confidence], confidence_quantile))
        keep &= finite_confidence & (confidence >= threshold)
    else:
        keep &= finite_confidence
    clean = np.asarray(points[keep], dtype=np.float64)
    if len(clean) < 20:
        raise ValueError(f"Only {len(clean)} robust point samples remain")

    provisional_center = np.median(clean, axis=0)
    eigenvalues, axes = np.linalg.eigh(np.cov(clean - provisional_center, rowvar=False))
    axes = axes[:, np.argsort(eigenvalues)[::-1]]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = (clean - provisional_center) @ axes
    low = np.quantile(local, extent_quantile, axis=0)
    high = np.quantile(local, 1.0 - extent_quantile, axis=0)
    center = provisional_center + (0.5 * (low + high)) @ axes.T
    extents = (high - low) / (1.0 - 2.0 * extent_quantile)
    order = np.argsort(extents)[::-1]
    axes = axes[:, order]
    extents = extents[order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return RobustFrame(center=center, axes=axes, extents=extents, inliers=keep)


def _mesh_frame(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center, axes, and descending extents of the mesh OBB."""

    to_obb, extents = trimesh.bounds.oriented_bounds(mesh)
    obb_to_mesh = np.linalg.inv(to_obb)
    center = obb_to_mesh[:3, 3]
    axes = obb_to_mesh[:3, :3]
    order = np.argsort(extents)[::-1]
    axes = axes[:, order]
    extents = np.asarray(extents, dtype=np.float64)[order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return center, axes, extents


def _axis_scales(
    mesh_extents: np.ndarray,
    point_extents: np.ndarray,
    *,
    mode: str,
    thickness_weight: float,
    max_anisotropy: float,
) -> np.ndarray:
    """Fit isotropic or bounded anisotropic scales in OBB-axis order."""

    ratios = point_extents / mesh_extents
    if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0):
        raise ValueError(f"Invalid mesh-to-point scale ratios: {ratios}")
    weights = np.array([1.0, 1.0, thickness_weight], dtype=np.float64)
    if mode == "isotropic":
        scale = float(np.exp(np.average(np.log(ratios), weights=weights)))
        return np.full(3, scale, dtype=np.float64)
    log_ratios = np.log(ratios)
    if ratios.max() / ratios.min() > max_anisotropy:
        midpoint = 0.5 * (log_ratios.max() + log_ratios.min())
        half_range = 0.5 * np.log(max_anisotropy)
        log_ratios = np.clip(log_ratios, midpoint - half_range, midpoint + half_range)
    return np.exp(log_ratios)


def _scale_transform(
    center: np.ndarray, axes: np.ndarray, axis_scales: np.ndarray
) -> np.ndarray:
    linear = axes @ np.diag(axis_scales) @ axes.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    transform[:3, 3] = -linear @ center
    return transform


