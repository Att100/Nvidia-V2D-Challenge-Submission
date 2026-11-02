"""Geometry and optimization primitives for hand projection alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PnPResult:
    rotation: np.ndarray
    translation: np.ndarray
    inliers: np.ndarray
    reprojection_error: np.ndarray


@dataclass(frozen=True)
class HandAlignmentResult:
    scale: float
    rotations: np.ndarray
    translations: np.ndarray
    camera_joints: np.ndarray
    initial_scale: float
    initial_rotations: np.ndarray
    initial_translations: np.ndarray
    loss_history: np.ndarray


def solve_hand_pnp(
    local_joints: np.ndarray,
    image_joints: np.ndarray,
    intrinsics: np.ndarray,
    *,
    ransac_threshold: float = 8.0,
    iterations: int = 200,
) -> PnPResult:
    """Place one wrist-relative MANO hand in a calibrated full-image camera."""
    object_points = np.asarray(local_joints, dtype=np.float64)
    image_points = np.asarray(image_joints, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    finite = np.isfinite(object_points).all(axis=1) & np.isfinite(image_points).all(axis=1)
    if finite.sum() < 6:
        raise ValueError("At least six finite MANO/2D correspondences are required for PnP")
    object_selected = np.ascontiguousarray(object_points[finite])
    image_selected = np.ascontiguousarray(image_points[finite])
    success, rotation_vector, translation, inlier_indices = cv2.solvePnPRansac(
        object_selected,
        image_selected,
        k,
        None,
        flags=cv2.SOLVEPNP_EPNP,
        iterationsCount=max(1, int(iterations)),
        reprojectionError=float(ransac_threshold),
        confidence=0.999,
    )
    if not success:
        raise RuntimeError("OpenCV solvePnPRansac failed")
    if inlier_indices is None or len(inlier_indices) < 6:
        inlier_indices = np.arange(len(object_selected), dtype=np.int32)[:, None]
    selected_inliers = inlier_indices.reshape(-1)
    rotation_vector, translation = cv2.solvePnPRefineLM(
        object_selected[selected_inliers],
        image_selected[selected_inliers],
        k,
        None,
        rotation_vector,
        translation,
    )
    rotation = cv2.Rodrigues(rotation_vector)[0]
    camera = object_points @ rotation.T + translation.reshape(1, 3)
    projected_h = camera @ k.T
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    errors = np.linalg.norm(projected - image_points, axis=1)
    inliers = np.zeros(len(object_points), dtype=bool)
    finite_indices = np.flatnonzero(finite)
    inliers[finite_indices[selected_inliers]] = True
    if not np.isfinite(camera).all() or np.median(camera[:, 2]) <= 0.0:
        raise RuntimeError("PnP produced invalid or behind-camera hand geometry")
    return PnPResult(
        rotation.astype(np.float32),
        translation.reshape(3).astype(np.float32),
        inliers,
        errors.astype(np.float32),
    )


def robust_sequence_scale(
    unit_camera_joints: np.ndarray,
    observed_depth: np.ndarray,
    *,
    minimum_frame_support: int = 6,
) -> tuple[float, np.ndarray]:
    """Estimate one hand scale from per-frame metric surface-depth observations."""
    unit_z = np.asarray(unit_camera_joints, dtype=np.float64)[..., 2]
    depth = np.asarray(observed_depth, dtype=np.float64)
    valid = np.isfinite(unit_z) & np.isfinite(depth) & (unit_z > 1e-6) & (depth > 1e-6)
    frame_scales = np.full(len(unit_z), np.nan, dtype=np.float64)
    for frame_id in range(len(unit_z)):
        if valid[frame_id].sum() < minimum_frame_support:
            continue
        ratios = depth[frame_id, valid[frame_id]] / unit_z[frame_id, valid[frame_id]]
        frame_scales[frame_id] = np.median(ratios)
    candidates = frame_scales[np.isfinite(frame_scales) & (frame_scales > 0.0)]
    if not len(candidates):
        raise RuntimeError("No frame has enough reliable metric depths to estimate hand scale")
    center = float(np.median(candidates))
    mad = float(np.median(np.abs(candidates - center)))
    if mad > 1e-9:
        keep = np.abs(frame_scales - center) <= 3.5 * 1.4826 * mad
        selected = frame_scales[keep & np.isfinite(frame_scales)]
        if len(selected):
            center = float(np.median(selected))
    return center, frame_scales.astype(np.float32)


def _skew_torch(vectors: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vectors[..., 0])
    x, y, z = vectors.unbind(dim=-1)
    return torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1
    ).reshape(vectors.shape[:-1] + (3, 3))


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
    return identity + coefficient_a[..., None] * skew + coefficient_b[..., None] * (skew @ skew)


def _smooth_l1_mean(values: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    return F.smooth_l1_loss(values, torch.zeros_like(values), beta=beta, reduction="mean")


@torch.enable_grad()
def optimize_hand_sequence(
    local_joints: np.ndarray,
    image_joints: np.ndarray,
    observed_depth: np.ndarray,
    intrinsics: np.ndarray,
    bbox_scale: np.ndarray,
    initial_rotations: np.ndarray,
    initial_unit_translations: np.ndarray,
    *,
    iterations: int = 800,
    learning_rate: float = 0.025,
    depth_weight: float = 1.0,
    rotation_prior_weight: float = 0.03,
    translation_prior_weight: float = 0.002,
    translation_acceleration_weight: float = 0.04,
    rotation_acceleration_weight: float = 0.01,
    scale_prior_weight: float = 0.005,
    fixed_scale: float | None = None,
    temporal_group_ids: np.ndarray | None = None,
    device: str = "cuda",
    progress: Callable[[int, int, float], None] | None = None,
) -> HandAlignmentResult:
    """Optimize fixed scale and framewise MANO global pose in the left camera.

    Gradient mode is enabled locally so inference-only foundation models cannot
    accidentally disable autograd for this downstream optimization stage.
    """
    joints_np = np.asarray(local_joints, dtype=np.float32)
    pixels_np = np.asarray(image_joints, dtype=np.float32)
    depths_np = np.asarray(observed_depth, dtype=np.float32)
    rotations_np = np.asarray(initial_rotations, dtype=np.float32)
    translations_np = np.asarray(initial_unit_translations, dtype=np.float32)
    if joints_np.ndim != 3 or joints_np.shape[-1] != 3:
        raise ValueError(f"Expected FxJx3 local joints, got {joints_np.shape}")
    if pixels_np.shape != joints_np.shape[:-1] + (2,) or depths_np.shape != joints_np.shape[:-1]:
        raise ValueError("2D joints and depth observations must match local joints")
    if rotations_np.shape != (len(joints_np), 3, 3) or translations_np.shape != (len(joints_np), 3):
        raise ValueError("Initial PnP rotations/translations have invalid shapes")
    if temporal_group_ids is None:
        temporal_groups_np = np.zeros(len(joints_np), dtype=np.int64)
    else:
        temporal_groups_np = np.asarray(temporal_group_ids, dtype=np.int64)
        if temporal_groups_np.shape != (len(joints_np),):
            raise ValueError(
                "temporal_group_ids must have one entry per frame/sample"
            )
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        device = "cpu"
    torch_device = torch.device(device)
    joints = torch.as_tensor(joints_np, device=torch_device)
    pixels = torch.as_tensor(pixels_np, device=torch_device)
    depth = torch.as_tensor(depths_np, device=torch_device)
    k = torch.as_tensor(np.asarray(intrinsics, dtype=np.float32), device=torch_device)
    box = torch.as_tensor(np.asarray(bbox_scale, dtype=np.float32), device=torch_device).clamp_min(8.0)
    base_rotations = torch.as_tensor(rotations_np, device=torch_device)
    base_translations = torch.as_tensor(translations_np, device=torch_device)
    temporal_groups = torch.as_tensor(temporal_groups_np, device=torch_device)

    unit_initial = joints @ base_rotations.transpose(-1, -2) + base_translations[:, None, :]
    if fixed_scale is None:
        initial_scale, _ = robust_sequence_scale(
            unit_initial.detach().cpu().numpy(), depths_np
        )
    else:
        initial_scale = float(fixed_scale)
        if not np.isfinite(initial_scale) or initial_scale <= 0.0:
            raise ValueError(f"fixed_scale must be positive and finite, got {fixed_scale}")
    valid_depth = torch.isfinite(depth) & (depth > 0.0)
    finite_pixels = torch.isfinite(pixels).all(dim=-1)
    uses_depth = float(depth_weight) > 0.0
    if uses_depth and int(valid_depth.sum()) < 12:
        raise RuntimeError("Too few visible stereo hand-depth samples for sequence optimization")
    if uses_depth:
        depth_center = torch.median(depth[valid_depth]).detach().clamp_min(1e-3)
    else:
        # RGB-only calibrated PnP has metric translation because MANO geometry
        # is metric.  It has no observed depth term, so temporal normalization
        # uses the robust PnP depth and scale must remain explicitly fixed.
        if fixed_scale is None:
            raise ValueError("RGB-only hand projection requires fixed_scale")
        positive_z = unit_initial[..., 2][torch.isfinite(unit_initial[..., 2]) & (unit_initial[..., 2] > 1e-3)]
        depth_center = (
            torch.median(positive_z).detach().clamp_min(1e-3)
            if positive_z.numel()
            else joints.new_tensor(1.0)
        )
    depth_sigma = (0.025 * depth_center).clamp_min(0.005)

    rotation_delta = torch.nn.Parameter(torch.zeros((len(joints_np), 3), device=torch_device))
    unit_translation = torch.nn.Parameter(base_translations.clone())
    log_scale = (
        torch.nn.Parameter(
            torch.tensor(np.log(initial_scale), dtype=torch.float32, device=torch_device)
        )
        if fixed_scale is None
        else None
    )
    parameters = [rotation_delta, unit_translation]
    if log_scale is not None:
        parameters.append(log_scale)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(iterations)), eta_min=learning_rate * 0.05
    )
    history: list[float] = []

    for step in range(max(1, int(iterations))):
        optimizer.zero_grad(set_to_none=True)
        correction = axis_angle_to_matrix(rotation_delta)
        rotations = correction @ base_rotations
        unit_camera = joints @ rotations.transpose(-1, -2) + unit_translation[:, None, :]
        scale = (
            torch.exp(log_scale)
            if log_scale is not None
            else joints.new_tensor(initial_scale)
        )
        camera = scale * unit_camera
        projected_h = camera @ k.T
        projected = projected_h[..., :2] / projected_h[..., 2:3].clamp_min(1e-5)

        pixel_sigma = (0.02 * box).clamp(min=3.0, max=8.0)
        reprojection_residual = (projected - pixels) / pixel_sigma[:, None, None]
        reprojection_loss = _smooth_l1_mean(reprojection_residual[finite_pixels], beta=1.0)
        if uses_depth:
            depth_residual = (camera[..., 2] - depth) / depth_sigma
            depth_loss = _smooth_l1_mean(depth_residual[valid_depth], beta=1.0)
        else:
            depth_loss = camera.new_zeros(())
        rotation_prior = _smooth_l1_mean(rotation_delta / 0.25, beta=1.0)
        translation_prior = _smooth_l1_mean(
            (unit_translation - base_translations) / 0.05, beta=1.0
        )
        if len(joints_np) >= 3:
            scene_translation = scale * unit_translation
            translation_acceleration_all = (
                scene_translation[2:] - 2.0 * scene_translation[1:-1] + scene_translation[:-2]
            ) / (0.015 * depth_center)
            rotation_acceleration_all = (
                rotations[2:] - 2.0 * rotations[1:-1] + rotations[:-2]
            ) / 0.08
            temporal_triplet = (
                (temporal_groups[2:] == temporal_groups[1:-1])
                & (temporal_groups[1:-1] == temporal_groups[:-2])
            )
            translation_acceleration = translation_acceleration_all[temporal_triplet]
            rotation_acceleration = rotation_acceleration_all[temporal_triplet]
            translation_smoothness = _smooth_l1_mean(translation_acceleration, beta=1.0)
            rotation_smoothness = _smooth_l1_mean(rotation_acceleration, beta=1.0)
        else:
            translation_smoothness = camera.new_zeros(())
            rotation_smoothness = camera.new_zeros(())
        scale_prior = (
            (log_scale - float(np.log(initial_scale))).square()
            if log_scale is not None
            else camera.new_zeros(())
        )
        loss = (
            reprojection_loss
            + depth_weight * depth_loss
            + rotation_prior_weight * rotation_prior
            + translation_prior_weight * translation_prior
            + translation_acceleration_weight * translation_smoothness
            + rotation_acceleration_weight * rotation_smoothness
            + scale_prior_weight * scale_prior
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
        optimizer.step()
        scheduler.step()
        history.append(float(loss.detach().cpu()))
        if progress is not None and (step == 0 or (step + 1) % 50 == 0 or step + 1 == iterations):
            progress(step + 1, int(iterations), history[-1])

    with torch.no_grad():
        rotations = axis_angle_to_matrix(rotation_delta) @ base_rotations
        scale = (
            float(torch.exp(log_scale).cpu())
            if log_scale is not None
            else initial_scale
        )
        translations = scale * unit_translation
        camera_joints = scale * (
            joints @ rotations.transpose(-1, -2) + unit_translation[:, None, :]
        )
    return HandAlignmentResult(
        scale=scale,
        rotations=rotations.cpu().numpy().astype(np.float32),
        translations=translations.cpu().numpy().astype(np.float32),
        camera_joints=camera_joints.cpu().numpy().astype(np.float32),
        initial_scale=initial_scale,
        initial_rotations=rotations_np,
        initial_translations=(initial_scale * translations_np).astype(np.float32),
        loss_history=np.asarray(history, dtype=np.float32),
    )


def apply_framewise_hand_pose(
    local_points: np.ndarray,
    scale: float,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    points = np.asarray(local_points, dtype=np.float32)
    rotations = np.asarray(rotations, dtype=np.float32)
    translations = np.asarray(translations, dtype=np.float32)
    if points.ndim != 3 or rotations.shape != (len(points), 3, 3) or translations.shape != (len(points), 3):
        raise ValueError("Framewise points, rotations, and translations have incompatible shapes")
    return float(scale) * (points @ rotations.transpose(0, 2, 1)) + translations[:, None, :]


def alignment_metrics(
    camera_joints: np.ndarray,
    image_joints: np.ndarray,
    observed_depth: np.ndarray,
    intrinsics: np.ndarray,
) -> dict[str, float | int | None]:
    camera = np.asarray(camera_joints, dtype=np.float64)
    pixels = np.asarray(image_joints, dtype=np.float64)
    depth = np.asarray(observed_depth, dtype=np.float64)
    projected_h = camera @ np.asarray(intrinsics, dtype=np.float64).T
    projected = projected_h[..., :2] / projected_h[..., 2:3]
    reprojection = np.linalg.norm(projected - pixels, axis=-1)
    reprojection = reprojection[np.isfinite(reprojection)]
    depth_valid = np.isfinite(depth) & np.isfinite(camera[..., 2])
    depth_error = np.abs(camera[..., 2] - depth)[depth_valid]
    translation = np.median(camera, axis=1)
    acceleration = np.linalg.norm(
        translation[2:] - 2.0 * translation[1:-1] + translation[:-2], axis=-1
    )

    def stats(values: np.ndarray) -> dict[str, float | int | None]:
        finite = values[np.isfinite(values)]
        if not len(finite):
            return {"count": 0, "median": None, "p90": None, "mean": None}
        return {
            "count": int(len(finite)),
            "median": float(np.median(finite)),
            "p90": float(np.percentile(finite, 90)),
            "mean": float(np.mean(finite)),
        }

    return {
        "reprojection": stats(reprojection),
        "visible_depth_absolute_error": stats(depth_error),
        "joint_center_acceleration": stats(acceleration),
    }
