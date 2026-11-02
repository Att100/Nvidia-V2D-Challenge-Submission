"""Visibility-aware stereo surface alignment for RGB-initialized MANO hands."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np

from pipeline.utils import PipelineRequest, StageResult, StageStatus, all_artifacts_exist

from .rigid_alignment import (
    _apply_wrist_se3,
    _clip_rotation_vector,
    _clip_translation,
    _constrain_se3_reprojection,
    _robust_rigid_alignment,
    _rotation_vector,
    _smooth_segmented_vectors,
    _visible_segment_ids,
)
from .surface_geometry import (
    BatchedSurfaceRasterizer,
    SurfaceCorrespondences,
    camera_ray,
    robust_ray_offset,
)


@dataclass(frozen=True)
class StereoHandSurfaceConfig:
    """Settings for projective visible-surface MANO refinement."""

    # Retained for configuration compatibility. Metric scale is optimized by
    # StereoDepthHandProjectionOptimizer before this stage.
    scale_mode: str = "global"
    minimum_correspondences: int = 96
    object_mask_margin: int = 2
    mask_erosion: int = 0
    raster_batch_size: int = 8
    raster_device: str = "auto"
    maximum_rotation_degrees: float = 12.0
    maximum_relative_translation: float = 0.30
    maximum_reprojection_increase_pixels: float = 2.0
    # The preceding stereo-depth projection already establishes metric hand
    # depth. Surface ICP must not trade that reliable depth anchor for a lower
    # one-sided visible-surface residual (which can be biased by occlusion).
    maximum_depth_median_increase_m: float = 0.002
    maximum_depth_p90_increase_m: float = 0.005
    minimum_depth_target_support: int = 5
    smoothness_weight: float = 6.0
    prior_weight: float = 0.04
    trim_iterations: int = 5
    maximum_correspondences: int = 12000
    hand_masks_path: Path | None = None
    # ``sequence_global`` estimates one wrist-centred rigid correction per
    # anatomical hand for the complete visible sequence.  It preserves the
    # accurate UmeTrack motion instead of allowing depth noise to create sliding.
    se3_scope: str = "sequence_global"


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return an independent weighted median for each vector coordinate."""

    result = np.zeros(values.shape[1], dtype=np.float64)
    for axis in range(values.shape[1]):
        order = np.argsort(values[:, axis])
        cumulative = np.cumsum(weights[order])
        index = np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left")
        result[axis] = values[order[min(index, len(order) - 1)], axis]
    return result.astype(np.float32)


def _reprojection_medians(
    joints: np.ndarray,
    targets: np.ndarray,
    intrinsic: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Measure one robust reprojection error per anatomical hand."""

    projected_h = joints @ intrinsic.T
    projected = projected_h[..., :2] / projected_h[..., 2:3]
    errors = np.linalg.norm(projected - targets, axis=-1)
    result = np.full(valid.shape[1], np.nan, dtype=np.float32)
    for hand_id in range(valid.shape[1]):
        selected = (
            valid[:, hand_id, None]
            & np.isfinite(errors[:, hand_id])
            & np.isfinite(joints[:, hand_id, :, 2])
            & (joints[:, hand_id, :, 2] > 1.0e-5)
        )
        if selected.any():
            result[hand_id] = float(np.median(errors[:, hand_id][selected]))
    return result


def _depth_target_frame_errors(
    joints: np.ndarray,
    targets: np.ndarray | None,
    support: np.ndarray | None,
    valid: np.ndarray,
    minimum_support: int,
) -> np.ndarray:
    """Return per-frame median joint-depth errors against stereo targets."""

    result = np.full(valid.shape, np.nan, dtype=np.float32)
    if targets is None or support is None or targets.shape != joints.shape[:-1]:
        return result
    if support.shape != targets.shape:
        return result
    errors = np.abs(joints[..., 2] - targets)
    selected = (
        valid[..., None]
        & np.isfinite(errors)
        & np.isfinite(targets)
        & (support >= minimum_support)
    )
    for frame_id, hand_id in np.argwhere(valid):
        frame_values = errors[frame_id, hand_id][selected[frame_id, hand_id]]
        if len(frame_values):
            result[frame_id, hand_id] = float(np.median(frame_values))
    return result


def _unpack(path: Path) -> np.ndarray:
    with np.load(path) as archive:
        width = int(np.asarray(archive["mask_width"]).item())
        packed = np.asarray(archive["packed_masks"], dtype=np.uint8)
    return np.unpackbits(packed, axis=-1)[..., :width].astype(bool)


def _depth_to_pointmap(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    y, x = np.mgrid[:height, :width]
    result = np.stack(
        (
            (x - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (y - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
        ),
        axis=-1,
    ).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0.10) & (depth < 10.0)
    result[~valid] = np.nan
    return result


def _human_masks(run_root: Path, config: StereoHandSurfaceConfig) -> np.ndarray:
    if config.hand_masks_path is not None:
        mask_path = config.hand_masks_path
        if not mask_path.is_absolute():
            mask_path = run_root / mask_path
        with np.load(mask_path) as archive:
            width = int(np.asarray(archive["mask_width"]).item())
            masks = np.unpackbits(
                np.asarray(archive["packed_masks"], dtype=np.uint8), axis=-1
            )[..., :width].astype(bool)
        if masks.ndim != 4 or masks.shape[1] != 2:
            raise ValueError(f"Expected frame,left/right,H,W hand masks, got {masks.shape}")
        return masks
    dynamic = _unpack(run_root / "dynamic_mask/dynamic_mask.npz")
    object_masks = _unpack(run_root / "tracks/sam3/refined_masks/object_masks.npz")
    objects = object_masks.any(axis=0) if len(object_masks) else np.zeros_like(dynamic)
    if config.object_mask_margin > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * config.object_mask_margin + 1, 2 * config.object_mask_margin + 1),
        )
        objects = np.stack(
            [cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) for mask in objects]
        )
    human = dynamic & ~objects
    if config.mask_erosion > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * config.mask_erosion + 1, 2 * config.mask_erosion + 1),
        )
        human = np.stack(
            [cv2.erode(mask.astype(np.uint8), kernel).astype(bool) for mask in human]
        )
    return human


def _rasterize_sequence(
    run_root: Path,
    vertices: np.ndarray,
    faces: tuple[np.ndarray, np.ndarray],
    valid: np.ndarray,
    human: np.ndarray,
    intrinsic: np.ndarray,
    config: StereoHandSurfaceConfig,
) -> list[list[SurfaceCorrespondences | None]]:
    """Match only z-buffer-visible MANO surface to depth at identical pixels."""

    frame_count, hand_count = valid.shape
    height, width = human.shape[-2:]
    rasterizer = BatchedSurfaceRasterizer(
        intrinsic, (height, width), device=config.raster_device
    )
    result: list[list[SurfaceCorrespondences | None]] = [
        [None for _ in range(hand_count)] for _ in range(frame_count)
    ]
    for hand_id in range(hand_count):
        ids = np.flatnonzero(valid[:, hand_id])
        for start in range(0, len(ids), config.raster_batch_size):
            batch_ids = ids[start : start + config.raster_batch_size]
            pointmaps = []
            for frame_id in batch_ids:
                depth = np.asarray(
                    np.load(run_root / "depth/frames" / f"{int(frame_id):06d}.npy"),
                    dtype=np.float32,
                )
                pointmaps.append(_depth_to_pointmap(depth, intrinsic))
            pairs = rasterizer(
                vertices[batch_ids, hand_id],
                faces[hand_id],
                np.stack(pointmaps),
                np.ones((len(batch_ids), height, width), dtype=np.float32),
                confidence_threshold=0.5,
                sample_masks=(
                    human[batch_ids, hand_id]
                    if human.ndim == 4
                    else human[batch_ids]
                ),
            )
            for frame_id, pair in zip(batch_ids, pairs, strict=True):
                result[int(frame_id)][hand_id] = pair
    return result


def _pair_errors(
    pairs: list[list[SurfaceCorrespondences | None]], valid: np.ndarray
) -> np.ndarray:
    values = np.full(valid.shape, np.nan, dtype=np.float32)
    for frame_id, hand_id in np.argwhere(valid):
        pair = pairs[frame_id][hand_id]
        if pair is not None and len(pair.predicted):
            values[frame_id, hand_id] = float(
                np.median(np.linalg.norm(pair.predicted - pair.observed, axis=1))
            )
    return values


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "median": None, "p90": None, "mean": None}
    return {
        "count": int(len(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "mean": float(np.mean(finite)),
    }


def _convex_silhouette_iou(
    vertices: np.ndarray,
    masks: np.ndarray,
    valid: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Measure a lightweight MANO-projection IoU against each tracked hand mask."""

    result = np.full(valid.shape, np.nan, dtype=np.float32)
    height, width = masks.shape[-2:]
    for frame_id, hand_id in np.argwhere(valid):
        target = masks[frame_id, hand_id] if masks.ndim == 4 else masks[frame_id]
        points = vertices[frame_id, hand_id]
        homogeneous = points @ intrinsic.T
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        finite = np.isfinite(pixels).all(axis=1) & (points[:, 2] > 1e-5)
        if finite.sum() < 3 or not target.any():
            continue
        hull = cv2.convexHull(np.rint(pixels[finite]).astype(np.int32))
        rendered = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(rendered, hull, 1)
        rendered = rendered.astype(bool)
        intersection = int((rendered & target).sum())
        union = int((rendered | target).sum())
        result[frame_id, hand_id] = intersection / max(1, union)
    return result


def run_stereo_surface_alignment(
    run_root: Path,
    input_motion: Path,
    output_dir: Path,
    config: StereoHandSurfaceConfig,
) -> None:
    """Refine hands against visible stereo surfaces without changing visibility."""

    with np.load(input_motion, allow_pickle=True) as archive:
        motion = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(run_root / "calibration/calibration.npz") as archive:
        intrinsic = np.asarray(archive["K_working"], dtype=np.float64)

    vertices_before = np.asarray(motion["vertices_camera"], dtype=np.float32)
    joints_before = np.asarray(motion["joints_camera"], dtype=np.float32)
    wrist_joint_index = int(np.asarray(motion.get("wrist_joint_index", 0)).item())
    target_pixels = np.asarray(motion["joints_2d_native"], dtype=np.float32)
    detection_valid = np.asarray(
        motion.get("detection_valid", motion["valid"]), dtype=bool
    ).copy()
    frame_count, hand_count = detection_valid.shape
    human = _human_masks(run_root, config)
    faces = (
        np.asarray(motion["faces_left"], dtype=np.int32),
        np.asarray(motion["faces_right"], dtype=np.int32),
    )

    initial_pairs = _rasterize_sequence(
        run_root, vertices_before, faces, detection_valid, human, intrinsic, config
    )
    pre_error = _pair_errors(initial_pairs, detection_valid)
    silhouette_iou_before = _convex_silhouette_iou(
        vertices_before, human, detection_valid, intrinsic
    )
    stereo_depth_target = (
        np.asarray(motion["stereo_depth_target"], dtype=np.float32)
        if "stereo_depth_target" in motion
        else None
    )
    stereo_depth_support = (
        np.asarray(motion["stereo_depth_support"], dtype=np.int32)
        if "stereo_depth_support" in motion
        else None
    )
    depth_error_before = _depth_target_frame_errors(
        joints_before,
        stereo_depth_target,
        stereo_depth_support,
        detection_valid,
        config.minimum_depth_target_support,
    )

    raw_rotation = np.full((frame_count, hand_count, 3), np.nan, dtype=np.float32)
    raw_translation = np.full((frame_count, hand_count, 3), np.nan, dtype=np.float32)
    residual = np.full((frame_count, hand_count), np.nan, dtype=np.float32)
    support = np.zeros((frame_count, hand_count), dtype=np.int32)
    for frame_id, hand_id in np.argwhere(detection_valid):
        pair = initial_pairs[frame_id][hand_id]
        if pair is None or len(pair.predicted) < config.minimum_correspondences:
            continue
        wrist = joints_before[frame_id, hand_id, wrist_joint_index].astype(np.float64)
        rotation, translation, count, error = _robust_rigid_alignment(
            pair.predicted,
            pair.observed,
            wrist,
            minimum_count=config.minimum_correspondences,
            trim_iterations=config.trim_iterations,
            maximum_correspondences=config.maximum_correspondences,
        )
        if not np.isfinite(translation).all():
            continue
        frame_targets = target_pixels[frame_id, hand_id]
        finite_targets = np.isfinite(frame_targets).all(axis=1)
        if not finite_targets.any():
            continue
        # The wrist can be hidden by the manipulated object even when several
        # palm/finger observations remain reliable.  Use their robust center
        # for the camera ray instead of invalidating the complete hand frame.
        ray_pixel = (
            frame_targets[0]
            if finite_targets[0]
            else np.median(frame_targets[finite_targets], axis=0)
        )
        ray = camera_ray(ray_pixel, intrinsic)
        radial_offset, radial_count = robust_ray_offset(
            pair, 1.0, ray, minimum_count=config.minimum_correspondences
        )
        if np.isfinite(radial_offset):
            tangential = translation - float(translation @ ray) * ray
            translation = tangential + radial_offset * ray
            count = min(count, radial_count)
        observed_depth = float(np.median(pair.observed[:, 2]))
        vector = _clip_rotation_vector(
            _rotation_vector(rotation), config.maximum_rotation_degrees
        )
        translation = _clip_translation(
            translation, observed_depth, config.maximum_relative_translation
        )
        vector, translation, _, _, _ = _constrain_se3_reprojection(
            vector,
            translation,
            wrist,
            joints_before[frame_id, hand_id],
            target_pixels[frame_id, hand_id],
            intrinsic,
            config.maximum_reprojection_increase_pixels,
        )
        raw_rotation[frame_id, hand_id] = vector
        raw_translation[frame_id, hand_id] = translation
        residual[frame_id, hand_id] = error
        support[frame_id, hand_id] = count

    surface_alignment_valid = support > 0
    if config.se3_scope not in {"sequence_global", "per_frame"}:
        raise ValueError(f"Unsupported hand surface SE3 scope: {config.se3_scope}")
    if config.se3_scope == "sequence_global":
        shared_rotation = np.zeros((hand_count, 3), dtype=np.float32)
        shared_translation = np.zeros((hand_count, 3), dtype=np.float32)
        for hand_id in range(hand_count):
            selected = (
                detection_valid[:, hand_id]
                & surface_alignment_valid[:, hand_id]
                & np.isfinite(raw_rotation[:, hand_id]).all(axis=1)
                & np.isfinite(raw_translation[:, hand_id]).all(axis=1)
            )
            if selected.any():
                weights = np.sqrt(np.maximum(support[selected, hand_id], 1.0))
                shared_rotation[hand_id] = _weighted_median(
                    raw_rotation[selected, hand_id], weights
                )
                shared_translation[hand_id] = _weighted_median(
                    raw_translation[selected, hand_id], weights
                )
        # Apply one common acceptance fraction per hand.  A per-frame gate here
        # would silently reintroduce framewise motion into the global model.
        baseline = _reprojection_medians(
            joints_before, target_pixels, intrinsic, detection_valid
        )
        accepted_fraction = np.zeros(hand_count, dtype=np.float32)
        for hand_id in range(hand_count):
            best_fraction = 0.0
            best_surface = float("inf")
            for fraction in np.linspace(0.0, 1.0, 51):
                candidate = joints_before.copy()
                for frame_id in np.flatnonzero(detection_valid[:, hand_id]):
                    candidate[frame_id, hand_id] = _apply_wrist_se3(
                        joints_before[frame_id, hand_id],
                        joints_before[frame_id, hand_id, wrist_joint_index],
                        shared_rotation[hand_id] * fraction,
                        shared_translation[hand_id] * fraction,
                    )
                error = _reprojection_medians(
                    candidate, target_pixels, intrinsic, detection_valid
                )[hand_id]
                if not np.isfinite(error) or error > baseline[hand_id] + config.maximum_reprojection_increase_pixels:
                    continue
                candidate_depth_error = _depth_target_frame_errors(
                    candidate,
                    stereo_depth_target,
                    stereo_depth_support,
                    detection_valid,
                    config.minimum_depth_target_support,
                )[:, hand_id]
                common_depth = np.isfinite(depth_error_before[:, hand_id]) & np.isfinite(
                    candidate_depth_error
                )
                if common_depth.any():
                    baseline_depth = depth_error_before[common_depth, hand_id]
                    candidate_depth = candidate_depth_error[common_depth]
                    if (
                        np.median(candidate_depth)
                        > np.median(baseline_depth)
                        + config.maximum_depth_median_increase_m
                        or np.percentile(candidate_depth, 90)
                        > np.percentile(baseline_depth, 90)
                        + config.maximum_depth_p90_increase_m
                    ):
                        continue
                frame_surface = []
                for frame_id in np.flatnonzero(detection_valid[:, hand_id]):
                    pair = initial_pairs[frame_id][hand_id]
                    if pair is None or len(pair.predicted) < config.minimum_correspondences:
                        continue
                    transformed = _apply_wrist_se3(
                        pair.predicted,
                        joints_before[frame_id, hand_id, wrist_joint_index],
                        shared_rotation[hand_id] * fraction,
                        shared_translation[hand_id] * fraction,
                    )
                    frame_surface.append(
                        float(np.median(np.linalg.norm(transformed - pair.observed, axis=1)))
                    )
                surface = float(np.median(frame_surface)) if frame_surface else float("inf")
                if surface < best_surface:
                    best_surface = surface
                    best_fraction = float(fraction)
            accepted_fraction[hand_id] = best_fraction
            shared_rotation[hand_id] *= accepted_fraction[hand_id]
            shared_translation[hand_id] *= accepted_fraction[hand_id]
        smooth_rotation = np.broadcast_to(
            shared_rotation[None], raw_rotation.shape
        ).copy()
        smooth_translation = np.broadcast_to(
            shared_translation[None], raw_translation.shape
        ).copy()
        segments = _visible_segment_ids(detection_valid, motion)
    else:
        accepted_fraction = np.ones(hand_count, dtype=np.float32)
        segments = _visible_segment_ids(detection_valid, motion)
        smooth_rotation = _smooth_segmented_vectors(
            raw_rotation,
            support,
            detection_valid,
            segments,
            smoothness_weight=config.smoothness_weight,
            prior_weight=config.prior_weight,
        )
        smooth_translation = _smooth_segmented_vectors(
            raw_translation,
            support,
            detection_valid,
            segments,
            smoothness_weight=config.smoothness_weight,
            prior_weight=config.prior_weight,
        )

    alignment_source = np.zeros(detection_valid.shape, dtype=np.uint8)
    alignment_source[surface_alignment_valid] = 1
    for hand_id in range(hand_count):
        for segment_id in np.unique(segments[:, hand_id]):
            if segment_id < 0:
                continue
            ids = np.flatnonzero(
                detection_valid[:, hand_id] & (segments[:, hand_id] == segment_id)
            )
            missing = ids[~surface_alignment_valid[ids, hand_id]]
            if len(missing) and surface_alignment_valid[ids, hand_id].any():
                alignment_source[missing, hand_id] = 2

    vertices_after = np.full_like(vertices_before, np.nan)
    joints_after = np.full_like(joints_before, np.nan)
    for frame_id, hand_id in np.argwhere(detection_valid):
        wrist = joints_before[frame_id, hand_id, wrist_joint_index]
        if config.se3_scope == "sequence_global":
            vector = smooth_rotation[frame_id, hand_id]
            translation = smooth_translation[frame_id, hand_id]
        else:
            vector, translation, _, _, _ = _constrain_se3_reprojection(
                smooth_rotation[frame_id, hand_id],
                smooth_translation[frame_id, hand_id],
                wrist,
                joints_before[frame_id, hand_id],
                target_pixels[frame_id, hand_id],
                intrinsic,
                config.maximum_reprojection_increase_pixels,
            )
        smooth_rotation[frame_id, hand_id] = vector
        smooth_translation[frame_id, hand_id] = translation
        vertices_after[frame_id, hand_id] = _apply_wrist_se3(
            vertices_before[frame_id, hand_id], wrist, vector, translation
        )
        joints_after[frame_id, hand_id] = _apply_wrist_se3(
            joints_before[frame_id, hand_id], wrist, vector, translation
        )

    refined_pairs = _rasterize_sequence(
        run_root, vertices_after, faces, detection_valid, human, intrinsic, config
    )
    post_error = _pair_errors(refined_pairs, detection_valid)
    silhouette_iou_after = _convex_silhouette_iou(
        vertices_after, human, detection_valid, intrinsic
    )
    depth_error_after = _depth_target_frame_errors(
        joints_after,
        stereo_depth_target,
        stereo_depth_support,
        detection_valid,
        config.minimum_depth_target_support,
    )

    camera_to_world = np.asarray(motion["camera_to_world"], dtype=np.float32)
    vertices_world = np.full_like(vertices_after, np.nan)
    joints_world = np.full_like(joints_after, np.nan)
    for frame_id, hand_id in np.argwhere(detection_valid):
        rotation = camera_to_world[frame_id, :3, :3]
        translation = camera_to_world[frame_id, :3, 3]
        vertices_world[frame_id, hand_id] = (
            vertices_after[frame_id, hand_id] @ rotation.T + translation
        )
        joints_world[frame_id, hand_id] = (
            joints_after[frame_id, hand_id] @ rotation.T + translation
        )

    if "mano_global_orient" in motion:
        orient = np.asarray(motion["mano_global_orient"], dtype=np.float32).copy()
        for frame_id, hand_id in np.argwhere(detection_valid):
            correction, _ = cv2.Rodrigues(
                smooth_rotation[frame_id, hand_id].astype(np.float64)
            )
            orient[frame_id, hand_id] = correction.astype(np.float32) @ orient[frame_id, hand_id]
        motion["mano_global_orient"] = orient

    motion.update(
        vertices_camera=vertices_after,
        joints_camera=joints_after,
        vertices_world=vertices_world,
        joints_world=joints_world,
        camera_translation=joints_after[:, :, wrist_joint_index],
        detection_valid=detection_valid,
        surface_alignment_valid=surface_alignment_valid,
        surface_alignment_source=alignment_source,
        surface_refine_radial_scale=np.ones(hand_count, dtype=np.float32),
        surface_refine_rotation_vector_raw=raw_rotation,
        surface_refine_rotation_vector=smooth_rotation,
        surface_refine_translation_raw=raw_translation,
        surface_refine_translation=smooth_translation,
        surface_refine_se3_support=support,
        surface_refine_se3_residual=residual,
        surface_refine_pre_visible_error=pre_error,
        surface_refine_post_visible_error=post_error,
        hand_mask_silhouette_iou_before=silhouette_iou_before,
        hand_mask_silhouette_iou_after=silhouette_iou_after,
        stereo_depth_anchor_error_before=depth_error_before,
        stereo_depth_anchor_error_after=depth_error_after,
        surface_refine_alignment_mode=np.asarray(
            "sequence-global-per-hand-se3"
            if config.se3_scope == "sequence_global"
            else "projective-visible-per-frame-se3"
        ),
        surface_refine_global_acceptance_fraction=accepted_fraction,
        valid=detection_valid,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "hand_motion.npz", **motion)
    diagnostics = {
        "format": "umetrack-foundationstereo-visible-surface-v3",
        "coordinate_system": "rectified-left camera and VIPE world",
        "input": str(input_motion),
        "correspondence": "z-buffer-visible MANO surface to FoundationStereo point at identical pixel",
        "surface_sample_mask": (
            "wearer-specific left/right hand-only masks"
            if config.hand_masks_path is not None
            else "all dynamic human pixels excluding objects"
        ),
        "scale": (
            "fixed or optionally optimized by preceding stereo depth projection stage"
        ),
        "mano_articulation_frozen": True,
        "optimized_variables": [
            "one sequence-global wrist-centred SE(3) per anatomical hand"
            if config.se3_scope == "sequence_global"
            else "per-frame wrist-centred SE(3)"
        ],
        "se3_scope": config.se3_scope,
        "validity": {
            "detection_valid_is_output_valid": True,
            "surface_failure_policy": "temporally interpolate correction or preserve depth-projected input",
            "detection_frames": {
                str(label): int(detection_valid[:, hand_id].sum())
                for hand_id, label in enumerate(motion["hand_labels"].tolist())
            },
            "measured_surface_frames": {
                str(label): int(surface_alignment_valid[:, hand_id].sum())
                for hand_id, label in enumerate(motion["hand_labels"].tolist())
            },
            "temporal_fallback_frames": {
                str(label): int((alignment_source[:, hand_id] == 2).sum())
                for hand_id, label in enumerate(motion["hand_labels"].tolist())
            },
        },
        "visible_surface_3d_error_before_m": _stats(pre_error),
        "visible_surface_3d_error_after_m": _stats(post_error),
        "hand_mask_silhouette_iou_before": _stats(silhouette_iou_before),
        "hand_mask_silhouette_iou_after": _stats(silhouette_iou_after),
        "stereo_depth_anchor_error_before_m": _stats(depth_error_before),
        "stereo_depth_anchor_error_after_m": _stats(depth_error_after),
        "stereo_depth_anchor_gate": {
            "maximum_median_increase_m": config.maximum_depth_median_increase_m,
            "maximum_p90_increase_m": config.maximum_depth_p90_increase_m,
            "minimum_target_support": config.minimum_depth_target_support,
        },
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )


class UmeTrackHandSurfaceAlignmentOptimizer:
    """Per-frame SE(3) surface alignment for MANO fitted from UmeTrack."""

    name = "umetrack_mano_surface_alignment"

    def __init__(self, config: StereoHandSurfaceConfig | None = None) -> None:
        self.config = config or StereoHandSurfaceConfig(se3_scope="per_frame")
        if self.config.se3_scope != "per_frame":
            raise ValueError("The UmeTrack stereo branch requires per-frame SE(3)")

    def run(self, request: PipelineRequest) -> StageResult:
        output = request.run_root / "hands/umetrack_mano_surface"
        artifacts = output / "hand_motion.npz", output / "diagnostics.json"
        if request.resume and not request.force and all_artifacts_exist(artifacts):
            return StageResult(self.name, StageStatus.REUSED, artifacts)
        if request.dry_run:
            return StageResult(self.name, StageStatus.SKIPPED, artifacts)
        started = time.perf_counter()
        run_stereo_surface_alignment(
            request.run_root,
            request.run_root / "hands/umetrack_mano_projection/hand_motion.npz",
            output,
            self.config,
        )
        if not all_artifacts_exist(artifacts):
            raise RuntimeError("UmeTrack MANO surface alignment did not produce artifacts")
        return StageResult(
            self.name,
            StageStatus.COMPLETED,
            artifacts,
            elapsed_seconds=time.perf_counter() - started,
        )
