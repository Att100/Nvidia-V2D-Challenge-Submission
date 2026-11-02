"""Shared robust SE(3) utilities for Track-3 hand and object alignment."""

from __future__ import annotations

import cv2
import numpy as np

from .helpers import project_camera_points


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    vector, _ = cv2.Rodrigues(np.asarray(matrix, dtype=np.float64))
    return vector.reshape(3)


def _rotation_matrix(vector: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.Rodrigues(np.asarray(vector, dtype=np.float64).reshape(3))
    return matrix


def _robust_rigid_alignment(
    predicted: np.ndarray,
    observed: np.ndarray,
    pivot: np.ndarray,
    *,
    minimum_count: int,
    trim_iterations: int,
    maximum_correspondences: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Estimate Q ~= R(P-pivot)+pivot+t using trimmed point-to-point Kabsch."""

    source = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(observed, dtype=np.float64)
    center = np.asarray(pivot, dtype=np.float64).reshape(3)
    keep = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target = source[keep], target[keep]
    if len(source) > maximum_correspondences:
        selected = np.linspace(0, len(source) - 1, maximum_correspondences, dtype=np.int64)
        source, target = source[selected], target[selected]
    if len(source) < minimum_count:
        return np.eye(3), np.full(3, np.nan), len(source), float("nan")

    active = np.ones(len(source), dtype=bool)
    rotation = np.eye(3, dtype=np.float64)
    translation_standard = np.zeros(3, dtype=np.float64)
    for _ in range(max(1, int(trim_iterations))):
        source_active, target_active = source[active], target[active]
        if len(source_active) < minimum_count:
            break
        source_mean = source_active.mean(axis=0)
        target_mean = target_active.mean(axis=0)
        covariance = (source_active - source_mean).T @ (target_active - target_mean)
        u, _, vt = np.linalg.svd(covariance, full_matrices=False)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        translation_standard = target_mean - rotation @ source_mean
        residual = np.linalg.norm(source @ rotation.T + translation_standard - target, axis=1)
        active_residual = residual[active]
        median = float(np.median(active_residual))
        mad = float(np.median(np.abs(active_residual - median)))
        threshold = max(0.004, median + 3.0 * 1.4826 * mad)
        updated = residual <= threshold
        if updated.sum() < minimum_count:
            break
        if np.array_equal(updated, active):
            active = updated
            break
        active = updated

    residual = np.linalg.norm(source @ rotation.T + translation_standard - target, axis=1)
    inlier_count = int(active.sum())
    median_residual = float(np.median(residual[active])) if inlier_count else float("nan")
    # Convert the standard affine translation into a displacement of the fixed
    # wrist pivot, which can be smoothed and regularized independently of R.
    pivot_translation = rotation @ center + translation_standard - center
    return rotation, pivot_translation, inlier_count, median_residual


def _clip_rotation_vector(vector: np.ndarray, maximum_degrees: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(value))
    maximum = float(np.deg2rad(maximum_degrees))
    if angle > maximum > 0:
        value *= maximum / angle
    return value.astype(np.float32)


def _clip_translation(
    translation: np.ndarray, observed_depth: float, maximum_relative: float
) -> np.ndarray:
    value = np.asarray(translation, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(value))
    maximum = float(maximum_relative) * float(observed_depth)
    if length > maximum > 0:
        value *= maximum / length
    return value.astype(np.float32)


def _visible_segment_ids(valid: np.ndarray, motion: dict[str, np.ndarray]) -> np.ndarray:
    if "visible_segment_id" in motion:
        return np.asarray(motion["visible_segment_id"][: len(valid)], dtype=np.int32)
    segments = np.full(valid.shape, -1, dtype=np.int32)
    for hand_id in range(valid.shape[1]):
        segment = -1
        previous = False
        for frame_id, current in enumerate(valid[:, hand_id]):
            if current and not previous:
                segment += 1
            if current:
                segments[frame_id, hand_id] = segment
            previous = bool(current)
    return segments


def _smooth_segmented_vectors(
    raw: np.ndarray,
    support: np.ndarray,
    valid: np.ndarray,
    segments: np.ndarray,
    *,
    smoothness_weight: float,
    prior_weight: float,
) -> np.ndarray:
    result = np.zeros_like(raw, dtype=np.float32)
    for hand_id in range(valid.shape[1]):
        for segment_id in np.unique(segments[:, hand_id]):
            if segment_id < 0:
                continue
            selected = np.flatnonzero(valid[:, hand_id] & (segments[:, hand_id] == segment_id))
            if not len(selected):
                continue
            start, end = int(selected[0]), int(selected[-1]) + 1
            for axis in range(3):
                result[start:end, hand_id, axis] = smooth_scalar_offsets(
                    raw[start:end, hand_id, axis],
                    support[start:end, hand_id],
                    smoothness_weight=smoothness_weight,
                    prior_weight=prior_weight,
                )
    return result


def _apply_wrist_se3(
    points: np.ndarray,
    pivot: np.ndarray,
    rotation_vector: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    rotation = _rotation_matrix(rotation_vector)
    values = np.asarray(points, dtype=np.float64)
    center = np.asarray(pivot, dtype=np.float64).reshape(3)
    return ((values - center) @ rotation.T + center + translation).astype(np.float32)


def _median_reprojection_error(
    joints: np.ndarray, pixels: np.ndarray, intrinsics: np.ndarray
) -> float:
    points = np.asarray(joints, dtype=np.float64)
    target = np.asarray(pixels, dtype=np.float64)
    projected_h = points @ np.asarray(intrinsics, dtype=np.float64).T
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    valid = (
        np.isfinite(projected).all(axis=1)
        & np.isfinite(target).all(axis=1)
        & (points[:, 2] > 1e-5)
    )
    if not valid.any():
        return float("inf")
    return float(np.median(np.linalg.norm(projected[valid] - target[valid], axis=1)))


def _constrain_se3_reprojection(
    rotation_vector: np.ndarray,
    translation: np.ndarray,
    pivot: np.ndarray,
    joints: np.ndarray,
    target_pixels: np.ndarray,
    intrinsics: np.ndarray,
    maximum_increase: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Keep the largest SE(3) correction allowed by the 2D reprojection bound.

    Radial wrist translation is not perfectly projection preserving for an
    articulated hand: points away from the wrist ray still move in the image.
    The previous implementation therefore discarded the *entire* radial
    correction whenever it exceeded the pixel budget by any amount.  This made
    the result discontinuous and left tens of millimetres of correctable depth
    error.  We first solve a bounded one-dimensional radial step, then spend the
    remaining reprojection budget on rotation and tangential translation.
    """

    vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    shift = np.asarray(translation, dtype=np.float64).reshape(3)
    center = np.asarray(pivot, dtype=np.float64).reshape(3)
    ray = center / (np.linalg.norm(center) + 1e-12)
    radial = float(shift @ ray) * ray
    tangential = shift - radial
    baseline = _median_reprojection_error(joints, target_pixels, intrinsics)
    allowed = baseline + float(maximum_increase)

    def error(radial_alpha: float, tangential_alpha: float) -> float:
        transformed = _apply_wrist_se3(
            joints,
            center,
            vector * tangential_alpha,
            radial * radial_alpha + tangential * tangential_alpha,
        )
        return _median_reprojection_error(transformed, target_pixels, intrinsics)

    full_error = error(1.0, 1.0)
    if full_error <= allowed:
        return vector.astype(np.float32), shift.astype(np.float32), 1.0, baseline, full_error

    radial_alpha = 1.0
    if error(1.0, 0.0) > allowed:
        low, high = 0.0, 1.0
        for _ in range(16):
            middle = 0.5 * (low + high)
            if error(middle, 0.0) <= allowed:
                low = middle
            else:
                high = middle
        radial_alpha = low

    accepted_radial = radial * radial_alpha
    if error(radial_alpha, 1.0) <= allowed:
        return (
            vector.astype(np.float32),
            (accepted_radial + tangential).astype(np.float32),
            float(radial_alpha),
            baseline,
            error(radial_alpha, 1.0),
        )

    low, high = 0.0, 1.0
    for _ in range(14):
        middle = 0.5 * (low + high)
        if error(radial_alpha, middle) <= allowed:
            low = middle
        else:
            high = middle
    constrained_vector = vector * low
    constrained_translation = accepted_radial + tangential * low
    constrained_error = error(radial_alpha, low)
    return (
        constrained_vector.astype(np.float32),
        constrained_translation.astype(np.float32),
        float(min(radial_alpha, low)),
        baseline,
        constrained_error,
    )


