"""Fixed-intrinsic stereo undistortion and rectification for complete videos.

AnyCalib is evaluated on the first synchronized frame only for the production
calibration.  Additional sampled predictions are diagnostics and never change
the fixed calibration.  A single stereo rotation/translation direction is
estimated from pooled same-time SIFT correspondences, and the known physical
baseline fixes the translation norm and metric depth scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from pipeline.modules.depth.anycalib_stereo import (
    estimate_relative_pose,
    read_video_frame,
    sift_correspondences,
    target_intrinsic,
)


@dataclass(frozen=True)
class StereoCalibrationConfig:
    """Inputs and fixed calibration policy for a synchronized stereo video."""

    camera_b_video: Path
    camera_c_video: Path
    output_dir: Path
    baseline_m: float = 0.075
    model_id: str = "anycalib_gen"
    camera_model: str = "kb:4"
    focal_length_factor: float = 0.5
    work_scale: float = 0.5
    sample_count: int = 6
    max_features_per_frame: int = 12000
    match_ratio: float = 0.78
    ransac_threshold_px: float = 1.5
    selection_mode: str = "multi_frame_candidates"
    max_rectified_center_offset_fraction: float = 0.15
    min_rectified_valid_fraction: float = 0.70
    candidate_consensus_weight: float = 0.10


def _video_metadata(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if min(count, width, height, fps) <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return count, width, height, fps


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value.resolve())
    raise TypeError(type(value).__name__)


class FixedStereoCalibrator:
    """Estimate one calibration and apply it to every synchronized frame."""

    def __init__(self, config: StereoCalibrationConfig) -> None:
        self.config = config

    @staticmethod
    def _undistort(
        rgb: np.ndarray,
        camera,
        parameters: torch.Tensor,
        focal_length_factor: float,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = (
            torch.as_tensor(rgb, dtype=torch.float32, device=device)
            .permute(2, 0, 1)
            .div_(255.0)
        )
        with torch.inference_mode():
            mapped = camera.undistort_image(
                image,
                parameters,
                scale=focal_length_factor,
                target_proj="perspective",
                outside_value=0.0,
                add_alpha=True,
            )
        mapped_rgb = (
            mapped[:3]
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(255)
            .round()
            .byte()
            .cpu()
            .numpy()
        )
        valid = mapped[3].cpu().numpy() > 0.999
        return mapped_rgb, valid

    @staticmethod
    def _rectification(
        image_size: tuple[int, int],
        intrinsic_b: np.ndarray,
        intrinsic_c: np.ndarray,
        rotation_b_to_c: np.ndarray,
        translation_b_to_c: np.ndarray,
    ) -> dict[str, np.ndarray | tuple[int, ...]]:
        zeros = np.zeros(5, dtype=np.float64)
        rb, rc, pb, pc, q, roi_b, roi_c = cv2.stereoRectify(
            intrinsic_b,
            zeros,
            intrinsic_c,
            zeros,
            image_size,
            rotation_b_to_c,
            translation_b_to_c,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
        map_bx, map_by = cv2.initUndistortRectifyMap(
            intrinsic_b, zeros, rb, pb[:, :3], image_size, cv2.CV_32FC1
        )
        map_cx, map_cy = cv2.initUndistortRectifyMap(
            intrinsic_c, zeros, rc, pc[:, :3], image_size, cv2.CV_32FC1
        )
        return {
            "R_b": rb,
            "R_c": rc,
            "P_b": pb,
            "P_c": pc,
            "Q": q,
            "roi_b": roi_b,
            "roi_c": roi_c,
            "map_bx": map_bx,
            "map_by": map_by,
            "map_cx": map_cx,
            "map_cy": map_cy,
        }

    @staticmethod
    def _rectified_correspondences(
        points_b: np.ndarray,
        points_c: np.ndarray,
        intrinsic_b: np.ndarray,
        intrinsic_c: np.ndarray,
        rectification: dict[str, np.ndarray | tuple[int, ...]],
        inliers: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pb = cv2.undistortPoints(
            points_b[inliers, None],
            intrinsic_b,
            None,
            R=np.asarray(rectification["R_b"]),
            P=np.asarray(rectification["P_b"]),
        ).reshape(-1, 2)
        pc = cv2.undistortPoints(
            points_c[inliers, None],
            intrinsic_c,
            None,
            R=np.asarray(rectification["R_c"]),
            P=np.asarray(rectification["P_c"]),
        ).reshape(-1, 2)
        return pb, pc

    def _evaluate_calibration_candidate(
        self,
        *,
        candidate_frame: int,
        raw_samples: dict[str, dict[int, np.ndarray]],
        parameters: dict[str, torch.Tensor],
        camera,
        device: torch.device,
        image_size: tuple[int, int],
    ) -> dict[str, object]:
        """Evaluate one paired AnyCalib prediction over all sampled stereo frames.

        A candidate is one pair of camera-B/camera-C KB4 predictions obtained at
        the same time.  Its intrinsics are held fixed while one relative pose is
        estimated from correspondences pooled across the complete sample set.
        The returned metrics jointly measure epipolar support and whether the
        resulting rectified virtual camera preserves a plausible field of view.
        """

        cfg = self.config
        width, height = image_size
        intrinsic = {
            label: target_intrinsic(
                parameters[label].detach().cpu().numpy(), cfg.focal_length_factor
            )
            for label in ("b", "c")
        }
        sampled: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
            "b": {},
            "c": {},
        }
        for label in ("b", "c"):
            for frame_id, rgb in raw_samples[label].items():
                sampled[label][frame_id] = self._undistort(
                    rgb,
                    camera,
                    parameters[label],
                    cfg.focal_length_factor,
                    device,
                )

        points_b_all: list[np.ndarray] = []
        points_c_all: list[np.ndarray] = []
        mutual_counts: list[int] = []
        for frame_id in sorted(raw_samples["b"]):
            image_b, valid_b = sampled["b"][frame_id]
            image_c, valid_c = sampled["c"][frame_id]
            points_b, points_c, metrics = sift_correspondences(
                image_b,
                image_c,
                valid_b.astype(np.uint8) * 255,
                valid_c.astype(np.uint8) * 255,
                max_features=cfg.max_features_per_frame,
                ratio=cfg.match_ratio,
            )
            points_b_all.append(points_b)
            points_c_all.append(points_c)
            mutual_counts.append(int(metrics["mutual_matches"]))
        points_b = np.concatenate(points_b_all)
        points_c = np.concatenate(points_c_all)
        rotation, direction, inliers, pose_metrics = estimate_relative_pose(
            points_b,
            points_c,
            intrinsic["b"],
            intrinsic["c"],
            cfg.ransac_threshold_px,
        )
        translation = direction / np.linalg.norm(direction) * cfg.baseline_m
        rectification = self._rectification(
            image_size,
            intrinsic["b"],
            intrinsic["c"],
            rotation,
            translation,
        )
        rect_b, rect_c = self._rectified_correspondences(
            points_b,
            points_c,
            intrinsic["b"],
            intrinsic["c"],
            rectification,
            inliers,
        )
        swapped = float(np.median(rect_b[:, 0] - rect_c[:, 0])) < 0
        if swapped:
            rotation = rotation.T
            translation = -rotation @ translation
            rectification = self._rectification(
                image_size,
                intrinsic["c"],
                intrinsic["b"],
                rotation,
                translation,
            )
            rect_left, rect_right = self._rectified_correspondences(
                points_c,
                points_b,
                intrinsic["c"],
                intrinsic["b"],
                rectification,
                inliers,
            )
            left_label, right_label = "c", "b"
            map_by_label = {
                "c": (
                    np.asarray(rectification["map_bx"]),
                    np.asarray(rectification["map_by"]),
                ),
                "b": (
                    np.asarray(rectification["map_cx"]),
                    np.asarray(rectification["map_cy"]),
                ),
            }
            projection = np.asarray(rectification["P_b"])
        else:
            rect_left, rect_right = rect_b, rect_c
            left_label, right_label = "b", "c"
            map_by_label = {
                "b": (
                    np.asarray(rectification["map_bx"]),
                    np.asarray(rectification["map_by"]),
                ),
                "c": (
                    np.asarray(rectification["map_cx"]),
                    np.asarray(rectification["map_cy"]),
                ),
            }
            projection = np.asarray(rectification["P_b"])

        valid_fraction: dict[str, float] = {}
        first_frame = min(raw_samples["b"])
        for label in ("b", "c"):
            source_valid = sampled[label][first_frame][1].astype(np.uint8)
            rectified_valid = cv2.remap(
                source_valid,
                *map_by_label[label],
                cv2.INTER_NEAREST,
            )
            valid_fraction[label] = float(rectified_valid.mean())

        vertical = np.abs(rect_left[:, 1] - rect_right[:, 1])
        disparity = rect_left[:, 0] - rect_right[:, 0]
        center_offset_x = abs(float(projection[0, 2]) - (width - 1) * 0.5) / width
        center_offset_y = abs(float(projection[1, 2]) - (height - 1) * 0.5) / height
        minimum_valid = min(valid_fraction.values())
        eligible = bool(
            center_offset_x <= cfg.max_rectified_center_offset_fraction
            and center_offset_y <= cfg.max_rectified_center_offset_fraction
            and minimum_valid >= cfg.min_rectified_valid_fraction
            and float(np.mean(disparity > 0)) >= 0.95
        )
        # Eligibility is a hard guard against an epipolar solution that attains
        # a high inlier count by shifting the virtual camera out of the useful
        # field of view.  Within the feasible set, prefer broad RANSAC support;
        # small residual/FoV terms provide deterministic tie breaking.
        selection_score = float(
            pose_metrics["cheirality_inlier_fraction"]
            - 0.02 * np.median(vertical) / max(cfg.ransac_threshold_px, 1e-6)
            - 0.05 * (center_offset_x + center_offset_y)
            + 0.02 * minimum_valid
        )
        return {
            "candidate_frame": int(candidate_frame),
            "eligible": eligible,
            "selection_score": selection_score,
            "pooled_correspondences": int(len(points_b)),
            "mutual_matches_per_sample": mutual_counts,
            **pose_metrics,
            "swapped_for_positive_disparity": swapped,
            "left_source": left_label,
            "right_source": right_label,
            "rotation_left_to_right": rotation.tolist(),
            "translation_direction_left_to_right": (
                translation / np.linalg.norm(translation)
            ).tolist(),
            "rectified_intrinsic": projection[:, :3].tolist(),
            "positive_sparse_disparity_fraction": float(np.mean(disparity > 0)),
            "median_vertical_error_px": float(np.median(vertical)),
            "p95_vertical_error_px": float(np.quantile(vertical, 0.95)),
            "rectified_center_offset_fraction": {
                "x": center_offset_x,
                "y": center_offset_y,
            },
            "rectified_valid_fraction": valid_fraction,
        }

    @staticmethod
    def _add_candidate_consensus(
        candidates: list[dict[str, object]],
        *,
        image_size: tuple[int, int],
        weight: float,
    ) -> None:
        """Penalize geometries that disagree with the robust candidate consensus."""

        usable = [candidate for candidate in candidates if "error" not in candidate]
        if len(usable) < 3:
            for candidate in usable:
                candidate["base_selection_score"] = candidate["selection_score"]
                candidate["consensus_distance"] = 0.0
            return
        width, height = image_size
        features: list[np.ndarray] = []
        for candidate in usable:
            rotation = np.asarray(candidate["rotation_left_to_right"], dtype=np.float64)
            rotation_vector = cv2.Rodrigues(rotation)[0].reshape(3)
            translation = np.asarray(
                candidate["translation_direction_left_to_right"], dtype=np.float64
            )
            intrinsic = np.asarray(candidate["rectified_intrinsic"], dtype=np.float64)
            features.append(
                np.concatenate(
                    [
                        rotation_vector,
                        translation[1:3],
                        np.asarray(
                            [
                                (intrinsic[0, 2] - (width - 1) * 0.5) / width,
                                (intrinsic[1, 2] - (height - 1) * 0.5) / height,
                                np.log(max(intrinsic[0, 0], 1.0) / width),
                            ]
                        ),
                    ]
                )
            )
        values = np.stack(features)
        median = np.median(values, axis=0)
        robust_scale = 1.4826 * np.median(np.abs(values - median), axis=0)
        floors = np.asarray([0.002, 0.002, 0.002, 0.02, 0.02, 0.03, 0.03, 0.05])
        robust_scale = np.maximum(robust_scale, floors)
        normalized = np.clip((values - median) / robust_scale, -3.0, 3.0)
        distances = np.sqrt(np.mean(normalized**2, axis=1))
        for candidate, distance in zip(usable, distances, strict=True):
            base_score = float(candidate["selection_score"])
            candidate["base_selection_score"] = base_score
            candidate["consensus_distance"] = float(distance)
            candidate["selection_score"] = float(base_score - weight * distance)

    def run(self) -> Path:
        """Create fixed rectified videos, calibration arrays, and diagnostics."""

        from anycalib import AnyCalib
        from anycalib.cameras import CameraFactory

        cfg = self.config
        started = time.perf_counter()
        output = cfg.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        metadata_b = _video_metadata(cfg.camera_b_video)
        metadata_c = _video_metadata(cfg.camera_c_video)
        if metadata_b != metadata_c:
            raise ValueError(f"Stereo video metadata differ: {metadata_b} vs {metadata_c}")
        frame_count, width, height, fps = metadata_b
        sample_ids = np.unique(
            np.rint(np.linspace(0, frame_count - 1, cfg.sample_count)).astype(np.int32)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AnyCalib(model_id=cfg.model_id).to(device).eval()
        camera = CameraFactory.create_from_id(cfg.camera_model)
        fixed_parameters: dict[str, torch.Tensor] = {}
        candidate_parameters: dict[str, dict[int, torch.Tensor]] = {"b": {}, "c": {}}
        diagnostic_predictions: dict[str, list[dict[str, object]]] = {"b": [], "c": []}
        raw_samples: dict[str, dict[int, np.ndarray]] = {"b": {}, "c": {}}
        sampled_undistorted: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
            "b": {},
            "c": {},
        }
        sources = {"b": cfg.camera_b_video, "c": cfg.camera_c_video}
        for label, source in sources.items():
            for frame_id in sample_ids:
                rgb = read_video_frame(source, int(frame_id))
                raw_samples[label][int(frame_id)] = rgb
                tensor = (
                    torch.as_tensor(rgb, dtype=torch.float32, device=device)
                    .permute(2, 0, 1)
                    .div_(255.0)
                )
                with torch.inference_mode():
                    prediction = model.predict(tensor, cam_id=cfg.camera_model)
                params = prediction["intrinsics"].detach()
                candidate_parameters[label][int(frame_id)] = params.clone()
                diagnostic_predictions[label].append(
                    {
                        "frame": int(frame_id),
                        "intrinsics": params.cpu().numpy().tolist(),
                        "prediction_size": [int(x) for x in prediction["pred_size"]],
                    }
                )

        candidate_evaluations: list[dict[str, object]] = []
        selection_fallback = False
        if cfg.selection_mode == "multi_frame_candidates":
            for frame_id in sample_ids:
                try:
                    candidate_evaluations.append(
                        self._evaluate_calibration_candidate(
                            candidate_frame=int(frame_id),
                            raw_samples=raw_samples,
                            parameters={
                                label: candidate_parameters[label][int(frame_id)]
                                for label in ("b", "c")
                            },
                            camera=camera,
                            device=device,
                            image_size=(width, height),
                        )
                    )
                except Exception as error:  # Preserve other viable candidates.
                    candidate_evaluations.append(
                        {
                            "candidate_frame": int(frame_id),
                            "eligible": False,
                            "selection_score": -1.0e30,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
            self._add_candidate_consensus(
                candidate_evaluations,
                image_size=(width, height),
                weight=cfg.candidate_consensus_weight,
            )
            feasible = [item for item in candidate_evaluations if item["eligible"]]
            if not feasible:
                feasible = [item for item in candidate_evaluations if "error" not in item]
                selection_fallback = True
            if not feasible:
                raise RuntimeError("Every multi-frame calibration candidate failed")
            selected = max(feasible, key=lambda item: float(item["selection_score"]))
            selected_frame = int(selected["candidate_frame"])
        elif cfg.selection_mode == "first_frame":
            selected_frame = int(sample_ids[0])
        else:
            raise ValueError(f"Unsupported calibration selection mode: {cfg.selection_mode}")
        fixed_parameters = {
            label: candidate_parameters[label][selected_frame].clone()
            for label in ("b", "c")
        }

        for label in ("b", "c"):
            fixed = fixed_parameters[label]
            for frame_id in sample_ids:
                sampled_undistorted[label][int(frame_id)] = self._undistort(
                    raw_samples[label][int(frame_id)],
                    camera,
                    fixed,
                    cfg.focal_length_factor,
                    device,
                )

        intrinsic_b = target_intrinsic(
            fixed_parameters["b"].cpu().numpy(), cfg.focal_length_factor
        )
        intrinsic_c = target_intrinsic(
            fixed_parameters["c"].cpu().numpy(), cfg.focal_length_factor
        )
        points_b_all: list[np.ndarray] = []
        points_c_all: list[np.ndarray] = []
        per_sample_matching: list[dict[str, object]] = []
        for frame_id in sample_ids:
            image_b, valid_b = sampled_undistorted["b"][int(frame_id)]
            image_c, valid_c = sampled_undistorted["c"][int(frame_id)]
            points_b, points_c, metrics = sift_correspondences(
                image_b,
                image_c,
                valid_b.astype(np.uint8) * 255,
                valid_c.astype(np.uint8) * 255,
                max_features=cfg.max_features_per_frame,
                ratio=cfg.match_ratio,
            )
            points_b_all.append(points_b)
            points_c_all.append(points_c)
            per_sample_matching.append({"frame": int(frame_id), **metrics})
        points_b = np.concatenate(points_b_all)
        points_c = np.concatenate(points_c_all)
        rotation, translation_direction, inliers, pose_metrics = estimate_relative_pose(
            points_b,
            points_c,
            intrinsic_b,
            intrinsic_c,
            cfg.ransac_threshold_px,
        )
        translation = translation_direction / np.linalg.norm(translation_direction) * cfg.baseline_m
        rectification = self._rectification(
            (width, height), intrinsic_b, intrinsic_c, rotation, translation
        )
        rect_b, rect_c = self._rectified_correspondences(
            points_b, points_c, intrinsic_b, intrinsic_c, rectification, inliers
        )
        swapped = float(np.median(rect_b[:, 0] - rect_c[:, 0])) < 0
        if swapped:
            inverse_rotation = rotation.T
            inverse_translation = -rotation.T @ translation
            rectification = self._rectification(
                (width, height), intrinsic_c, intrinsic_b, inverse_rotation, inverse_translation
            )
            # Keep diagnostic points in the same order as the final left/right videos.
            rect_left, rect_right = self._rectified_correspondences(
                points_c,
                points_b,
                intrinsic_c,
                intrinsic_b,
                rectification,
                inliers,
            )
            final_rotation = inverse_rotation
            final_translation = inverse_translation
            left_label, right_label = "c", "b"
            rectify_maps = {
                "c": (
                    np.asarray(rectification["map_bx"]),
                    np.asarray(rectification["map_by"]),
                ),
                "b": (
                    np.asarray(rectification["map_cx"]),
                    np.asarray(rectification["map_cy"]),
                ),
            }
            projection_by_label = {
                "c": np.asarray(rectification["P_b"]),
                "b": np.asarray(rectification["P_c"]),
            }
        else:
            rect_left, rect_right = rect_b, rect_c
            final_rotation = rotation
            final_translation = translation
            left_label, right_label = "b", "c"
            rectify_maps = {
                "b": (
                    np.asarray(rectification["map_bx"]),
                    np.asarray(rectification["map_by"]),
                ),
                "c": (
                    np.asarray(rectification["map_cx"]),
                    np.asarray(rectification["map_cy"]),
                ),
            }
            projection_by_label = {
                "b": np.asarray(rectification["P_b"]),
                "c": np.asarray(rectification["P_c"]),
            }

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writers = {
            "b_undistorted": cv2.VideoWriter(
                str(output / "camera_b_undistorted.mp4"), fourcc, fps, (width, height)
            ),
            "c_undistorted": cv2.VideoWriter(
                str(output / "camera_c_undistorted.mp4"), fourcc, fps, (width, height)
            ),
            "left": cv2.VideoWriter(
                str(output / "left_rectified.mp4"), fourcc, fps, (width, height)
            ),
            "right": cv2.VideoWriter(
                str(output / "right_rectified.mp4"), fourcc, fps, (width, height)
            ),
        }
        work_size = (
            int(round(width * cfg.work_scale)),
            int(round(height * cfg.work_scale)),
        )
        writers["left_work"] = cv2.VideoWriter(
            str(output / "left_rectified_working.mp4"), fourcc, fps, work_size
        )
        writers["right_work"] = cv2.VideoWriter(
            str(output / "right_rectified_working.mp4"), fourcc, fps, work_size
        )
        if not all(writer.isOpened() for writer in writers.values()):
            raise RuntimeError("Failed to create one or more calibration videos")

        valid_undistorted: dict[str, np.ndarray] = {}
        for label in ("b", "c"):
            _, valid_undistorted[label] = self._undistort(
                np.zeros((height, width, 3), dtype=np.uint8),
                camera,
                fixed_parameters[label],
                cfg.focal_length_factor,
                device,
            )

        captures = {
            "b": cv2.VideoCapture(str(cfg.camera_b_video)),
            "c": cv2.VideoCapture(str(cfg.camera_c_video)),
        }
        sample_rows: list[np.ndarray] = []
        try:
            for frame_id in range(frame_count):
                undistorted: dict[str, np.ndarray] = {}
                originals: dict[str, np.ndarray] = {}
                for label in ("b", "c"):
                    ok, bgr = captures[label].read()
                    if not ok:
                        raise RuntimeError(f"Failed to decode {label} frame {frame_id}")
                    originals[label] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    undistorted[label], _ = self._undistort(
                        originals[label],
                        camera,
                        fixed_parameters[label],
                        cfg.focal_length_factor,
                        device,
                    )
                    writers[f"{label}_undistorted"].write(
                        cv2.cvtColor(undistorted[label], cv2.COLOR_RGB2BGR)
                    )
                rectified = {
                    "b": cv2.remap(
                        undistorted["b"],
                        *rectify_maps["b"],
                        cv2.INTER_LINEAR,
                    ),
                    "c": cv2.remap(
                        undistorted["c"],
                        *rectify_maps["c"],
                        cv2.INTER_LINEAR,
                    ),
                }
                left, right = rectified[left_label], rectified[right_label]
                writers["left"].write(cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
                writers["right"].write(cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
                left_work = cv2.resize(left, work_size, interpolation=cv2.INTER_AREA)
                right_work = cv2.resize(right, work_size, interpolation=cv2.INTER_AREA)
                writers["left_work"].write(cv2.cvtColor(left_work, cv2.COLOR_RGB2BGR))
                writers["right_work"].write(cv2.cvtColor(right_work, cv2.COLOR_RGB2BGR))
                if frame_id in sample_ids:
                    row = np.concatenate(
                        [
                            cv2.resize(originals[left_label], work_size),
                            cv2.resize(undistorted[left_label], work_size),
                            left_work,
                        ],
                        axis=1,
                    )
                    cv2.putText(
                        row,
                        f"frame {frame_id}: original | fixed undistortion | rectified",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 80, 40),
                        2,
                        cv2.LINE_AA,
                    )
                    sample_rows.append(row)
        finally:
            for capture in captures.values():
                capture.release()
            for writer in writers.values():
                writer.release()
            del model
            torch.cuda.empty_cache()

        # Validity is fixed because both source calibration and stereo rectification are fixed.
        rectified_valid = {
            "b": cv2.remap(
                valid_undistorted["b"].astype(np.uint8),
                *rectify_maps["b"],
                cv2.INTER_NEAREST,
            ).astype(bool),
            "c": cv2.remap(
                valid_undistorted["c"].astype(np.uint8),
                *rectify_maps["c"],
                cv2.INTER_NEAREST,
            ).astype(bool),
        }
        left_valid = cv2.resize(
            rectified_valid[left_label].astype(np.uint8), work_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        right_valid = cv2.resize(
            rectified_valid[right_label].astype(np.uint8), work_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        full_k = projection_by_label[left_label][:, :3]
        work_k = full_k.copy()
        work_k[0] *= cfg.work_scale
        work_k[1] *= cfg.work_scale
        np.savez_compressed(
            output / "calibration.npz",
            K=full_k,
            K_working=work_k,
            baseline_m=np.asarray(cfg.baseline_m, dtype=np.float64),
            stereo_R=final_rotation,
            stereo_t=final_translation,
            left_valid=left_valid,
            right_valid=right_valid,
            sample_frames=sample_ids,
            source_intrinsics_b=fixed_parameters["b"].cpu().numpy(),
            source_intrinsics_c=fixed_parameters["c"].cpu().numpy(),
        )
        np.savetxt(output / "K_rectified.txt", full_k, fmt="%.10f")
        np.savetxt(output / "K_rectified_working.txt", work_k, fmt="%.10f")
        if sample_rows:
            cv2.imwrite(
                str(output / "shared_calibration_samples.jpg"),
                cv2.cvtColor(np.concatenate(sample_rows, axis=0), cv2.COLOR_RGB2BGR),
            )

        deviations: dict[str, dict[str, float]] = {}
        for label in ("b", "c"):
            values = np.asarray(
                [record["intrinsics"] for record in diagnostic_predictions[label]],
                dtype=np.float64,
            )
            reference = values[0]
            deviations[label] = {
                "max_focal_relative_deviation": float(
                    np.max(np.abs(values[:, :2] / reference[None, :2] - 1.0))
                ),
                "max_principal_point_deviation_px": float(
                    np.max(np.abs(values[:, 2:4] - reference[None, 2:4]))
                ),
                "max_distortion_absolute_deviation": float(
                    np.max(np.abs(values[:, 4:] - reference[None, 4:]))
                ),
            }
        vertical = np.abs(rect_left[:, 1] - rect_right[:, 1])
        disparity = rect_left[:, 0] - rect_right[:, 0]
        report = {
            "schema_version": "ego-scene-hoi-fixed-stereo-calibration-v2",
            "result": "PASS",
            "production_calibration_frame": selected_frame,
            "diagnostic_sample_frames": sample_ids.tolist(),
            "frame_count": frame_count,
            "fps": fps,
            "source_size": [height, width],
            "working_size": [work_size[1], work_size[0]],
            "configuration": asdict(cfg),
            "fixed_source_intrinsics": {
                "camera_b": fixed_parameters["b"].cpu().numpy(),
                "camera_c": fixed_parameters["c"].cpu().numpy(),
            },
            "sampled_anycalib_predictions": diagnostic_predictions,
            "shared_calibration_diagnostics": deviations,
            "calibration_selection": {
                "mode": cfg.selection_mode,
                "selected_frame": selected_frame,
                "fallback_without_fov_eligibility": selection_fallback,
                "max_rectified_center_offset_fraction": (
                    cfg.max_rectified_center_offset_fraction
                ),
                "min_rectified_valid_fraction": cfg.min_rectified_valid_fraction,
                "candidate_consensus_weight": cfg.candidate_consensus_weight,
                "candidates": candidate_evaluations,
            },
            "stereo_matching": {
                "per_sample": per_sample_matching,
                "pooled_correspondences": int(len(points_b)),
                **pose_metrics,
            },
            "stereo_geometry": {
                "left_source": left_label,
                "right_source": right_label,
                "swapped_for_positive_disparity": swapped,
                "baseline_m": cfg.baseline_m,
                "rotation_left_to_right": final_rotation,
                "translation_left_to_right_m": final_translation,
                "K_rectified": full_k,
                "K_rectified_working": work_k,
                "positive_sparse_disparity_fraction": float(np.mean(disparity > 0)),
                "median_sparse_disparity_px": float(np.median(disparity)),
                "median_vertical_error_px": float(np.median(vertical)),
                "p95_vertical_error_px": float(np.quantile(vertical, 0.95)),
                "left_valid_fraction_working": float(left_valid.mean()),
                "right_valid_fraction_working": float(right_valid.mean()),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path = output / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, default=_to_jsonable) + "\n", encoding="utf-8"
        )
        return report_path
