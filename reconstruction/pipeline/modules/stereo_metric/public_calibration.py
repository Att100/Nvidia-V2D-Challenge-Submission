"""Rectify the NVIDIA Track-3 ego stereo videos with the released calibration.

The released ego videos are stored in their display orientation (180 degrees
relative to the OAK sensor mounting).  The metadata consequently defines
``ego_cam_c`` as the displayed left camera and ``ego_cam_b`` as the displayed
right camera.  This module deliberately follows that released convention and
does not estimate any camera parameter from image content.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class PublicStereoCalibrationConfig:
    """Inputs for deterministic rectification from public camera metadata."""

    left_video: Path
    right_video: Path
    calibration_json: Path
    output_dir: Path
    work_scale: float = 0.5
    sample_count: int = 12
    rectification_alpha: float = 0.0
    max_features_per_frame: int = 12000
    match_ratio: float = 0.78


def _metadata(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    try:
        values = (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        capture.release()
    if min(values) <= 0:
        raise ValueError(f"Invalid video metadata for {path}: {values}")
    return values


def _read_frame(path: Path, frame_id: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_id} from {path}")
    return frame


def _sift_matches(
    left: np.ndarray,
    right: np.ndarray,
    *,
    maximum_features: int,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    detector = cv2.SIFT_create(nfeatures=maximum_features)
    keypoints_l, descriptors_l = detector.detectAndCompute(left, None)
    keypoints_r, descriptors_r = detector.detectAndCompute(right, None)
    if descriptors_l is None or descriptors_r is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        descriptors_l, descriptors_r, k=2
    )
    accepted = [first for first, second in candidates if first.distance < ratio * second.distance]
    points_l = np.asarray(
        [keypoints_l[item.queryIdx].pt for item in accepted], dtype=np.float32
    )
    points_r = np.asarray(
        [keypoints_r[item.trainIdx].pt for item in accepted], dtype=np.float32
    )
    return points_l, points_r


class PublicStereoCalibrator:
    """Apply the released metric intrinsics, distortion and stereo extrinsics."""

    def __init__(self, config: PublicStereoCalibrationConfig) -> None:
        self.config = config

    def run(self) -> Path:
        """Write pipeline-compatible rectified videos and calibration arrays."""

        cfg = self.config
        started = time.perf_counter()
        output = cfg.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        meta_l = _metadata(cfg.left_video)
        meta_r = _metadata(cfg.right_video)
        if meta_l != meta_r:
            raise ValueError(f"Stereo video metadata differ: {meta_l} vs {meta_r}")
        frame_count, width, height, fps = meta_l

        metadata = json.loads(cfg.calibration_json.read_text(encoding="utf-8"))
        pair = next(
            item for item in metadata["stereo_pairs"] if item["camera"] == "ego"
        )
        if pair["left"] != "ego_cam_c" or pair["right"] != "ego_cam_b":
            raise ValueError(f"Unexpected released ego stereo order: {pair}")
        left_camera = metadata["cameras"][pair["left"]]
        right_camera = metadata["cameras"][pair["right"]]
        expected_size = tuple(int(value) for value in left_camera["resolution"])
        if expected_size != (width, height):
            raise ValueError(
                f"Calibration resolution {expected_size} does not match video {(width, height)}"
            )

        intrinsic_l = np.asarray(left_camera["intrinsic_matrix"], dtype=np.float64)
        intrinsic_r = np.asarray(right_camera["intrinsic_matrix"], dtype=np.float64)
        distortion_l = np.asarray(
            left_camera["distortion_coefficients"], dtype=np.float64
        )
        distortion_r = np.asarray(
            right_camera["distortion_coefficients"], dtype=np.float64
        )
        transform = np.asarray(pair["left_to_right"], dtype=np.float64)
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        baseline = float(pair["baseline_m"])
        if not np.isclose(np.linalg.norm(translation), baseline, rtol=0.02, atol=1.0e-4):
            raise ValueError(
                f"Released translation norm {np.linalg.norm(translation)} disagrees with "
                f"baseline {baseline}"
            )

        rect_l, rect_r, proj_l, proj_r, q, roi_l, roi_r = cv2.stereoRectify(
            intrinsic_l,
            distortion_l,
            intrinsic_r,
            distortion_r,
            (width, height),
            rotation,
            translation,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=float(cfg.rectification_alpha),
        )
        map_lx, map_ly = cv2.initUndistortRectifyMap(
            intrinsic_l,
            distortion_l,
            rect_l,
            proj_l[:, :3],
            (width, height),
            cv2.CV_32FC1,
        )
        map_rx, map_ry = cv2.initUndistortRectifyMap(
            intrinsic_r,
            distortion_r,
            rect_r,
            proj_r[:, :3],
            (width, height),
            cv2.CV_32FC1,
        )

        full_k = proj_l[:, :3].copy()
        work_size = (
            int(round(width * cfg.work_scale)),
            int(round(height * cfg.work_scale)),
        )
        work_k = full_k.copy()
        work_k[0] *= cfg.work_scale
        work_k[1] *= cfg.work_scale
        source_valid = np.ones((height, width), dtype=np.uint8)
        left_valid_full = cv2.remap(
            source_valid, map_lx, map_ly, cv2.INTER_NEAREST, borderValue=0
        ).astype(bool)
        right_valid_full = cv2.remap(
            source_valid, map_rx, map_ry, cv2.INTER_NEAREST, borderValue=0
        ).astype(bool)
        left_valid = cv2.resize(
            left_valid_full.astype(np.uint8), work_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        right_valid = cv2.resize(
            right_valid_full.astype(np.uint8), work_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_specs = {
            "camera_c_undistorted.mp4": (width, height),
            "camera_b_undistorted.mp4": (width, height),
            "left_rectified.mp4": (width, height),
            "right_rectified.mp4": (width, height),
            "left_rectified_working.mp4": work_size,
            "right_rectified_working.mp4": work_size,
        }
        writers = {
            name: cv2.VideoWriter(str(output / name), fourcc, fps, size)
            for name, size in writer_specs.items()
        }
        if not all(writer.isOpened() for writer in writers.values()):
            raise RuntimeError("Failed to create one or more public-calibration videos")
        capture_l = cv2.VideoCapture(str(cfg.left_video))
        capture_r = cv2.VideoCapture(str(cfg.right_video))
        sample_ids = np.unique(
            np.rint(np.linspace(0, frame_count - 1, cfg.sample_count)).astype(np.int32)
        )
        sample_set = set(int(value) for value in sample_ids)
        diagnostics: list[dict[str, float | int]] = []
        sample_rows: list[np.ndarray] = []
        try:
            for frame_id in range(frame_count):
                ok_l, image_l = capture_l.read()
                ok_r, image_r = capture_r.read()
                if not ok_l or not ok_r:
                    raise RuntimeError(f"Failed to decode stereo frame {frame_id}")
                undistorted_l = cv2.undistort(
                    image_l, intrinsic_l, distortion_l, None, intrinsic_l
                )
                undistorted_r = cv2.undistort(
                    image_r, intrinsic_r, distortion_r, None, intrinsic_r
                )
                rectified_l = cv2.remap(
                    image_l, map_lx, map_ly, cv2.INTER_LINEAR, borderValue=0
                )
                rectified_r = cv2.remap(
                    image_r, map_rx, map_ry, cv2.INTER_LINEAR, borderValue=0
                )
                work_l = cv2.resize(rectified_l, work_size, interpolation=cv2.INTER_AREA)
                work_r = cv2.resize(rectified_r, work_size, interpolation=cv2.INTER_AREA)
                writers["camera_c_undistorted.mp4"].write(undistorted_l)
                writers["camera_b_undistorted.mp4"].write(undistorted_r)
                writers["left_rectified.mp4"].write(rectified_l)
                writers["right_rectified.mp4"].write(rectified_r)
                writers["left_rectified_working.mp4"].write(work_l)
                writers["right_rectified_working.mp4"].write(work_r)

                if frame_id in sample_set:
                    points_l, points_r = _sift_matches(
                        rectified_l,
                        rectified_r,
                        maximum_features=cfg.max_features_per_frame,
                        ratio=cfg.match_ratio,
                    )
                    if len(points_l):
                        vertical = np.abs(points_l[:, 1] - points_r[:, 1])
                        disparity = points_l[:, 0] - points_r[:, 0]
                        diagnostics.append(
                            {
                                "frame": frame_id,
                                "matches": int(len(points_l)),
                                "median_vertical_error_px": float(np.median(vertical)),
                                "p95_vertical_error_px": float(np.quantile(vertical, 0.95)),
                                "median_disparity_px": float(np.median(disparity)),
                                "positive_disparity_fraction": float(np.mean(disparity > 0)),
                            }
                        )
                    row = np.concatenate(
                        [
                            cv2.resize(image_l, work_size),
                            work_l,
                            work_r,
                        ],
                        axis=1,
                    )
                    cv2.putText(
                        row,
                        f"frame {frame_id}: ego_cam_c | rectified left | rectified right",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (30, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    sample_rows.append(row)
        finally:
            capture_l.release()
            capture_r.release()
            for writer in writers.values():
                writer.release()

        np.savez_compressed(
            output / "calibration.npz",
            K=full_k,
            K_working=work_k,
            baseline_m=np.asarray(baseline, dtype=np.float64),
            stereo_R=rotation,
            stereo_t=translation,
            left_valid=left_valid,
            right_valid=right_valid,
            sample_frames=sample_ids,
            source_intrinsics_b=np.concatenate(
                [intrinsic_r[[0, 1], [0, 1]], intrinsic_r[:2, 2], distortion_r]
            ),
            source_intrinsics_c=np.concatenate(
                [intrinsic_l[[0, 1], [0, 1]], intrinsic_l[:2, 2], distortion_l]
            ),
        )
        np.savetxt(output / "K_rectified.txt", full_k, fmt="%.10f")
        np.savetxt(output / "K_rectified_working.txt", work_k, fmt="%.10f")
        if sample_rows:
            cv2.imwrite(str(output / "shared_calibration_samples.jpg"), np.concatenate(sample_rows, axis=0))

        pooled_vertical = [row["median_vertical_error_px"] for row in diagnostics]
        pooled_positive = [row["positive_disparity_fraction"] for row in diagnostics]
        result = "PASS" if diagnostics and float(np.median(pooled_positive)) >= 0.90 else "FAIL"
        report = {
            "schema_version": "ego-scene-hoi-public-stereo-calibration-v1",
            "result": result,
            "calibration_source": str(cfg.calibration_json.resolve()),
            "camera_order": {"left": pair["left"], "right": pair["right"]},
            "frame_count": frame_count,
            "fps": fps,
            "source_size": [height, width],
            "working_size": [work_size[1], work_size[0]],
            "configuration": asdict(cfg),
            "stereo_geometry": {
                "baseline_m": baseline,
                "rotation_left_to_right": rotation,
                "translation_left_to_right_m": translation,
                "K_rectified": full_k,
                "K_rectified_working": work_k,
                "left_valid_fraction_working": float(left_valid.mean()),
                "right_valid_fraction_working": float(right_valid.mean()),
                "median_sample_vertical_error_px": (
                    float(np.median(pooled_vertical)) if pooled_vertical else None
                ),
                "median_sample_positive_disparity_fraction": (
                    float(np.median(pooled_positive)) if pooled_positive else None
                ),
                "Q": q,
                "roi_left": roi_l,
                "roi_right": roi_r,
            },
            "sample_diagnostics": diagnostics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path = output / "report.json"
        report_path.write_text(
            json.dumps(
                report,
                indent=2,
                default=lambda value: value.tolist()
                if isinstance(value, np.ndarray)
                else str(value),
            )
            + "\n",
            encoding="utf-8",
        )
        if result != "PASS":
            raise RuntimeError(f"Released calibration validation failed; see {report_path}")
        return report_path
