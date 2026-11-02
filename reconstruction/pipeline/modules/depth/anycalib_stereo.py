#!/usr/bin/env python3
"""Prepare an uncalibrated stereo pair for FoundationStereo.

The module estimates a KB4 intrinsic model independently for both input
images with AnyCalib, maps both images to pinhole projection, estimates the
relative camera pose from static image correspondences, and performs calibrated
stereo rectification.  Translation is recovered only up to scale, so the
written baseline is one arbitrary unit and downstream depth is non-metric.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


@dataclass(frozen=True)
class CameraCalibration:
    """Single-view calibration and target pinhole parameters."""

    source: str
    model_id: str
    camera_model: str
    source_intrinsics: list[float]
    target_intrinsic: list[list[float]]
    prediction_size: list[int]


def read_video_frame(video: Path, frame_id: int) -> np.ndarray:
    """Decode one zero-based RGB frame without changing its geometry."""

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, bgr = capture.read()
    capture.release()
    if not ok or bgr is None:
        raise RuntimeError(f"Cannot decode frame {frame_id} from {video}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def target_intrinsic(params: np.ndarray, focal_factor: float) -> np.ndarray:
    """Return the pinhole matrix induced by AnyCalib undistortion."""

    intrinsic = np.eye(3, dtype=np.float64)
    intrinsic[0, 0] = float(params[0]) * focal_factor
    intrinsic[1, 1] = float(params[1]) * focal_factor
    intrinsic[0, 2] = float(params[2])
    intrinsic[1, 2] = float(params[3])
    return intrinsic


def sift_correspondences(
    left: np.ndarray,
    right: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    *,
    max_features: int,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Find mutual SIFT correspondences inside valid undistorted pixels."""

    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.015)
    gray_left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    key_left, desc_left = sift.detectAndCompute(gray_left, left_valid)
    key_right, desc_right = sift.detectAndCompute(gray_right, right_valid)
    if desc_left is None or desc_right is None:
        raise RuntimeError("SIFT found no descriptors in one of the images")
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def ratio_matches(a: np.ndarray, b: np.ndarray) -> dict[tuple[int, int], cv2.DMatch]:
        selected: dict[tuple[int, int], cv2.DMatch] = {}
        for pair in matcher.knnMatch(a, b, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                match = pair[0]
                selected[(match.queryIdx, match.trainIdx)] = match
        return selected

    forward = ratio_matches(desc_left, desc_right)
    backward = ratio_matches(desc_right, desc_left)
    mutual = [
        match
        for (query, train), match in forward.items()
        if (train, query) in backward
    ]
    if len(mutual) < 20:
        raise RuntimeError(f"Only {len(mutual)} mutual SIFT matches; need at least 20")
    points_left = np.float64([key_left[m.queryIdx].pt for m in mutual])
    points_right = np.float64([key_right[m.trainIdx].pt for m in mutual])
    return points_left, points_right, {
        "left_keypoints": len(key_left),
        "right_keypoints": len(key_right),
        "forward_ratio_matches": len(forward),
        "backward_ratio_matches": len(backward),
        "mutual_matches": len(mutual),
    }


def estimate_relative_pose(
    points_left: np.ndarray,
    points_right: np.ndarray,
    intrinsic_left: np.ndarray,
    intrinsic_right: np.ndarray,
    ransac_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Estimate left-to-right pose from normalized correspondences."""

    normalized_left = cv2.undistortPoints(
        points_left[:, None], intrinsic_left, None
    ).reshape(-1, 2)
    normalized_right = cv2.undistortPoints(
        points_right[:, None], intrinsic_right, None
    ).reshape(-1, 2)
    mean_focal = float(
        np.mean(
            [
                intrinsic_left[0, 0],
                intrinsic_left[1, 1],
                intrinsic_right[0, 0],
                intrinsic_right[1, 1],
            ]
        )
    )
    threshold = ransac_threshold_px / max(mean_focal, 1.0)
    essential, mask = cv2.findEssentialMat(
        normalized_left,
        normalized_right,
        np.eye(3),
        method=cv2.USAC_MAGSAC,
        prob=0.999,
        threshold=threshold,
    )
    if essential is None or mask is None:
        raise RuntimeError("Essential-matrix estimation failed")
    if essential.shape != (3, 3):
        essential = essential[:3]
    inliers, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        normalized_left,
        normalized_right,
        np.eye(3),
        mask=mask,
    )
    retained = pose_mask.reshape(-1) > 0
    if int(inliers) < 20:
        raise RuntimeError(f"Only {inliers} relative-pose inliers; need at least 20")
    return rotation, translation.reshape(3), retained, {
        "ransac_threshold_px": ransac_threshold_px,
        "essential_inliers": int(mask.astype(bool).sum()),
        "cheirality_inliers": int(inliers),
        "cheirality_inlier_fraction": float(inliers / len(points_left)),
    }


def rectify_pair(
    left: np.ndarray,
    right: np.ndarray,
    intrinsic_left: np.ndarray,
    intrinsic_right: np.ndarray,
    rotation_left_to_right: np.ndarray,
    translation_left_to_right: np.ndarray,
    points_left: np.ndarray,
    points_right: np.ndarray,
    inliers: np.ndarray,
) -> dict[str, np.ndarray | tuple[int, ...]]:
    """Rectify a pinhole pair using a unit-norm translation baseline."""

    height, width = left.shape[:2]
    image_size = (width, height)
    zeros = np.zeros(5, dtype=np.float64)
    r_left, r_right, p_left, p_right, q, roi_left, roi_right = cv2.stereoRectify(
        intrinsic_left,
        zeros,
        intrinsic_right,
        zeros,
        image_size,
        rotation_left_to_right,
        translation_left_to_right,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map_lx, map_ly = cv2.initUndistortRectifyMap(
        intrinsic_left, zeros, r_left, p_left[:, :3], image_size, cv2.CV_32FC1
    )
    map_rx, map_ry = cv2.initUndistortRectifyMap(
        intrinsic_right, zeros, r_right, p_right[:, :3], image_size, cv2.CV_32FC1
    )
    rectified_left = cv2.remap(left, map_lx, map_ly, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right, map_rx, map_ry, cv2.INTER_LINEAR)
    selected_left = points_left[inliers]
    selected_right = points_right[inliers]
    rectified_points_left = cv2.undistortPoints(
        selected_left[:, None], intrinsic_left, None, R=r_left, P=p_left
    ).reshape(-1, 2)
    rectified_points_right = cv2.undistortPoints(
        selected_right[:, None], intrinsic_right, None, R=r_right, P=p_right
    ).reshape(-1, 2)
    disparity = rectified_points_left[:, 0] - rectified_points_right[:, 0]
    vertical_error = np.abs(rectified_points_left[:, 1] - rectified_points_right[:, 1])
    return {
        "left": rectified_left,
        "right": rectified_right,
        "R_left": r_left,
        "R_right": r_right,
        "P_left": p_left,
        "P_right": p_right,
        "Q": q,
        "roi_left": roi_left,
        "roi_right": roi_right,
        "disparities": disparity,
        "vertical_errors": vertical_error,
    }


def rectification_preview(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack rectified images and draw shared horizontal scan lines."""

    preview = np.concatenate([left, right], axis=0)
    for y in range(40, left.shape[0], 80):
        cv2.line(preview, (0, y), (left.shape[1] - 1, y), (255, 64, 64), 2)
        y_right = y + left.shape[0]
        cv2.line(preview, (0, y_right), (left.shape[1] - 1, y_right), (255, 64, 64), 2)
    return preview


class AnyCalibStereoRectifier:
    """Calibrate, undistort, and rectify one synchronized stereo frame."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments

    def run(self) -> Path:
        from anycalib import AnyCalib
        from anycalib.cameras import CameraFactory

        args = self.arguments
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        rgb_a = read_video_frame(args.camera_a_video.expanduser().resolve(), args.frame_id)
        rgb_b = read_video_frame(args.camera_b_video.expanduser().resolve(), args.frame_id)
        if rgb_a.shape != rgb_b.shape:
            raise ValueError(f"Stereo image shapes differ: {rgb_a.shape} vs {rgb_b.shape}")
        imageio.imwrite(output / "camera_a_original.png", rgb_a)
        imageio.imwrite(output / "camera_b_original.png", rgb_b)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AnyCalib(model_id=args.model_id).to(device).eval()
        camera = CameraFactory.create_from_id(args.camera_model)
        calibrations: list[CameraCalibration] = []
        undistorted: list[np.ndarray] = []
        valid_masks: list[np.ndarray] = []
        target_intrinsics: list[np.ndarray] = []
        for label, source, rgb in (
            ("camera_a", args.camera_a_video, rgb_a),
            ("camera_b", args.camera_b_video, rgb_b),
        ):
            image_tensor = (
                torch.as_tensor(rgb, dtype=torch.float32, device=device)
                .permute(2, 0, 1)
                .div_(255.0)
            )
            prediction = model.predict(image_tensor, cam_id=args.camera_model)
            params = prediction["intrinsics"].detach()
            mapped = camera.undistort_image(
                image_tensor,
                params,
                scale=args.focal_length_factor,
                target_proj="perspective" if args.target_projection == "pinhole" else args.target_projection,
                outside_value=0.0,
                add_alpha=True,
            )
            mapped_rgb = (
                mapped[:3].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
            )
            valid = mapped[3].cpu().numpy() > 0.5
            intrinsic = target_intrinsic(params.cpu().numpy(), args.focal_length_factor)
            imageio.imwrite(output / f"{label}_undistorted.png", mapped_rgb)
            imageio.imwrite(output / f"{label}_valid_mask.png", valid.astype(np.uint8) * 255)
            calibrations.append(
                CameraCalibration(
                    source=str(Path(source).expanduser().resolve()),
                    model_id=args.model_id,
                    camera_model=args.camera_model,
                    source_intrinsics=[float(value) for value in params.cpu().numpy()],
                    target_intrinsic=intrinsic.tolist(),
                    prediction_size=[int(value) for value in prediction["pred_size"]],
                )
            )
            undistorted.append(mapped_rgb)
            valid_masks.append(valid.astype(np.uint8) * 255)
            target_intrinsics.append(intrinsic)
        del model
        torch.cuda.empty_cache()

        points_a, points_b, match_report = sift_correspondences(
            undistorted[0],
            undistorted[1],
            valid_masks[0],
            valid_masks[1],
            max_features=args.max_features,
            ratio=args.match_ratio,
        )
        rotation, translation, inliers, pose_report = estimate_relative_pose(
            points_a,
            points_b,
            target_intrinsics[0],
            target_intrinsics[1],
            args.ransac_threshold_px,
        )
        rectified = rectify_pair(
            undistorted[0],
            undistorted[1],
            target_intrinsics[0],
            target_intrinsics[1],
            rotation,
            translation,
            points_a,
            points_b,
            inliers,
        )
        swapped = float(np.median(rectified["disparities"])) < 0
        if swapped:
            rotation_inverse = rotation.T
            translation_inverse = -rotation.T @ translation
            rectified = rectify_pair(
                undistorted[1],
                undistorted[0],
                target_intrinsics[1],
                target_intrinsics[0],
                rotation_inverse,
                translation_inverse,
                points_b,
                points_a,
                inliers,
            )

        rectified_left = np.asarray(rectified["left"])
        rectified_right = np.asarray(rectified["right"])
        imageio.imwrite(output / "left_rectified.png", rectified_left)
        imageio.imwrite(output / "right_rectified.png", rectified_right)
        imageio.imwrite(
            output / "rectification_preview.png",
            rectification_preview(rectified_left, rectified_right),
        )
        p_left = np.asarray(rectified["P_left"])
        p_right = np.asarray(rectified["P_right"])
        baseline = abs(float(p_right[0, 3] / p_right[0, 0]))
        np.savetxt(output / "rectified_K.txt", p_left[:, :3], fmt="%.10f")
        with (output / "foundationstereo_intrinsic.txt").open("w", encoding="utf-8") as handle:
            handle.write(" ".join(f"{value:.10f}" for value in p_left[:, :3].reshape(-1)))
            handle.write(f"\n{baseline:.10f}\n")
        disparities = np.asarray(rectified["disparities"])
        vertical_errors = np.asarray(rectified["vertical_errors"])
        report = {
            "frame_id": args.frame_id,
            "configuration": {
                "model_id": args.model_id,
                "camera_model": args.camera_model,
                "distortion_coefficients": 4,
                "focal_length_factor": args.focal_length_factor,
                "target_projection": args.target_projection,
            },
            "calibrations": [asdict(calibration) for calibration in calibrations],
            "matching": match_report | pose_report,
            "relative_pose": {
                "camera_a_to_camera_b_rotation": rotation.tolist(),
                "camera_a_to_camera_b_translation_direction": translation.tolist(),
                "translation_norm": float(np.linalg.norm(translation)),
            },
            "rectification": {
                "foundationstereo_left_source": str(
                    (args.camera_b_video if swapped else args.camera_a_video)
                    .expanduser()
                    .resolve()
                ),
                "foundationstereo_right_source": str(
                    (args.camera_a_video if swapped else args.camera_b_video)
                    .expanduser()
                    .resolve()
                ),
                "swapped_for_positive_disparity": swapped,
                "rectified_intrinsic": p_left[:, :3].tolist(),
                "arbitrary_baseline": baseline,
                "scale_status": "unknown_global_scale; baseline normalized by recoverPose",
                "median_sparse_disparity_px": float(np.median(disparities)),
                "positive_sparse_disparity_fraction": float(np.mean(disparities > 0)),
                "median_vertical_error_px": float(np.median(vertical_errors)),
                "p95_vertical_error_px": float(np.quantile(vertical_errors, 0.95)),
                "roi_left": [int(value) for value in rectified["roi_left"]],
                "roi_right": [int(value) for value in rectified["roi_right"]],
            },
        }
        report_path = output / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(report_path)
        return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-a-video", type=Path, required=True)
    parser.add_argument("--camera-b-video", type=Path, required=True)
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="anycalib_gen")
    parser.add_argument("--camera-model", default="kb:4")
    parser.add_argument("--focal-length-factor", type=float, default=0.5)
    parser.add_argument("--target-projection", default="pinhole")
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--match-ratio", type=float, default=0.78)
    parser.add_argument("--ransac-threshold-px", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    AnyCalibStereoRectifier(parse_args()).run()


if __name__ == "__main__":
    main()
