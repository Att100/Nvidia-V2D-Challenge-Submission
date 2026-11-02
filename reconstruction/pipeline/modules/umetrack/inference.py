#!/usr/bin/env python3
"""Run UmeTrack on rectified egocentric stereo videos.

SAM3 left-view hand masks provide crop prompts; UmeTrack predicts articulation,
metric skeleton scale, and the rectified-left camera-space wrist transform.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
UMETRACK_ROOT = ROOT / "thirdparty" / "UmeTrack"
sys.path.insert(0, str(UMETRACK_ROOT))

from lib.common import camera as um_camera  # noqa: E402
from lib.common.crop import gen_crop_parameters_from_points  # noqa: E402
from lib.common.hand import HandModel, scaled_hand_model  # noqa: E402
from lib.common.hand_skinning import _skin_points, skin_landmarks  # noqa: E402
from lib.models.model_loader import load_pretrained_model  # noqa: E402
from lib.tracker.tracker import (  # noqa: E402
    HandTracker,
    HandTrackerOpts,
    InputFrame,
    ViewData,
)
from lib.tracker.video_pose_data import (  # noqa: E402
    _load_json,
    load_hand_model_from_dict,
)
from pipeline.utils import parse_worker_args  # noqa: E402


HAND_NAMES = ("left", "right")
HAND_COLORS = ((70, 220, 80), (40, 160, 255))


@dataclass(frozen=True)
class SequenceInputs:
    run_root: Path
    left_video: Path
    right_video: Path
    masks: Path
    calibration: Path
    depth_dir: Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-margin", type=float, default=1.20)
    parser.add_argument("--min-mask-area", type=int, default=80)
    parser.add_argument("--max-calibration-samples", type=int, default=80)
    parser.add_argument(
        "--mode",
        choices=("stereo", "single"),
        default="stereo",
        help="Stereo predicts a subject skeleton scale; single uses the generic skeleton.",
    )
    return parser


def resolve_inputs(run_root: Path) -> SequenceInputs:
    values = SequenceInputs(
        run_root=run_root,
        left_video=run_root / "calibration" / "left_rectified_working.mp4",
        right_video=run_root / "calibration" / "right_rectified_working.mp4",
        masks=run_root / "tracks" / "sam3_hands_left" / "hand_masks.npz",
        calibration=run_root / "calibration" / "calibration.npz",
        depth_dir=run_root / "depth" / "frames",
    )
    required = [values.left_video, values.masks, values.calibration, values.depth_dir]
    if not values.right_video.exists():
        required.append(values.right_video)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing UmeTrack experiment inputs:\n" + "\n".join(missing))
    return values


def read_video(path: Path) -> Tuple[List[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 20.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    return frames, fps


def load_masks(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[str, str]]:
    archive = np.load(path, allow_pickle=True)
    width = int(archive["mask_width"])
    masks = np.unpackbits(
        np.asarray(archive["packed_masks"], dtype=np.uint8), axis=-1, count=width
    ).astype(bool)
    valid = np.asarray(archive["valid"], dtype=bool)
    labels = tuple(str(x) for x in archive["hand_labels"].tolist())
    if labels != HAND_NAMES:
        order = [labels.index(name) for name in HAND_NAMES]
        masks = masks[:, order]
        valid = valid[:, order]
        labels = HAND_NAMES
    return masks, valid, labels


def mask_bbox(mask: np.ndarray, margin: float) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    side = max(x1 - x0, y1 - y0) * margin
    return np.array([cx - side * 0.5, cy - side * 0.5, cx + side * 0.5, cy + side * 0.5])


def bbox_world_points(bbox: np.ndarray, k: np.ndarray, depth_m: float) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    pixels = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [(x0 + x1) / 2, (y0 + y1) / 2]],
        dtype=np.float64,
    )
    homo = np.concatenate([pixels, np.ones((len(pixels), 1))], axis=1)
    rays = homo @ np.linalg.inv(k).T
    # The public UmeTrack tracker API represents camera geometry in millimeters
    # and converts it to meters immediately before invoking the network.
    return rays * (float(depth_m) * 1000.0)


def robust_mask_depth(mask: np.ndarray, depth: np.ndarray, fallback: float) -> float:
    values = depth[mask & np.isfinite(depth) & (depth > 0.08) & (depth < 3.0)]
    if values.size < 32:
        return fallback
    return float(np.median(values))


def make_crop_cameras(
    bbox: np.ndarray,
    depth_m: float,
    hand_idx: int,
    source_cameras: List[um_camera.PinholePlaneCameraModel],
    k: np.ndarray,
    stereo: bool,
) -> Dict[int, um_camera.PinholePlaneCameraModel]:
    points_world = bbox_world_points(bbox, k, depth_m)
    indices = range(2 if stereo else 1)
    return {
        index: gen_crop_parameters_from_points(
            source_cameras[index],
            points_world,
            (96, 96),
            mirror_img_x=hand_idx == 1,
            camera_angle=0.0,
            focal_multiplier=0.92,
        )
        for index in indices
    }


def skin_mesh(hand_model: HandModel, angles: np.ndarray, wrist_xf: np.ndarray, hand_idx: int) -> np.ndarray:
    xf = wrist_xf.copy()
    if hand_idx == 1:
        # Match UmeTrack's official landmarks_from_hand_pose convention.
        xf[:, 0] *= -1
    joint_angles = torch.from_numpy(angles).float()
    wrist = torch.from_numpy(xf).float()
    dense_weights = hand_model.dense_bone_weights
    assert dense_weights is not None and hand_model.mesh_vertices is not None
    vertices = _skin_points(
        hand_model.joint_rest_positions,
        hand_model.joint_rotation_axes,
        dense_weights.reshape(1, dense_weights.shape[-2], dense_weights.shape[-1]),
        joint_angles,
        hand_model.mesh_vertices,
        wrist,
    )
    return vertices.detach().cpu().numpy()


def project(points_mm: np.ndarray, k: np.ndarray) -> np.ndarray:
    points = np.asarray(points_mm)
    uvw = points @ k.T
    return uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)


def rasterize_mesh(vertices_mm: np.ndarray, faces: np.ndarray, k: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    uv = project(vertices_mm, k)
    z = vertices_mm[:, 2]
    output = np.zeros((h, w), dtype=np.uint8)
    # Back-to-front ordering gives a deterministic diagnostic silhouette.
    valid_faces = faces[np.all(z[faces] > 1.0, axis=1)]
    if len(valid_faces) == 0:
        return output.astype(bool)
    depths = z[valid_faces].mean(axis=1)
    for face in valid_faces[np.argsort(depths)[::-1]]:
        polygon_float = uv[face]
        if np.all(np.isfinite(polygon_float)):
            polygon = np.rint(polygon_float).astype(np.int32)
            cv2.fillConvexPoly(output, polygon, 1)
    return output.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def draw_mask(frame: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.35) -> None:
    color_layer = np.empty_like(frame)
    color_layer[:] = color
    frame[mask] = cv2.addWeighted(frame, 1.0 - alpha, color_layer, alpha, 0)[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, color, 1, cv2.LINE_AA)


def write_overlay(
    path: Path,
    frames: List[np.ndarray],
    fps: float,
    target_masks: np.ndarray,
    predicted_masks: np.ndarray,
    keypoints: np.ndarray,
    valid: np.ndarray,
    k: np.ndarray,
) -> None:
    h, w = frames[0].shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))
    for frame_idx, gray in enumerate(frames):
        left = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        right = left.copy()
        for hand_idx, color in enumerate(HAND_COLORS):
            if target_masks[frame_idx, hand_idx].any():
                draw_mask(left, target_masks[frame_idx, hand_idx], color)
            if valid[frame_idx, hand_idx]:
                draw_mask(right, predicted_masks[frame_idx, hand_idx], color, alpha=0.42)
                uv = project(keypoints[frame_idx, hand_idx], k)
                for x, y in uv:
                    if np.isfinite(x) and np.isfinite(y):
                        cv2.circle(right, (int(round(x)), int(round(y))), 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(left, "SAM3 hand masks", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(right, "UmeTrack mesh + 21 landmarks", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(right, f"frame {frame_idx:06d}", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        writer.write(np.concatenate([left, right], axis=1))
    writer.release()


def summarize(values: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    selected = values[valid & np.isfinite(values)]
    if selected.size == 0:
        return {"count": 0}
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p10": float(np.percentile(selected, 10)),
        "p90": float(np.percentile(selected, 90)),
    }


def run_sequence(args: argparse.Namespace) -> None:
    inputs = resolve_inputs(args.run_root.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    left_frames, fps = read_video(inputs.left_video)
    right_frames, right_fps = read_video(inputs.right_video)
    masks, mask_valid, _ = load_masks(inputs.masks)
    frame_count = min(len(left_frames), len(right_frames), len(masks))
    left_frames, right_frames, masks, mask_valid = (
        left_frames[:frame_count],
        right_frames[:frame_count],
        masks[:frame_count],
        mask_valid[:frame_count],
    )
    calibration = np.load(inputs.calibration)
    k = np.asarray(calibration["K_working"], dtype=np.float64)
    baseline_m = float(calibration["baseline_m"])
    h, w = left_frames[0].shape
    left_camera = um_camera.PinholePlaneCameraModel(w, h, (k[0, 0], k[1, 1]), (k[0, 2], k[1, 2]), [])
    right_c2w = np.eye(4)
    # UmeTrack's CameraModel convention is millimeters (see tracker.MM_TO_M).
    right_c2w[0, 3] = baseline_m * 1000.0
    right_camera = um_camera.PinholePlaneCameraModel(
        w, h, (k[0, 0], k[1, 1]), (k[0, 2], k[1, 2]), [], camera_to_world_xf=right_c2w
    )
    source_cameras = [left_camera, right_camera]

    model = load_pretrained_model(str(UMETRACK_ROOT / "pretrained_models" / "pretrained_weights.torch"))
    model.eval().to(args.device)
    generic = load_hand_model_from_dict(_load_json(str(UMETRACK_ROOT / "dataset" / "generic_hand_model.json")))

    # Precompute mask-driven perspective crop cameras and robust stereo depths.
    crop_cameras: List[Dict[int, Dict[int, um_camera.PinholePlaneCameraModel]]] = []
    mask_depths = np.full((frame_count, 2), np.nan, dtype=np.float32)
    last_depth = np.array([0.55, 0.55], dtype=np.float64)
    areas = masks.reshape(frame_count, 2, -1).sum(axis=-1)
    for frame_idx in range(frame_count):
        depth = np.load(inputs.depth_dir / f"{frame_idx:06d}.npy")
        per_hand: Dict[int, Dict[int, um_camera.PinholePlaneCameraModel]] = {}
        for hand_idx in range(2):
            if not mask_valid[frame_idx, hand_idx] or areas[frame_idx, hand_idx] < args.min_mask_area:
                continue
            bbox = mask_bbox(masks[frame_idx, hand_idx], args.crop_margin)
            if bbox is None:
                continue
            depth_m = robust_mask_depth(masks[frame_idx, hand_idx], depth, float(last_depth[hand_idx]))
            last_depth[hand_idx] = depth_m
            mask_depths[frame_idx, hand_idx] = depth_m
            try:
                per_hand[hand_idx] = make_crop_cameras(
                    bbox, depth_m, hand_idx, source_cameras, k, args.mode == "stereo"
                )
            except ValueError:
                continue
        crop_cameras.append(per_hand)

    scale = 1.0
    scale_samples: List[float] = []
    if args.mode == "stereo":
        calibration_tracker = HandTracker(model, HandTrackerOpts())
        candidates = np.argsort(areas.max(axis=1))[::-1]
        selected = sorted(int(index) for index in candidates[: args.max_calibration_samples])
        for frame_idx in selected:
            crops = crop_cameras[frame_idx]
            crops = {key: value for key, value in crops.items() if len(value) == 2}
            if not crops:
                continue
            sample = InputFrame(
                views=[
                    ViewData(left_frames[frame_idx], left_camera, 0.0),
                    ViewData(right_frames[frame_idx], right_camera, 0.0),
                ]
            )
            result = calibration_tracker.track_frame_and_calibrate_scale(sample, crops)
            scale_samples.extend(float(value) for value in result.predicted_scales.values())
        if scale_samples:
            raw_scale = float(np.median(scale_samples))
            scale = float(np.clip(raw_scale, 0.70, 1.30))
    calibrated = scaled_hand_model(generic, scale)
    tracker = HandTracker(model, HandTrackerOpts())
    joint_angles = np.zeros((frame_count, 2, 22), dtype=np.float32)
    wrist_xfs = np.tile(np.eye(4, dtype=np.float32), (frame_count, 2, 1, 1))
    keypoints = np.zeros((frame_count, 2, 21, 3), dtype=np.float32)
    vertices = np.zeros((frame_count, 2, 788, 3), dtype=np.float32)
    valid = np.zeros((frame_count, 2), dtype=bool)
    predicted_masks = np.zeros_like(masks)
    per_frame_iou = np.full((frame_count, 2), np.nan, dtype=np.float32)
    wrist_depth_error_m = np.full((frame_count, 2), np.nan, dtype=np.float32)

    started = time.perf_counter()
    with torch.inference_mode():
        for frame_idx in range(frame_count):
            views = [ViewData(left_frames[frame_idx], left_camera, 0.0)]
            if args.mode == "stereo":
                views.append(ViewData(right_frames[frame_idx], right_camera, 0.0))
            result = tracker.track_frame(InputFrame(views=views), calibrated, crop_cameras[frame_idx])
            for hand_idx, pose in result.hand_poses.items():
                valid[frame_idx, hand_idx] = True
                joint_angles[frame_idx, hand_idx] = pose.joint_angles
                wrist_xfs[frame_idx, hand_idx] = pose.wrist_xform
                landmark_xf = pose.wrist_xform.copy()
                if hand_idx == 1:
                    landmark_xf[:, 0] *= -1
                cur_keypoints = skin_landmarks(
                    calibrated,
                    torch.from_numpy(pose.joint_angles).float(),
                    torch.from_numpy(landmark_xf).float(),
                ).cpu().numpy()
                cur_vertices = skin_mesh(calibrated, pose.joint_angles, pose.wrist_xform, hand_idx)
                keypoints[frame_idx, hand_idx] = cur_keypoints
                vertices[frame_idx, hand_idx] = cur_vertices
                cur_mask = rasterize_mesh(
                    cur_vertices,
                    np.asarray(calibrated.mesh_triangles, dtype=np.int32),
                    k,
                    (h, w),
                )
                predicted_masks[frame_idx, hand_idx] = cur_mask
                per_frame_iou[frame_idx, hand_idx] = iou(cur_mask, masks[frame_idx, hand_idx])
                wrist_depth_error_m[frame_idx, hand_idx] = abs(
                    float(pose.wrist_xform[2, 3]) / 1000.0 - float(mask_depths[frame_idx, hand_idx])
                )
    elapsed = time.perf_counter() - started

    faces = np.asarray(calibrated.mesh_triangles, dtype=np.int32)
    np.savez_compressed(
        output / "umetrack_results.npz",
        joint_angles=joint_angles,
        wrist_xfs_left_camera_mm=wrist_xfs,
        keypoints_left_camera_mm=keypoints,
        vertices_left_camera_mm=vertices,
        faces=faces,
        valid=valid,
        mask_depth_m=mask_depths,
        silhouette_iou=per_frame_iou,
        wrist_depth_error_m=wrist_depth_error_m,
        hand_names=np.asarray(HAND_NAMES),
        skeleton_scale=np.asarray(scale, dtype=np.float32),
        K=k.astype(np.float32),
    )
    write_overlay(
        output / "umetrack_overlay.mp4",
        left_frames,
        fps,
        masks,
        predicted_masks,
        keypoints,
        valid,
        k,
    )
    report = {
        "schema_version": "umetrack-rectified-stereo-experiment-v1",
        "result": "PASS",
        "run_root": str(inputs.run_root),
        "mode": args.mode,
        "input": {
            "frame_count": frame_count,
            "fps": fps,
            "image_size_hw": [h, w],
            "K": k.tolist(),
            "baseline_m": baseline_m,
            "crop_source": str(inputs.masks),
        },
        "model": {
            "weights": str(UMETRACK_ROOT / "pretrained_models" / "pretrained_weights.torch"),
            "native_crop_size": [96, 96],
            "skeleton_scale": scale,
            "raw_scale_sample_summary": summarize(
                np.asarray(scale_samples, dtype=np.float32),
                np.ones(len(scale_samples), dtype=bool),
            ),
        },
        "metrics": {
            hand: {
                "valid_frames": int(valid[:, hand_idx].sum()),
                "target_mask_frames": int(mask_valid[:, hand_idx].sum()),
                "silhouette_iou": summarize(per_frame_iou[:, hand_idx], valid[:, hand_idx]),
                "absolute_wrist_vs_stereo_mask_depth_error_m": summarize(
                    wrist_depth_error_m[:, hand_idx], valid[:, hand_idx]
                ),
            }
            for hand_idx, hand in enumerate(HAND_NAMES)
        },
        "runtime": {
            "inference_seconds": elapsed,
            "inference_fps": frame_count / elapsed if elapsed > 0 else None,
            "device": args.device,
        },
        "outputs": {
            "arrays": "umetrack_results.npz",
            "overlay": "umetrack_overlay.mp4",
        },
        "notes": [
            "Coordinates are the rectified-left camera frame; lengths are millimeters.",
            "UmeTrack uses its native 22-angle/20-effective-DoF skeleton, not MANO.",
            "SAM3 masks are used only to construct perspective crops and evaluation silhouettes.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


class UmeTrackHandReconstructor:
    """Worker entry point for native rectified-stereo hand reconstruction."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments

    @classmethod
    def from_worker_job(cls) -> "UmeTrackHandReconstructor":
        return cls(parse_worker_args(_parser()))

    def run(self) -> None:
        run_sequence(self.arguments)


def main() -> None:
    UmeTrackHandReconstructor.from_worker_job().run()


if __name__ == "__main__":
    main()
