"""Dynamic-mask and metric stereo-depth inference for rectified videos."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import torch

from pipeline.modules.depth.foundation_stereo import load_trusted_checkpoint


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
    return count, width, height, fps


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _object_prompt_candidates(name: str, prompt: str) -> tuple[str, ...]:
    """Keep descriptive grounding first and add conservative category aliases."""

    values = [prompt, name]
    lowered = f"{name} {prompt}".lower()
    if "spoon" in lowered:
        values.extend(("wooden spoon", "spoon"))
    elif "saucepan" in lowered or "pan" in lowered:
        values.extend(("white saucepan", "saucepan"))
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _depth_color(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    color = np.zeros((*depth.shape, 3), dtype=np.uint8)
    values = depth[valid]
    if not len(values):
        return color
    near, far = np.quantile(values, [0.02, 0.98])
    normalized = np.clip((depth - near) / max(float(far - near), 1e-6), 0, 1)
    mapped = cv2.applyColorMap(
        np.uint8(np.round((1.0 - normalized) * 255)), cv2.COLORMAP_TURBO
    )
    color[valid] = mapped[valid]
    return color


@dataclass(frozen=True)
class DynamicMaskConfig:
    """SAM3 text-prompt dynamic-union configuration."""

    video: Path
    reasoning_json: Path
    output_dir: Path
    repository_root: Path
    object_tracks_dir: Path | None = None
    hand_masks_npz: Path | None = None
    checkpoint: Path | None = None
    confidence_threshold: float = 0.3
    minimum_mask_area: int = 64
    dilation_pixels: int = 4
    body_prompts: tuple[str, ...] = (
        "human",
        "human hand",
        "human arm",
        "human forearm",
        "human sleeve",
    )


class SAM3DynamicMaskBuilder:
    """Build one same-camera dynamic union from tracks and body segmentation."""

    def __init__(self, config: DynamicMaskConfig) -> None:
        self.config = config

    def run(self) -> Path:
        sam3_parent = self.config.repository_root.resolve() / "thirdparty"
        if str(sam3_parent) not in sys.path:
            sys.path.insert(0, str(sam3_parent))
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        cfg = self.config
        output = cfg.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        reasoning = json.loads(cfg.reasoning_json.read_text(encoding="utf-8"))
        object_records = list(reasoning.get("objects", []))
        object_prompts = [
            str(record.get("sam3_prompt") or record["name"])
            for record in object_records
        ]
        object_ids = [
            str(record.get("global_object_id") or f"object_{index:03d}")
            for index, record in enumerate(object_records)
        ]
        object_prompt_variants = [
            _object_prompt_candidates(str(record["name"]), prompt)
            for record, prompt in zip(object_records, object_prompts)
        ]
        use_tracked_objects = cfg.object_tracks_dir is not None
        prompts = tuple(
            dict.fromkeys(
                list(cfg.body_prompts)
                if use_tracked_objects
                else [prompt for variants in object_prompt_variants for prompt in variants]
                + list(cfg.body_prompts)
            )
        )
        if not prompts:
            raise ValueError("No object or body prompts were configured")
        frame_count, width, height, fps = _video_metadata(cfg.video)
        kwargs: dict[str, Any] = {"device": "cuda"}
        if cfg.checkpoint is not None:
            kwargs.update(checkpoint_path=str(cfg.checkpoint.resolve()), load_from_HF=False)
        model = build_sam3_image_model(**kwargs)
        processor = Sam3Processor(
            model,
            device="cuda",
            confidence_threshold=cfg.confidence_threshold,
        )
        masks = np.zeros((frame_count, height, width), dtype=bool)
        body_masks = np.zeros((frame_count, height, width), dtype=bool)
        object_masks = np.zeros(
            (len(object_prompts), frame_count, height, width), dtype=bool
        )
        tracked_object_sources: dict[str, str] = {}
        if use_tracked_objects:
            tracks_dir = cfg.object_tracks_dir.resolve()
            for object_index, object_id in enumerate(object_ids):
                track_path = tracks_dir / "tracklets" / f"{object_id}.npz"
                if not track_path.is_file():
                    raise FileNotFoundError(track_path)
                with np.load(track_path) as archive:
                    track_width = int(np.asarray(archive["mask_width"]).item())
                    tracked = np.unpackbits(
                        np.asarray(archive["packed_masks"], dtype=np.uint8), axis=2
                    )[..., :track_width].astype(bool)
                    source_frames = np.asarray(archive["source_frames"], dtype=np.int64)
                if tracked.shape != (frame_count, height, width):
                    raise ValueError(
                        f"Tracked mask shape {tracked.shape} != "
                        f"{(frame_count, height, width)} for {object_id}"
                    )
                if not np.array_equal(source_frames, np.arange(frame_count)):
                    raise ValueError(f"Non-dense source frames in {track_path}")
                object_masks[object_index] = tracked
                tracked_object_sources[object_id] = str(track_path)
        tracked_hand_masks: np.ndarray | None = None
        tracked_hand_source: str | None = None
        if cfg.hand_masks_npz is not None:
            hand_path = cfg.hand_masks_npz.resolve()
            if not hand_path.is_file():
                raise FileNotFoundError(hand_path)
            with np.load(hand_path) as archive:
                hand_width = int(np.asarray(archive["mask_width"]).item())
                tracked_hand_masks = np.unpackbits(
                    np.asarray(archive["packed_masks"], dtype=np.uint8),
                    axis=-1,
                    count=hand_width,
                ).astype(bool)
            expected_hand_shape = (frame_count, 2, height, width)
            if tracked_hand_masks.shape != expected_hand_shape:
                raise ValueError(
                    f"Tracked hand mask shape {tracked_hand_masks.shape} != "
                    f"{expected_hand_shape}"
                )
            tracked_hand_source = str(hand_path)
        object_indices_by_prompt: dict[str, list[int]] = {}
        if not use_tracked_objects:
            for object_index, variants in enumerate(object_prompt_variants):
                for prompt in variants:
                    object_indices_by_prompt.setdefault(prompt, []).append(object_index)
        prompt_frame_hits = {prompt: 0 for prompt in prompts}
        prompt_pixel_counts = {prompt: 0 for prompt in prompts}
        capture = cv2.VideoCapture(str(cfg.video))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output / "dynamic_mask_overlay.mp4"), fourcc, fps, (width, height)
        )
        if not capture.isOpened() or not writer.isOpened():
            raise RuntimeError("Could not open dynamic-mask video input/output")
        started = time.perf_counter()
        try:
            for frame_id in range(frame_count):
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode frame {frame_id}")
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                state = processor.set_image(image)
                combined = (
                    np.logical_or.reduce(object_masks[:, frame_id])
                    if len(object_masks)
                    else np.zeros((height, width), dtype=bool)
                )
                body_combined = np.zeros((height, width), dtype=bool)
                if tracked_hand_masks is not None:
                    tracked_hands = np.logical_or.reduce(tracked_hand_masks[frame_id])
                    combined |= tracked_hands
                    body_combined |= tracked_hands
                for prompt in prompts:
                    state = processor.set_text_prompt(prompt=prompt, state=state)
                    candidates = _as_numpy(state.get("masks", [])).astype(bool)
                    if candidates.ndim == 4 and candidates.shape[1] == 1:
                        candidates = candidates[:, 0]
                    if candidates.ndim == 2:
                        candidates = candidates[None]
                    accepted = []
                    for candidate in candidates:
                        if candidate.shape != (height, width):
                            candidate = cv2.resize(
                                candidate.astype(np.uint8),
                                (width, height),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        if int(candidate.sum()) >= cfg.minimum_mask_area:
                            accepted.append(candidate)
                    if accepted:
                        prompt_mask = np.logical_or.reduce(accepted)
                        combined |= prompt_mask
                        if prompt in cfg.body_prompts:
                            body_combined |= prompt_mask
                        for object_index in object_indices_by_prompt.get(prompt, []):
                            object_masks[object_index, frame_id] |= prompt_mask
                        prompt_frame_hits[prompt] += 1
                        prompt_pixel_counts[prompt] += int(prompt_mask.sum())
                    processor.reset_all_prompts(state)
                if cfg.dilation_pixels > 0 and combined.any():
                    radius = cfg.dilation_pixels
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
                    )
                    combined = cv2.dilate(combined.astype(np.uint8), kernel).astype(bool)
                masks[frame_id] = combined
                body_masks[frame_id] = body_combined
                overlay = bgr.copy()
                overlay[combined] = (
                    0.35 * overlay[combined] + 0.65 * np.array([0, 0, 255])
                ).astype(np.uint8)
                cv2.putText(
                    overlay,
                    f"frame {frame_id} dynamic={100.0 * combined.mean():.1f}%",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(overlay)
        finally:
            capture.release()
            writer.release()
            del processor, model
            gc.collect()
            torch.cuda.empty_cache()

        mask_path = output / "dynamic_mask.npz"
        np.savez_compressed(
            mask_path,
            packed_masks=np.packbits(masks, axis=2),
            mask_width=np.asarray(width, dtype=np.int32),
            source_indices=np.arange(frame_count, dtype=np.int32),
        )
        body_mask_path = output / "body_masks.npz"
        np.savez_compressed(
            body_mask_path,
            packed_masks=np.packbits(body_masks, axis=2),
            mask_width=np.asarray(width, dtype=np.int32),
            source_indices=np.arange(frame_count, dtype=np.int32),
        )
        object_mask_path = output / "object_masks.npz"
        np.savez_compressed(
            object_mask_path,
            packed_masks=np.packbits(object_masks, axis=3),
            mask_width=np.asarray(width, dtype=np.int32),
            object_ids=np.asarray(object_ids),
            object_prompts=np.asarray(object_prompts),
            source_indices=np.arange(frame_count, dtype=np.int32),
        )
        per_frame_fraction = masks.mean(axis=(1, 2))
        report = {
            "schema_version": "ego-scene-hoi-stereo-dynamic-mask-v1",
            "result": "PASS",
            "video": str(cfg.video.resolve()),
            "reasoning": str(cfg.reasoning_json.resolve()),
            "object_prompts": object_prompts,
            "object_prompt_variants": [list(values) for values in object_prompt_variants],
            "object_mask_mode": (
                "codex_bbox_seeded_full_video_tracks"
                if use_tracked_objects
                else "independent_text_segmentation"
            ),
            "tracked_object_sources": tracked_object_sources,
            "hand_mask_mode": (
                "sam3_tracked_bbox_second_pass"
                if tracked_hand_masks is not None
                else "body_text_prompts_only"
            ),
            "tracked_hand_source": tracked_hand_source,
            "body_prompts": list(cfg.body_prompts),
            "all_prompts": list(prompts),
            "frame_count": frame_count,
            "shape": [frame_count, height, width],
            "prompt_frame_hits": prompt_frame_hits,
            "prompt_pixel_counts": prompt_pixel_counts,
            "dynamic_fraction": {
                "mean": float(per_frame_fraction.mean()),
                "median": float(np.median(per_frame_fraction)),
                "max": float(per_frame_fraction.max()),
            },
            "dilation_pixels": cfg.dilation_pixels,
            "mask_file": str(mask_path),
            "body_mask_file": str(body_mask_path),
            "object_mask_file": str(object_mask_path),
            "overlay_video": str((output / "dynamic_mask_overlay.mp4").resolve()),
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path = output / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report_path


@dataclass(frozen=True)
class FoundationStereoSequenceConfig:
    """FoundationStereo inference settings for one fixed rectified sequence."""

    left_video: Path
    right_video: Path
    calibration_npz: Path
    repository: Path
    checkpoint: Path
    output_dir: Path
    valid_iters: int = 32
    z_near_m: float = 0.10
    z_far_m: float = 10.0
    sample_count: int = 6
    point_stride: int = 2


class FoundationStereoSequenceEstimator:
    """Load FoundationStereo once and infer metric left depth for every frame."""

    def __init__(self, config: FoundationStereoSequenceConfig) -> None:
        self.config = config

    @staticmethod
    def _sample_cloud(
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        output: Path,
        stride: int,
    ) -> int:
        yy, xx = np.meshgrid(
            np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing="ij"
        )
        valid = depth > 0
        sampling = (yy % stride == 0) & (xx % stride == 0)
        valid &= sampling
        z = depth[valid]
        x = (xx[valid] - intrinsic[0, 2]) / intrinsic[0, 0] * z
        y = (yy[valid] - intrinsic[1, 2]) / intrinsic[1, 1] * z
        points = np.stack([x, y, z], axis=-1)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(
            rgb[valid][:, ::-1].astype(np.float64) / 255.0
        )
        o3d.io.write_point_cloud(str(output), cloud)
        return int(len(points))

    def run(self) -> Path:
        from omegaconf import OmegaConf

        cfg = self.config
        output = cfg.output_dir.resolve()
        depth_dir = output / "frames"
        sample_dir = output / "samples"
        depth_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        left_meta = _video_metadata(cfg.left_video)
        right_meta = _video_metadata(cfg.right_video)
        if left_meta != right_meta:
            raise ValueError(f"Rectified stereo metadata differ: {left_meta} vs {right_meta}")
        frame_count, width, height, fps = left_meta
        calibration = np.load(cfg.calibration_npz)
        intrinsic = np.asarray(calibration["K_working"], dtype=np.float32)
        baseline = float(calibration["baseline_m"])
        left_valid = np.asarray(calibration["left_valid"], dtype=bool)
        right_valid = np.asarray(calibration["right_valid"], dtype=bool)
        if left_valid.shape != (height, width):
            raise ValueError(f"Validity shape {left_valid.shape} != video {(height, width)}")

        repository = cfg.repository.resolve()
        sys.path.insert(0, str(repository))
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder

        model_config = OmegaConf.load(cfg.checkpoint.parent / "cfg.yaml")
        if "vit_size" not in model_config:
            model_config["vit_size"] = "vitl"
        model_config.valid_iters = cfg.valid_iters
        model_config.hiera = 0
        model = FoundationStereo(OmegaConf.create(model_config))
        checkpoint = load_trusted_checkpoint(cfg.checkpoint)
        model.load_state_dict(checkpoint["model"])
        model.cuda().eval()
        torch.cuda.reset_peak_memory_stats()

        left_capture = cv2.VideoCapture(str(cfg.left_video))
        right_capture = cv2.VideoCapture(str(cfg.right_video))
        sample_ids = set(
            np.unique(np.rint(np.linspace(0, frame_count - 1, cfg.sample_count)).astype(int)).tolist()
        )
        per_frame: list[dict[str, float | int]] = []
        sample_records: list[dict[str, object]] = []
        started = time.perf_counter()
        try:
            for frame_id in range(frame_count):
                ok_left, left_bgr = left_capture.read()
                ok_right, right_bgr = right_capture.read()
                if not ok_left or not ok_right:
                    raise RuntimeError(f"Failed to decode rectified pair {frame_id}")
                left_tensor = (
                    torch.as_tensor(left_bgr[:, :, ::-1].copy()).cuda().float()[None].permute(0, 3, 1, 2)
                )
                right_tensor = (
                    torch.as_tensor(right_bgr[:, :, ::-1].copy()).cuda().float()[None].permute(0, 3, 1, 2)
                )
                padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
                padded_left, padded_right = padder.pad(left_tensor, right_tensor)
                inference_start = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    disparity_tensor = model.forward(
                        padded_left,
                        padded_right,
                        iters=cfg.valid_iters,
                        test_mode=True,
                    )
                disparity = (
                    padder.unpad(disparity_tensor.float())
                    .cpu()
                    .numpy()
                    .reshape(height, width)
                )
                xx = np.arange(width)[None]
                valid = (
                    np.isfinite(disparity)
                    & (disparity > 0)
                    & ((xx - disparity) >= 0)
                    & left_valid
                )
                # A conservative right-valid check at the matched horizontal location.
                finite_disparity = np.where(np.isfinite(disparity), disparity, 0.0)
                matched_x = np.clip(
                    np.rint(xx - finite_disparity).astype(np.int32), 0, width - 1
                )
                rows = np.arange(height)[:, None]
                valid &= right_valid[rows, matched_x]
                depth = np.zeros((height, width), dtype=np.float32)
                depth[valid] = intrinsic[0, 0] * baseline / disparity[valid]
                valid &= (depth >= cfg.z_near_m) & (depth <= cfg.z_far_m)
                depth[~valid] = 0.0
                np.save(depth_dir / f"{frame_id:06d}.npy", depth)
                values = depth[depth > 0]
                per_frame.append(
                    {
                        "frame": frame_id,
                        "inference_seconds": time.perf_counter() - inference_start,
                        "valid_fraction": float(valid.mean()),
                        "median_depth_m": float(np.median(values)) if len(values) else 0.0,
                    }
                )
                if frame_id in sample_ids:
                    color = _depth_color(depth, valid)
                    preview = np.concatenate([left_bgr, color], axis=1)
                    cv2.putText(
                        preview,
                        f"frame {frame_id} | FoundationStereo metric depth",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    preview_path = sample_dir / f"frame_{frame_id:06d}_depth.jpg"
                    cv2.imwrite(str(preview_path), preview)
                    cloud_path = sample_dir / f"frame_{frame_id:06d}_camera.ply"
                    point_count = self._sample_cloud(
                        left_bgr, depth, intrinsic, cloud_path, cfg.point_stride
                    )
                    sample_records.append(
                        {
                            "frame": frame_id,
                            "depth_preview": str(preview_path.resolve()),
                            "pointcloud": str(cloud_path.resolve()),
                            "point_count": point_count,
                        }
                    )
        finally:
            left_capture.release()
            right_capture.release()
            del model
            gc.collect()
            torch.cuda.empty_cache()

        medians = np.asarray([row["median_depth_m"] for row in per_frame], dtype=np.float64)
        valid_fractions = np.asarray([row["valid_fraction"] for row in per_frame], dtype=np.float64)
        timings = np.asarray([row["inference_seconds"] for row in per_frame], dtype=np.float64)
        report = {
            "schema_version": "ego-scene-hoi-foundation-stereo-sequence-v1",
            "result": "PASS",
            "left_video": str(cfg.left_video.resolve()),
            "right_video": str(cfg.right_video.resolve()),
            "frame_count": frame_count,
            "shape": [height, width],
            "fps": fps,
            "K": intrinsic.tolist(),
            "baseline_m": baseline,
            "depth_equation": "Z_m = fx_px * baseline_m / disparity_px",
            "depth_range_m": [cfg.z_near_m, cfg.z_far_m],
            "valid_iters": cfg.valid_iters,
            "statistics": {
                "mean_inference_seconds": float(timings.mean()),
                "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "mean_valid_fraction": float(valid_fractions.mean()),
                "median_frame_median_depth_m": float(np.median(medians[medians > 0])),
            },
            "samples": sample_records,
            "per_frame": per_frame,
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path = output / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report_path
