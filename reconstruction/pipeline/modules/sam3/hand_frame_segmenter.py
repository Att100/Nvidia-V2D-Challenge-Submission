"""Independent box-and-text-prompted SAM 3 hand segmentation.

Unlike the object tracker, second-pass masks are segmented independently. A
mask exists only when the coarse identity track provides a valid prompt for the
same source frame and SAM 3 accepts it.
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from pipeline.utils import write_json


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1.0)


def _normalized_center_distance(a: np.ndarray, b: np.ndarray) -> float:
    center_a = (a[:2] + a[2:]) * 0.5
    center_b = (b[:2] + b[2:]) * 0.5
    size = 0.5 * (np.linalg.norm(a[2:] - a[:2]) + np.linalg.norm(b[2:] - b[:2]))
    return float(np.linalg.norm(center_a - center_b) / max(size, 1.0))


def _visible_runs(valid: np.ndarray, maximum_internal_gap: int = 2) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    chunks: list[list[int]] = [[int(indices[0])]]
    for raw in indices[1:]:
        frame = int(raw)
        if frame - chunks[-1][-1] - 1 > maximum_internal_gap:
            chunks.append([frame])
        else:
            chunks[-1].append(frame)
    return [np.asarray(chunk, dtype=np.int32) for chunk in chunks]


def suppress_ambiguous_handedness(
    valid: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    duplicate_iou: float,
    maximum_flip_run: int,
    identity_gap: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Reject duplicate slots and short handedness flips conservatively.

    Coarse tracking can occasionally emit the same physical hand in both slots,
    or changes its handedness label for a few boundary frames.  Since a wrong
    MANO chirality cannot be repaired by relabeling, ambiguous detections are
    invalidated instead of copied into the opposite slot.
    """

    cleaned = np.asarray(valid, dtype=bool).copy()
    records: list[dict[str, Any]] = []
    frame_count = len(cleaned)

    # Resolve same-frame duplicate boxes using temporal identity support.  The
    # source slot with nearby exclusive detections wins; score is only a tie
    # breaker because the other chirality may receive a higher confidence.
    for frame in range(frame_count):
        if not cleaned[frame].all() or _box_iou(boxes[frame, 0], boxes[frame, 1]) < duplicate_iou:
            continue
        support = np.zeros(2, dtype=np.float32)
        for hand in range(2):
            for direction in (-1, 1):
                for gap in range(1, identity_gap + 1):
                    other_frame = frame + direction * gap
                    if not 0 <= other_frame < frame_count:
                        break
                    if cleaned[other_frame, hand] and not cleaned[other_frame, 1 - hand]:
                        distance = _normalized_center_distance(
                            boxes[frame, hand], boxes[other_frame, hand]
                        )
                        if distance < 0.8:
                            support[hand] += np.exp(-gap / 4.0) * (1.0 - distance / 0.8)
                        break
        keep = int(np.argmax(support)) if not np.isclose(*support) else int(np.nanargmax(scores[frame]))
        reject = 1 - keep
        cleaned[frame, reject] = False
        records.append(
            {
                "frame": frame,
                "reason": "duplicate_left_right_boxes",
                "kept_slot": keep,
                "rejected_slot": reject,
                "box_iou": _box_iou(boxes[frame, 0], boxes[frame, 1]),
            }
        )

    # A short run is considered a handedness flip only when the other slot is
    # the dominant track and has a spatially continuous detection immediately
    # before or after the run.  We drop it rather than mirror a MANO mesh.
    counts = cleaned.sum(axis=0)
    for hand in range(2):
        other = 1 - hand
        if counts[other] < max(3 * max(int(counts[hand]), 1), 20):
            continue
        for run in _visible_runs(cleaned[:, hand]):
            if len(run) > maximum_flip_run:
                continue
            boundary_match = False
            for edge in (int(run[0]), int(run[-1])):
                for direction in (-1, 1):
                    for gap in range(1, identity_gap + 1):
                        candidate = edge + direction * gap
                        if candidate in run:
                            continue
                        if not 0 <= candidate < frame_count:
                            break
                        if cleaned[candidate, other]:
                            if _normalized_center_distance(
                                boxes[edge, hand], boxes[candidate, other]
                            ) < 0.45:
                                boundary_match = True
                            break
            if not boundary_match:
                continue
            cleaned[run, hand] = False
            records.append(
                {
                    "frames": run.tolist(),
                    "reason": "short_handedness_flip",
                    "rejected_slot": hand,
                }
            )
    return cleaned, records


@dataclass(frozen=True)
class HandFrameSegmentationConfig:
    """Settings for independent hand segmentation and identity gating."""

    confidence_threshold: float = 0.3
    minimum_mask_area: int = 64
    box_expand_ratio: float = 1.15
    duplicate_box_iou: float = 0.65
    maximum_flip_run: int = 12
    identity_gap: int = 5
    device: str = "cuda"
    checkpoint: Path | None = None
    schema_version: str = "ego-scene-hoi-global-object-masks-v2"


class SAM3HandFrameSegmenter:
    """Refine coarse hand tracks using independent per-frame SAM3 prompts."""

    def __init__(self, config: HandFrameSegmentationConfig, repository_root: Path) -> None:
        self.config = config
        self.repository_root = repository_root.resolve()

    def run(self, *, video_path: Path, hand_motion_path: Path, output_dir: Path) -> Path:
        import torch

        sys.path.insert(0, str(self.repository_root / "thirdparty"))
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        with np.load(hand_motion_path) as archive:
            valid = np.asarray(archive["valid"], dtype=bool)
            boxes = np.asarray(archive["bbox_xyxy"], dtype=np.float32)
            score_key = "detection_score" if "detection_score" in archive.files else "scores"
            scores = np.asarray(archive[score_key], dtype=np.float32)
            labels = [str(value).strip().lower() for value in archive["hand_labels"]]
            source_indices = np.asarray(
                archive["source_indices"]
                if "source_indices" in archive.files
                else np.arange(len(valid)),
                dtype=np.int32,
            )
        if labels != ["left", "right"]:
            raise ValueError(f"Expected [left, right] hand slots, got {labels}")
        accepted_input, identity_rejections = suppress_ambiguous_handedness(
            valid,
            boxes,
            scores,
            duplicate_iou=self.config.duplicate_box_iou,
            maximum_flip_run=self.config.maximum_flip_run,
            identity_gap=self.config.identity_gap,
        )

        capture = cv2.VideoCapture(str(video_path.resolve()))
        if not capture.isOpened():
            raise OSError(f"Could not open {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = len(valid)
        if len(source_indices) != frame_count:
            raise ValueError("Track source-index count differs from valid detections")
        masks = np.zeros((frame_count, 2, height, width), dtype=bool)
        mask_scores = np.full((frame_count, 2), -np.inf, dtype=np.float32)

        prompt_boxes = boxes.copy()
        prompt_valid = accepted_input.copy()

        kwargs: dict[str, Any] = {"device": self.config.device}
        if self.config.checkpoint is not None:
            kwargs.update(checkpoint_path=str(self.config.checkpoint.resolve()), load_from_HF=False)
        model = build_sam3_image_model(**kwargs)
        processor = Sam3Processor(
            model,
            device=self.config.device,
            confidence_threshold=self.config.confidence_threshold,
        )
        started = time.perf_counter()
        decoded = 0
        try:
            for frame in np.flatnonzero(prompt_valid.any(axis=1)):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_indices[frame]))
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode source frame {source_indices[frame]}")
                decoded += 1
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                state = processor.set_image(image)
                for hand in np.flatnonzero(prompt_valid[frame]):
                    processor.reset_all_prompts(state)
                    # The coarse temporal track supplies identity and locality.
                    text_prompt = f"{labels[hand]} hand"
                    state = processor.set_text_prompt(
                        prompt=text_prompt,
                        state=state,
                    )
                    x1, y1, x2, y2 = prompt_boxes[frame, hand]
                    box = [
                        float((x1 + x2) * 0.5 / width),
                        float((y1 + y2) * 0.5 / height),
                        float((x2 - x1) / width),
                        float((y2 - y1) / height),
                    ]
                    state = processor.add_geometric_prompt(box=box, label=True, state=state)
                    candidates = _as_numpy(state.get("masks", [])).astype(bool)
                    candidate_scores = _as_numpy(state.get("scores", [])).reshape(-1)
                    if candidates.ndim == 4 and candidates.shape[1] == 1:
                        candidates = candidates[:, 0]
                    if candidates.ndim == 2:
                        candidates = candidates[None]
                    if not len(candidates):
                        continue
                    if candidates.shape[1:] != (height, width):
                        candidates = np.stack(
                            [
                                cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
                                for mask in candidates
                            ]
                        )
                    # Keep the proposal most concentrated around the detector box.
                    ix1, iy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
                    ix2, iy2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
                    overlaps = np.asarray(
                        [mask[iy1:iy2, ix1:ix2].sum() / max(int(mask.sum()), 1) for mask in candidates]
                    )
                    best = int(np.argmax(overlaps))
                    mask = candidates[best]
                    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                    bw = (x2 - x1) * self.config.box_expand_ratio
                    bh = (y2 - y1) * self.config.box_expand_ratio
                    ex1, ey1 = max(0, int(cx - bw * 0.5)), max(0, int(cy - bh * 0.5))
                    ex2, ey2 = min(width, int(np.ceil(cx + bw * 0.5))), min(height, int(np.ceil(cy + bh * 0.5)))
                    clipped = np.zeros_like(mask)
                    clipped[ey1:ey2, ex1:ex2] = mask[ey1:ey2, ex1:ex2]
                    if int(clipped.sum()) < self.config.minimum_mask_area:
                        continue
                    masks[frame, hand] = clipped
                    mask_scores[frame, hand] = (
                        float(candidate_scores[best]) if best < len(candidate_scores) else 1.0
                    )
        finally:
            capture.release()
            del processor, model
            gc.collect()
            torch.cuda.empty_cache()

        output_dir.mkdir(parents=True, exist_ok=True)
        # Stereo consumers use one dense [frame,left/right,H,W] archive.  Keep
        # the per-hand files below for compact downstream access
        # manifest, but make the refined two-pass masks the authoritative
        # artifact for hand geometry.
        boxes_refined = np.full((frame_count, 2, 4), np.nan, dtype=np.float32)
        for frame, hand in np.argwhere(masks.any(axis=(-1, -2))):
            yy, xx = np.nonzero(masks[frame, hand])
            boxes_refined[frame, hand] = [
                xx.min(), yy.min(), xx.max() + 1, yy.max() + 1
            ]
        np.savez_compressed(
            output_dir / "hand_masks.npz",
            packed_masks=np.packbits(masks, axis=-1),
            mask_width=np.asarray(width, dtype=np.int32),
            scores=mask_scores,
            bbox_xyxy=boxes_refined,
            bbox_xyxy_native=boxes_refined,
            valid=masks.any(axis=(-1, -2)),
            provenance=np.full((frame_count, 2), -1, dtype=np.int16),
            prompt_bbox_xyxy=prompt_boxes,
            prompt_valid=prompt_valid,
            hand_labels=np.asarray(labels),
            source_indices=source_indices,
            native_size=np.asarray([width, height], dtype=np.int32),
        )
        records = []
        # Keep right first for stable compatibility track IDs.
        for hand in (1, 0):
            label = labels[hand]
            path = output_dir / f"{label}_hand.npz"
            visible = masks[:, hand].reshape(frame_count, -1).any(axis=1)
            np.savez_compressed(
                path,
                packed_masks=np.packbits(masks[:, hand], axis=2),
                mask_width=np.asarray(width, dtype=np.int32),
                scores=mask_scores[:, hand],
                visible=visible,
                accepted_track_detection=accepted_input[:, hand],
                source_indices=source_indices,
            )
            visible_ids = np.flatnonzero(visible)
            records.append(
                {
                    "global_object_id": f"{label}_hand",
                    "object_name": f"{label} hand",
                    "mask_file": str(path.resolve()),
                    "visible_frame_count": len(visible_ids),
                    "first_visible_source_frame": int(source_indices[visible_ids[0]]) if len(visible_ids) else None,
                    "last_visible_source_frame": int(source_indices[visible_ids[-1]]) if len(visible_ids) else None,
                }
            )
        manifest = output_dir / "manifest.json"
        write_json(
            manifest,
            {
                "schema_version": self.config.schema_version,
                "strategy": "tracked_bbox_independent_frame_segmentation",
                "temporal_tracking": False,
                "prompt_mode": "short_side_text_plus_tracking_box",
                "text_prompts": {
                    label: f"{label} hand"
                    for label in labels
                },
                "coarse_tracking_source": str(hand_motion_path.resolve()),
                "video": str(video_path.resolve()),
                "hand_motion": str(hand_motion_path.resolve()),
                "objects": records,
                "decoded_frame_count": decoded,
                "input_valid_counts": dict(zip(labels, valid.sum(axis=0).astype(int).tolist())),
                "identity_gated_counts": dict(zip(labels, accepted_input.sum(axis=0).astype(int).tolist())),
                "identity_rejections": identity_rejections,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        return manifest

