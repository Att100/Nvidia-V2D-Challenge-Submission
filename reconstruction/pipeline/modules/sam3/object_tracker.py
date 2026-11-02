#!/usr/bin/env python3
"""Track each Codex-discovered physical object once through the complete video.

Codex supplies a globally associated object registry and one canonical frame with
an original-resolution bounding box per physical instance. SAM3 jointly uses that
box and the concise instance phrase, then independently propagates forward and
backward. There are no window-local tracklets or post-hoc identity association.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.utils import parse_worker_args
from .engine import SAM3VideoEngine, SAM3VideoEngineConfig


@dataclass
class Tracklet:
    """One full-video SAM3 track for one globally identified physical object."""

    tracklet_id: str
    object_id: str
    object_name: str
    seed_source_frame: int
    selected_sam3_object_id: int
    seed_score: float
    seed_candidates: dict[int, dict[str, float | int]]
    masks: np.ndarray
    scores: np.ndarray
    reasoning_object: dict[str, Any]

    @property
    def source_frames(self) -> np.ndarray:
        return np.arange(len(self.masks), dtype=np.int32)

    def metadata(self) -> dict[str, Any]:
        visible = np.flatnonzero(self.masks.any(axis=(1, 2)))
        return {
            "tracklet_id": self.tracklet_id,
            "clip_id": 0,
            "event_index": int(self.reasoning_object["id"]),
            "global_object_id": self.object_id,
            "object_name": self.object_name,
            "prompt": self.reasoning_object["sam3_prompt"],
            "seed_source_frame": self.seed_source_frame,
            "seed_window_frame": self.seed_source_frame,
            "window_start_frame": 0,
            "window_end_frame_exclusive": len(self.masks),
            "selected_sam3_object_id": self.selected_sam3_object_id,
            "seed_score": self.seed_score,
            "candidate_seed_scores": {
                str(key): float(value["score"])
                for key, value in self.seed_candidates.items()
            },
            "seed_candidates": {
                str(key): value for key, value in self.seed_candidates.items()
            },
            "visible_mask_count": int(len(visible)),
            "first_visible_source_frame": int(visible[0]) if len(visible) else None,
            "last_visible_source_frame": int(visible[-1]) if len(visible) else None,
            "mask_storage": f"tracklets/{self.tracklet_id}.npz",
            "reasoning_object": self.reasoning_object,
        }


def parse_args() -> argparse.Namespace:
    """Parse a full-video canonical-seed tracking job."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min-mask-area", type=int, default=64)
    parser.add_argument("--min-mask-area-fraction", type=float, default=0.0)
    parser.add_argument(
        "--prompt-mode",
        choices=("bbox", "bbox_short_text"),
        default="bbox",
        help="Use the canonical bbox alone or jointly with the concise SAM3 phrase.",
    )
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Segment only each canonical frame without temporal propagation.",
    )
    parser.add_argument("--schema-version", required=True)
    parser.add_argument(
        "--registry-key",
        choices=("objects", "support_objects"),
        default="objects",
        help="Codex registry to segment; support objects remain outside dynamic masks.",
    )
    return parse_worker_args(parser)


def as_numpy(value: Any) -> np.ndarray:
    """Convert tensor-like SAM3 values to NumPy arrays."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def normalize_masks(value: Any) -> np.ndarray:
    """Normalize SAM3 masks to N,H,W booleans."""

    masks = as_numpy(value).astype(bool, copy=False)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected N,H,W masks, got {masks.shape}")
    return masks


def video_geometry(path: Path) -> tuple[int, int, int, float]:
    """Return frame count, height, width, and FPS."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(path)
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if min(count, height, width, fps) <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return count, height, width, fps


def visual_prompt_request(
    session_id: str,
    frame_index: int,
    bbox_xyxy: list[int],
    width: int,
    height: int,
    text: str | None = None,
) -> dict[str, Any]:
    """Build a normalized positive-box prompt, optionally conditioned by short text."""

    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    return {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": int(frame_index),
        "text": text or "visual",
        "bounding_boxes": [
            [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]
        ],
        "bounding_box_labels": [1],
    }


def update_outputs(
    output: dict[str, Any] | None,
    frame_index: int,
    *,
    frame_count: int,
    height: int,
    width: int,
    masks_by_id: dict[int, np.ndarray],
    scores_by_id: dict[int, np.ndarray],
) -> None:
    """Accumulate one streamed SAM3 output into dense per-instance arrays."""

    if output is None:
        return
    ids = as_numpy(output["out_obj_ids"]).astype(np.int64).reshape(-1)
    masks = normalize_masks(output["out_binary_masks"])
    scores = as_numpy(output["out_probs"]).astype(np.float32).reshape(-1)
    if not (len(ids) == len(masks) == len(scores)):
        raise ValueError("SAM3 output fields have inconsistent lengths")
    for raw_id_value, mask, score in zip(ids, masks, scores, strict=True):
        raw_id = int(raw_id_value)
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if raw_id not in masks_by_id:
            masks_by_id[raw_id] = np.zeros((frame_count, height, width), dtype=bool)
            scores_by_id[raw_id] = np.full(frame_count, np.nan, dtype=np.float32)
        masks_by_id[raw_id][frame_index] = mask
        scores_by_id[raw_id][frame_index] = float(score)


def seed_candidate_metrics(
    masks_by_id: dict[int, np.ndarray],
    scores_by_id: dict[int, np.ndarray],
    seed_frame: int,
    bbox_xyxy: list[int],
) -> dict[int, dict[str, float | int]]:
    """Score seed instances by overlap with Codex's physical-instance box."""

    x0, y0, x1, y1 = bbox_xyxy
    box_area = max(1, (x1 - x0) * (y1 - y0))
    candidates: dict[int, dict[str, float | int]] = {}
    for raw_id, masks in masks_by_id.items():
        mask = masks[seed_frame]
        score = float(scores_by_id[raw_id][seed_frame])
        if not mask.any() or not math.isfinite(score):
            continue
        intersection = int(mask[y0:y1, x0:x1].sum())
        mask_area = int(mask.sum())
        union = mask_area + box_area - intersection
        candidates[raw_id] = {
            "score": score,
            "area": mask_area,
            "bbox_intersection": intersection,
            "bbox_iou": float(intersection / max(1, union)),
            "bbox_coverage": float(intersection / box_area),
            "mask_containment": float(intersection / max(1, mask_area)),
        }
    return candidates


def select_candidate(candidates: dict[int, dict[str, float | int]]) -> int:
    """Select the instance best constrained by the canonical bounding box."""

    if not candidates:
        raise RuntimeError("SAM3 returned no valid instance at the canonical frame")
    return max(
        candidates,
        key=lambda raw_id: (
            float(candidates[raw_id]["bbox_iou"]),
            float(candidates[raw_id]["bbox_coverage"]),
            float(candidates[raw_id]["score"]),
        ),
    )


def propagate_direction(
    predictor: Any,
    session_id: str,
    *,
    direction: str,
    seed_frame: int,
    bbox_xyxy: list[int],
    frame_count: int,
    height: int,
    width: int,
    prompt_text: str | None = None,
) -> tuple[int, dict[int, dict[str, float | int]], np.ndarray, np.ndarray]:
    """Start from a fresh visual seed and propagate in exactly one direction."""

    predictor.handle_request({"type": "reset_session", "session_id": session_id})
    response = predictor.handle_request(
        visual_prompt_request(
            session_id,
            seed_frame,
            bbox_xyxy,
            width,
            height,
            prompt_text,
        )
    )
    masks_by_id: dict[int, np.ndarray] = {}
    scores_by_id: dict[int, np.ndarray] = {}
    update_outputs(
        response.get("outputs"),
        seed_frame,
        frame_count=frame_count,
        height=height,
        width=width,
        masks_by_id=masks_by_id,
        scores_by_id=scores_by_id,
    )
    candidates = seed_candidate_metrics(
        masks_by_id, scores_by_id, seed_frame, bbox_xyxy
    )
    selected = select_candidate(candidates)
    request = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": direction,
        "start_frame_index": seed_frame,
        "max_frame_num_to_track": frame_count,
    }
    for record in predictor.handle_stream_request(request):
        update_outputs(
            record.get("outputs"),
            int(record["frame_index"]),
            frame_count=frame_count,
            height=height,
            width=width,
            masks_by_id=masks_by_id,
            scores_by_id=scores_by_id,
        )
    return selected, candidates, masks_by_id[selected], scores_by_id[selected]


def run_object_track(
    predictor: Any,
    session_id: str,
    object_record: dict[str, Any],
    *,
    frame_count: int,
    height: int,
    width: int,
    minimum_area: int,
    prompt_mode: str = "bbox",
) -> Tracklet:
    """Track one Codex-associated object from its canonical frame in both directions."""

    canonical = object_record["canonical"]
    seed_frame = int(canonical["frame"])
    bbox = [int(value) for value in canonical["bbox_xyxy"]]
    prompt_text = str(object_record["sam3_prompt"]) if prompt_mode == "bbox_short_text" else None
    forward_id, candidates, forward_masks, forward_scores = propagate_direction(
        predictor,
        session_id,
        direction="forward",
        seed_frame=seed_frame,
        bbox_xyxy=bbox,
        frame_count=frame_count,
        height=height,
        width=width,
        prompt_text=prompt_text,
    )
    _, _, backward_masks, backward_scores = propagate_direction(
        predictor,
        session_id,
        direction="backward",
        seed_frame=seed_frame,
        bbox_xyxy=bbox,
        frame_count=frame_count,
        height=height,
        width=width,
        prompt_text=prompt_text,
    )
    masks = forward_masks.copy()
    scores = forward_scores.copy()
    replace = backward_masks.any(axis=(1, 2)) & (
        ~masks.any(axis=(1, 2))
        | (
            np.nan_to_num(backward_scores, nan=-np.inf)
            > np.nan_to_num(scores, nan=-np.inf)
        )
    )
    masks[replace] = backward_masks[replace]
    scores[replace] = backward_scores[replace]
    keep = masks.sum(axis=(1, 2)) >= minimum_area
    masks[~keep] = False
    scores[~keep] = np.nan
    if not masks[seed_frame].any():
        raise RuntimeError(
            f"Canonical seed mask is smaller than {minimum_area} pixels"
        )
    object_id = str(object_record["global_object_id"])
    return Tracklet(
        tracklet_id=object_id,
        object_id=object_id,
        object_name=str(object_record["name"]),
        seed_source_frame=seed_frame,
        selected_sam3_object_id=forward_id,
        seed_score=float(candidates[forward_id]["score"]),
        seed_candidates=candidates,
        masks=masks,
        scores=scores,
        reasoning_object=object_record,
    )


def run_object_seed(
    predictor: Any,
    session_id: str,
    object_record: dict[str, Any],
    *,
    frame_count: int,
    height: int,
    width: int,
    minimum_area: int,
    prompt_mode: str = "bbox",
) -> Tracklet:
    """Segment one canonical RGB frame from its transferred Codex box."""

    canonical = object_record["canonical"]
    seed_frame = int(canonical["frame"])
    bbox = [int(value) for value in canonical["bbox_xyxy"]]
    predictor.handle_request({"type": "reset_session", "session_id": session_id})
    prompt_text = str(object_record["sam3_prompt"]) if prompt_mode == "bbox_short_text" else None
    response = predictor.handle_request(
        visual_prompt_request(session_id, seed_frame, bbox, width, height, prompt_text)
    )
    masks_by_id: dict[int, np.ndarray] = {}
    scores_by_id: dict[int, np.ndarray] = {}
    update_outputs(
        response.get("outputs"),
        seed_frame,
        frame_count=frame_count,
        height=height,
        width=width,
        masks_by_id=masks_by_id,
        scores_by_id=scores_by_id,
    )
    candidates = seed_candidate_metrics(masks_by_id, scores_by_id, seed_frame, bbox)
    selected = select_candidate(candidates)
    masks = masks_by_id[selected]
    scores = scores_by_id[selected]
    if int(masks[seed_frame].sum()) < minimum_area:
        raise RuntimeError(f"Canonical seed mask is smaller than {minimum_area} pixels")
    object_id = str(object_record["global_object_id"])
    return Tracklet(
        tracklet_id=object_id,
        object_id=object_id,
        object_name=str(object_record["name"]),
        seed_source_frame=seed_frame,
        selected_sam3_object_id=selected,
        seed_score=float(scores[seed_frame]),
        seed_candidates=candidates,
        masks=masks,
        scores=scores,
        reasoning_object=object_record,
    )


def save_tracklet(tracklet: Tracklet, output_dir: Path) -> None:
    """Store a compact bit-packed full-video mask sequence and metadata."""

    directory = output_dir / "tracklets"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / f"{tracklet.tracklet_id}.npz",
        packed_masks=np.packbits(tracklet.masks, axis=2),
        mask_width=np.asarray(tracklet.masks.shape[2], dtype=np.int32),
        scores=tracklet.scores,
        source_frames=tracklet.source_frames,
    )
    (directory / f"{tracklet.tracklet_id}.json").write_text(
        json.dumps(tracklet.metadata(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_overlay(video: Path, tracklets: list[Tracklet], output: Path) -> None:
    """Render masks, boxes, and stable global object IDs on the source video."""

    count, height, width, fps = video_geometry(video)
    capture = cv2.VideoCapture(str(video))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    colors = [(235, 85, 69), (74, 190, 112), (66, 135, 245), (220, 175, 65)]
    try:
        for frame_index in range(count):
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"Cannot read frame {frame_index}")
            for index, tracklet in enumerate(tracklets):
                mask = tracklet.masks[frame_index]
                if not mask.any():
                    continue
                color = np.asarray(colors[index % len(colors)], dtype=np.uint8)
                frame[mask] = (0.55 * frame[mask] + 0.45 * color).astype(np.uint8)
                ys, xs = np.nonzero(mask)
                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = int(xs.max()), int(ys.max())
                cv2.rectangle(frame, (x0, y0), (x1, y1), tuple(map(int, color)), 3)
                cv2.putText(
                    frame,
                    f"{tracklet.object_id} {tracklet.object_name}",
                    (x0, max(24, y0 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    tuple(map(int, color)),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        capture.release()
        writer.release()


def run_object_tracking(args: argparse.Namespace) -> None:
    """Run one canonical-seeded full-video SAM3 track per Codex object."""

    reasoning_path = args.reasoning.expanduser().resolve()
    video = args.video.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reasoning = json.loads(reasoning_path.read_text(encoding="utf-8"))
    frame_count, height, width, _ = video_geometry(video)
    declared = int(reasoning["video"]["num_frames"])
    if frame_count != declared:
        raise ValueError(f"Video has {frame_count} frames but reasoning declares {declared}")
    minimum_area = max(
        int(args.min_mask_area),
        int(math.ceil(args.min_mask_area_fraction * height * width)),
    )
    config = SAM3VideoEngineConfig(
        repository_root=Path(__file__).resolve().parents[3],
        gpu=args.gpu,
        checkpoint=args.checkpoint,
        apply_temporal_disambiguation=False,
    )
    tracklets: list[Tracklet] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    with SAM3VideoEngine(config) as predictor:
        response = predictor.handle_request(
            {"type": "start_session", "resource_path": str(video)}
        )
        session_id = response["session_id"]
        try:
            registry = reasoning.get(args.registry_key)
            if not isinstance(registry, list):
                raise ValueError(f"Reasoning has no {args.registry_key!r} registry")
            for object_record in registry:
                object_start = time.perf_counter()
                try:
                    tracklet_runner = run_object_seed if args.seed_only else run_object_track
                    tracklet = tracklet_runner(
                        predictor,
                        session_id,
                        object_record,
                        frame_count=frame_count,
                        height=height,
                        width=width,
                        minimum_area=minimum_area,
                        prompt_mode=args.prompt_mode,
                    )
                    save_tracklet(tracklet, output)
                    tracklets.append(tracklet)
                    print(
                        f"{tracklet.object_id} {tracklet.object_name!r}: "
                        f"canonical={tracklet.seed_source_frame}, "
                        f"visible={tracklet.metadata()['visible_mask_count']}, "
                        f"seconds={time.perf_counter() - object_start:.1f}",
                        flush=True,
                    )
                except Exception as error:  # noqa: BLE001 - isolate one object failure
                    failure = {
                        "global_object_id": object_record["global_object_id"],
                        "object_name": object_record["name"],
                        "canonical_frame": object_record["canonical"]["frame"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                    failures.append(failure)
                    print(json.dumps(failure), flush=True)
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
    if not tracklets and args.registry_key != "support_objects":
        raise RuntimeError(f"SAM3 failed for every discovered object: {failures}")
    if not args.no_overlay:
        write_overlay(video, tracklets, output / "overlays/full_video.mp4")

    successful = {tracklet.object_id: tracklet for tracklet in tracklets}
    global_tracks = [
        {
            "global_object_id": tracklet.object_id,
            "canonical_object_name": tracklet.object_name,
            "observed_object_names": [tracklet.object_name],
            "tracklet_ids": [tracklet.tracklet_id],
            "identity_status": "codex_global_physical_object",
            "canonical_frame": tracklet.seed_source_frame,
            "canonical_bbox_xyxy": tracklet.reasoning_object["canonical"]["bbox_xyxy"],
        }
        for tracklet in tracklets
    ]
    episodes = []
    for interaction in reasoning["interactions"]:
        object_id = str(interaction["global_object_id"])
        if object_id not in successful:
            continue
        tracklet = successful[object_id]
        start, end = interaction["frame_interval"]
        episodes.append(
            {
                "episode_id": f"interaction_{int(interaction['id']):04d}",
                "global_object_id": object_id,
                "object": tracklet.object_name,
                "start_frame": int(start),
                "end_frame": int(end),
                "start_time_sec": float(interaction["time_interval_sec"][0]),
                "end_time_sec": float(interaction["time_interval_sec"][1]),
                "source_tracklets": [tracklet.tracklet_id],
                "source_clips": [],
                "observed_object_names": [tracklet.object_name],
                "prompts": [tracklet.reasoning_object["sam3_prompt"]],
                "atoms": [
                    {
                        "hand": interaction["hand"],
                        "start_frame": int(start),
                        "end_frame": int(end),
                        "start_time_sec": float(interaction["time_interval_sec"][0]),
                        "end_time_sec": float(interaction["time_interval_sec"][1]),
                        "physical_support": "unknown",
                    }
                ],
                "confidence": 1.0,
                "status": "codex_associated",
            }
        )
    result = {
        "schema_version": args.schema_version,
        "reasoning_source": str(reasoning_path),
        "video": str(video),
        "tracking_policy": {
            "registry_key": args.registry_key,
            "mode": (
                "codex_canonical_seed_only"
                if args.seed_only
                else "codex_canonical_full_video"
            ),
            "seed": "Codex global object canonical frame and bbox",
            "propagation_direction": (
                "none" if args.seed_only else "independent forward and backward passes"
            ),
            "propagation_extent": "canonical frame only" if args.seed_only else "complete video",
            "minimum_mask_area_pixels": minimum_area,
            "prompt_mode": args.prompt_mode,
        },
        "association_policy": {
            "mode": "none",
            "reason": "Codex supplies one globally associated physical-object registry",
        },
        "tracklets": [tracklet.metadata() for tracklet in tracklets],
        "global_object_tracks": global_tracks,
        "contact_episodes": episodes,
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = output / "contact_tracks.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(result_path)


class SAM3ObjectTracker:
    """Worker-facing Codex-canonical SAM3 tracker."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments

    @classmethod
    def from_worker_job(cls) -> "SAM3ObjectTracker":
        return cls(parse_args())

    def run(self) -> None:
        run_object_tracking(self.arguments)


def main() -> None:
    SAM3ObjectTracker.from_worker_job().run()


if __name__ == "__main__":
    main()
