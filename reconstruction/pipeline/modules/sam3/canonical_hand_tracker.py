#!/usr/bin/env python3
"""Track wearer hands from one or more Codex canonical bbox observations."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline.utils import parse_worker_args

from .engine import SAM3VideoEngine, SAM3VideoEngineConfig
from .object_tracker import run_object_track, video_geometry


SIDES = ("left", "right")
COLOURS = ((60, 220, 80), (220, 80, 220))


def parse_args() -> argparse.Namespace:
    """Parse one canonical-seeded hand-tracking worker job."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reasoning", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--minimum-area", type=int, default=64)
    parser.add_argument("--maximum-area-fraction", type=float, default=0.15)
    parser.add_argument(
        "--prompt-mode", choices=("bbox", "bbox_short_text"), default="bbox_short_text"
    )
    return parse_worker_args(parser)


def merge_seed_tracks(
    tracks: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    minimum_area: int,
    maximum_area: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the temporally nearest valid canonical track at every frame."""

    if not tracks:
        raise ValueError("At least one canonical track is required")
    frame_count, height, width = tracks[0][1].shape
    masks = np.zeros((frame_count, height, width), dtype=bool)
    scores = np.full(frame_count, np.nan, dtype=np.float32)
    provenance = np.full(frame_count, -1, dtype=np.int16)
    seeds = np.asarray([item[0] for item in tracks], dtype=np.int32)
    for frame_id in range(frame_count):
        for seed_index in np.argsort(np.abs(seeds - frame_id), kind="stable"):
            mask = tracks[int(seed_index)][1][frame_id]
            area = int(mask.sum())
            if minimum_area <= area <= maximum_area:
                masks[frame_id] = mask
                scores[frame_id] = tracks[int(seed_index)][2][frame_id]
                provenance[frame_id] = int(seed_index)
                break
    return masks, scores, provenance


def _boxes(masks: np.ndarray) -> np.ndarray:
    output = np.full((len(masks), 4), np.nan, dtype=np.float32)
    for frame_id in np.flatnonzero(masks.any(axis=(1, 2))):
        yy, xx = np.nonzero(masks[frame_id])
        output[frame_id] = [xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]
    return output


def _write_overlay(video: Path, masks: np.ndarray, boxes: np.ndarray, output: Path) -> None:
    frame_count, height, width, fps = video_geometry(video)
    capture = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not capture.isOpened() or not writer.isOpened():
        raise OSError(output)
    try:
        for frame_id in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"Cannot read frame {frame_id}")
            for hand_id, side in enumerate(SIDES):
                mask = masks[frame_id, hand_id]
                if not mask.any():
                    continue
                colour = np.asarray(COLOURS[hand_id], dtype=np.uint8)
                frame[mask] = (0.55 * frame[mask] + 0.45 * colour).astype(np.uint8)
                x0, y0, x1, y1 = np.rint(boxes[frame_id, hand_id]).astype(int)
                cv2.rectangle(frame, (x0, y0), (x1, y1), tuple(map(int, colour)), 2)
                cv2.putText(
                    frame,
                    f"wearer {side}",
                    (x0, max(20, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    tuple(map(int, colour)),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        capture.release()
        writer.release()


class CanonicalHandTracker:
    """Run independent bidirectional tracks from every hand canonical observation."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments

    @classmethod
    def from_worker_job(cls) -> "CanonicalHandTracker":
        return cls(parse_args())

    def run(self) -> Path:
        args = self.arguments
        video = args.video.expanduser().resolve()
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        reasoning = json.loads(args.reasoning.expanduser().resolve().read_text(encoding="utf-8"))
        hand_records = reasoning.get("hands")
        if not isinstance(hand_records, list):
            raise ValueError("Reasoning has no hands registry")
        by_side = {str(item["side"]): item for item in hand_records}
        frame_count, height, width, _ = video_geometry(video)
        masks = np.zeros((frame_count, 2, height, width), dtype=bool)
        scores = np.full((frame_count, 2), np.nan, dtype=np.float32)
        provenance = np.full((frame_count, 2), -1, dtype=np.int16)
        seed_manifest: dict[str, list[dict]] = {side: [] for side in SIDES}
        maximum_area = max(args.minimum_area, int(args.maximum_area_fraction * height * width))
        config = SAM3VideoEngineConfig(
            repository_root=Path(__file__).resolve().parents[3],
            checkpoint=args.checkpoint,
            gpu=args.gpu,
            apply_temporal_disambiguation=False,
        )
        started = time.perf_counter()
        with SAM3VideoEngine(config) as predictor:
            response = predictor.handle_request(
                {"type": "start_session", "resource_path": str(video)}
            )
            session_id = response["session_id"]
            try:
                for hand_id, side in enumerate(SIDES):
                    canonical_tracks: list[tuple[int, np.ndarray, np.ndarray]] = []
                    for canonical in by_side.get(side, {}).get("canonicals", []):
                        seed_id = int(canonical["id"])
                        record = {
                            "id": seed_id,
                            "global_object_id": f"hand_{side}_{seed_id:03d}",
                            "name": f"wearer {side} hand",
                            "sam3_prompt": f"{side} hand",
                            "canonical": {
                                "frame": int(canonical["frame"]),
                                "bbox_xyxy": [int(value) for value in canonical["bbox_xyxy"]],
                            },
                        }
                        track = run_object_track(
                            predictor,
                            session_id,
                            record,
                            frame_count=frame_count,
                            height=height,
                            width=width,
                            minimum_area=args.minimum_area,
                            prompt_mode=args.prompt_mode,
                        )
                        canonical_tracks.append((track.seed_source_frame, track.masks, track.scores))
                        seed_manifest[side].append(record["canonical"])
                    if canonical_tracks:
                        masks[:, hand_id], scores[:, hand_id], provenance[:, hand_id] = (
                            merge_seed_tracks(
                                canonical_tracks,
                                minimum_area=args.minimum_area,
                                maximum_area=maximum_area,
                            )
                        )
                    del canonical_tracks
                    gc.collect()
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
        boxes = np.stack([_boxes(masks[:, hand_id]) for hand_id in range(2)], axis=1)
        np.savez_compressed(
            output / "hand_masks.npz",
            packed_masks=np.packbits(masks, axis=-1),
            mask_width=np.asarray(width, dtype=np.int32),
            scores=scores,
            bbox_xyxy=boxes,
            bbox_xyxy_native=boxes,
            valid=masks.any(axis=(-1, -2)),
            provenance=provenance,
            hand_labels=np.asarray(SIDES),
            native_size=np.asarray([width, height], dtype=np.int32),
        )
        _write_overlay(video, masks, boxes, output / "overlay.mp4")
        report = {
            "schema_version": "codex-sam3-wearer-hand-tracks-v2",
            "video": str(video),
            "reasoning": str(args.reasoning.expanduser().resolve()),
            "prompt_mode": args.prompt_mode,
            "merge_policy": "temporally nearest valid canonical track",
            "visible_frames": {
                side: int(masks[:, hand_id].any(axis=(1, 2)).sum())
                for hand_id, side in enumerate(SIDES)
            },
            "canonical_seeds": seed_manifest,
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path = output / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(report_path)
        return report_path


def main() -> None:
    CanonicalHandTracker.from_worker_job().run()


if __name__ == "__main__":
    main()
