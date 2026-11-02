#!/usr/bin/env python3
"""Transfer canonical object boxes between synchronized camera views with Codex.

The source box is only an identity cue.  Codex must inspect the synchronized RGB
frame and localize the same physical instance in that image; no geometric resize
or homography is assumed between the cameras.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.utils import parse_worker_args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--rgb-video", type=Path, required=True)
    parser.add_argument("--source-reasoning", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-schema", type=Path, required=True)
    parser.add_argument("--prompt-template", required=True)
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--rgb-validation-masks", type=Path)
    parser.add_argument(
        "--exclude-hands",
        action="store_true",
        help="Map only interacting/support objects; hands remain in rectified left.",
    )
    return parse_worker_args(parser)


def _video_metadata(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    try:
        values = (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()
    if min(values) <= 0:
        raise ValueError(f"Invalid video metadata for {path}: {values}")
    return values


def _read_frame(capture: cv2.VideoCapture, frame_id: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame_id}")
    return frame


def _validate_bbox(value: Any, width: int, height: int, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must contain four coordinates")
    box = [int(item) for item in value]
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"{label} is outside {width}x{height}: {box}")
    return box


def _box_iou(first: list[int], second: list[int]) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return float(intersection / max(first_area + second_area - intersection, 1))


def _mask_for_frame(path: Path, frame_id: int) -> np.ndarray:
    with np.load(path) as archive:
        width = int(np.asarray(archive["mask_width"]).item())
        source_indices = np.asarray(archive["source_indices"], dtype=np.int64)
        matches = np.flatnonzero(source_indices == frame_id)
        if len(matches) != 1:
            raise ValueError(f"Frame {frame_id} is not uniquely represented in {path}")
        packed = np.asarray(archive["packed_masks"][int(matches[0])], dtype=np.uint8)
    return np.unpackbits(packed, axis=1)[:, :width].astype(bool)


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]


class CrossViewCanonicalBBoxMapper:
    """Map rectified-left canonical boxes to synchronized native RGB frames."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments

    @classmethod
    def from_worker_job(cls) -> "CrossViewCanonicalBBoxMapper":
        return cls(_parse_args())

    def _prepare_evidence(
        self,
        reasoning: dict[str, Any],
        source_video: Path,
        rgb_video: Path,
        output: Path,
    ) -> list[dict[str, Any]]:
        source_capture = cv2.VideoCapture(str(source_video))
        rgb_capture = cv2.VideoCapture(str(rgb_video))
        evidence_dir = output / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        try:
            registry_records = [
                *(
                    {
                        "record_id": str(item["global_object_id"]),
                        "name": str(item["name"]),
                        "registry_key": "objects",
                        "canonical": item["canonical"],
                    }
                    for item in reasoning["objects"]
                ),
                *(
                    {
                        "record_id": str(item["global_object_id"]),
                        "name": str(item["name"]),
                        "registry_key": "support_objects",
                        "canonical": item["canonical"],
                    }
                    for item in reasoning.get("support_objects", [])
                ),
                *(
                    {
                        "record_id": f"hand_{hand['side']}_{int(canonical['id']):03d}",
                        "name": f"camera wearer {hand['side']} hand",
                        "registry_key": "hands",
                        "side": str(hand["side"]),
                        "canonical": canonical,
                    }
                    for hand in reasoning.get("hands", [])
                    for canonical in hand.get("canonicals", [])
                    if not self.arguments.exclude_hands
                ),
            ]
            for item in registry_records:
                object_id = str(item["record_id"])
                frame_id = int(item["canonical"]["frame"])
                source_box = [int(value) for value in item["canonical"]["bbox_xyxy"]]
                source = _read_frame(source_capture, frame_id)
                rgb = _read_frame(rgb_capture, frame_id)
                x0, y0, x1, y1 = source_box
                overlay = source.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 235, 40), 4)
                cv2.putText(
                    overlay,
                    f"SOURCE {object_id}: {item['name']}",
                    (max(8, x0), max(30, y0 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (20, 235, 40),
                    2,
                    cv2.LINE_AA,
                )
                source_path = evidence_dir / f"{object_id}_frame_{frame_id:06d}_source_box.jpg"
                rgb_path = evidence_dir / f"{object_id}_frame_{frame_id:06d}_target_rgb.jpg"
                if not cv2.imwrite(str(source_path), overlay) or not cv2.imwrite(str(rgb_path), rgb):
                    raise OSError(f"Could not write evidence for {object_id}")
                records.append(
                    {
                        "global_object_id": object_id,
                        "name": str(item["name"]),
                        "registry_key": str(item["registry_key"]),
                        "side": item.get("side"),
                        "canonical_id": int(item["canonical"].get("id", 0)),
                        "frame": frame_id,
                        "source_bbox_xyxy": source_box,
                        "source_box_image": str(source_path.resolve()),
                        "target_rgb_image": str(rgb_path.resolve()),
                    }
                )
        finally:
            source_capture.release()
            rgb_capture.release()
        manifest = output / "evidence_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "objects": [item for item in records if item["registry_key"] == "objects"],
                    "support_objects": [
                        item for item in records if item["registry_key"] == "support_objects"
                    ],
                    "hands": [item for item in records if item["registry_key"] == "hands"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return records

    def run(self) -> Path:
        args = self.arguments
        source_video = args.source_video.expanduser().resolve()
        rgb_video = args.rgb_video.expanduser().resolve()
        source_reasoning = args.source_reasoning.expanduser().resolve()
        output = args.output_dir.expanduser().resolve()
        repository = args.repository_root.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        source_meta = _video_metadata(source_video)
        rgb_meta = _video_metadata(rgb_video)
        if source_meta[0] != rgb_meta[0]:
            raise ValueError(f"Synchronized videos have different frame counts: {source_meta} vs {rgb_meta}")
        if abs(source_meta[1] - rgb_meta[1]) > 1e-3:
            raise ValueError(f"Synchronized videos have different FPS: {source_meta} vs {rgb_meta}")
        reasoning = json.loads(source_reasoning.read_text(encoding="utf-8"))
        evidence = self._prepare_evidence(reasoning, source_video, rgb_video, output)
        evidence_path = output / "evidence_manifest.json"
        prompt = (
            args.prompt_template.replace("{evidence_manifest}", str(evidence_path.resolve()))
            .replace("{rgb_width}", str(rgb_meta[2]))
            .replace("{rgb_height}", str(rgb_meta[3]))
            .replace("{item_count}", str(len(evidence)))
            .replace(
                "{object_count}",
                str(sum(item["registry_key"] == "objects" for item in evidence)),
            )
            .replace(
                "{support_count}",
                str(sum(item["registry_key"] == "support_objects" for item in evidence)),
            )
            .replace(
                "{hand_canonical_count}",
                str(sum(item["registry_key"] == "hands" for item in evidence)),
            )
        )
        prompt_path = output / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        raw_path = output / "codex_raw_response.json"
        stdout_path = output / "codex_stdout.jsonl"
        stderr_path = output / "codex_stderr.log"
        command = [
            shutil.which(args.codex_binary) or args.codex_binary,
            "exec",
            "--model",
            args.model,
            "--sandbox",
            "workspace-write",
            "--cd",
            str(repository),
            "--skip-git-repo-check",
            "--ephemeral",
            "--json",
            "--config",
            f'model_reasoning_effort="{args.reasoning_effort}"',
            "--config",
            'approval_policy="never"',
            "--output-schema",
            str(args.output_schema.expanduser().resolve()),
            "--output-last-message",
            str(raw_path),
            "-",
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
            check=False,
            cwd=repository,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"codex exec failed with {completed.returncode}; see {stderr_path}")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        expected = {record["global_object_id"]: record for record in evidence}
        returned_objects = raw.get("objects")
        returned_support = raw.get("support_objects")
        returned_hands = raw.get("hands", [])
        if not all(isinstance(value, list) for value in (returned_objects, returned_support)):
            raise TypeError("Codex bbox-transfer output must contain objects/support_objects")
        if not args.exclude_hands and not isinstance(returned_hands, list):
            raise TypeError("Codex bbox-transfer output must contain hands")
        actual: dict[str, dict[str, Any]] = {
            str(record["global_object_id"]): record
            for record in [*returned_objects, *returned_support]
        }
        for hand in returned_hands:
            side = str(hand["side"])
            for canonical in hand["canonicals"]:
                actual[f"hand_{side}_{int(canonical['id']):03d}"] = canonical
        if set(actual) != set(expected):
            raise ValueError(f"Mapped canonical IDs {sorted(actual)} != expected {sorted(expected)}")

        rgb_reasoning = json.loads(json.dumps(reasoning))
        if args.exclude_hands:
            # Hand reconstruction stays entirely in rectified-left coordinates.
            # Never expose source-view boxes as if they were native-RGB boxes.
            rgb_reasoning["hands"] = []
        rgb_by_id = {
            str(item["global_object_id"]): item
            for registry_key in ("objects", "support_objects")
            for item in rgb_reasoning.get(registry_key, [])
        }
        rgb_hands = {str(item["side"]): item for item in rgb_reasoning.get("hands", [])}
        validation_records: list[dict[str, Any]] = []
        rgb_capture = cv2.VideoCapture(str(rgb_video))
        cards: list[np.ndarray] = []
        try:
            for object_id, source_record in expected.items():
                mapped = actual[object_id]
                if int(mapped["frame"]) != int(source_record["frame"]):
                    raise ValueError(f"Codex changed synchronized frame for {object_id}")
                bbox = _validate_bbox(
                    mapped["rgb_bbox_xyxy"], rgb_meta[2], rgb_meta[3], f"{object_id} RGB bbox"
                )
                visibility = str(mapped["visibility"])
                confidence = float(mapped["confidence"])
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError(f"Invalid confidence for {object_id}: {confidence}")
                if source_record["registry_key"] == "hands":
                    side = str(source_record["side"])
                    canonical_id = int(source_record["canonical_id"])
                    matches = [
                        item
                        for item in rgb_hands[side]["canonicals"]
                        if int(item["id"]) == canonical_id
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"Expected one {side} hand canonical {canonical_id}, got {len(matches)}"
                        )
                    canonical = matches[0]
                else:
                    canonical = rgb_by_id[object_id]["canonical"]
                canonical["bbox_xyxy"] = bbox
                canonical["source_rectified_left"] = {
                    "bbox_xyxy": source_record["source_bbox_xyxy"],
                    "video": str(source_video),
                }
                canonical["view"] = "native_rgb"
                canonical["visibility"] = visibility
                canonical["mapping_confidence"] = confidence

                frame = _read_frame(rgb_capture, int(source_record["frame"]))
                x0, y0, x1, y1 = bbox
                cv2.rectangle(frame, (x0, y0), (x1, y1), (20, 235, 40), 5)
                validation: dict[str, Any] = {
                    "canonical_record_id": object_id,
                    "registry_key": source_record["registry_key"],
                    "name": source_record["name"],
                    "frame": int(source_record["frame"]),
                    "rgb_bbox_xyxy": bbox,
                    "visibility": visibility,
                    "codex_confidence": confidence,
                    "in_bounds": True,
                }
                if args.rgb_validation_masks is not None:
                    mask_path = args.rgb_validation_masks.expanduser().resolve() / f"{object_id}.npz"
                    if mask_path.is_file():
                        mask = _mask_for_frame(mask_path, int(source_record["frame"]))
                        if mask.shape != frame.shape[:2]:
                            raise ValueError(f"RGB validation mask shape mismatch for {object_id}")
                        mask_box = _mask_bbox(mask)
                        if mask_box is not None:
                            inside = np.zeros_like(mask)
                            inside[y0:y1, x0:x1] = True
                            overlap = int((mask & inside).sum())
                            mask_pixels = int(mask.sum())
                            box_area = int((x1 - x0) * (y1 - y0))
                            center_box = np.asarray([(x0 + x1) / 2, (y0 + y1) / 2])
                            center_mask = np.asarray(
                                [(mask_box[0] + mask_box[2]) / 2, (mask_box[1] + mask_box[3]) / 2]
                            )
                            diagonal = max(float(np.hypot(x1 - x0, y1 - y0)), 1.0)
                            validation.update(
                                {
                                    "validation_mask": str(mask_path),
                                    "validation_mask_bbox_xyxy": mask_box,
                                    "bbox_iou_with_mask_bbox": _box_iou(bbox, mask_box),
                                    "mask_containment_in_bbox": overlap / max(mask_pixels, 1),
                                    "bbox_fill_by_mask": overlap / max(box_area, 1),
                                    "normalized_center_error": float(
                                        np.linalg.norm(center_box - center_mask) / diagonal
                                    ),
                                }
                            )
                            overlay = frame.copy()
                            overlay[mask] = np.rint(
                                0.70 * overlay[mask] + 0.30 * np.asarray([255, 0, 255])
                            ).astype(np.uint8)
                            frame = overlay
                            cv2.rectangle(
                                frame,
                                (mask_box[0], mask_box[1]),
                                (mask_box[2], mask_box[3]),
                                (0, 220, 255),
                                3,
                            )
                cv2.putText(
                    frame,
                    f"{object_id} mapped RGB bbox (green)",
                    (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (20, 235, 40),
                    2,
                    cv2.LINE_AA,
                )
                target = output / f"{object_id}_rgb_bbox_validation.jpg"
                cv2.imwrite(str(target), frame)
                validation["visualization"] = str(target.resolve())
                validation_records.append(validation)
                scale = min(900 / frame.shape[1], 620 / frame.shape[0])
                cards.append(cv2.resize(frame, None, fx=scale, fy=scale))
        finally:
            rgb_capture.release()

        rgb_reasoning["video"] = {
            "path": str(rgb_video),
            "num_frames": rgb_meta[0],
            "fps": rgb_meta[1],
            "width": rgb_meta[2],
            "height": rgb_meta[3],
            "view": "native_rgb",
        }
        rgb_reasoning["canonical_bbox_mapping"] = {
            "source_reasoning": str(source_reasoning),
            "source_view": "rectified_left",
            "target_view": "native_rgb",
            "method": "codex_synchronized_cross_view_visual_localization",
        }
        result_path = output / "contact_events_rgb.json"
        result_path.write_text(json.dumps(rgb_reasoning, indent=2) + "\n", encoding="utf-8")
        montage_path = output / "rgb_bbox_validation_montage.jpg"
        if cards:
            width = max(card.shape[1] for card in cards)
            padded = [
                cv2.copyMakeBorder(card, 0, 0, 0, width - card.shape[1], cv2.BORDER_CONSTANT)
                for card in cards
            ]
            cv2.imwrite(str(montage_path), np.vstack(padded))
        metrics = [item for item in validation_records if "bbox_iou_with_mask_bbox" in item]
        thresholds = {
            "minimum_bbox_iou_with_mask_bbox": 0.50,
            "minimum_mask_containment_in_bbox": 0.80,
            "maximum_normalized_center_error": 0.25,
        }
        for item in metrics:
            item["accuracy_check"] = (
                "pass"
                if item["bbox_iou_with_mask_bbox"]
                >= thresholds["minimum_bbox_iou_with_mask_bbox"]
                and item["mask_containment_in_bbox"]
                >= thresholds["minimum_mask_containment_in_bbox"]
                and item["normalized_center_error"]
                <= thresholds["maximum_normalized_center_error"]
                else "fail"
            )
        all_accuracy_checks_pass = bool(metrics) and all(
            item["accuracy_check"] == "pass" for item in metrics
        )
        report = {
            "schema_version": "ego-scene-hoi-cross-view-canonical-bbox-v2",
            "result": "PASS",
            "source_video": str(source_video),
            "rgb_video": str(rgb_video),
            "frame_synchronization": "strict_equal_frame_index_and_fps",
            "canonical_count": len(validation_records),
            "object_count": sum(item["registry_key"] == "objects" for item in evidence),
            "support_count": sum(
                item["registry_key"] == "support_objects" for item in evidence
            ),
            "hand_canonical_count": sum(item["registry_key"] == "hands" for item in evidence),
            "validation": validation_records,
            "accuracy_validation": {
                "reference": "independent native-RGB full-video SAM3 track mask",
                "thresholds": thresholds,
                "status": (
                    "pass"
                    if all_accuracy_checks_pass
                    else "not_available" if not metrics else "fail"
                ),
            },
            "aggregate_validation": (
                {
                    "count": len(metrics),
                    "median_bbox_iou_with_mask_bbox": float(
                        np.median([item["bbox_iou_with_mask_bbox"] for item in metrics])
                    ),
                    "median_mask_containment_in_bbox": float(
                        np.median([item["mask_containment_in_bbox"] for item in metrics])
                    ),
                    "median_normalized_center_error": float(
                        np.median([item["normalized_center_error"] for item in metrics])
                    ),
                }
                if metrics
                else {"count": 0}
            ),
            "artifacts": {
                "rgb_reasoning": str(result_path.resolve()),
                "montage": str(montage_path.resolve()),
                "evidence_manifest": str(evidence_path.resolve()),
            },
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path = output / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(report_path)
        return report_path


def main() -> None:
    CrossViewCanonicalBBoxMapper.from_worker_job().run()


if __name__ == "__main__":
    main()
