"""Occlusion-aware refinement of persistent rigid-object masks.

SAM video tracking can absorb a hand or a second rigid object when their
silhouettes touch.  Downstream RGB-D pose estimation must only receive pixels
that are unambiguously owned by the target object.  This module keeps the raw
SAM tracks intact and writes a refined, pairwise-disjoint mask set.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from pipeline.config import get_config
from pipeline.utils import (
    PipelineRequest,
    StageResult,
    StageStatus,
    all_artifacts_exist,
    file_sha256,
    read_json,
    value_sha256,
    write_json,
)


@dataclass(frozen=True)
class ObjectMaskRefinementConfig:
    """Controls conservative object-mask visibility and ownership filtering."""

    minimum_area: int = 64
    containment_threshold: float = 0.50


def _resolve_mask_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    candidate = manifest_path.parent / path.name
    if candidate.is_file():
        return candidate
    if not path.is_absolute():
        candidate = manifest_path.parent / path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def _unpack_mask_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        width = int(np.asarray(archive["mask_width"]).item())
        masks = np.unpackbits(
            np.asarray(archive["packed_masks"], dtype=np.uint8), axis=2
        )[..., :width].astype(bool)
        source = np.asarray(
            archive["source_indices"]
            if "source_indices" in archive.files
            else np.arange(len(masks)),
            dtype=np.int32,
        )
        scores = np.asarray(
            archive["scores"]
            if "scores" in archive.files
            else np.ones(len(masks), dtype=np.float32),
            dtype=np.float32,
        )
        visible = np.asarray(
            archive["visible"]
            if "visible" in archive.files
            else masks.any(axis=(1, 2)),
            dtype=bool,
        )
    return masks, scores, visible, source


def _unpack_plain_masks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        width = int(np.asarray(archive["mask_width"]).item())
        masks = np.unpackbits(
            np.asarray(archive["packed_masks"], dtype=np.uint8), axis=2
        )[..., :width].astype(bool)
        source = np.asarray(
            archive["source_indices"]
            if "source_indices" in archive.files
            else np.arange(len(masks)),
            dtype=np.int32,
        )
    return masks, source


def _resize_sequence(masks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if masks.shape[1:] == shape:
        return masks
    return np.stack(
        [
            cv2.resize(
                mask.astype(np.uint8),
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for mask in masks
        ]
    )


def _manifest_union(
    manifest_path: Path,
    source_indices: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    """Load and union all masks from a hand/non-rigid manifest."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = np.zeros((len(source_indices), *shape), dtype=bool)
    target_by_source = {int(value): index for index, value in enumerate(source_indices)}
    for record in manifest.get("objects", manifest.get("categories", [])):
        path = _resolve_mask_path(manifest_path, str(record["mask_file"]))
        masks, _, _, source = _unpack_mask_archive(path)
        masks = _resize_sequence(masks, shape)
        for local_index, source_frame in enumerate(source):
            target = target_by_source.get(int(source_frame))
            if target is not None:
                result[target] |= masks[local_index]
    return result


def _manifest_mask_files(manifest_path: Path) -> list[Path]:
    """Resolve mask archives so cache invalidation covers array contents."""

    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        _resolve_mask_path(manifest_path, str(record["mask_file"]))
        for record in manifest.get("objects", manifest.get("categories", []))
    ]


def refine_object_masks(
    raw_masks: np.ndarray,
    nonrigid_masks: np.ndarray,
    *,
    containment_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint masks plus pixels removed by non-rigid/rigid occluders.

    Highly nested masks are resolved in favour of the more specific instance
    (the mask with the larger overlap fraction, usually the smaller object).
    Ambiguous partial intersections are removed from both objects because the
    visible owner cannot be inferred reliably from 2D masks alone.
    """

    if raw_masks.ndim != 4:
        raise ValueError(f"Expected [object, frame, height, width], got {raw_masks.shape}")
    if nonrigid_masks.shape != raw_masks.shape[1:]:
        raise ValueError(
            f"Non-rigid masks {nonrigid_masks.shape} != object timeline {raw_masks.shape[1:]}"
        )
    refined = raw_masks & ~nonrigid_masks[None]
    removed_nonrigid = raw_masks & nonrigid_masks[None]
    before_rigid = refined.copy()
    object_count, frame_count = refined.shape[:2]
    for frame_id in range(frame_count):
        for first in range(object_count):
            for second in range(first + 1, object_count):
                overlap = refined[first, frame_id] & refined[second, frame_id]
                overlap_area = int(overlap.sum())
                if not overlap_area:
                    continue
                first_area = int(refined[first, frame_id].sum())
                second_area = int(refined[second, frame_id].sum())
                first_fraction = overlap_area / max(first_area, 1)
                second_fraction = overlap_area / max(second_area, 1)
                if max(first_fraction, second_fraction) >= containment_threshold:
                    if first_fraction > second_fraction:
                        refined[second, frame_id, overlap] = False
                    elif second_fraction > first_fraction:
                        refined[first, frame_id, overlap] = False
                    elif first_area <= second_area:
                        refined[second, frame_id, overlap] = False
                    else:
                        refined[first, frame_id, overlap] = False
                else:
                    refined[first, frame_id, overlap] = False
                    refined[second, frame_id, overlap] = False
    removed_rigid = before_rigid & ~refined
    if np.any(refined.sum(axis=0) > 1):
        raise AssertionError("Refined rigid-object masks are not pairwise disjoint")
    return refined, removed_nonrigid, removed_rigid


class ObjectMaskRefinementModule:
    """Create the canonical downstream mask set from raw SAM3 tracks."""

    name = "object_mask_refinement"

    def __init__(self, config: ObjectMaskRefinementConfig | None = None) -> None:
        self.config = config or ObjectMaskRefinementConfig()

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        root = request.run_root / "tracks/sam3/refined_masks"
        return root / "manifest.json", root / "object_masks.npz", root / "report.json"

    def _inputs(self, request: PipelineRequest) -> tuple[Path, Path | None, Path | None]:
        raw = request.run_root / "tracks/sam3/global_masks/manifest.json"
        stereo = (
            str(get_config(request.config, "dataset.input_mode", ""))
            == "rgb_plus_stereo_grayscale"
        )
        if stereo:
            return raw, None, request.run_root / "dynamic_mask/body_masks.npz"
        return raw, request.run_root / "tracks/sam3_hands_full/global_masks/manifest.json", None

    def run(self, request: PipelineRequest) -> StageResult:
        artifacts = self.expected_artifacts(request)
        raw_manifest_path, nonrigid_manifest, nonrigid_archive = self._inputs(request)
        inputs = [raw_manifest_path]
        inputs.extend(
            path for path in (nonrigid_manifest, nonrigid_archive) if path is not None
        )
        missing = [str(path) for path in inputs if not path.is_file()]
        if missing and not request.dry_run:
            raise FileNotFoundError(f"Missing object-mask refinement inputs: {missing}")
        dependency_files = list(inputs)
        dependency_files.extend(_manifest_mask_files(raw_manifest_path))
        if nonrigid_manifest is not None:
            dependency_files.extend(_manifest_mask_files(nonrigid_manifest))
        fingerprint = value_sha256(
            {
                "config": self.config.__dict__,
                "inputs": {
                    str(path): file_sha256(path) if path.is_file() else None
                    for path in dependency_files
                },
            }
        )
        if request.resume and not request.force and all_artifacts_exist(artifacts):
            if read_json(artifacts[2]).get("fingerprint") == fingerprint:
                return StageResult(self.name, StageStatus.REUSED, artifacts)
        if request.dry_run:
            return StageResult(self.name, StageStatus.SKIPPED, artifacts)

        started = time.perf_counter()
        raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
        records = list(raw_manifest.get("objects", []))
        if not records:
            raise RuntimeError("Raw SAM3 manifest contains no object masks")
        raw_sequences: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        source_indices: np.ndarray | None = None
        for record in records:
            path = _resolve_mask_path(raw_manifest_path, str(record["mask_file"]))
            masks, item_scores, _, item_source = _unpack_mask_archive(path)
            if source_indices is None:
                source_indices = item_source
            elif not np.array_equal(source_indices, item_source):
                raise ValueError("Rigid-object masks do not share one source timeline")
            raw_sequences.append(masks)
            scores.append(item_scores)
        assert source_indices is not None
        raw_masks = np.stack(raw_sequences)
        shape = raw_masks.shape[2:]

        if nonrigid_manifest is not None:
            nonrigid = _manifest_union(nonrigid_manifest, source_indices, shape)
            nonrigid_source = str(nonrigid_manifest.resolve())
        else:
            assert nonrigid_archive is not None
            nonrigid, body_source = _unpack_plain_masks(nonrigid_archive)
            if not np.array_equal(source_indices, body_source):
                raise ValueError("Body and rigid-object masks use different timelines")
            nonrigid = _resize_sequence(nonrigid, shape)
            nonrigid_source = str(nonrigid_archive.resolve())

        refined, removed_nonrigid, removed_rigid = refine_object_masks(
            raw_masks,
            nonrigid,
            containment_threshold=self.config.containment_threshold,
        )
        output = request.run_root / "tracks/sam3/refined_masks"
        output.mkdir(parents=True, exist_ok=True)
        summaries: list[dict[str, Any]] = []
        per_object_report: list[dict[str, Any]] = []
        score_array = np.stack(scores)
        for index, record in enumerate(records):
            object_id = str(record["global_object_id"])
            visible = refined[index].sum(axis=(1, 2)) >= self.config.minimum_area
            refined[index, ~visible] = False
            item_scores = score_array[index].copy()
            item_scores[~visible] = -np.inf
            path = output / f"{object_id.replace('/', '_')}.npz"
            np.savez_compressed(
                path,
                packed_masks=np.packbits(refined[index], axis=2),
                mask_width=np.asarray(shape[1], dtype=np.int32),
                scores=item_scores.astype(np.float32),
                visible=visible,
                source_indices=source_indices,
            )
            visible_indices = np.flatnonzero(visible)
            summaries.append(
                {
                    "global_object_id": object_id,
                    "object_name": str(record["object_name"]),
                    "mask_file": str(path.resolve()),
                    "visible_frame_count": int(visible.sum()),
                    "first_visible_source_frame": (
                        int(source_indices[visible_indices[0]]) if len(visible_indices) else None
                    ),
                    "last_visible_source_frame": (
                        int(source_indices[visible_indices[-1]]) if len(visible_indices) else None
                    ),
                }
            )
            before = raw_masks[index].sum(axis=(1, 2))
            after = refined[index].sum(axis=(1, 2))
            per_object_report.append(
                {
                    "global_object_id": object_id,
                    "object_name": str(record["object_name"]),
                    "raw_pixels": int(before.sum()),
                    "refined_pixels": int(after.sum()),
                    "removed_nonrigid_pixels": int(removed_nonrigid[index].sum()),
                    "removed_other_object_pixels": int(removed_rigid[index].sum()),
                    "frames_dropped": int(((before > 0) & ~visible).sum()),
                    "retained_fraction": float(after.sum() / max(int(before.sum()), 1)),
                }
            )

        manifest_path = output / "manifest.json"
        write_json(
            manifest_path,
            {
                "schema_version": "ego-scene-hoi-refined-object-masks-v1",
                "source_manifest": str(raw_manifest_path.resolve()),
                "nonrigid_source": nonrigid_source,
                "ownership_policy": "nested-specific-instance; ambiguous-overlap-excluded",
                "pairwise_disjoint": True,
                "minimum_mask_area_pixels": self.config.minimum_area,
                "containment_threshold": self.config.containment_threshold,
                "objects": summaries,
            },
        )
        aggregate_path = output / "object_masks.npz"
        np.savez_compressed(
            aggregate_path,
            packed_masks=np.packbits(refined, axis=3),
            mask_width=np.asarray(shape[1], dtype=np.int32),
            object_ids=np.asarray([str(item["global_object_id"]) for item in records]),
            object_names=np.asarray([str(item["object_name"]) for item in records]),
            source_indices=source_indices,
        )
        report_path = output / "report.json"
        write_json(
            report_path,
            {
                "schema_version": "ego-scene-hoi-object-mask-refinement-report-v1",
                "result": "PASS",
                "fingerprint": fingerprint,
                "raw_manifest": str(raw_manifest_path.resolve()),
                "nonrigid_source": nonrigid_source,
                "object_count": len(records),
                "frame_count": len(source_indices),
                "pairwise_overlap_pixels_after": int(
                    np.maximum(refined.sum(axis=0).astype(np.int16) - 1, 0).sum()
                ),
                "objects": per_object_report,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        return StageResult(
            self.name,
            StageStatus.COMPLETED,
            artifacts,
            elapsed_seconds=time.perf_counter() - started,
            metrics={"object_count": len(records)},
        )

