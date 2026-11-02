#!/usr/bin/env python3
"""Consolidate window-local SAM3 tracklets into persistent per-object masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pipeline.utils import parse_worker_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--camera-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compat-tracking-dir",
        type=Path,
        required=True,
        help="Write reference-pipeline-compatible tracking_arrays.npz and instances.json.",
    )
    parser.add_argument(
        "--merge-by-canonical-name",
        action="store_true",
        help=(
            "Merge disconnected global groups with the same canonical name. "
            "Use this only for persistent semantic identities such as left/right hand."
        ),
    )
    parser.add_argument("--min-mask-area", type=int, default=0)
    parser.add_argument("--min-mask-area-fraction", type=float, default=0.0)
    parser.add_argument("--schema-version", required=True)
    return parse_worker_args(parser)


def unpack(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(path)
    width = int(archive["mask_width"])
    masks = np.unpackbits(archive["packed_masks"], axis=2)[:, :, :width].astype(bool)
    return masks, np.asarray(archive["scores"], dtype=np.float32), np.asarray(
        archive["source_frames"], dtype=np.int32
    )


def main() -> None:
    args = parse_args()
    tracks_path = args.tracks.expanduser().resolve()
    tracks_dir = tracks_path.parent
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    camera = np.load(args.camera_data.expanduser().resolve())
    source_indices = np.asarray(camera["source_indices"], dtype=np.int32)
    source_to_local = {int(value): index for index, value in enumerate(source_indices)}
    metadata = {item["tracklet_id"]: item for item in tracks["tracklets"]}
    summary = []
    consolidated: list[tuple[int, str, np.ndarray, np.ndarray]] = []
    global_tracks = tracks["global_object_tracks"]
    if args.merge_by_canonical_name:
        grouped: dict[str, list[dict]] = {}
        for item in global_tracks:
            grouped.setdefault(str(item["canonical_object_name"]), []).append(item)
        global_tracks = [
            {
                "global_object_id": name.lower().replace(" ", "_"),
                "canonical_object_name": name,
                "tracklet_ids": [
                    tracklet_id
                    for item in items
                    for tracklet_id in item["tracklet_ids"]
                ],
            }
            for name, items in grouped.items()
        ]
    for item in global_tracks:
        object_id = str(item["global_object_id"])
        masks_out = scores_out = None
        visible = np.zeros(len(source_indices), dtype=bool)
        for tracklet_id in item["tracklet_ids"]:
            meta = metadata[tracklet_id]
            path = Path(meta["mask_storage"])
            if not path.is_absolute():
                path = tracks_dir / path
            masks, scores, frames = unpack(path)
            if masks_out is None:
                masks_out = np.zeros((len(source_indices), *masks.shape[1:]), dtype=bool)
                scores_out = np.full(len(source_indices), -np.inf, dtype=np.float32)
            minimum_area = max(
                int(args.min_mask_area),
                int(np.ceil(args.min_mask_area_fraction * masks.shape[1] * masks.shape[2])),
            )
            for mask, score, source_frame in zip(masks, scores, frames):
                local = source_to_local.get(int(source_frame))
                if (
                    local is None
                    or not mask.any()
                    or int(mask.sum()) < minimum_area
                    or not np.isfinite(score)
                ):
                    continue
                if score > scores_out[local]:
                    masks_out[local] = mask
                    scores_out[local] = score
                    visible[local] = True
        if masks_out is None or scores_out is None or not visible.any():
            continue
        safe_id = object_id.replace("/", "_")
        path = output_dir / f"{safe_id}.npz"
        np.savez_compressed(
            path,
            packed_masks=np.packbits(masks_out, axis=2),
            mask_width=np.asarray(masks_out.shape[2], dtype=np.int32),
            scores=scores_out,
            visible=visible,
            source_indices=source_indices,
        )
        summary.append(
            {
                "global_object_id": object_id,
                "object_name": item["canonical_object_name"],
                "mask_file": str(path),
                "visible_frame_count": int(visible.sum()),
                "first_visible_source_frame": int(source_indices[np.flatnonzero(visible)[0]]),
                "last_visible_source_frame": int(source_indices[np.flatnonzero(visible)[-1]]),
            }
        )
        consolidated.append((len(consolidated) + 1, item["canonical_object_name"], masks_out, scores_out))
    manifest = {
        "schema_version": args.schema_version,
        "source_tracks": str(tracks_path),
        "camera_data": str(args.camera_data.expanduser().resolve()),
        "objects": summary,
        "merge_by_canonical_name": args.merge_by_canonical_name,
        "minimum_mask_area_pixels": args.min_mask_area,
        "minimum_mask_area_fraction": args.min_mask_area_fraction,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    compat = args.compat_tracking_dir.expanduser().resolve()
    compat.mkdir(parents=True, exist_ok=True)
    if consolidated:
        height, width = consolidated[0][2].shape[1:]
        instance_map = np.zeros((len(source_indices), height, width), dtype=np.uint16)
        winning_score = np.full((len(source_indices), height, width), -np.inf, dtype=np.float32)
        labels = {}
        for track_id, label, object_masks, object_scores in consolidated:
            labels[str(track_id)] = label
            for frame_id in np.flatnonzero(object_masks.any(axis=(1, 2))):
                candidate = object_masks[frame_id] & (
                    object_scores[frame_id] > winning_score[frame_id]
                )
                instance_map[frame_id][candidate] = track_id
                winning_score[frame_id][candidate] = object_scores[frame_id]
        np.savez_compressed(
            compat / "tracking_arrays.npz",
            instance_map=instance_map,
            source_indices=source_indices,
        )
        instance_dir = compat / "instance_map"
        instance_dir.mkdir(parents=True, exist_ok=True)
        for frame_id, frame_map in enumerate(instance_map):
            if not cv2.imwrite(str(instance_dir / f"{frame_id:06d}.png"), frame_map):
                raise OSError(f"Could not write instance map {frame_id}")
        (compat / "instances.json").write_text(
            json.dumps(
                {
                    "format": "ego-scene-hoi-reference-tracking-compat-v1",
                    "track_labels": labels,
                    "source_global_masks": str(path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(path)


if __name__ == "__main__":
    main()
