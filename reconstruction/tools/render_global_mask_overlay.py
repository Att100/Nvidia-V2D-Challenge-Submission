#!/usr/bin/env python3
"""Render a frame-aligned video from packed global SAM3 object masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COLORS_BGR = (
    (255, 120, 40),
    (40, 190, 255),
    (100, 220, 80),
    (220, 80, 220),
    (80, 80, 255),
    (255, 200, 80),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    manifest = json.loads(args.mask_manifest.read_text(encoding="utf-8"))
    objects: list[dict[str, object]] = []
    frame_count: int | None = None
    output_size: tuple[int, int] | None = None
    for index, record in enumerate(manifest.get("objects", [])):
        archive = np.load(Path(str(record["mask_file"])))
        width = int(archive["mask_width"])
        packed = np.asarray(archive["packed_masks"], dtype=np.uint8)
        masks = np.unpackbits(packed, axis=2)[:, :, :width].astype(bool, copy=False)
        source_indices = np.asarray(archive["source_indices"], dtype=np.int64)
        if frame_count is None:
            frame_count = len(masks)
            output_size = (width, masks.shape[1])
        elif len(masks) != frame_count or output_size != (width, masks.shape[1]):
            raise ValueError("All global mask archives must share frame count and image size")
        objects.append(
            {
                "name": str(record.get("object_name", record.get("global_object_id", index))),
                "masks": masks,
                "source_indices": source_indices,
                "color": COLORS_BGR[index % len(COLORS_BGR)],
            }
        )
    if frame_count is None or output_size is None:
        raise ValueError("Mask manifest has no objects")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {args.output}")

    try:
        for frame_id in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended before frame {frame_id}")
            frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
            overlay = frame.copy()
            labels: list[tuple[str, tuple[int, int, int]]] = []
            for obj in objects:
                source_indices = obj["source_indices"]
                if int(source_indices[frame_id]) != frame_id:
                    raise ValueError("Global masks are not aligned to sequential source frames")
                mask = obj["masks"][frame_id]
                if not np.any(mask):
                    continue
                color = obj["color"]
                overlay[mask] = color
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(frame, contours, -1, color, 2, cv2.LINE_AA)
                labels.append((str(obj["name"]), color))
            frame = cv2.addWeighted(overlay, args.alpha, frame, 1.0 - args.alpha, 0.0)
            for row, (label, color) in enumerate(labels):
                y = 24 + 24 * row
                cv2.putText(
                    frame,
                    label,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    print(args.output.resolve())


if __name__ == "__main__":
    main()
