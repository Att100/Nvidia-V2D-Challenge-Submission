#!/usr/bin/env python3
"""Render global object-track state, masks, and boxes beside an RGB video."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np


COLORS_BGR = (
    (65, 158, 255),
    (80, 210, 120),
    (235, 120, 75),
    (210, 100, 220),
    (80, 215, 225),
    (225, 180, 75),
    (125, 105, 245),
    (190, 190, 80),
)
VISIBLE_BGR = (72, 220, 95)
INVISIBLE_BGR = (75, 75, 235)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=390)
    parser.add_argument("--mask-alpha", type=float, default=0.42)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--preview-frames", type=int, nargs="*", default=[])
    return parser.parse_args()


class PackedMaskTrack:
    """One persistent object mask sequence stored as packed bits."""

    def __init__(self, item: dict[str, Any], manifest_dir: Path, index: int) -> None:
        path = Path(item["mask_file"])
        if not path.is_absolute():
            path = manifest_dir / path
        archive = np.load(path)
        self.global_object_id = str(item["global_object_id"])
        self.object_name = str(item["object_name"])
        self.display_number = index + 1
        self.color = COLORS_BGR[index % len(COLORS_BGR)]
        self.packed_masks = np.asarray(archive["packed_masks"], dtype=np.uint8)
        self.mask_width = int(archive["mask_width"])
        self.visible = np.asarray(archive["visible"], dtype=bool)
        self.source_indices = np.asarray(archive["source_indices"], dtype=np.int32)

    def mask(self, local_frame: int, height: int, width: int) -> np.ndarray | None:
        if local_frame >= len(self.visible) or not self.visible[local_frame]:
            return None
        mask = np.unpackbits(self.packed_masks[local_frame], axis=1)[
            :, : self.mask_width
        ].astype(bool)
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return mask if mask.any() else None


class H264Writer:
    """Small PyAV H.264 writer retaining the source spatial resolution."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream(
            "libx264", rate=Fraction(fps).limit_denominator(100_000)
        )
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(crf), "preset": preset}

    def write(self, bgr: np.ndarray) -> None:
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def draw_mask_and_box(
    frame: np.ndarray,
    mask: np.ndarray,
    track: PackedMaskTrack,
    mask_alpha: float,
) -> None:
    """Alpha blend one object and draw its numbered bounding box."""

    color = np.asarray(track.color, dtype=np.float32)
    pixels = frame[mask].astype(np.float32)
    frame[mask] = np.clip((1.0 - mask_alpha) * pixels + mask_alpha * color, 0, 255).astype(
        np.uint8
    )
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    thickness = max(2, round(min(frame.shape[:2]) / 500))
    cv2.rectangle(frame, (x0, y0), (x1, y1), track.color, thickness, cv2.LINE_AA)
    label = f"{track.display_number:02d}"
    font_scale = max(0.65, min(frame.shape[:2]) / 1200)
    text_size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    label_y0 = max(0, y0 - text_size[1] - baseline - 8)
    label_x1 = min(frame.shape[1] - 1, x0 + text_size[0] + 12)
    cv2.rectangle(frame, (x0, label_y0), (label_x1, y0), track.color, -1)
    luminance = 0.114 * track.color[0] + 0.587 * track.color[1] + 0.299 * track.color[2]
    text_color = (20, 20, 20) if luminance > 150 else (255, 255, 255)
    cv2.putText(
        frame,
        label,
        (x0 + 6, max(text_size[1] + 1, y0 - baseline - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(
    height: int,
    width: int,
    tracks: list[PackedMaskTrack],
    visible: list[bool],
    source_frame: int,
) -> np.ndarray:
    """Draw the ordered global-track list with per-frame visibility state."""

    panel = np.full((height, width, 3), (24, 26, 30), dtype=np.uint8)
    margin = 25
    title_scale = max(0.72, min(1.0, width / 420))
    cv2.putText(
        panel,
        "UPDATED GLOBAL TRACKS",
        (margin, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        title_scale,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"Frame {source_frame:06d}",
        (margin, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (155, 160, 170),
        1,
        cv2.LINE_AA,
    )
    cv2.line(panel, (margin, 94), (width - margin, 94), (70, 73, 80), 1)
    line_height = max(42, min(58, (height - 130) // max(1, len(tracks))))
    y = 130
    for track, is_visible in zip(tracks, visible):
        state_color = VISIBLE_BGR if is_visible else INVISIBLE_BGR
        cv2.circle(panel, (margin + 7, y - 7), 6, state_color, -1, cv2.LINE_AA)
        label = f"{track.display_number:02d}  {track.object_name}"
        cv2.putText(
            panel,
            label,
            (margin + 24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.67,
            state_color,
            2,
            cv2.LINE_AA,
        )
        y += line_height
    cv2.putText(
        panel,
        "GREEN: visible",
        (margin, height - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        VISIBLE_BGR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "RED: not visible",
        (margin, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        INVISIBLE_BGR,
        1,
        cv2.LINE_AA,
    )
    return panel


def main(args: argparse.Namespace) -> None:
    manifest_path = args.mask_manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tracks = [
        PackedMaskTrack(item, manifest_path.parent, index)
        for index, item in enumerate(manifest["objects"])
    ]
    if not tracks:
        raise ValueError("Global mask manifest contains no objects")
    source_indices = tracks[0].source_indices
    if any(not np.array_equal(track.source_indices, source_indices) for track in tracks[1:]):
        raise ValueError("Global tracks do not share one source-frame timeline")

    capture = cv2.VideoCapture(str(args.video.expanduser().resolve()))
    if not capture.isOpened():
        raise OSError(f"Cannot open {args.video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count != len(source_indices):
        raise ValueError(
            f"Video has {frame_count} frames but masks have {len(source_indices)}"
        )
    output = args.output.expanduser().resolve()
    writer = H264Writer(
        output, width + args.panel_width, height, fps, args.crf, args.preset
    )
    preview_set = set(args.preview_frames)
    preview_dir = output.parent / f"{output.stem}_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    try:
        local_frame = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            visible: list[bool] = []
            for track in tracks:
                mask = track.mask(local_frame, height, width)
                is_visible = mask is not None
                visible.append(is_visible)
                if is_visible:
                    draw_mask_and_box(frame, mask, track, args.mask_alpha)
            source_frame = int(source_indices[local_frame])
            panel = draw_panel(
                height, args.panel_width, tracks, visible, source_frame
            )
            composed = np.concatenate((panel, frame), axis=1)
            writer.write(composed)
            if source_frame in preview_set:
                cv2.imwrite(str(preview_dir / f"frame_{source_frame:06d}.jpg"), composed)
            local_frame += 1
            if local_frame == 1 or local_frame % 300 == 0 or local_frame == frame_count:
                print(f"[global-track-overlay] {local_frame}/{frame_count}", flush=True)
    finally:
        capture.release()
        writer.close()
    if local_frame != frame_count:
        raise RuntimeError(f"Rendered {local_frame}/{frame_count} frames")
    output_manifest = {
        "schema_version": "global-track-overlay-v1",
        "video": str(args.video.expanduser().resolve()),
        "mask_manifest": str(manifest_path),
        "output": str(output),
        "frame_count": frame_count,
        "fps": fps,
        "video_size": [height, width],
        "output_size": [height, width + args.panel_width],
        "mask_alpha": args.mask_alpha,
        "tracks": [
            {
                "display_number": track.display_number,
                "global_object_id": track.global_object_id,
                "object_name": track.object_name,
                "color_bgr": list(track.color),
                "visible_frame_count": int(track.visible.sum()),
            }
            for track in tracks
        ],
        "preview_directory": str(preview_dir),
    }
    manifest_output = output.with_suffix(".json")
    manifest_output.write_text(json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    ARGS = parse_args()
    if not 0.0 <= ARGS.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be in [0,1]")
    main(ARGS)
