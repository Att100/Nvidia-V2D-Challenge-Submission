#!/usr/bin/env python3
"""Create an evenly sampled video contact sheet for visual QA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError(f"cannot read video metadata: {args.video}")
    indices = np.linspace(0, frame_count - 1, min(args.samples, frame_count)).round().astype(int)
    tiles = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {index}")
        height = round(frame.shape[0] * args.width / frame.shape[1])
        frame = cv2.resize(frame, (args.width, height), interpolation=cv2.INTER_AREA)
        cv2.putText(
            frame,
            f"frame {index:04d}  t={index / fps:.2f}s",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"frame {index:04d}  t={index / fps:.2f}s",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        tiles.append(frame)
    capture.release()
    columns = args.columns
    rows = math.ceil(len(tiles) / columns)
    blank = np.zeros_like(tiles[0])
    tiles.extend([blank] * (rows * columns - len(tiles)))
    sheet = np.vstack([np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise RuntimeError(f"failed to write {args.output}")
    print(args.output)


if __name__ == "__main__":
    main()
