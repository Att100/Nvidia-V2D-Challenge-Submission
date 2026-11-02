"""Worker entry point for coarse-track-gated independent SAM3 hand masks."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.modules.sam3.hand_frame_segmenter import (
    HandFrameSegmentationConfig,
    SAM3HandFrameSegmenter,
)
from pipeline.utils import parse_worker_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hand-motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, required=True)
    parser.add_argument("--min-mask-area", type=int, required=True)
    parser.add_argument("--box-expand-ratio", type=float, required=True)
    parser.add_argument("--duplicate-box-iou", type=float, required=True)
    parser.add_argument("--maximum-flip-run", type=int, required=True)
    parser.add_argument("--identity-gap", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--schema-version", required=True)
    return parse_worker_args(parser)


def main() -> None:
    args = parse_args()
    config = HandFrameSegmentationConfig(
        confidence_threshold=args.confidence_threshold,
        minimum_mask_area=args.min_mask_area,
        box_expand_ratio=args.box_expand_ratio,
        duplicate_box_iou=args.duplicate_box_iou,
        maximum_flip_run=args.maximum_flip_run,
        identity_gap=args.identity_gap,
        device=args.device,
        checkpoint=args.checkpoint,
        schema_version=args.schema_version,
    )
    result = SAM3HandFrameSegmenter(config, args.repository_root).run(
        video_path=args.video,
        hand_motion_path=args.hand_motion,
        output_dir=args.output_dir,
    )
    print(result)


if __name__ == "__main__":
    main()
