#!/usr/bin/env python3
"""Validate the frozen Track-3 hand-off without loading GPU models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REQUIRED_PATHS = (
    "run.py",
    "tools/gradio_4d_pointcloud_viewer.py",
    "tools/view_stereo_geometry_gradio.py",
    "configs/dataset/rgb_stereo_gray.yaml",
    "configs/pipeline/v2.yaml",
    "docs/CHECKPOINTS.md",
    "scripts/download_checkpoints.sh",
    "thirdparty/FoundationStereo/pretrained_models/23-51-11/cfg.yaml",
    "environment/ego_scene_hoi.yaml",
    "environment/umetrack.yaml",
)

CHECKPOINT_PATHS = (
    "thirdparty/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth",
    "thirdparty/FoundationPose/weights/2023-10-28-18-33-37/config.yml",
    "thirdparty/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth",
    "thirdparty/FoundationPose/weights/2024-01-11-20-02-45/config.yml",
    "thirdparty/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth",
    "thirdparty/HACO_RELEASE/release_checkpoint/haco_neurips_vit_b_checkpoint.ckpt",
    "thirdparty/UmeTrack/pretrained_models/pretrained_weights.torch",
    "thirdparty/MANO/mano_data/MANO_LEFT.pkl",
    "thirdparty/MANO/mano_data/MANO_RIGHT.pkl",
    "thirdparty/MANO/mano_data/mano_mean_params.npz",
    "thirdparty/HACO_RELEASE/data/base_data/human_models/mano/MANO_LEFT.pkl",
    "thirdparty/HACO_RELEASE/data/base_data/human_models/mano/MANO_RIGHT.pkl",
    "thirdparty/HACO_RELEASE/data/base_data/human_models/mano/V_regressor_84.npy",
    "thirdparty/HACO_RELEASE/data/base_data/human_models/mano/V_regressor_336.npy",
    "thirdparty/sam-3d-objects/checkpoints/hf/ss_generator.ckpt",
    "thirdparty/sam-3d-objects/checkpoints/hf/pipeline.yaml",
    "thirdparty/sam-3d-objects/checkpoints/hf/slat_generator.ckpt",
    "thirdparty/sam-3d-objects/checkpoints/hf/ss_decoder.ckpt",
    "thirdparty/sam-3d-objects/checkpoints/hf/slat_decoder_gs.ckpt",
    "thirdparty/sam-3d-objects/checkpoints/hf/slat_decoder_gs_4.ckpt",
    "thirdparty/sam-3d-objects/checkpoints/hf/slat_decoder_mesh.ckpt",
    "thirdparty/sam-3d-objects/checkpoints/hf/slat_encoder.ckpt",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--require-checkpoints",
        action="store_true",
        help="also require every manually restored checkpoint",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []

    for relative in REQUIRED_PATHS:
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required file: {relative}")

    if args.require_checkpoints:
        for relative in CHECKPOINT_PATHS:
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing or empty checkpoint: {relative}")

    broken_links = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink() and not path.exists()]
    errors.extend(f"broken symlink: {path}" for path in broken_links)

    forbidden_workspace_references = (
        "/root/" + "workspace/ego-scene-hoi",
        "agentic-human-" + "interaction-gen",
    )
    for path in root.rglob("*"):
        if not path.is_file() or any(part == "__pycache__" for part in path.parts):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for reference in forbidden_workspace_references:
            if reference in content:
                errors.append(
                    "non-portable source-workspace reference: "
                    f"{path.relative_to(root)} -> {reference}"
                )

    if errors:
        print("Bundle validation FAILED", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    checkpoint_status = "present" if args.require_checkpoints else "not required"
    print(
        "Bundle validation PASS: source/configuration complete, "
        f"checkpoints {checkpoint_status}, no broken links."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
