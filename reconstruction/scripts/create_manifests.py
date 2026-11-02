#!/usr/bin/env python3
"""Create compact machine-readable model and result inventories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files() -> list[Path]:
    roots = (
        ROOT / "thirdparty/FoundationStereo/pretrained_models",
        ROOT / "thirdparty/FoundationPose/weights",
        ROOT / "thirdparty/HACO_RELEASE/release_checkpoint",
        ROOT / "thirdparty/HACO_RELEASE/data/base_data/human_models/mano",
        ROOT / "thirdparty/UmeTrack/pretrained_models",
        ROOT / "thirdparty/MANO/mano_data",
        ROOT / "thirdparty/sam-3d-objects/checkpoints/hf",
    )
    suffixes = {
        ".pth", ".pt", ".torch", ".ckpt", ".pkl", ".onnx", ".safetensors", ".bin"
    }
    files: list[Path] = []
    for directory in roots:
        for path in directory.rglob("*"):
            if path.is_file() and path.stat().st_size > 0 and path.suffix.lower() in suffixes:
                files.append(path)
    return sorted(files)


def model_manifest() -> dict[str, object]:
    records = []
    for path in checkpoint_files():
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema_version": "track3-model-inventory-v1",
        "checkpoints": records,
        "download_guide": "docs/CHECKPOINTS.md",
        "download_command": "bash scripts/download_checkpoints.sh all",
        "validation_command": "python scripts/validate_bundle.py --require-checkpoints",
        "automatic_downloads": [
            "facebook/sam3",
            "rootlocalghost/FLUX.2-klein-9B-FP8",
            "Ruicheng/moge-vitl",
            "AnyCalib anycalib_gen",
            "VIPE DROID-SLAM",
        ],
        "external_services": [
            {
                "name": "Codex CLI",
                "model": "gpt-6-astra",
                "reasoning_effort": "high",
                "local_weights": False,
            }
        ],
    }


def main() -> None:
    destination = ROOT / "manifests"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "models.json").write_text(json.dumps(model_manifest(), indent=2) + "\n")
    print(f"Wrote manifests to {destination}")


if __name__ == "__main__":
    main()
