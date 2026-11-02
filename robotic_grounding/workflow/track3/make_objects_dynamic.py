#!/usr/bin/env python3
"""Remove legacy kinematic-object metadata without changing policy tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_dynamic(source: Path, destination: Path) -> dict[str, object]:
    from safetensors import safe_open
    from safetensors.numpy import save_file

    with safe_open(source, framework="numpy") as checkpoint:
        metadata = dict(checkpoint.metadata() or {})
        tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}

    key = "user.resolved_config_json"
    if key not in metadata:
        raise ValueError(f"checkpoint lacks {key}: {source}")
    config = json.loads(metadata[key])
    scene = config.get("scene")
    if not isinstance(scene, dict):
        raise ValueError(f"checkpoint resolved config lacks scene mapping: {source}")
    original = list(scene.get("kinematic_object_names", []))
    scene["kinematic_object_names"] = []
    metadata[key] = json.dumps(config, sort_keys=True, separators=(",", ":"))
    metadata["user.evaluation_scene_override"] = "all_scored_objects_dynamic"
    metadata["user.evaluation_source_checkpoint_sha256"] = sha256(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, destination, metadata=metadata)
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "checkpoint": str(destination),
        "checkpoint_sha256": sha256(destination),
        "original_kinematic_object_names": original,
        "evaluation_kinematic_object_names": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    record = make_dynamic(args.source.expanduser().resolve(), args.destination.expanduser())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

