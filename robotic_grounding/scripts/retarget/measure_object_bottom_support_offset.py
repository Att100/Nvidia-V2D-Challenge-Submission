#!/usr/bin/env python3
"""Measure one object's vertical support offset over a selected frame range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--support-height", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    table = pq.read_table(args.parquet.resolve())
    if table.num_rows != 1:
        raise ValueError(f"expected one row, got {table.num_rows}")
    row = table.to_pylist()[0]
    names = [str(value) for value in row["object_body_names"]]
    if args.object_name not in names:
        raise ValueError(f"{args.object_name!r} not found in {names}")
    index = names.index(args.object_name)
    positions = np.asarray(row["object_body_position"], dtype=np.float64)[:, index]
    wxyz = np.asarray(row["object_body_wxyz"], dtype=np.float64)[:, index]
    frame_count = len(positions)
    if not (0 <= args.start_frame <= args.end_frame < frame_count):
        raise ValueError("invalid frame range")

    loaded = trimesh.load(row["object_mesh_paths"][index], force="scene", process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("object mesh is not triangular")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    matrices = Rotation.from_quat(wxyz, scalar_first=True).as_matrix()
    selected = np.arange(args.start_frame, args.end_frame + 1)
    bottom = np.empty(len(selected), dtype=np.float64)
    for output_index, frame in enumerate(selected):
        world = vertices @ matrices[frame].T + positions[frame]
        bottom[output_index] = float(world[:, 2].min())
    residual = bottom - float(args.support_height)
    correction = -float(np.median(residual))
    report = {
        "schema_version": "object-bottom-support-offset-v1",
        "parquet": str(args.parquet.resolve()),
        "object_name": args.object_name,
        "object_index": index,
        "frame_range_inclusive": [args.start_frame, args.end_frame],
        "support_height_m": float(args.support_height),
        "bottom_z_m": {
            "median": float(np.median(bottom)),
            "minimum": float(np.min(bottom)),
            "maximum": float(np.max(bottom)),
        },
        "support_residual_mm": {
            "median": float(1000.0 * np.median(residual)),
            "p05": float(1000.0 * np.percentile(residual, 5)),
            "p95": float(1000.0 * np.percentile(residual, 95)),
        },
        "recommended_translation_m": [0.0, 0.0, correction],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

