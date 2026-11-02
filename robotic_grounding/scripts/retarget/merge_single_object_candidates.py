#!/usr/bin/env python3
"""Merge independently selected rigid-object candidates into one reference.

Each candidate must contain exactly one object whose name matches one body in
the source multi-object parquet.  Object pose tracks, URDF paths, mesh paths,
and mesh radii are replaced per body.  Robot, hand, and contact columns are
preserved verbatim for a later contact-aware hand optimization stage.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def _read_row(path: Path) -> tuple[pa.Table, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != 1:
        raise ValueError(f"expected one row in {path}")
    table = parquet.read()
    return table, {name: table[name][0].as_py() for name in table.column_names}


def _partition_value(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}=... partition in {path}")


def _rotation_error_deg(a_wxyz: np.ndarray, b_wxyz: np.ndarray) -> float:
    a = Rotation.from_quat(a_wxyz, scalar_first=True)
    b = Rotation.from_quat(b_wxyz, scalar_first=True)
    return float(np.degrees((a.inv() * b).magnitude()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--table-height", type=float, default=0.75)
    args = parser.parse_args()

    source = args.source_parquet.resolve()
    source_table, source_cells = _read_row(source)
    names = [str(value) for value in source_cells["object_body_names"]]
    if len(names) < 2:
        raise ValueError("source reference must contain at least two objects")
    source_position = np.asarray(source_cells["object_body_position"], dtype=np.float64)
    source_quaternion = np.asarray(source_cells["object_body_wxyz"], dtype=np.float64)
    position = source_position.copy()
    quaternion = source_quaternion.copy()
    urdfs = list(source_cells["object_urdf_paths"])
    meshes = list(source_cells["object_mesh_paths"])
    radii = list(source_cells["object_mesh_radius"])

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate_path in args.candidate:
        candidate = candidate_path.resolve()
        _, cells = _read_row(candidate)
        candidate_names = [str(value) for value in cells["object_body_names"]]
        if len(candidate_names) != 1:
            raise ValueError(f"candidate must contain one object: {candidate}")
        name = candidate_names[0]
        if name not in names:
            raise ValueError(f"candidate object {name!r} not in source names {names}")
        if name in seen:
            raise ValueError(f"duplicate candidate for {name}")
        seen.add(name)
        body = names.index(name)
        candidate_position = np.asarray(cells["object_body_position"], dtype=np.float64)
        candidate_quaternion = np.asarray(cells["object_body_wxyz"], dtype=np.float64)
        if candidate_position.shape != (len(position), 1, 3) or candidate_quaternion.shape != (
            len(position),
            1,
            4,
        ):
            raise ValueError(f"candidate pose shape mismatch for {name}: {candidate}")
        position[:, body] = candidate_position[:, 0]
        quaternion[:, body] = candidate_quaternion[:, 0]
        urdfs[body] = cells["object_urdf_paths"][0]
        meshes[body] = cells["object_mesh_paths"][0]
        radii[body] = cells["object_mesh_radius"][0]
        merged.append(
            {
                "object_index": body,
                "object_name": name,
                "candidate_parquet": str(candidate),
                "candidate_urdf": urdfs[body],
                "frame0_translation_delta_mm": (
                    1000.0 * (position[0, body] - source_position[0, body])
                ).tolist(),
                "frame0_rotation_delta_deg": _rotation_error_deg(
                    source_quaternion[0, body], quaternion[0, body]
                ),
            }
        )
    missing = sorted(set(names) - seen)
    if missing:
        raise ValueError(f"missing candidates for source objects: {missing}")

    output_cells = dict(source_cells)
    output_cells["object_body_position"] = position.tolist()
    output_cells["object_body_wxyz"] = quaternion.tolist()
    output_cells["object_urdf_paths"] = urdfs
    output_cells["object_mesh_paths"] = meshes
    output_cells["object_mesh_radius"] = radii
    if output_cells.get("object_root_position") is not None:
        output_cells["object_root_position"] = position[:, 0].tolist()
    if output_cells.get("object_root_axis_angle") is not None:
        output_cells["object_root_axis_angle"] = Rotation.from_quat(
            quaternion[:, 0], scalar_first=True
        ).as_rotvec().tolist()

    sequence = _partition_value(source, "sequence_id")
    robot = _partition_value(source, "robot_name")
    output_root = args.output_root.resolve()
    output = (
        output_root
        / "selected_retarget"
        / f"sequence_id={sequence}"
        / f"robot_name={robot}"
        / "part-0.parquet"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([output_cells], schema=source_table.schema),
        output,
        compression="snappy",
    )
    support_output = output_root / "reconstructed_stage" / f"{sequence}_support.usda"
    support_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.support_usda.resolve(), support_output)
    report = {
        "schema_version": "merged-single-object-candidates-v1",
        "status": "GENERATED_REVIEW",
        "source_parquet": str(source),
        "source_unchanged": True,
        "output_parquet": str(output),
        "support_usda": str(support_output),
        "common_support_plane_height_m": float(args.table_height),
        "objects": sorted(merged, key=lambda item: item["object_index"]),
        "hands_refined": False,
        "contact_columns_refined": False,
        "warning": "object-only merge; hand/contact alignment must be optimized after object physics acceptance",
    }
    (output_root / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
