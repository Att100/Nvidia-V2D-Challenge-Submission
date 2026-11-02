#!/usr/bin/env python3
"""Extract one rigid object from a multi-object ManoSharpa reference.

The result is intended for isolated object-on-support physics validation.  The
source parquet is never modified.  Object-indexed fields are sliced, contact
part IDs are remapped to the one-body convention (0=no contact, 1=selected
object), and contacts belonging to other objects are disabled.
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
        raise ValueError(f"expected one row in {path}, got {parquet.metadata.num_rows}")
    table = parquet.read()
    return table, {name: table[name][0].as_py() for name in table.column_names}


def _partition_value(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}=... partition in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--object", required=True, help="Exact object body name or integer body index")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.parquet.resolve()
    support = args.support_usda.resolve()
    table, cells = _read_row(source)
    names = [str(value) for value in cells["object_body_names"]]
    try:
        index = int(args.object)
    except ValueError:
        if args.object not in names:
            raise ValueError(f"object {args.object!r} not in {names}") from None
        index = names.index(args.object)
    if index < 0 or index >= len(names):
        raise IndexError(f"object index {index} outside [0, {len(names)})")

    selected_name = names[index]
    output_cells = dict(cells)
    for key in (
        "object_body_names",
        "safe_object_body_names",
        "object_mesh_paths",
        "object_urdf_paths",
        "object_mesh_radius",
    ):
        values = output_cells.get(key)
        if values is not None:
            if len(values) != len(names):
                raise ValueError(f"{key} length {len(values)} does not match {len(names)} objects")
            output_cells[key] = [values[index]]

    position = np.asarray(cells["object_body_position"], dtype=np.float64)
    quaternion = np.asarray(cells["object_body_wxyz"], dtype=np.float64)
    if position.shape[1] != len(names) or quaternion.shape[1] != len(names):
        raise ValueError("object pose axes do not match object_body_names")
    output_cells["object_body_position"] = position[:, index : index + 1].tolist()
    output_cells["object_body_wxyz"] = quaternion[:, index : index + 1].tolist()
    if output_cells.get("object_root_position") is not None:
        output_cells["object_root_position"] = position[:, index].tolist()
    if output_cells.get("object_root_axis_angle") is not None:
        output_cells["object_root_axis_angle"] = Rotation.from_quat(
            quaternion[:, index], scalar_first=True
        ).as_rotvec().tolist()
    output_cells["object_name"] = selected_name
    if "safe_object_name" in output_cells:
        safe_names = cells.get("safe_object_body_names")
        output_cells["safe_object_name"] = (
            str(safe_names[index]) if safe_names is not None else selected_name
        )

    selected_part_id = index + 1
    contact_summary: dict[str, Any] = {}
    for side in ("left", "right"):
        key = f"mano_{side}_object_contact_part_ids"
        if output_cells.get(key) is None:
            continue
        original = np.asarray(output_cells[key], dtype=np.int32)
        remapped = np.where(original == selected_part_id, 1, 0).astype(np.int32)
        output_cells[key] = remapped.tolist()
        contact_summary[side] = {
            "original_active_slots": int(np.count_nonzero(original)),
            "selected_active_slots": int(np.count_nonzero(remapped)),
        }

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
        pa.Table.from_pylist([output_cells], schema=table.schema),
        output,
        compression="snappy",
    )
    support_output = output_root / "reconstructed_stage" / f"{sequence}_support.usda"
    support_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(support, support_output)

    report = {
        "schema_version": "isolated-single-object-reference-v1",
        "source_parquet": str(source),
        "source_unchanged": True,
        "selected_object_index": index,
        "selected_object_name": selected_name,
        "output_parquet": str(output),
        "support_usda": str(support_output),
        "contact_id_remap": contact_summary,
        "hands_refined": False,
        "purpose": "isolated object-on-support physics validation only",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "extraction_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
