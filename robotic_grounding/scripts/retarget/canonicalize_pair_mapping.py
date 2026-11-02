#!/usr/bin/env python3
"""Reorder a mislabeled two-object reference without changing either trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def partition(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}= partition in {path}")


def remap(values: Any, old_pot: int, old_spoon: int) -> Any:
    source = np.asarray(values, dtype=np.int64)
    output = source.copy()
    output[source == old_pot + 1] = 1
    output[source == old_spoon + 1] = 2
    return output.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--old-pot-index", type=int, default=1)
    parser.add_argument("--old-spoon-index", type=int, default=0)
    args = parser.parse_args()

    source = args.source_parquet.resolve()
    # Read the physical file directly so Hive partition names in parent
    # directories are not injected as extra schema columns.
    table = pq.ParquetFile(source).read()
    if table.num_rows != 1:
        raise ValueError("reference parquet must contain exactly one row")
    row = table.to_pylist()[0]
    order = [args.old_pot_index, args.old_spoon_index]
    if sorted(order) != [0, 1] or len(row["object_body_names"]) != 2:
        raise ValueError("expected a two-object reference and a permutation of [0, 1]")
    output = dict(row)
    output["object_body_names"] = ["white_pot", "wooden_spoon"]
    output["safe_object_body_names"] = ["white_pot", "wooden_spoon"]
    for key in ("object_mesh_paths", "object_urdf_paths", "object_mesh_radius"):
        output[key] = [row[key][index] for index in order]
    position = np.asarray(row["object_body_position"], dtype=np.float64)[:, order]
    quaternion = np.asarray(row["object_body_wxyz"], dtype=np.float64)[:, order]
    output["object_body_position"] = position.tolist()
    output["object_body_wxyz"] = quaternion.tolist()
    output["object_root_position"] = position[:, 0].tolist()
    output["object_root_axis_angle"] = Rotation.from_quat(
        quaternion[:, 0], scalar_first=True
    ).as_rotvec().tolist()
    remapped = []
    for side in ("left", "right"):
        key = f"mano_{side}_object_contact_part_ids"
        output[key] = remap(row[key], args.old_pot_index, args.old_spoon_index)
        remapped.append(key)

    sequence = partition(source, "sequence_id")
    robot = partition(source, "robot_name")
    root = args.output_root.resolve()
    output_path = root / "selected_retarget" / f"sequence_id={sequence}" / f"robot_name={robot}" / "part-0.parquet"
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    output_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([output], schema=table.schema), output_path, compression="snappy")
    report = {
        "schema_version": "canonical-pair-mapping-v1",
        "source_parquet": str(source),
        "source_unchanged": True,
        "output_parquet": str(output_path),
        "old_body_order": row["object_body_names"],
        "new_body_order": output["object_body_names"],
        "old_to_new_body": {str(args.old_pot_index): 0, str(args.old_spoon_index): 1},
        "old_to_new_contact_part_id": {str(args.old_pot_index + 1): 1, str(args.old_spoon_index + 1): 2},
        "object_trajectories_changed": False,
        "hand_trajectories_changed": False,
        "contact_part_id_columns_remapped": remapped,
    }
    (root / "canonicalization_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
