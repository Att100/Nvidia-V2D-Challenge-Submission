#!/usr/bin/env python3
"""Apply a selected object-only settled state as a fixed SE(3) track correction."""

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


def partition(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}= partition in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--rollout-npz", type=Path, required=True)
    parser.add_argument("--settled-index", type=int, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--body-index",
        type=int,
        action="append",
        help="Only refine these body indices; defaults to every body.",
    )
    args = parser.parse_args()

    source = args.parquet.resolve()
    table = pq.read_table(source)
    if table.num_rows != 1:
        raise ValueError("reference parquet must have exactly one row")
    cells: dict[str, Any] = table.to_pylist()[0]
    position = np.asarray(cells["object_body_position"], dtype=np.float64)
    quaternion = np.asarray(cells["object_body_wxyz"], dtype=np.float64)
    rollout = np.load(args.rollout_npz.resolve())
    settled_position = np.asarray(rollout["positions"], dtype=np.float64)
    settled_xyzw = np.asarray(rollout["quaternions_xyzw"], dtype=np.float64)
    names = [str(value) for value in cells["object_body_names"]]
    rollout_names = [str(value) for value in rollout["names"]]
    if names != rollout_names:
        raise ValueError(f"body order mismatch: reference={names}, rollout={rollout_names}")
    index = args.settled_index
    if index < 1 or index >= len(settled_position):
        raise ValueError(f"settled index {index} outside [1, {len(settled_position)-1}]")

    output_position = position.copy()
    output_quaternion = quaternion.copy()
    corrections = []
    selected = set(range(len(names))) if args.body_index is None else set(args.body_index)
    if not selected.issubset(set(range(len(names)))):
        raise ValueError(f"invalid body indices {sorted(selected)} for {len(names)} bodies")
    for body, name in enumerate(names):
        if body not in selected:
            continue
        source_r0 = Rotation.from_quat(quaternion[0, body], scalar_first=True)
        target_r0 = Rotation.from_quat(settled_xyzw[index, body])
        delta_r = target_r0 * source_r0.inv()
        source_p0 = position[0, body]
        target_p0 = settled_position[index, body]
        output_position[:, body] = delta_r.apply(position[:, body] - source_p0) + target_p0
        output_quaternion[:, body] = (delta_r * Rotation.from_quat(
            quaternion[:, body], scalar_first=True
        )).as_quat(scalar_first=True)
        corrections.append(
            {
                "body_index": body,
                "name": name,
                "translation_mm": (1000.0 * (target_p0 - source_p0)).tolist(),
                "rotation_deg": float(np.degrees(delta_r.magnitude())),
            }
        )

    output = dict(cells)
    output["object_body_position"] = output_position.tolist()
    output["object_body_wxyz"] = output_quaternion.tolist()
    output["object_root_position"] = output_position[:, 0].tolist()
    output["object_root_axis_angle"] = Rotation.from_quat(
        output_quaternion[:, 0], scalar_first=True
    ).as_rotvec().tolist()

    sequence = partition(source, "sequence_id")
    robot = partition(source, "robot_name")
    root = args.output_root.resolve()
    parquet_out = root / "selected_retarget" / f"sequence_id={sequence}" / f"robot_name={robot}" / "part-0.parquet"
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    parquet_out.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([output], schema=table.schema), parquet_out, compression="snappy")
    support_out = root / "reconstructed_stage" / f"{sequence}_support.usda"
    support_out.parent.mkdir(parents=True)
    shutil.copy2(args.support_usda.resolve(), support_out)
    report = {
        "schema_version": "multi-object-settled-se3-refinement-v1",
        "source_parquet": str(source),
        "source_unchanged": True,
        "rollout_npz": str(args.rollout_npz.resolve()),
        "settled_index": index,
        "settled_time_s": float(index / float(rollout["fps"])),
        "output_parquet": str(parquet_out),
        "corrections": corrections,
        "hands_refined": False,
        "contact_positions_refined": False,
    }
    (root / "refinement_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
