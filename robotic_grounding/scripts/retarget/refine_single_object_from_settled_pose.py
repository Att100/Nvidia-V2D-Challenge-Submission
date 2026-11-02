#!/usr/bin/env python3
"""Create a second-pass object reference from a settled free-simulation tail."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def _partition_value(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}= partition in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--simulation-npz", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tail-frames", type=int, default=20)
    args = parser.parse_args()

    source_path = args.parquet.resolve()
    parquet = pq.ParquetFile(source_path)
    table = parquet.read()
    cells = {name: table[name][0].as_py() for name in table.column_names}
    position = np.asarray(cells["object_body_position"], dtype=np.float64)
    quaternion = np.asarray(cells["object_body_wxyz"], dtype=np.float64)
    if position.shape[1] != 1:
        raise ValueError("settled-pose refinement supports exactly one object")

    simulation = np.load(args.simulation_npz.resolve())
    simulated_position = np.asarray(simulation["positions"], dtype=np.float64)[:, 0]
    simulated_quaternion = np.asarray(simulation["quaternions_xyzw"], dtype=np.float64)[:, 0]
    tail = min(int(args.tail_frames), len(simulated_position))
    if tail < 2:
        raise ValueError("settled simulation tail must contain at least two frames")
    desired_position = np.median(simulated_position[-tail:], axis=0)
    desired_rotation = Rotation.from_quat(simulated_quaternion[-tail:]).mean()

    initial_position = position[0, 0].copy()
    initial_rotation = Rotation.from_quat(quaternion[0, 0], scalar_first=True)
    position_offset = desired_position - initial_position
    delta_rotation = desired_rotation * initial_rotation.inv()

    position[:, 0] += position_offset
    rotation = Rotation.from_quat(quaternion[:, 0], scalar_first=True)
    quaternion[:, 0] = (delta_rotation * rotation).as_quat(scalar_first=True)
    cells["object_body_position"] = position.tolist()
    cells["object_body_wxyz"] = quaternion.tolist()
    if cells.get("object_root_position") is not None:
        root_position = np.asarray(cells["object_root_position"], dtype=np.float64)
        cells["object_root_position"] = (root_position + position_offset).tolist()
    if cells.get("object_root_axis_angle") is not None:
        root_rotation = Rotation.from_rotvec(np.asarray(cells["object_root_axis_angle"], dtype=np.float64))
        cells["object_root_axis_angle"] = (delta_rotation * root_rotation).as_rotvec().tolist()

    sequence_id = _partition_value(source_path, "sequence_id")
    robot_name = _partition_value(source_path, "robot_name")
    output_root = args.output_root.resolve()
    output_path = (
        output_root
        / "selected_retarget"
        / f"sequence_id={sequence_id}"
        / f"robot_name={robot_name}"
        / "part-0.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([cells], schema=table.schema), output_path, compression="snappy")
    support_output = output_root / "reconstructed_stage" / f"{sequence_id}_support.usda"
    support_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.support_usda.resolve(), support_output)

    report = {
        "schema_version": "single-object-settled-fixed-point-v1",
        "source_parquet": str(source_path),
        "simulation_npz": str(args.simulation_npz.resolve()),
        "tail_frames": tail,
        "position_offset_mm": (1000.0 * position_offset).tolist(),
        "rotation_offset_deg": float(np.degrees(delta_rotation.magnitude())),
        "output_parquet": str(output_path),
        "hand_columns_refined": False,
    }
    (output_root / "generation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
