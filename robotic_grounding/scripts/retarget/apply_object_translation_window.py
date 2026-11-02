#!/usr/bin/env python3
"""Apply a smooth, bounded translation window to one object trajectory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def partition(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}= partition in {path}")


def cosine_ramp(length: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    phase = np.arange(1, length + 1, dtype=np.float64) / (length + 1)
    return 0.5 - 0.5 * np.cos(np.pi * phase)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--translation", type=float, nargs=3, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--full-start-frame", type=int, required=True)
    parser.add_argument("--full-end-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    args = parser.parse_args()

    source = args.source_parquet.resolve()
    table = pq.read_table(source)
    if table.num_rows != 1:
        raise ValueError(f"expected one row, found {table.num_rows}")
    row = table.to_pylist()[0]
    names = list(row["object_body_names"])
    if args.object_name not in names:
        raise ValueError(f"{args.object_name!r} not found in {names}")
    object_index = names.index(args.object_name)
    position = np.asarray(row["object_body_position"], dtype=np.float64)
    frames = len(position)
    bounds = (
        args.start_frame,
        args.full_start_frame,
        args.full_end_frame,
        args.end_frame,
    )
    if not (0 <= bounds[0] <= bounds[1] <= bounds[2] <= bounds[3] < frames):
        raise ValueError(f"invalid frame bounds {bounds} for {frames} frames")

    weight = np.zeros(frames, dtype=np.float64)
    weight[args.start_frame : args.full_start_frame] = cosine_ramp(
        args.full_start_frame - args.start_frame
    )
    weight[args.full_start_frame : args.full_end_frame + 1] = 1.0
    down = cosine_ramp(args.end_frame - args.full_end_frame)
    weight[args.full_end_frame + 1 : args.end_frame + 1] = 1.0 - down
    delta = weight[:, None] * np.asarray(args.translation, dtype=np.float64)[None, :]
    position[:, object_index] += delta

    output = dict(row)
    output["object_body_position"] = position.tolist()
    if object_index == 0:
        output["object_root_position"] = position[:, 0].tolist()

    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    sequence = partition(source, "sequence_id")
    robot = partition(source, "robot_name")
    output_parquet = (
        root
        / "selected_retarget"
        / f"sequence_id={sequence}"
        / f"robot_name={robot}"
        / "part-0.parquet"
    )
    output_parquet.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([output], schema=table.schema),
        output_parquet,
        compression="snappy",
    )
    support_output = root / "reconstructed_stage" / f"{sequence}_support.usda"
    support_output.parent.mkdir(parents=True)
    shutil.copy2(args.support_usda.resolve(), support_output)
    report = {
        "schema_version": "smooth-object-translation-window-v1",
        "source_parquet": str(source),
        "output_parquet": str(output_parquet),
        "object_name": args.object_name,
        "object_index": object_index,
        "translation_m": args.translation,
        "frame_bounds": {
            "start": args.start_frame,
            "full_start": args.full_start_frame,
            "full_end": args.full_end_frame,
            "end": args.end_frame,
        },
        "maximum_step_mm": float(1000.0 * np.linalg.norm(np.diff(delta, axis=0), axis=-1).max()),
        "hands_modified": False,
        "contact_data_modified": False,
    }
    (root / "translation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
