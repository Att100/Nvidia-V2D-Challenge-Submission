#!/usr/bin/env python3
"""Retime a hand's pre-contact translation to reduce controller lag.

The contact pose and every frame after it are kept exactly unchanged.  Before
contact, the recorded wrist path is resampled at constant arc-length speed and
the resulting translation is applied rigidly to all robot-hand frames.  Finger
joint angles and orientations stay unchanged, preserving exact hand
articulation while giving the physical controller more time to reach the grasp.
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


def _read(path: Path) -> tuple[pa.Table, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != 1:
        raise ValueError(f"expected one row in {path}")
    table = parquet.read()
    return table, {name: table[name][0].as_py() for name in table.column_names}


def _partition(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}= partition in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--contact-frame", type=int, required=True)
    args = parser.parse_args()

    source_path = args.source_parquet.resolve()
    table, source = _read(source_path)
    side = args.side
    wrist_key = f"robot_{side}_wrist_position"
    frames_key = f"robot_{side}_frames"
    wrist = np.asarray(source[wrist_key], dtype=np.float64)
    frames = np.asarray(source[frames_key], dtype=np.float64)
    start = int(args.start_frame)
    end = int(args.contact_frame)
    if not 0 <= start < end < len(wrist):
        raise ValueError(f"invalid retiming window [{start}, {end}] for {len(wrist)} frames")

    path = wrist[start : end + 1]
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    total = float(cumulative[-1])
    if total <= 1.0e-9:
        raise ValueError("pre-contact wrist path has zero arc length")
    # Remove repeated arc coordinates before interpolation.
    keep = np.r_[True, np.diff(cumulative) > 1.0e-12]
    arc = cumulative[keep]
    points = path[keep]
    target_arc = np.linspace(0.0, total, end - start + 1)
    retimed = np.column_stack(
        [np.interp(target_arc, arc, points[:, axis]) for axis in range(3)]
    )
    # Endpoints are an explicit invariant, avoiding floating-point drift at the
    # first contact pose.
    retimed[0] = path[0]
    retimed[-1] = path[-1]

    correction = np.zeros_like(wrist)
    correction[start : end + 1] = retimed - path
    output = dict(source)
    output[wrist_key] = (wrist + correction).tolist()
    output_frames = frames.copy()
    output_frames[..., :3] += correction[:, None, :]
    output[frames_key] = output_frames.tolist()

    output_root = args.output_root.resolve()
    sequence_id = _partition(source_path, "sequence_id")
    robot_name = _partition(source_path, "robot_name")
    output_path = (
        output_root
        / "selected_retarget"
        / f"sequence_id={sequence_id}"
        / f"robot_name={robot_name}"
        / "part-0.parquet"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([output], schema=table.schema),
        output_path,
        compression="snappy",
    )
    support_output = output_root / "reconstructed_stage" / f"{sequence_id}_support.usda"
    support_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.support_usda.resolve(), support_output)

    original_step = 1000.0 * np.linalg.norm(np.diff(wrist, axis=0), axis=1)
    updated_wrist = wrist + correction
    updated_step = 1000.0 * np.linalg.norm(np.diff(updated_wrist, axis=0), axis=1)
    correction_norm = 1000.0 * np.linalg.norm(correction, axis=1)
    correction_step = 1000.0 * np.linalg.norm(np.diff(correction, axis=0), axis=1)
    report = {
        "schema_version": "constant-arc-precontact-hand-retiming-v1",
        "source_parquet": str(source_path),
        "output_parquet": str(output_path),
        "side": side,
        "start_frame": start,
        "contact_frame": end,
        "contact_pose_exactly_preserved": bool(
            np.array_equal(updated_wrist[end], wrist[end])
        ),
        "post_contact_trajectory_exactly_preserved": bool(
            np.array_equal(updated_wrist[end:], wrist[end:])
        ),
        "object_trajectory_fixed": True,
        "finger_joint_angles_changed": False,
        "hand_orientation_changed": False,
        "path_arc_length_mm": 1000.0 * total,
        "maximum_wrist_step_before_mm": float(np.max(original_step[start:end])),
        "maximum_wrist_step_after_mm": float(np.max(updated_step[start:end])),
        "maximum_applied_translation_mm": float(np.max(correction_norm)),
        "maximum_translation_correction_step_mm": float(np.max(correction_step)),
    }
    report_path = output_root / "reports" / "hand_contact_approach_retiming.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
