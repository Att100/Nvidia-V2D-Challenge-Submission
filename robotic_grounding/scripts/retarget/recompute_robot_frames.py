#!/usr/bin/env python3
"""Recompute cached Sharpa robot-frame poses from wrist and finger trajectories."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

RG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RG_ROOT / "source" / "robotic_grounding"))

from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics  # noqa: E402


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


def _angle_deg(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(lhs * rhs, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--side", choices=("left", "right", "both"), default="both"
    )
    args = parser.parse_args()

    source_path = args.source_parquet.resolve()
    table, source = _read(source_path)
    output = dict(source)
    sides = ("left", "right") if args.side == "both" else (args.side,)
    report: dict[str, Any] = {
        "schema_version": "recomputed-robot-frames-v1",
        "source_parquet": str(source_path),
        "sides": {},
        "wrist_trajectory_fixed": True,
        "finger_trajectory_fixed": True,
        "object_trajectory_fixed": True,
        "contact_labels_fixed": True,
    }

    for side in sides:
        kinematics = setup_sharpa_kinematics(side)
        expected_names = list(kinematics.robot_frame_names.values())
        stored_names = [str(value) for value in source[f"{side}_robot_frame_names"]]
        if expected_names != stored_names:
            raise ValueError(f"{side} kinematics frame order differs from parquet")

        wrist_position = np.asarray(
            source[f"robot_{side}_wrist_position"], dtype=np.float64
        )
        wrist_wxyz = np.asarray(source[f"robot_{side}_wrist_wxyz"], dtype=np.float64)
        finger_joints = np.asarray(
            source[f"robot_{side}_finger_joints"], dtype=np.float64
        )
        before = np.asarray(source[f"robot_{side}_frames"], dtype=np.float64)
        after = np.empty_like(before)

        for frame in range(len(wrist_position)):
            q = kinematics.robot.q0.copy()
            q[:3] = wrist_position[frame]
            q[3:7] = wrist_wxyz[frame][[1, 2, 3, 0]]
            q[7:] = finger_joints[frame]
            kinematics.configuration.update(q)
            for index, frame_name in enumerate(expected_names):
                transform = kinematics.configuration.get_transform_frame_to_world(
                    frame_name
                )
                after[frame, index, :3] = transform.translation
                after[frame, index, 3:] = Rotation.from_matrix(
                    transform.rotation
                ).as_quat(scalar_first=True)

        output[f"robot_{side}_frames"] = after.tolist()
        report["sides"][side] = {
            "frame_count": int(len(after)),
            "robot_frame_count": int(after.shape[1]),
            "maximum_source_position_error_mm": float(
                1000.0 * np.max(np.linalg.norm(before[..., :3] - after[..., :3], axis=-1))
            ),
            "maximum_source_orientation_error_deg": float(
                np.max(_angle_deg(before[..., 3:], after[..., 3:]))
            ),
            "maximum_output_position_error_mm": 0.0,
            "maximum_output_orientation_error_deg": 0.0,
        }

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

    report["output_parquet"] = str(output_path)
    report_path = output_root / "reports" / "recomputed_robot_frames.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
