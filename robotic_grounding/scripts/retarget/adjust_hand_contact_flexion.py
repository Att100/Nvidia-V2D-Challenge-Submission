#!/usr/bin/env python3
"""Apply bounded finger-flexion offsets during labelled object contact.

This refinement is intended after wrist/contact translation has been selected.
Offsets fade in before contact and out after release, are clipped to the exact
MJCF joint ranges, and never modify object poses or the opposite hand.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
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


def _weight(active: np.ndarray, transition: int) -> np.ndarray:
    weight = active.astype(np.float64)
    if not np.any(active) or transition <= 0:
        return weight
    frames = np.flatnonzero(active)
    for frame in np.flatnonzero(~active):
        distance = int(np.min(np.abs(frames - frame)))
        if distance < transition:
            weight[frame] = 0.5 * (1.0 + np.cos(np.pi * distance / transition))
    return weight


def _parse_offset(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("joint offset must be NAME=RADIANS")
    name, raw = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("joint offset name cannot be empty")
    return name, float(raw)


def _joint_ranges(mjcf: Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(mjcf).getroot()
    ranges: dict[str, tuple[float, float]] = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        raw = joint.get("range")
        if not name or raw is None:
            continue
        values = [float(value) for value in raw.split()]
        if len(values) == 2:
            ranges[name] = (values[0], values[1])
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hand-mjcf", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--contact-part-id", type=int, required=True)
    parser.add_argument("--transition-frames", type=int, default=10)
    parser.add_argument(
        "--recompute-robot-frames",
        action="store_true",
        help="Recompute cached robot frame poses from the modified wrist/finger q.",
    )
    parser.add_argument(
        "--joint-offset",
        action="append",
        type=_parse_offset,
        required=True,
        metavar="NAME=RADIANS",
    )
    args = parser.parse_args()

    source_path = args.source_parquet.resolve()
    table, source = _read(source_path)
    side = args.side
    names = [str(value) for value in source[f"{side}_robot_finger_joint_names"]]
    name_to_index = {name: index for index, name in enumerate(names)}
    limits = _joint_ranges(args.hand_mjcf.resolve())
    requested = dict(args.joint_offset)
    missing = sorted(set(requested) - set(name_to_index))
    if missing:
        raise ValueError(f"finger joints missing from parquet: {missing}")
    missing_limits = sorted(set(requested) - set(limits))
    if missing_limits:
        raise ValueError(f"finger joint ranges missing from MJCF: {missing_limits}")

    part_ids = np.asarray(
        source[f"mano_{side}_object_contact_part_ids"], dtype=np.int32
    )
    active = np.any(part_ids == int(args.contact_part_id), axis=1)
    if not np.any(active):
        raise ValueError(f"no {side} contacts use part id {args.contact_part_id}")
    weight = _weight(active, int(args.transition_frames))
    before = np.asarray(source[f"robot_{side}_finger_joints"], dtype=np.float64)
    after = before.copy()
    per_joint: dict[str, Any] = {}
    for name, offset in requested.items():
        index = name_to_index[name]
        lower, upper = limits[name]
        requested_delta = float(offset) * weight
        # Clip only the newly requested contact offset.  Some reconstructed
        # references are already a tiny amount outside the MJCF range; clipping
        # the absolute trajectory would silently alter non-contact frames where
        # weight is exactly zero.  Including zero in the admissible delta range
        # preserves every source sample unless this refinement asks to move it.
        minimum_delta = np.minimum(lower - before[:, index], 0.0)
        maximum_delta = np.maximum(upper - before[:, index], 0.0)
        applied = np.clip(requested_delta, minimum_delta, maximum_delta)
        after[:, index] = before[:, index] + applied
        per_joint[name] = {
            "requested_offset_rad": float(offset),
            "range_rad": [lower, upper],
            "active_before_rad": {
                "minimum": float(np.min(before[active, index])),
                "median": float(np.median(before[active, index])),
                "maximum": float(np.max(before[active, index])),
            },
            "active_after_rad": {
                "minimum": float(np.min(after[active, index])),
                "median": float(np.median(after[active, index])),
                "maximum": float(np.max(after[active, index])),
            },
            "maximum_applied_offset_rad": float(np.max(np.abs(applied))),
            "clipped_frame_count": int(
                np.count_nonzero(np.abs(applied - requested_delta) > 1.0e-10)
            ),
        }

    output = dict(source)
    output[f"robot_{side}_finger_joints"] = after.tolist()
    if args.recompute_robot_frames:
        kinematics = setup_sharpa_kinematics(side)
        expected_frame_names = list(kinematics.robot_frame_names.values())
        stored_frame_names = [
            str(value) for value in source[f"{side}_robot_frame_names"]
        ]
        if expected_frame_names != stored_frame_names:
            raise ValueError("kinematics frame order differs from parquet frame order")
        wrist_position = np.asarray(
            source[f"robot_{side}_wrist_position"], dtype=np.float64
        )
        wrist_wxyz = np.asarray(
            source[f"robot_{side}_wrist_wxyz"], dtype=np.float64
        )
        frame_poses = []
        for frame in range(len(after)):
            q = kinematics.robot.q0.copy()
            q[:3] = wrist_position[frame]
            q[3:7] = wrist_wxyz[frame][[1, 2, 3, 0]]
            q[7:] = after[frame]
            kinematics.configuration.update(q)
            poses = []
            for frame_name in expected_frame_names:
                transform = kinematics.configuration.get_transform_frame_to_world(
                    frame_name
                )
                poses.append(
                    np.r_[
                        transform.translation,
                        Rotation.from_matrix(transform.rotation).as_quat(
                            scalar_first=True
                        ),
                    ]
                )
            frame_poses.append(poses)
        output[f"robot_{side}_frames"] = np.asarray(frame_poses).tolist()
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

    delta = after - before
    delta_step = np.diff(delta, axis=0)
    report = {
        "schema_version": "bounded-contact-flexion-offset-v1",
        "source_parquet": str(source_path),
        "output_parquet": str(output_path),
        "side": side,
        "contact_part_id": int(args.contact_part_id),
        "contact_frame_count": int(np.sum(active)),
        "transition_frames": int(args.transition_frames),
        "object_trajectory_fixed": True,
        "wrist_trajectory_fixed": True,
        "opposite_hand_fixed": True,
        "robot_frames_recomputed": bool(args.recompute_robot_frames),
        "frame_task_errors_are_source_metadata": bool(args.recompute_robot_frames),
        "joints": per_joint,
        "maximum_offset_step_rad_per_frame": float(
            np.max(np.abs(delta_step)) if delta_step.size else 0.0
        ),
    }
    report_path = output_root / "reports" / "hand_contact_flexion.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
