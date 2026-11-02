#!/usr/bin/env python3
"""Stabilize one object's initial pose and fade the correction before contact.

The closest convex-hull stable attitude is yaw-aligned to the reconstructed
frame-0 attitude.  Its mesh bottom is placed on the table and the resulting
SE(3) correction is smoothly reduced to identity at ``--end-frame``.  This is
intended for an object which is initially unsupported by a hand: hand and
contact columns remain unchanged because the correction is zero at contact.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
from trimesh.poses import compute_stable_poses


def partition(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}= partition in {path}")


def nearest_yaw(reference: Rotation, candidate: Rotation) -> tuple[Rotation, float]:
    result = minimize_scalar(
        lambda yaw: float(
            (
                reference.inv()
                * (Rotation.from_rotvec([0.0, 0.0, yaw]) * candidate)
            ).magnitude()
        ),
        bounds=(-math.pi, math.pi),
        method="bounded",
    )
    aligned = Rotation.from_rotvec([0.0, 0.0, float(result.x)]) * candidate
    return aligned, float(result.fun)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--table-height", type=float, default=0.75)
    parser.add_argument("--xy-offset", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--maximum-correction-deg", type=float, default=30.0)
    args = parser.parse_args()

    source = args.source_parquet.resolve()
    table = pq.read_table(source)
    if table.num_rows != 1:
        raise ValueError(f"expected one row, found {table.num_rows}")
    cells: dict[str, Any] = table.to_pylist()[0]
    names = list(cells["object_body_names"])
    if args.object_name not in names:
        raise ValueError(f"{args.object_name!r} not found in {names}")
    body = names.index(args.object_name)
    positions = np.asarray(cells["object_body_position"], dtype=np.float64)
    quaternions = np.asarray(cells["object_body_wxyz"], dtype=np.float64)
    frames = len(positions)
    if not 1 <= args.end_frame < frames:
        raise ValueError(f"end-frame must be in [1, {frames - 1}]")

    mesh_path = Path(cells["object_mesh_paths"][body]).resolve()
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"invalid object mesh: {mesh_path}")

    reference_rotation = Rotation.from_quat(quaternions[0, body], scalar_first=True)
    urdf_path = Path(cells["object_urdf_paths"][body]).resolve()
    inertial = ET.parse(urdf_path).getroot().find(".//inertial")
    origin = None if inertial is None else inertial.find("origin")
    center_mass = np.zeros(3, dtype=np.float64)
    if origin is not None:
        center_mass = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    transforms, probabilities = compute_stable_poses(
        mesh.convex_hull, center_mass=center_mass, sigma=0.0, n_samples=1
    )
    candidates = []
    for stable_id, (transform, probability) in enumerate(
        zip(transforms, probabilities, strict=True)
    ):
        rotation, error = nearest_yaw(
            reference_rotation, Rotation.from_matrix(transform[:3, :3])
        )
        error_deg = float(np.degrees(error))
        if error_deg <= args.maximum_correction_deg:
            candidates.append((error_deg, -float(probability), stable_id, rotation))
    if not candidates:
        raise RuntimeError(
            f"no stable pose within {args.maximum_correction_deg:g} degrees"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    error_deg, negative_probability, stable_id, target_rotation = candidates[0]

    rotated_vertices = target_rotation.apply(np.asarray(mesh.vertices, dtype=np.float64))
    target_position = positions[0, body].copy()
    target_position[:2] += np.asarray(args.xy_offset, dtype=np.float64)
    target_position[2] = float(args.table_height) - float(rotated_vertices[:, 2].min())
    translation = target_position - positions[0, body]
    delta_rotation = target_rotation * reference_rotation.inv()

    weight = np.zeros(frames, dtype=np.float64)
    phase = np.arange(args.end_frame + 1, dtype=np.float64) / args.end_frame
    weight[: args.end_frame + 1] = 0.5 * (1.0 + np.cos(np.pi * phase))
    output_positions = positions.copy()
    output_quaternions = quaternions.copy()
    output_positions[:, body] += weight[:, None] * translation[None, :]
    trajectory_rotation = Rotation.from_quat(quaternions[:, body], scalar_first=True)
    weighted_delta = Rotation.from_rotvec(
        weight[:, None] * delta_rotation.as_rotvec()[None, :]
    )
    output_quaternions[:, body] = (
        weighted_delta * trajectory_rotation
    ).as_quat(scalar_first=True)

    output = dict(cells)
    output["object_body_position"] = output_positions.tolist()
    output["object_body_wxyz"] = output_quaternions.tolist()
    if body == 0:
        output["object_root_position"] = output_positions[:, 0].tolist()
        output["object_root_axis_angle"] = Rotation.from_quat(
            output_quaternions[:, 0], scalar_first=True
        ).as_rotvec().tolist()

    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    sequence = partition(source, "sequence_id")
    robot = partition(source, "robot_name")
    destination = (
        root
        / "selected_retarget"
        / f"sequence_id={sequence}"
        / f"robot_name={robot}"
        / "part-0.parquet"
    )
    destination.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([output], schema=table.schema),
        destination,
        compression="snappy",
    )
    support_destination = root / "reconstructed_stage" / f"{sequence}_support.usda"
    support_destination.parent.mkdir(parents=True)
    shutil.copy2(args.support_usda.resolve(), support_destination)

    report = {
        "schema_version": "precontact-stable-pose-v1",
        "source_parquet": str(source),
        "output_parquet": str(destination),
        "object_name": args.object_name,
        "object_index": body,
        "mesh": str(mesh_path),
        "urdf": str(urdf_path),
        "urdf_center_mass_local_m": center_mass.tolist(),
        "mesh_watertight": bool(mesh.is_watertight),
        "table_height_m": float(args.table_height),
        "stable_pose_id": int(stable_id),
        "stable_probability": float(-negative_probability),
        "frame0_translation_delta_mm": (1000.0 * translation).tolist(),
        "frame0_rotation_delta_deg": error_deg,
        "delta_rotation_wxyz": delta_rotation.as_quat(scalar_first=True).tolist(),
        "fade_end_frame": args.end_frame,
        "maximum_translation_step_mm": float(
            1000.0 * np.linalg.norm(np.diff(weight[:, None] * translation, axis=0), axis=1).max()
        ),
        "hands_modified": False,
        "contact_data_modified": False,
        "contact_alignment_note": "SE(3) correction is exactly identity at and after fade_end_frame",
    }
    (root / "stable_pose_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
