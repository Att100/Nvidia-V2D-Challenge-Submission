#!/usr/bin/env python3
"""Generate a joint frame-0 table-stable candidate for multiple rigid objects.

This stage assumes every object is independently table-supported at frame 0.
It finds the nearest convex-hull stable attitude for each object, preserves XY,
places every transformed mesh bottom on the same table plane, and applies each
object's fixed correction to its complete reference trajectory.  Hands and
contact annotations are intentionally left unchanged for the later contact pass.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from trimesh.poses import compute_stable_poses


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
    raise ValueError(f"missing {key}= partition in {path}")


def _nearest_yaw(reference: Rotation, candidate: Rotation) -> tuple[Rotation, float, float]:
    result = minimize_scalar(
        lambda yaw: float(
            (reference.inv() * (Rotation.from_rotvec([0.0, 0.0, yaw]) * candidate)).magnitude()
        ),
        bounds=(-math.pi, math.pi),
        method="bounded",
    )
    aligned = Rotation.from_rotvec([0.0, 0.0, float(result.x)]) * candidate
    return aligned, float(result.x), float(result.fun)


def _sample_world(mesh: trimesh.Trimesh, rotation: Rotation, position: np.ndarray) -> np.ndarray:
    count = min(5000, max(1000, len(mesh.faces)))
    sample, _ = trimesh.sample.sample_surface(mesh, count)
    return rotation.apply(sample) + position


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--table-height", type=float, default=0.75)
    parser.add_argument("--maximum-correction-deg", type=float, default=45.0)
    args = parser.parse_args()

    source_path = args.parquet.resolve()
    table, cells = _read_row(source_path)
    names = list(cells["object_body_names"])
    mesh_paths = [Path(value).resolve() for value in cells["object_mesh_paths"]]
    if len(names) < 2:
        raise ValueError("multi-object optimizer requires at least two objects")

    position = np.asarray(cells["object_body_position"], dtype=np.float64)
    quaternion = np.asarray(cells["object_body_wxyz"], dtype=np.float64)
    corrected_position = position.copy()
    corrected_quaternion = quaternion.copy()
    candidates: list[dict[str, Any]] = []
    meshes: list[trimesh.Trimesh] = []
    frame0_samples: list[np.ndarray] = []

    for body, (name, mesh_path) in enumerate(zip(names, mesh_paths, strict=True)):
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError(f"invalid object mesh: {mesh_path}")
        meshes.append(mesh)
        reference_position = position[0, body]
        reference_rotation = Rotation.from_quat(quaternion[0, body], scalar_first=True)
        transforms, probabilities = compute_stable_poses(mesh.convex_hull, sigma=0.0, n_samples=1)
        valid: list[dict[str, Any]] = []
        for stable_id, (transform, probability) in enumerate(zip(transforms, probabilities, strict=True)):
            aligned, yaw, error = _nearest_yaw(reference_rotation, Rotation.from_matrix(transform[:3, :3]))
            error_deg = float(np.degrees(error))
            if error_deg > float(args.maximum_correction_deg):
                continue
            rotated_vertices = aligned.apply(np.asarray(mesh.vertices, dtype=np.float64))
            desired_position = reference_position.copy()
            desired_position[2] = float(args.table_height) - float(rotated_vertices[:, 2].min())
            valid.append(
                {
                    "stable_pose_id": int(stable_id),
                    "stable_probability": float(probability),
                    "yaw_alignment_rad": yaw,
                    "correction_rotation_deg": error_deg,
                    "translation_delta_mm": (1000.0 * (desired_position - reference_position)).tolist(),
                    "desired_position": desired_position,
                    "desired_rotation": aligned,
                }
            )
        if not valid:
            raise RuntimeError(
                f"{name}: no stable pose within {args.maximum_correction_deg:g} degrees"
            )
        valid.sort(key=lambda item: (item["correction_rotation_deg"], -item["stable_probability"]))
        chosen = valid[0]
        desired_position = chosen.pop("desired_position")
        desired_rotation = chosen.pop("desired_rotation")
        delta_position = desired_position - reference_position
        delta_rotation = desired_rotation * reference_rotation.inv()
        corrected_position[:, body] += delta_position
        trajectory_rotation = Rotation.from_quat(quaternion[:, body], scalar_first=True)
        corrected_quaternion[:, body] = (delta_rotation * trajectory_rotation).as_quat(scalar_first=True)
        frame0_samples.append(_sample_world(mesh, desired_rotation, desired_position))
        candidates.append(
            {
                "object_index": body,
                "object_name": name,
                "mesh": str(mesh_path),
                **chosen,
                "delta_rotation_wxyz": delta_rotation.as_quat(scalar_first=True).tolist(),
            }
        )

    pair_metrics = []
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            d01, _ = cKDTree(frame0_samples[second]).query(frame0_samples[first], k=1)
            d10, _ = cKDTree(frame0_samples[first]).query(frame0_samples[second], k=1)
            pair_metrics.append(
                {
                    "objects": [names[first], names[second]],
                    "sampled_surface_distance_mm": float(1000.0 * min(d01.min(), d10.min())),
                    "warning": "sampled unsigned distance; physics gate determines penetration",
                }
            )

    output_cells = dict(cells)
    output_cells["object_body_position"] = corrected_position.tolist()
    output_cells["object_body_wxyz"] = corrected_quaternion.tolist()
    if output_cells.get("object_root_position") is not None:
        output_cells["object_root_position"] = corrected_position[:, 0].tolist()
    if output_cells.get("object_root_axis_angle") is not None:
        output_cells["object_root_axis_angle"] = Rotation.from_quat(
            corrected_quaternion[:, 0], scalar_first=True
        ).as_rotvec().tolist()

    sequence_id = _partition_value(source_path, "sequence_id")
    robot_name = _partition_value(source_path, "robot_name")
    output_root = args.output_root.resolve()
    output_path = output_root / "selected_retarget" / f"sequence_id={sequence_id}" / f"robot_name={robot_name}" / "part-0.parquet"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([output_cells], schema=table.schema), output_path, compression="snappy")
    support_output = output_root / "reconstructed_stage" / f"{sequence_id}_support.usda"
    support_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.support_usda.resolve(), support_output)

    report = {
        "schema_version": "multi-object-frame0-table-stable-v1",
        "status": "GENERATED_REVIEW",
        "source_parquet": str(source_path),
        "output_parquet": str(output_path),
        "assumption": "all objects are independently supported by the common table at frame 0",
        "table_height_m": float(args.table_height),
        "object_trajectory_correction": "fixed per-object position offset and attitude delta applied to all frames",
        "hands_refined": False,
        "objects": candidates,
        "frame0_object_pairs": pair_metrics,
    }
    report_path = output_root / "generation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
