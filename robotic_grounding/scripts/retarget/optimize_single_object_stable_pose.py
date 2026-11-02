#!/usr/bin/env python3
"""Generate auditable stable-pose candidates for one table-supported object.

This is the object-only first stage.  It never edits the source parquet or mesh.
Each output candidate applies one constant world-frame SE(3) correction to the
entire object trajectory, preserving the trajectory's relative motion.  Hand
and contact columns are deliberately left unchanged for a later contact-aware
hand optimization stage.
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
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from trimesh.poses import compute_stable_poses


def _partition_value(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}=... partition in {path}")


def _read_single_row(path: Path) -> tuple[pa.Table, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != 1:
        raise ValueError(f"expected one parquet row, got {parquet.metadata.num_rows}: {path}")
    table = parquet.read()
    return table, {name: table[name][0].as_py() for name in table.column_names}


def _wxyz_to_rotation(values: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.asarray(values, dtype=np.float64), scalar_first=True)


def _rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    return rotation.as_quat(scalar_first=True)


def _nearest_yaw(reference: Rotation, candidate: Rotation) -> tuple[Rotation, float, float]:
    """Yaw-align one stable pose to a reference and return pose/yaw/error."""

    def objective(yaw: float) -> float:
        aligned = Rotation.from_rotvec([0.0, 0.0, yaw]) * candidate
        return float((reference.inv() * aligned).magnitude())

    result = minimize_scalar(objective, bounds=(-math.pi, math.pi), method="bounded")
    aligned = Rotation.from_rotvec([0.0, 0.0, float(result.x)]) * candidate
    return aligned, float(result.x), float(result.fun)


def _support_metrics(points_world: np.ndarray, table_height: float, tolerance: float = 0.002) -> dict[str, Any]:
    height = points_world[:, 2] - float(table_height)
    support = points_world[height <= tolerance]
    area = 0.0
    if len(support) >= 3:
        try:
            area = float(ConvexHull(support[:, :2]).volume)
        except Exception:
            area = 0.0
    return {
        "minimum_gap_mm": float(1000.0 * height.min()),
        "support_vertex_count_2mm": int(len(support)),
        "support_hull_area_mm2": float(1.0e6 * area),
    }


def _urdf_inertial(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    inertial = root.find(".//inertial")
    if inertial is None:
        return {"present": False}
    origin = inertial.find("origin")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    return {
        "present": True,
        "origin_xyz": [float(value) for value in (origin.get("xyz", "0 0 0").split())] if origin is not None else [0.0, 0.0, 0.0],
        "mass_kg": float(mass.get("value")) if mass is not None else None,
        "inertia": dict(inertia.attrib) if inertia is not None else None,
    }


def _write_candidate(
    *,
    source_table: pa.Table,
    source_cells: dict[str, Any],
    destination: Path,
    delta_rotation: Rotation,
    position_offset: np.ndarray,
) -> None:
    cells = dict(source_cells)
    position = np.asarray(cells["object_body_position"], dtype=np.float64)
    quaternion = np.asarray(cells["object_body_wxyz"], dtype=np.float64)
    if position.shape[1] != 1:
        raise ValueError(f"single-object optimizer received {position.shape[1]} object bodies")

    rotation = _wxyz_to_rotation(quaternion[:, 0])
    # Correct the object's attitude without rotating its reconstructed world-space
    # center trajectory.  Rotating all positions around the world origin would
    # spuriously bend the demonstrated translation path.
    position[:, 0] = position[:, 0] + position_offset
    quaternion[:, 0] = _rotation_to_wxyz(delta_rotation * rotation)
    cells["object_body_position"] = position.tolist()
    cells["object_body_wxyz"] = quaternion.tolist()

    if "object_root_position" in cells and cells["object_root_position"] is not None:
        root_position = np.asarray(cells["object_root_position"], dtype=np.float64)
        root_position = root_position + position_offset
        cells["object_root_position"] = root_position.tolist()
    if "object_root_axis_angle" in cells and cells["object_root_axis_angle"] is not None:
        root_axis_angle = np.asarray(cells["object_root_axis_angle"], dtype=np.float64)
        root_rotation = Rotation.from_rotvec(root_axis_angle)
        cells["object_root_axis_angle"] = (delta_rotation * root_rotation).as_rotvec().tolist()

    destination.parent.mkdir(parents=True, exist_ok=True)
    output = pa.Table.from_pylist([cells], schema=source_table.schema)
    pq.write_table(output, destination, compression="snappy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--table-height", type=float, default=0.75)
    parser.add_argument("--maximum-correction-deg", type=float, default=15.0)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    source_path = args.parquet.resolve()
    support_path = args.support_usda.resolve()
    output_root = args.output_root.resolve()
    source_table, cells = _read_single_row(source_path)

    names = list(cells["object_body_names"])
    mesh_paths = [Path(value).resolve() for value in cells["object_mesh_paths"]]
    urdf_paths = [Path(value).resolve() for value in cells["object_urdf_paths"]]
    if len(names) != 1 or len(mesh_paths) != 1 or len(urdf_paths) != 1:
        raise ValueError("this optimizer is intentionally restricted to one rigid object")

    position = np.asarray(cells["object_body_position"], dtype=np.float64)
    quaternion = np.asarray(cells["object_body_wxyz"], dtype=np.float64)
    reference_position = position[0, 0]
    reference_rotation = _wxyz_to_rotation(quaternion[0, 0])

    mesh = trimesh.load(mesh_paths[0], force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"invalid triangle mesh: {mesh_paths[0]}")
    hull = mesh.convex_hull
    transforms, probabilities = compute_stable_poses(hull, sigma=0.0, n_samples=1)

    candidates: list[dict[str, Any]] = []
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    hull_com = np.asarray(hull.center_mass, dtype=np.float64)
    for stable_id, (transform, probability) in enumerate(zip(transforms, probabilities, strict=True)):
        candidate_rotation, yaw, angular_error = _nearest_yaw(
            reference_rotation,
            Rotation.from_matrix(transform[:3, :3]),
        )
        angular_error_deg = float(np.degrees(angular_error))
        if angular_error_deg > float(args.maximum_correction_deg):
            continue

        rotated = candidate_rotation.apply(vertices)
        candidate_position = reference_position.copy()
        candidate_position[2] = float(args.table_height) - float(rotated[:, 2].min())
        delta_rotation = candidate_rotation * reference_rotation.inv()
        position_offset = candidate_position - reference_position
        points_world = candidate_rotation.apply(vertices) + candidate_position
        com_world = candidate_rotation.apply(hull_com) + candidate_position
        metrics = _support_metrics(points_world, args.table_height)
        candidates.append(
            {
                "stable_pose_id": int(stable_id),
                "stable_probability": float(probability),
                "yaw_alignment_rad": yaw,
                "correction_rotation_deg": angular_error_deg,
                "frame0_translation_delta_mm": (1000.0 * (candidate_position - reference_position)).tolist(),
                "frame0_translation_norm_mm": float(1000.0 * np.linalg.norm(candidate_position - reference_position)),
                "frame0_position": candidate_position.tolist(),
                "frame0_wxyz": _rotation_to_wxyz(candidate_rotation).tolist(),
                "delta_rotation_wxyz": _rotation_to_wxyz(delta_rotation).tolist(),
                "position_offset": position_offset.tolist(),
                "convex_hull_com_world": com_world.tolist(),
                **metrics,
                "_delta_rotation": delta_rotation,
                "_position_offset": position_offset,
            }
        )

    candidates.sort(key=lambda item: (item["correction_rotation_deg"], -item["stable_probability"]))
    candidates = candidates[: max(1, int(args.top_k))]
    if not candidates:
        raise RuntimeError(
            f"no stable pose within {args.maximum_correction_deg:g} degrees of the reconstruction"
        )

    sequence_id = _partition_value(source_path, "sequence_id")
    robot_name = _partition_value(source_path, "robot_name")
    serializable_candidates = []
    for rank, candidate in enumerate(candidates):
        candidate_root = output_root / f"candidate_{rank:02d}"
        destination = (
            candidate_root
            / "selected_retarget"
            / f"sequence_id={sequence_id}"
            / f"robot_name={robot_name}"
            / "part-0.parquet"
        )
        _write_candidate(
            source_table=source_table,
            source_cells=cells,
            destination=destination,
            delta_rotation=candidate.pop("_delta_rotation"),
            position_offset=candidate.pop("_position_offset"),
        )
        support_destination = candidate_root / "reconstructed_stage" / f"{sequence_id}_support.usda"
        support_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(support_path, support_destination)
        candidate["rank"] = rank
        candidate["parquet"] = str(destination)
        candidate["support_usda"] = str(support_destination)
        serializable_candidates.append(candidate)

    inertia_eigenvalues = np.linalg.eigvalsh(np.asarray(hull.moment_inertia, dtype=np.float64))
    report = {
        "schema_version": "single-object-stable-pose-candidates-v1",
        "source_parquet": str(source_path),
        "source_unchanged": True,
        "object_name": names[0],
        "mesh": str(mesh_paths[0]),
        "urdf": str(urdf_paths[0]),
        "table_height_m": float(args.table_height),
        "mesh_audit": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "watertight": bool(mesh.is_watertight),
            "extents_mm": (1000.0 * np.asarray(mesh.extents)).tolist(),
            "convex_hull_com_local_mm": (1000.0 * hull_com).tolist(),
            "convex_hull_inertia_eigenvalues": inertia_eigenvalues.tolist(),
            "convex_hull_inertia_axis_ratio": float(inertia_eigenvalues.max() / inertia_eigenvalues.min()),
        },
        "urdf_inertial": _urdf_inertial(urdf_paths[0]),
        "candidate_count": len(serializable_candidates),
        "candidates": serializable_candidates,
        "hand_columns_refined": False,
        "warning": "candidate parquets are object-only; hand/contact columns remain in the original world frame",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "generation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
