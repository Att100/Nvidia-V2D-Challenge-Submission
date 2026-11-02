#!/usr/bin/env python3
"""Prepare a non-destructive rigid-object URDF/proxy candidate and parquet."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh


def _partition_value(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}= partition in {path}")


def _inertia(mesh: trimesh.Trimesh, target_mass: float) -> tuple[np.ndarray, np.ndarray]:
    hull = mesh.convex_hull
    if not np.isfinite(hull.mass) or hull.mass <= 0.0:
        raise ValueError("convex hull has invalid volume/mass")
    center = np.asarray(hull.center_mass, dtype=np.float64)
    tensor = np.asarray(hull.moment_inertia, dtype=np.float64) * (float(target_mass) / float(hull.mass))
    return center, tensor


def _segmented_hulls(mesh: trimesh.Trimesh, count: int) -> list[trimesh.Trimesh]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    centered = vertices - vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    coordinate = vertices @ axis
    edges = np.linspace(float(coordinate.min()), float(coordinate.max()), count + 1)
    hulls = []
    for index in range(count):
        lower, upper = edges[index], edges[index + 1]
        mask = (coordinate >= lower - 1.0e-6) & (coordinate <= upper + 1.0e-6)
        points = vertices[mask]
        if len(points) < 4:
            raise ValueError(f"segment {index} has only {len(points)} vertices")
        hulls.append(trimesh.Trimesh(vertices=points, process=False).convex_hull)
    return hulls


def _urdf(name: str, collision_files: list[str], mass: float, com: np.ndarray, inertia: np.ndarray) -> str:
    collisions = "\n".join(
        f'''      <collision name="collision_{index:02d}">
        <origin rpy="0 0 0" xyz="0 0 0"/>
        <geometry><mesh filename="./{filename}" scale="1 1 1"/></geometry>
      </collision>'''
        for index, filename in enumerate(collision_files)
    )
    return f'''<?xml version="1.0" ?>
<robot name="{name}">
  <mujoco><compiler meshdir="./" balanceinertia="true" discardvisual="false"/></mujoco>
  <link name="object">
    <inertial>
      <origin xyz="{com[0]:.12g} {com[1]:.12g} {com[2]:.12g}" rpy="0 0 0"/>
      <mass value="{mass:.12g}"/>
      <inertia ixx="{inertia[0,0]:.12g}" ixy="{inertia[0,1]:.12g}" ixz="{inertia[0,2]:.12g}"
               iyy="{inertia[1,1]:.12g}" iyz="{inertia[1,2]:.12g}" izz="{inertia[2,2]:.12g}"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="./visual.stl" scale="1 1 1"/></geometry>
    </visual>
{collisions}
  </link>
</robot>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("inertia_only", "segmented"), required=True)
    parser.add_argument("--segments", type=int, default=5)
    parser.add_argument("--mass-kg", type=float, default=0.05)
    args = parser.parse_args()
    if args.mass_kg <= 0.0:
        raise ValueError("--mass-kg must be positive")

    source_path = args.parquet.resolve()
    parquet = pq.ParquetFile(source_path)
    table = parquet.read()
    cells = {name: table[name][0].as_py() for name in table.column_names}
    names = list(cells["object_body_names"])
    meshes = list(cells["object_mesh_paths"])
    if len(names) != 1 or len(meshes) != 1:
        raise ValueError("proxy preparation supports exactly one rigid object")
    mesh = trimesh.load(meshes[0], force="mesh", process=False)
    output_root = args.output_root.resolve()
    asset_dir = output_root / "assets" / names[0]
    asset_dir.mkdir(parents=True, exist_ok=True)
    mesh.export(asset_dir / "visual.stl")

    if args.mode == "inertia_only":
        collision_meshes = [mesh]
    else:
        collision_meshes = _segmented_hulls(mesh, int(args.segments))
    collision_files = []
    for index, collision in enumerate(collision_meshes):
        filename = f"collision_part_{index:02d}.stl"
        collision.export(asset_dir / filename)
        collision_files.append(filename)

    com, tensor = _inertia(mesh, args.mass_kg)
    urdf_path = asset_dir / f"{names[0]}_rigid.urdf"
    urdf_path.write_text(_urdf(names[0], collision_files, args.mass_kg, com, tensor))
    cells["object_urdf_paths"] = [str(urdf_path)]

    sequence_id = _partition_value(source_path, "sequence_id")
    robot_name = _partition_value(source_path, "robot_name")
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

    eigenvalues = np.linalg.eigvalsh(tensor)
    report = {
        "schema_version": "single-object-proxy-candidate-v1",
        "source_parquet": str(source_path),
        "mode": args.mode,
        "segments": len(collision_meshes),
        "mass_kg": float(args.mass_kg),
        "com_local_mm": (1000.0 * com).tolist(),
        "inertia": tensor.tolist(),
        "inertia_axis_ratio": float(eigenvalues.max() / eigenvalues.min()),
        "visual_mesh_unchanged": True,
        "urdf": str(urdf_path),
        "output_parquet": str(output_path),
    }
    (output_root / "generation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
