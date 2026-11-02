#!/usr/bin/env python3
"""Build a single-object CoACD collision URDF with consistent inertia and support height.

The visual geometry and source reference are preserved.  CoACD produces explicit
convex collision parts, inertia is estimated from their voxelized union, and one
constant Z offset places the frame-0 collision proxy on the requested support
plane.  Hands and contact labels are deliberately left unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import coacd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog


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
    raise ValueError(f"missing {key}=... partition in {path}")


def _source_mass(path: Path) -> float:
    root = ET.parse(path).getroot()
    element = root.find(".//inertial/mass")
    mass = 0.3 if element is None else float(element.get("value", "0.3"))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"invalid source mass {mass} in {path}")
    return mass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repaired_part(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True, validate=True)
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise ValueError("CoACD returned a degenerate part")
    if not mesh.is_watertight or not mesh.is_volume:
        mesh = mesh.convex_hull
    if float(mesh.volume) < 0.0:
        mesh.invert()
    if not mesh.is_watertight or not mesh.is_volume or float(mesh.volume) <= 0.0:
        raise ValueError("CoACD part is not a positive watertight volume")
    return mesh


def _voxel_union_inertia(
    parts: list[trimesh.Trimesh], mass: float, pitch: float | None
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    vertices = np.concatenate([np.asarray(part.vertices) for part in parts], axis=0)
    extents = np.ptp(vertices, axis=0)
    if pitch is None:
        pitch = float(np.clip(float(extents.max()) / 96.0, 0.001, 0.003))
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("voxel pitch must be positive")
    origin = vertices.min(axis=0) - 2.0 * pitch
    occupied: set[tuple[int, int, int]] = set()
    for part in parts:
        points = np.asarray(part.voxelized(pitch).fill().points, dtype=np.float64)
        indices = np.rint((points - origin) / pitch).astype(np.int64)
        occupied.update(map(tuple, indices.tolist()))
    if not occupied:
        raise RuntimeError("collision voxel union is empty")
    indices = np.asarray(sorted(occupied), dtype=np.float64)
    centers = origin + indices * pitch
    com = centers.mean(axis=0)
    centered = centers - com
    point_mass = float(mass) / len(centers)
    squared = np.einsum("ni,ni->n", centered, centered)
    inertia = point_mass * (
        np.eye(3) * squared.sum() - np.einsum("ni,nj->ij", centered, centered)
    )
    # Add each voxel cube's own inertia about its center.
    inertia += np.eye(3) * (float(mass) * pitch * pitch / 6.0)
    eigenvalues = np.linalg.eigvalsh(inertia)
    if not np.all(np.isfinite(inertia)) or float(eigenvalues.min()) <= 0.0:
        raise ValueError("voxel-union inertia is not positive definite")
    return com, inertia, float(len(centers) * pitch**3), pitch, len(centers)


def _support_plate(
    parts: list[trimesh.Trimesh],
    rotation: Rotation,
    position: np.ndarray,
    *,
    patch_tolerance: float,
    depth: float,
    thickness: float,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Create a convex plate from the frame-0 bottom patch and return it in object-local coordinates."""
    if depth <= 0.0 or thickness <= 0.0 or thickness <= depth:
        raise ValueError("support plate requires 0 < depth < thickness")
    vertices_local = np.concatenate([np.asarray(part.vertices) for part in parts], axis=0)
    vertices_world = rotation.apply(vertices_local) + position
    minimum = float(vertices_world[:, 2].min())
    patch = vertices_world[vertices_world[:, 2] <= minimum + patch_tolerance]
    if len(patch) < 3:
        raise RuntimeError("support patch contains fewer than three vertices")
    hull = ConvexHull(patch[:, :2])
    polygon = patch[hull.vertices, :2]
    count = len(polygon)
    bottom_z = minimum - depth
    top_z = minimum + (thickness - depth)
    bottom = np.column_stack((polygon, np.full(count, bottom_z)))
    top = np.column_stack((polygon, np.full(count, top_z)))
    vertices_world = np.concatenate((bottom, top), axis=0)
    faces: list[list[int]] = []
    for index in range(1, count - 1):
        faces.append([0, index + 1, index])
        faces.append([count, count + index, count + index + 1])
    for index in range(count):
        following = (index + 1) % count
        faces.append([index, following, count + following])
        faces.append([index, count + following, count + index])
    vertices_local = rotation.inv().apply(vertices_world - position)
    plate = _repaired_part(vertices_local, np.asarray(faces, dtype=np.int64))
    return plate, {
        "patch_vertex_count": int(len(patch)),
        "polygon_vertex_count": int(count),
        "patch_hull_area_mm2": float(1.0e6 * hull.volume),
        "depth_below_proxy_mm": float(1000.0 * depth),
        "thickness_mm": float(1000.0 * thickness),
    }


def _support_cylinder(
    parts: list[trimesh.Trimesh],
    rotation: Rotation,
    position: np.ndarray,
    *,
    patch_tolerance: float,
    depth: float,
    thickness: float,
    radius_scale: float,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Fit a conservative cylinder inside the frame-0 bottom patch."""
    if depth <= 0.0 or thickness <= depth or not 0.0 < radius_scale <= 1.0:
        raise ValueError("support cylinder requires 0 < depth < thickness and radius scale in (0, 1]")
    vertices_local = np.concatenate([np.asarray(part.vertices) for part in parts], axis=0)
    vertices_world = rotation.apply(vertices_local) + position
    minimum = float(vertices_world[:, 2].min())
    patch = vertices_world[vertices_world[:, 2] <= minimum + patch_tolerance]
    if len(patch) < 3:
        raise RuntimeError("support patch contains fewer than three vertices")
    hull = ConvexHull(patch[:, :2])
    normals = hull.equations[:, :2]
    offsets = hull.equations[:, 2]
    norms = np.linalg.norm(normals, axis=1)
    result = linprog(
        c=np.asarray([0.0, 0.0, -1.0]),
        A_ub=np.column_stack((normals, norms)),
        b_ub=-offsets,
        bounds=((None, None), (None, None), (0.0, None)),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"support-cylinder Chebyshev-center fit failed: {result.message}")
    center_xy = np.asarray(result.x[:2], dtype=np.float64)
    radius = float(result.x[2]) * radius_scale
    if radius <= 0.0:
        raise RuntimeError("support-cylinder radius is nonpositive")
    bottom_z = minimum - depth
    center_world = np.asarray([center_xy[0], center_xy[1], bottom_z + 0.5 * thickness])
    world_mesh = trimesh.creation.cylinder(radius=radius, height=thickness, sections=32)
    world_mesh.apply_translation(center_world)
    vertices_local = rotation.inv().apply(np.asarray(world_mesh.vertices) - position)
    cylinder_local = _repaired_part(vertices_local, np.asarray(world_mesh.faces))
    local_center = rotation.inv().apply(center_world - position)
    local_rotation = rotation.inv()
    return cylinder_local, {
        "patch_vertex_count": int(len(patch)),
        "patch_hull_area_mm2": float(1.0e6 * hull.volume),
        "center_local_m": local_center.tolist(),
        "rpy_local_rad": local_rotation.as_euler("xyz").tolist(),
        "radius_m": radius,
        "height_m": thickness,
        "depth_below_proxy_mm": float(1000.0 * depth),
        "radius_scale": radius_scale,
    }


def _urdf(
    name: str,
    visual_filename: str,
    collision_filenames: list[str],
    mass: float,
    com: np.ndarray,
    inertia: np.ndarray,
    support_cylinder: dict[str, Any] | None,
) -> str:
    collisions = "\n".join(
        f'''    <collision name="collision_{index:03d}">
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="./{filename}" scale="1 1 1"/></geometry>
    </collision>'''
        for index, filename in enumerate(collision_filenames)
    )
    if support_cylinder is not None:
        xyz = support_cylinder["center_local_m"]
        rpy = support_cylinder["rpy_local_rad"]
        collisions += f'''\n    <collision name="support_cylinder">
      <origin xyz="{xyz[0]:.12g} {xyz[1]:.12g} {xyz[2]:.12g}" rpy="{rpy[0]:.12g} {rpy[1]:.12g} {rpy[2]:.12g}"/>
      <geometry><cylinder radius="{support_cylinder['radius_m']:.12g}" length="{support_cylinder['height_m']:.12g}"/></geometry>
    </collision>'''
    return f'''<?xml version="1.0"?>
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
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="./{visual_filename}" scale="1 1 1"/></geometry>
    </visual>
{collisions}
  </link>
</robot>
'''


def _support_metrics(
    vertices_local: np.ndarray,
    com_local: np.ndarray,
    rotation: Rotation,
    position: np.ndarray,
    table_height: float,
    tolerance: float,
) -> dict[str, Any]:
    vertices_world = rotation.apply(vertices_local) + position
    com_world = rotation.apply(com_local) + position
    bottom = vertices_world[vertices_world[:, 2] <= table_height + tolerance]
    area = 0.0
    inside = False
    margin = None
    if len(bottom) >= 3:
        hull = ConvexHull(bottom[:, :2])
        area = float(hull.volume)
        signed = hull.equations[:, :2] @ com_world[:2] + hull.equations[:, 2]
        inside = bool(np.all(signed <= 1.0e-9))
        margin = float(-signed.max())
    return {
        "minimum_gap_mm": float(1000.0 * (vertices_world[:, 2].min() - table_height)),
        "support_vertex_count": int(len(bottom)),
        "support_hull_area_mm2": float(1.0e6 * area),
        "com_world": com_world.tolist(),
        "com_projection_inside_support_hull": inside,
        "com_support_margin_mm": None if margin is None else float(1000.0 * margin),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-convex-hull", type=int, default=16)
    parser.add_argument("--preprocess-mode", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--preprocess-resolution", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--table-height", type=float, default=0.75)
    parser.add_argument("--support-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--voxel-pitch-mm", type=float, default=None)
    parser.add_argument("--minimum-part-volume-mm3", type=float, default=1.0)
    parser.add_argument("--support-plate-depth-mm", type=float, default=0.0)
    parser.add_argument("--support-plate-thickness-mm", type=float, default=2.0)
    parser.add_argument("--support-patch-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--support-cylinder-depth-mm", type=float, default=0.0)
    parser.add_argument("--support-cylinder-thickness-mm", type=float, default=2.5)
    parser.add_argument("--support-cylinder-radius-scale", type=float, default=0.9)
    args = parser.parse_args()

    source = args.parquet.resolve()
    support = args.support_usda.resolve()
    table, cells = _read_row(source)
    names = list(cells["object_body_names"])
    mesh_paths = [Path(value).resolve() for value in cells["object_mesh_paths"]]
    urdf_paths = [Path(value).resolve() for value in cells["object_urdf_paths"]]
    if len(names) != 1 or len(mesh_paths) != 1 or len(urdf_paths) != 1:
        raise ValueError("CoACD URDF preparation requires exactly one rigid object")
    mesh_path = mesh_paths[0]
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"invalid triangle mesh: {mesh_path}")

    coacd.set_log_level("error")
    raw_parts = coacd.run_coacd(
        coacd.Mesh(
            np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh.faces, dtype=np.int32),
        ),
        threshold=float(args.threshold),
        max_convex_hull=int(args.max_convex_hull),
        preprocess_mode=args.preprocess_mode,
        preprocess_resolution=int(args.preprocess_resolution),
        resolution=int(args.resolution),
        seed=int(args.seed),
        merge=True,
        decimate=True,
        max_ch_vertex=256,
    )
    minimum_volume = float(args.minimum_part_volume_mm3) * 1.0e-9
    parts = []
    discarded = []
    for index, (vertices, faces) in enumerate(raw_parts):
        part = _repaired_part(vertices, faces)
        if float(part.volume) <= minimum_volume:
            discarded.append({"index": index, "volume_mm3": float(part.volume * 1.0e9)})
        else:
            parts.append(part)
    if not parts:
        raise RuntimeError("CoACD returned no retained collision parts")

    position = np.asarray(cells["object_body_position"], dtype=np.float64)
    quaternion = np.asarray(cells["object_body_wxyz"], dtype=np.float64)
    rotation0 = Rotation.from_quat(quaternion[0, 0], scalar_first=True)
    support_plate_report = None
    support_cylinder_report = None
    support_cylinder_mesh = None
    if float(args.support_plate_depth_mm) > 0.0 and float(args.support_cylinder_depth_mm) > 0.0:
        raise ValueError("select either a support plate or a support cylinder, not both")
    if float(args.support_plate_depth_mm) > 0.0:
        plate, support_plate_report = _support_plate(
            parts,
            rotation0,
            position[0, 0],
            patch_tolerance=float(args.support_patch_tolerance_mm) / 1000.0,
            depth=float(args.support_plate_depth_mm) / 1000.0,
            thickness=float(args.support_plate_thickness_mm) / 1000.0,
        )
        parts.append(plate)

    if float(args.support_cylinder_depth_mm) > 0.0:
        support_cylinder_mesh, support_cylinder_report = _support_cylinder(
            parts,
            rotation0,
            position[0, 0],
            patch_tolerance=float(args.support_patch_tolerance_mm) / 1000.0,
            depth=float(args.support_cylinder_depth_mm) / 1000.0,
            thickness=float(args.support_cylinder_thickness_mm) / 1000.0,
            radius_scale=float(args.support_cylinder_radius_scale),
        )

    mass = _source_mass(urdf_paths[0])
    pitch = None if args.voxel_pitch_mm is None else float(args.voxel_pitch_mm) / 1000.0
    physical_parts = parts + ([] if support_cylinder_mesh is None else [support_cylinder_mesh])
    com, inertia, union_volume, pitch, voxel_count = _voxel_union_inertia(physical_parts, mass, pitch)
    output_root = args.output_root.resolve()
    asset = output_root / "assets" / names[0]
    asset.mkdir(parents=True, exist_ok=True)
    visual = asset / "visual.stl"
    mesh.export(visual)
    collision_files = []
    part_reports = []
    for index, part in enumerate(parts):
        filename = f"collision_part_{index:03d}.stl"
        path = asset / filename
        part.export(path)
        collision_files.append(filename)
        part_reports.append(
            {
                "index": index,
                "vertices": int(len(part.vertices)),
                "faces": int(len(part.faces)),
                "volume_mm3": float(part.volume * 1.0e9),
                "watertight": bool(part.is_watertight),
                "is_volume": bool(part.is_volume),
                "sha256": _sha256(path),
            }
        )
    urdf = asset / f"{names[0]}_coacd_rigid.urdf"
    urdf.write_text(
        _urdf(
            names[0],
            visual.name,
            collision_files,
            mass,
            com,
            inertia,
            support_cylinder_report,
        )
    )

    collision_vertices = np.concatenate([np.asarray(part.vertices) for part in physical_parts], axis=0)
    initial_minimum = float((rotation0.apply(collision_vertices) + position[0, 0])[:, 2].min())
    z_offset = float(args.table_height) - initial_minimum
    position[:, 0, 2] += z_offset
    output_cells = dict(cells)
    output_cells["object_body_position"] = position.tolist()
    output_cells["object_urdf_paths"] = [str(urdf)]
    if output_cells.get("object_root_position") is not None:
        root_position = np.asarray(output_cells["object_root_position"], dtype=np.float64)
        root_position[:, 2] += z_offset
        output_cells["object_root_position"] = root_position.tolist()

    sequence = _partition_value(source, "sequence_id")
    robot = _partition_value(source, "robot_name")
    output = (
        output_root
        / "selected_retarget"
        / f"sequence_id={sequence}"
        / f"robot_name={robot}"
        / "part-0.parquet"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([output_cells], schema=table.schema), output, compression="snappy")
    support_output = output_root / "reconstructed_stage" / f"{sequence}_support.usda"
    support_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(support, support_output)

    support_metrics = _support_metrics(
        collision_vertices,
        com,
        rotation0,
        position[0, 0],
        float(args.table_height),
        float(args.support_tolerance_mm) / 1000.0,
    )
    report = {
        "schema_version": "coacd-collision-inertia-support-candidate-v1",
        "source_parquet": str(source),
        "source_mesh": str(mesh_path),
        "source_mesh_sha256": _sha256(mesh_path),
        "source_mesh_watertight": bool(mesh.is_watertight),
        "source_unchanged": True,
        "visual_geometry_unchanged": True,
        "output_parquet": str(output),
        "output_urdf": str(urdf),
        "coacd": {
            "threshold": float(args.threshold),
            "max_convex_hull": int(args.max_convex_hull),
            "preprocess_mode": args.preprocess_mode,
            "preprocess_resolution": int(args.preprocess_resolution),
            "resolution": int(args.resolution),
            "seed": int(args.seed),
            "raw_part_count": len(raw_parts),
            "retained_part_count": len(parts),
            "discarded_parts": discarded,
            "parts": part_reports,
            "support_plate": support_plate_report,
            "support_cylinder": support_cylinder_report,
        },
        "mass_kg": mass,
        "voxel_union": {
            "pitch_mm": float(pitch * 1000.0),
            "voxel_count": voxel_count,
            "volume_mm3": float(union_volume * 1.0e9),
            "com_local_mm": (1000.0 * com).tolist(),
            "inertia_kg_m2": inertia.tolist(),
            "inertia_eigenvalues": np.linalg.eigvalsh(inertia).tolist(),
        },
        "support_pose": {
            "table_height_m": float(args.table_height),
            "frame0_z_offset_mm": float(1000.0 * z_offset),
            **support_metrics,
        },
        "hands_refined": False,
        "runtime_requirement": "set scene.decompose_objects=false to avoid a second CoACD pass",
    }
    (output_root / "generation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
