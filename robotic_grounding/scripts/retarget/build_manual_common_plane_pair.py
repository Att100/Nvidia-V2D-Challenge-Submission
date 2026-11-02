#!/usr/bin/env python3
"""Canonicalize a mislabeled pot/spoon pair and add a manual flat pot base.

This builder is intentionally non-destructive.  It takes one single-object pose
candidate for each semantic object, reorders the multi-object body axis, remaps
1-based contact part IDs, and writes canonical assets under a new output root.
The pot receives the same style of native thin-box support proxy and low center
of mass used by the accepted episode_000008 asset.
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
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def read_row(path: Path) -> tuple[pa.Schema, dict[str, Any]]:
    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ValueError(f"expected exactly one row: {path}")
    return table.schema, table.to_pylist()[0]


def partition(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}= partition in {path}")


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        raise ValueError(f"invalid mesh: {path}")
    return loaded


def candidate_track(path: Path, frames: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    _, row = read_row(path)
    position = np.asarray(row["object_body_position"], dtype=np.float64)
    quaternion = np.asarray(row["object_body_wxyz"], dtype=np.float64)
    if position.shape != (frames, 1, 3) or quaternion.shape != (frames, 1, 4):
        raise ValueError(f"single-object candidate has incompatible pose shape: {path}")
    return row, position[:, 0], quaternion[:, 0]


def inertial_from_urdf(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    inertial = root.find(".//inertial")
    if inertial is None:
        return 0.3, np.zeros(3), np.eye(3) * 0.001
    mass_node = inertial.find("mass")
    origin_node = inertial.find("origin")
    inertia_node = inertial.find("inertia")
    mass = 0.3 if mass_node is None else float(mass_node.get("value", "0.3"))
    com = np.zeros(3)
    if origin_node is not None:
        com = np.fromstring(origin_node.get("xyz", "0 0 0"), sep=" ")
    inertia = np.eye(3) * 0.001
    if inertia_node is not None:
        inertia = np.asarray(
            [
                [float(inertia_node.get("ixx", "0.001")), float(inertia_node.get("ixy", "0")), float(inertia_node.get("ixz", "0"))],
                [float(inertia_node.get("ixy", "0")), float(inertia_node.get("iyy", "0.001")), float(inertia_node.get("iyz", "0"))],
                [float(inertia_node.get("ixz", "0")), float(inertia_node.get("iyz", "0")), float(inertia_node.get("izz", "0.001"))],
            ]
        )
    return mass, com, inertia


def fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def write_urdf(
    path: Path,
    name: str,
    mass: float,
    com: np.ndarray,
    inertia: np.ndarray,
    *,
    box_center: np.ndarray | None = None,
    box_rpy: np.ndarray | None = None,
    box_size: np.ndarray | None = None,
) -> None:
    box = ""
    if box_center is not None and box_rpy is not None and box_size is not None:
        box = f"""
      <!-- Manually fitted horizontal pot-bottom proxy, following episode_000008. -->
      <collision>
        <origin xyz=\"{fmt(box_center)}\" rpy=\"{fmt(box_rpy)}\"/>
        <geometry><box size=\"{fmt(box_size)}\"/></geometry>
      </collision>"""
    text = f"""<?xml version=\"1.0\"?>
<robot name=\"{name}\">
  <mujoco><compiler meshdir=\"./\" balanceinertia=\"true\" discardvisual=\"false\"/></mujoco>
  <link name=\"object\">
    <inertial>
      <origin xyz=\"{fmt(com)}\" rpy=\"0 0 0\"/>
      <mass value=\"{mass:.10g}\"/>
      <inertia ixx=\"{inertia[0,0]:.10g}\" ixy=\"{inertia[0,1]:.10g}\" ixz=\"{inertia[0,2]:.10g}\"
               iyy=\"{inertia[1,1]:.10g}\" iyz=\"{inertia[1,2]:.10g}\" izz=\"{inertia[2,2]:.10g}\"/>
    </inertial>
    <visual><origin xyz=\"0 0 0\" rpy=\"0 0 0\"/><geometry><mesh filename=\"./{name}_visual.stl\"/></geometry></visual>
    <collision><origin xyz=\"0 0 0\" rpy=\"0 0 0\"/><geometry><mesh filename=\"./{name}_visual.stl\"/></geometry></collision>{box}
  </link>
</robot>
"""
    path.write_text(text)


def write_primitive_urdf(
    path: Path,
    name: str,
    com: np.ndarray,
    collisions: list[dict[str, Any]],
) -> None:
    blocks = []
    for index, collision in enumerate(collisions):
        if collision["type"] == "box":
            proxy = trimesh.creation.box(extents=np.asarray(collision["size"], dtype=np.float64))
        elif collision["type"] == "cylinder":
            proxy = trimesh.creation.cylinder(
                radius=float(collision["radius"]),
                height=float(collision["length"]),
                sections=32,
            )
        else:
            raise ValueError(f"unsupported primitive {collision['type']}")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", collision["rpy"]).as_matrix()
        transform[:3, 3] = np.asarray(collision["center"], dtype=np.float64)
        proxy.apply_transform(transform)
        filename = f"collision_proxy_{index:03d}.stl"
        proxy.export(path.parent / filename)
        blocks.append(
            f'<collision><origin xyz="0 0 0" rpy="0 0 0"/>'
            f'<geometry><mesh filename="./{filename}"/></geometry></collision>'
        )
    path.write_text(
        f'''<?xml version="1.0"?>
<robot name="{name}">
  <mujoco><compiler meshdir="./" balanceinertia="true" discardvisual="false"/></mujoco>
  <link name="object">
    <inertial><origin xyz="{fmt(com)}" rpy="0 0 0"/><mass value="0.3"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="./{name}_visual.stl"/></geometry></visual>
    {''.join(blocks)}
  </link>
</robot>
'''
    )


def world_pose_to_local(
    center_world: np.ndarray, rotation_world: Rotation, position: np.ndarray, rotation: Rotation
) -> tuple[np.ndarray, np.ndarray]:
    center = rotation.inv().apply(center_world - position)
    rpy = (rotation.inv() * rotation_world).as_euler("xyz")
    return center, rpy


def primitive_proxy(
    name: str,
    mesh: trimesh.Trimesh,
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    table_height: float,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    rotation = Rotation.from_quat(quaternion_wxyz, scalar_first=True)
    world = rotation.apply(np.asarray(mesh.vertices)) + position
    minimum = float(world[:, 2].min())
    patch = world[world[:, 2] <= minimum + 0.005]
    hull = ConvexHull(patch[:, :2])
    normals = hull.equations[:, :2]
    offsets = hull.equations[:, 2]
    fit = linprog(
        [0.0, 0.0, -1.0],
        A_ub=np.column_stack((normals, np.linalg.norm(normals, axis=1))),
        b_ub=-offsets,
        bounds=((None, None), (None, None), (0.0, None)),
        method="highs",
    )
    if not fit.success:
        raise RuntimeError(f"primitive proxy bottom fit failed: {fit.message}")
    support_center = np.asarray([fit.x[0], fit.x[1]])
    collisions: list[dict[str, Any]] = []
    identity = Rotation.identity()
    if name == "white_pot":
        radius = float(min(0.070, 0.90 * fit.x[2]))
        height = 0.080
        body_world = np.asarray([*support_center, table_height + height / 2.0])
        body_center, body_rpy = world_pose_to_local(body_world, identity, position, rotation)
        collisions.append({"type": "cylinder", "center": body_center, "rpy": body_rpy, "radius": radius, "length": height})
        relative = world[:, :2] - support_center
        distance = np.linalg.norm(relative, axis=1)
        direction = relative[int(np.argmax(distance))]
        direction /= np.linalg.norm(direction)
        far = float(distance.max())
        start = radius * 0.65
        end = max(start + 0.04, far * 0.95)
        length = end - start
        yaw = math.atan2(direction[1], direction[0])
        handle_world = np.asarray([*(support_center + direction * (start + end) / 2.0), table_height + 0.045])
        handle_rotation = Rotation.from_euler("z", yaw)
        handle_center, handle_rpy = world_pose_to_local(handle_world, handle_rotation, position, rotation)
        collisions.append({"type": "box", "center": handle_center, "rpy": handle_rpy, "size": [length, 0.030, 0.030]})
        com_world = np.asarray([*support_center, table_height + 0.040])
    else:
        xy = world[:, :2]
        xy_center = xy.mean(axis=0)
        covariance = np.cov(xy - xy_center, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        yaw = math.atan2(long_axis[1], long_axis[0])
        axes = np.column_stack((long_axis, np.asarray([-long_axis[1], long_axis[0]])))
        projected = (xy - xy_center) @ axes
        low = np.quantile(projected, 0.005, axis=0)
        high = np.quantile(projected, 0.995, axis=0)
        center_xy = xy_center + axes @ ((low + high) / 2.0)
        size_xy = high - low
        support_radius = float(max(0.008, 0.80 * fit.x[2]))
        support_height = 0.020
        support_world = np.asarray([*support_center, table_height + support_height / 2.0])
        support_local, support_rpy = world_pose_to_local(
            support_world, Rotation.identity(), position, rotation
        )
        collisions.append(
            {
                "type": "cylinder",
                "center": support_local,
                "rpy": support_rpy,
                "radius": support_radius,
                "length": support_height,
            }
        )
        height = max(0.012, min(0.025, float(np.ptp(world[:, 2])) * 0.8))
        box_world = np.asarray([*center_xy, table_height + 0.005 + height / 2.0])
        box_rotation = Rotation.from_euler("z", yaw)
        box_center, box_rpy = world_pose_to_local(box_world, box_rotation, position, rotation)
        collisions.append({"type": "box", "center": box_center, "rpy": box_rpy, "size": [float(size_xy[0]), float(size_xy[1]), height]})
        com_world = np.asarray([*support_center, table_height + 0.015])
    com_local = rotation.inv().apply(com_world - position)
    serializable = []
    for item in collisions:
        serializable.append({key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in item.items()})
    return com_local, collisions, {
        "type": "manual_primitive_proxy",
        "mesh_frame0_min_z_m": minimum,
        "bottom_patch_inscribed_radius_mm": float(fit.x[2] * 1000.0),
        "collisions": serializable,
        "com_world_m": com_world.tolist(),
    }


def copy_explicit_urdf(source: Path, destination: Path, name: str) -> None:
    """Copy a pre-decomposed URDF and every relative mesh it references."""
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("name", name)
    for node in root.findall(".//mesh"):
        filename = node.get("filename")
        if not filename:
            continue
        mesh_source = Path(filename)
        if not mesh_source.is_absolute():
            mesh_source = source.parent / mesh_source
        mesh_source = mesh_source.resolve()
        mesh_destination = destination.parent / mesh_source.name
        shutil.copy2(mesh_source, mesh_destination)
        node.set("filename", f"./{mesh_source.name}")
    tree.write(destination, encoding="unicode", xml_declaration=True)


def flat_box_fit(
    mesh: trimesh.Trimesh,
    position: np.ndarray,
    quaternion_wxyz: np.ndarray,
    table_height: float,
    side: float,
    height: float,
    com_height: float,
    box_bottom_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rotation = Rotation.from_quat(quaternion_wxyz, scalar_first=True)
    world = rotation.apply(np.asarray(mesh.vertices)) + position
    minimum = float(world[:, 2].min())
    patch = world[world[:, 2] <= minimum + 0.005]
    hull = ConvexHull(patch[:, :2])
    normals = hull.equations[:, :2]
    offsets = hull.equations[:, 2]
    norms = np.linalg.norm(normals, axis=1)
    fit = linprog(
        [0.0, 0.0, -1.0],
        A_ub=np.column_stack((normals, norms)),
        b_ub=-offsets,
        bounds=((None, None), (None, None), (0.0, None)),
        method="highs",
    )
    if not fit.success:
        raise RuntimeError(f"bottom-patch fit failed: {fit.message}")
    radius = float(fit.x[2])
    if side * math.sqrt(0.5) > radius:
        raise ValueError(
            f"requested {side:g} m square does not fit bottom patch radius {radius:g} m"
        )
    # episode_000008's accepted proxy is an internal pan-bottom surface: its
    # lower face is about 6 mm above the reconstructed mesh minimum.  Keeping
    # it off the support avoids a duplicate table contact manifold.
    center_world = np.asarray(
        [fit.x[0], fit.x[1], minimum + box_bottom_offset + 0.5 * height]
    )
    center_local = rotation.inv().apply(center_world - position)
    box_rpy_local = rotation.inv().as_euler("xyz")
    com_world = np.asarray([fit.x[0], fit.x[1], table_height + com_height])
    com_local = rotation.inv().apply(com_world - position)
    report = {
        "bottom_patch_tolerance_mm": 5.0,
        "bottom_patch_points": int(len(patch)),
        "bottom_patch_hull_area_cm2": float(hull.volume * 1.0e4),
        "bottom_patch_inscribed_radius_mm": float(radius * 1000.0),
        "box_world_center_m": center_world.tolist(),
        "box_world_rpy_rad": [0.0, 0.0, 0.0],
        "box_size_m": [side, side, height],
        "box_bottom_offset_from_mesh_min_mm": float(box_bottom_offset * 1000.0),
        "com_world_m": com_world.tolist(),
        "mesh_frame0_min_z_m": minimum,
    }
    return center_local, box_rpy_local, com_local, report


def remap_contact_ids(values: Any, old_pot_index: int, old_spoon_index: int) -> Any:
    array = np.asarray(values, dtype=np.int64)
    output = array.copy()
    output[array == old_pot_index + 1] = 1
    output[array == old_spoon_index + 1] = 2
    return output.tolist()


def rotation_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    ra = Rotation.from_quat(a, scalar_first=True)
    rb = Rotation.from_quat(b, scalar_first=True)
    return float(np.degrees((ra.inv() * rb).magnitude()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--pot-candidate", type=Path, required=True)
    parser.add_argument("--spoon-candidate", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--old-pot-index", type=int, default=1)
    parser.add_argument("--old-spoon-index", type=int, default=0)
    parser.add_argument("--table-height", type=float, default=0.75)
    parser.add_argument("--pot-box-side", type=float, default=0.10)
    parser.add_argument("--pot-box-height", type=float, default=0.004)
    parser.add_argument("--pot-box-bottom-offset", type=float, default=0.006)
    parser.add_argument(
        "--disable-pot-box",
        action="store_true",
        help="Keep the manual low COM but omit the pot box collision proxy.",
    )
    parser.add_argument("--pot-com-height", type=float, default=0.045)
    parser.add_argument(
        "--preserve-candidate-collision-urdfs",
        action="store_true",
        help="Copy explicit pre-decomposed collision URDFs from both candidates; do not add the native pot box.",
    )
    parser.add_argument(
        "--primitive-collision-proxies",
        action="store_true",
        help="Use deterministic hand-fitted primitive collisions for pot and spoon.",
    )
    parser.add_argument(
        "--pot-primitive-collision",
        action="store_true",
        help="Use the deterministic primitive proxy only for the pot; retain the spoon mesh collision.",
    )
    parser.add_argument(
        "--keep-source-spoon-track",
        action="store_true",
        help="Keep the source spoon SE(3) trajectory while using the candidate collision asset.",
    )
    args = parser.parse_args()

    source = args.source_parquet.resolve()
    schema, row = read_row(source)
    source_position = np.asarray(row["object_body_position"], dtype=np.float64)
    source_quaternion = np.asarray(row["object_body_wxyz"], dtype=np.float64)
    frames, bodies, _ = source_position.shape
    if bodies != 2 or {args.old_pot_index, args.old_spoon_index} != {0, 1}:
        raise ValueError("this builder requires exactly two objects with distinct indices 0 and 1")

    pot_row, pot_position, pot_quaternion = candidate_track(args.pot_candidate.resolve(), frames)
    spoon_row, spoon_position, spoon_quaternion = candidate_track(args.spoon_candidate.resolve(), frames)
    if args.keep_source_spoon_track:
        spoon_position = source_position[:, args.old_spoon_index].copy()
        spoon_quaternion = source_quaternion[:, args.old_spoon_index].copy()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    assets = output_root / "assets"
    assets.mkdir(parents=True)

    semantic = [
        ("white_pot", pot_row, pot_position, pot_quaternion),
        ("wooden_spoon", spoon_row, spoon_position, spoon_quaternion),
    ]
    mesh_paths: list[str] = []
    urdf_paths: list[str] = []
    radii: list[float] = []
    asset_reports: list[dict[str, Any]] = []
    for new_index, (name, candidate, position, quaternion) in enumerate(semantic):
        asset_dir = assets / name
        asset_dir.mkdir()
        source_mesh = Path(candidate["object_mesh_paths"][0]).resolve()
        mesh = load_mesh(source_mesh)
        obj_path = asset_dir / f"{name}.obj"
        stl_path = asset_dir / f"{name}_visual.stl"
        mesh.export(obj_path)
        mesh.export(stl_path)
        mass, com, inertia = inertial_from_urdf(Path(candidate["object_urdf_paths"][0]))
        proxy_report = None
        kwargs: dict[str, Any] = {}
        primitive_report = None
        primitive_collisions = None
        use_primitive = args.primitive_collision_proxies or (
            args.pot_primitive_collision and new_index == 0
        )
        if use_primitive:
            com, primitive_collisions, primitive_report = primitive_proxy(
                name, mesh, position[0], quaternion[0], args.table_height
            )
            mass = 0.3
            inertia = np.eye(3) * 0.001
        elif new_index == 0 and not args.preserve_candidate_collision_urdfs:
            center, rpy, com, proxy_report = flat_box_fit(
                mesh,
                position[0],
                quaternion[0],
                args.table_height,
                args.pot_box_side,
                args.pot_box_height,
                args.pot_com_height,
                args.pot_box_bottom_offset,
            )
            # Match the accepted 000008 hand-tuned asset rather than trusting a
            # non-watertight reconstruction's volume-derived tensor.
            mass = 0.3
            inertia = np.eye(3) * 0.001
            if not args.disable_pot_box:
                kwargs = {
                    "box_center": center,
                    "box_rpy": rpy,
                    "box_size": np.asarray([args.pot_box_side, args.pot_box_side, args.pot_box_height]),
                }
            else:
                proxy_report["collision_enabled"] = False
        urdf = asset_dir / f"{name}_rigid.urdf"
        if use_primitive:
            assert primitive_collisions is not None
            write_primitive_urdf(urdf, name, com, primitive_collisions)
        elif args.preserve_candidate_collision_urdfs:
            copy_explicit_urdf(Path(candidate["object_urdf_paths"][0]).resolve(), urdf, name)
        else:
            write_urdf(urdf, name, mass, com, inertia, **kwargs)
        mesh_paths.append(str(obj_path))
        urdf_paths.append(str(urdf))
        radii.append(float(candidate["object_mesh_radius"][0]))
        asset_reports.append(
            {
                "new_body_index": new_index,
                "semantic_name": name,
                "candidate_parquet": str((args.pot_candidate if new_index == 0 else args.spoon_candidate).resolve()),
                "source_mesh": str(source_mesh),
                "output_mesh": str(obj_path),
                "output_urdf": str(urdf),
                "mass_kg": mass,
                "com_local_m": com.tolist(),
                "inertia_kg_m2": inertia.tolist(),
                "pot_support_proxy": proxy_report,
                "primitive_proxy": primitive_report,
                "collision_representation": (
                    "manual_primitive_proxy"
                    if use_primitive
                    else "candidate_explicit_convex_parts"
                    if args.preserve_candidate_collision_urdfs
                    else (
                        "runtime_mesh_decomposition_low_com_no_box"
                        if args.disable_pot_box
                        else "runtime_mesh_decomposition_plus_native_pot_box"
                    )
                ),
            }
        )

    position = np.stack((pot_position, spoon_position), axis=1)
    quaternion = np.stack((pot_quaternion, spoon_quaternion), axis=1)
    output = dict(row)
    output["object_body_names"] = ["white_pot", "wooden_spoon"]
    output["safe_object_body_names"] = ["white_pot", "wooden_spoon"]
    output["object_body_position"] = position.tolist()
    output["object_body_wxyz"] = quaternion.tolist()
    output["object_mesh_paths"] = mesh_paths
    output["object_urdf_paths"] = urdf_paths
    output["object_mesh_radius"] = radii
    output["object_root_position"] = position[:, 0].tolist()
    output["object_root_axis_angle"] = Rotation.from_quat(
        quaternion[:, 0], scalar_first=True
    ).as_rotvec().tolist()
    remapped_columns: list[str] = []
    for hand in ("left", "right"):
        key = f"mano_{hand}_object_contact_part_ids"
        if key in output:
            output[key] = remap_contact_ids(
                output[key], args.old_pot_index, args.old_spoon_index
            )
            remapped_columns.append(key)

    sequence = partition(source, "sequence_id")
    robot = partition(source, "robot_name")
    parquet_out = (
        output_root
        / "selected_retarget"
        / f"sequence_id={sequence}"
        / f"robot_name={robot}"
        / "part-0.parquet"
    )
    parquet_out.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([output], schema=schema), parquet_out, compression="snappy")
    support_out = output_root / "reconstructed_stage" / f"{sequence}_support.usda"
    support_out.parent.mkdir(parents=True)
    shutil.copy2(args.support_usda.resolve(), support_out)

    old_names = list(row["object_body_names"])
    report = {
        "schema_version": "manual-common-plane-pair-v1",
        "status": "GENERATED_FOR_PHYSICS_VALIDATION",
        "source_parquet": str(source),
        "source_unchanged": True,
        "output_parquet": str(parquet_out),
        "support_usda": str(support_out),
        "mapping": {
            f"old_body_{args.old_pot_index}": {"old_name": old_names[args.old_pot_index], "new_body": 0, "canonical_name": "white_pot"},
            f"old_body_{args.old_spoon_index}": {"old_name": old_names[args.old_spoon_index], "new_body": 1, "canonical_name": "wooden_spoon"},
        },
        "contact_part_id_columns_remapped": remapped_columns,
        "contact_part_id_map": {str(args.old_pot_index + 1): 1, str(args.old_spoon_index + 1): 2},
        "common_support_plane_height_m": args.table_height,
        "runtime_decompose_objects": not args.preserve_candidate_collision_urdfs,
        "source_spoon_track_preserved": args.keep_source_spoon_track,
        "objects": asset_reports,
        "frame0_corrections": [
            {
                "name": "white_pot",
                "translation_mm": (1000.0 * (position[0, 0] - source_position[0, args.old_pot_index])).tolist(),
                "rotation_deg": rotation_delta_deg(source_quaternion[0, args.old_pot_index], quaternion[0, 0]),
            },
            {
                "name": "wooden_spoon",
                "translation_mm": (1000.0 * (position[0, 1] - source_position[0, args.old_spoon_index])).tolist(),
                "rotation_deg": rotation_delta_deg(source_quaternion[0, args.old_spoon_index], quaternion[0, 1]),
            },
        ],
        "hands_refined": False,
        "contact_positions_refined": False,
        "warning": "object-only correction; optimize hands/contact only after joint physics acceptance",
    }
    (output_root / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
