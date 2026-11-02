#!/usr/bin/env python3
"""Replace a solid cylindrical pot collision proxy with an explicit hollow proxy.

The reference trajectory and all hand data are copied verbatim.  Only asset paths
are redirected to a new, self-contained asset directory whose pot collision is a
thin bottom, a segmented wall, and the original handle proxy.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def fmt(values: np.ndarray | list[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def resolve_mesh(source_urdf: Path, filename: str) -> Path:
    path = Path(filename)
    if not path.is_absolute():
        path = source_urdf.parent / path
    return path.resolve()


def copy_urdf_with_assets(source: Path, destination: Path, name: str) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("name", name)
    for node in root.findall(".//mesh"):
        filename = node.get("filename")
        if not filename:
            continue
        mesh_source = resolve_mesh(source, filename)
        mesh_destination = destination.parent / mesh_source.name
        shutil.copy2(mesh_source, mesh_destination)
        node.set("filename", f"./{mesh_source.name}")
    tree.write(destination, encoding="unicode", xml_declaration=True)


def inertial_xml(source_urdf: Path, origin: np.ndarray | None = None) -> str:
    inertial = ET.parse(source_urdf).getroot().find(".//link/inertial")
    if inertial is None:
        raise ValueError(f"missing inertial element in {source_urdf}")
    if origin is not None:
        origin_node = inertial.find("origin")
        if origin_node is None:
            origin_node = ET.SubElement(inertial, "origin")
        origin_node.set("xyz", fmt(origin))
        origin_node.set("rpy", "0 0 0")
    return ET.tostring(inertial, encoding="unicode")


def visual_mesh(source_urdf: Path) -> Path:
    node = ET.parse(source_urdf).getroot().find(".//link/visual/geometry/mesh")
    if node is None or not node.get("filename"):
        raise ValueError(f"missing visual mesh in {source_urdf}")
    return resolve_mesh(source_urdf, node.get("filename", ""))


def collision_xml(kind: str, center: np.ndarray, rotation: Rotation, **geometry: float | list[float]) -> str:
    if kind == "box":
        body = f'<box size="{fmt(geometry["size"])}"/>'
    elif kind == "cylinder":
        body = (
            f'<cylinder radius="{float(geometry["radius"]):.12g}" '
            f'length="{float(geometry["length"]):.12g}"/>'
        )
    else:
        raise ValueError(f"unsupported collision primitive: {kind}")
    rpy = rotation.as_euler("xyz")
    return (
        f'    <collision><origin xyz="{fmt(center)}" rpy="{fmt(rpy)}"/>'
        f"<geometry>{body}</geometry></collision>"
    )


def write_hollow_pot_urdf(
    destination: Path,
    source_urdf: Path,
    name: str,
    cylinder: dict,
    handles: list[dict],
    *,
    inner_radius: float,
    outer_radius_override: float | None,
    bottom_thickness: float,
    wall_segments: int,
    overlap: float,
    bottom_box_side: float | None,
    com_height_from_bottom: float | None,
    handle_length_scale: float,
    handle_cross_section_scale: float,
    wall_base_center: np.ndarray | None,
    wall_rotation: Rotation | None,
    wall_height_override: float | None,
) -> dict:
    source_outer_radius = float(cylinder["radius"])
    outer_radius = source_outer_radius if outer_radius_override is None else outer_radius_override
    height = float(cylinder["length"])
    if not 0.0 < inner_radius < outer_radius:
        raise ValueError(f"inner radius must be in (0, {outer_radius}), got {inner_radius}")
    if not 0.0 < bottom_thickness < height:
        raise ValueError(f"bottom thickness must be in (0, {height}), got {bottom_thickness}")
    if wall_segments != 0 and wall_segments < 4:
        raise ValueError("wall segments must be zero (open proxy) or at least 4")
    if handle_length_scale <= 0.0 or handle_cross_section_scale <= 0.0:
        raise ValueError("handle scales must be positive")

    cylinder_center = np.asarray(cylinder["center"], dtype=np.float64)
    cylinder_rotation = Rotation.from_euler("xyz", cylinder["rpy"])
    wall_thickness = outer_radius - inner_radius
    wall_center_radius = 0.5 * (outer_radius + inner_radius)
    wall_height = (
        height - bottom_thickness
        if wall_height_override is None
        else wall_height_override
    )
    if wall_height <= 0.0:
        raise ValueError(f"wall height must be positive, got {wall_height}")
    tangent_length = (
        2.0 * outer_radius * math.tan(math.pi / wall_segments) * overlap
        if wall_segments
        else 0.0
    )

    collisions: list[str] = []
    bottom_local = np.asarray([0.0, 0.0, -height / 2.0 + bottom_thickness / 2.0])
    bottom_center = cylinder_center + cylinder_rotation.apply(bottom_local)
    if bottom_box_side is None:
        collisions.append(
            collision_xml(
                "cylinder",
                bottom_center,
                cylinder_rotation,
                radius=outer_radius,
                length=bottom_thickness,
            )
        )
        bottom_geometry = {"type": "cylinder", "radius_m": outer_radius}
    else:
        if not 0.0 < bottom_box_side <= 2.0 * inner_radius:
            raise ValueError(
                f"bottom box side must be in (0, {2.0 * inner_radius}], got {bottom_box_side}"
            )
        collisions.append(
            collision_xml(
                "box",
                bottom_center,
                cylinder_rotation,
                size=[bottom_box_side, bottom_box_side, bottom_thickness],
            )
        )
        bottom_geometry = {"type": "box", "side_m": bottom_box_side}
    if wall_base_center is None:
        wall_base_center = cylinder_center + cylinder_rotation.apply(
            np.asarray([0.0, 0.0, -height / 2.0])
        )
    if wall_rotation is None:
        wall_rotation = cylinder_rotation
    wall_center = wall_base_center + wall_rotation.apply(
        np.asarray([0.0, 0.0, wall_height / 2.0])
    )
    for index in range(wall_segments):
        angle = 2.0 * math.pi * index / wall_segments
        radial = np.asarray(
            [wall_center_radius * math.cos(angle), wall_center_radius * math.sin(angle), 0.0]
        )
        center = wall_center + wall_rotation.apply(radial)
        rotation = wall_rotation * Rotation.from_euler("z", angle)
        collisions.append(
            collision_xml(
                "box",
                center,
                rotation,
                size=[wall_thickness, tangent_length, wall_height],
            )
        )
    for handle in handles:
        handle_geometry = {
            key: value for key, value in handle.items() if key not in {"type", "center", "rpy"}
        }
        if handle["type"] == "box":
            size = np.asarray(handle_geometry["size"], dtype=np.float64)
            size[0] *= handle_length_scale
            size[1:] *= handle_cross_section_scale
            handle_geometry["size"] = size.tolist()
        collisions.append(
            collision_xml(
                handle["type"],
                np.asarray(handle["center"], dtype=np.float64),
                Rotation.from_euler("xyz", handle["rpy"]),
                **handle_geometry,
            )
        )

    visual_source = visual_mesh(source_urdf)
    visual_destination = destination.parent / visual_source.name
    shutil.copy2(visual_source, visual_destination)
    inertial_origin = None
    if com_height_from_bottom is not None:
        if not 0.0 < com_height_from_bottom < height:
            raise ValueError(
                f"COM height from bottom must be in (0, {height}), got {com_height_from_bottom}"
            )
        com_offset = np.asarray([0.0, 0.0, -height / 2.0 + com_height_from_bottom])
        inertial_origin = cylinder_center + cylinder_rotation.apply(com_offset)
    destination.write_text(
        "\n".join(
            [
                '<?xml version="1.0"?>',
                f'<robot name="{name}">',
                '  <mujoco><compiler meshdir="./" balanceinertia="true" discardvisual="false"/></mujoco>',
                '  <link name="object">',
                f"    {inertial_xml(source_urdf, inertial_origin)}",
                (
                    '    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry>'
                    f'<mesh filename="./{visual_destination.name}"/></geometry></visual>'
                ),
                *collisions,
                "  </link>",
                "</robot>",
                "",
            ]
        )
    )
    return {
        "outer_radius_m": outer_radius,
        "source_solid_radius_m": source_outer_radius,
        "inner_radius_m": inner_radius,
        "wall_thickness_m": wall_thickness,
        "height_m": height,
        "bottom_thickness_m": bottom_thickness,
        "bottom_geometry": bottom_geometry,
        "com_height_from_bottom_m": com_height_from_bottom,
        "inertial_origin_local_m": inertial_origin.tolist() if inertial_origin is not None else None,
        "wall_segments": wall_segments,
        "wall_height_m": wall_height,
        "wall_base_center_local_m": wall_base_center.tolist(),
        "wall_axis_local": wall_rotation.apply([0.0, 0.0, 1.0]).tolist(),
        "wall_tangent_length_m": tangent_length,
        "handle_primitives": len(handles),
        "handle_length_scale": handle_length_scale,
        "handle_cross_section_scale": handle_cross_section_scale,
        "total_collision_primitives": 1 + wall_segments + len(handles),
    }


def partition_value(path: Path, prefix: str) -> str:
    for parent in (path, *path.parents):
        if parent.name.startswith(prefix + "="):
            return parent.name.split("=", 1)[1]
    raise ValueError(f"cannot find {prefix}=... in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--source-build-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pot-name", default="white_pot")
    parser.add_argument("--inner-radius", type=float, default=0.062)
    parser.add_argument(
        "--outer-radius",
        type=float,
        help="Override the old solid-cylinder radius for the hollow wall.",
    )
    parser.add_argument("--bottom-thickness", type=float, default=0.006)
    parser.add_argument("--wall-segments", type=int, default=16)
    parser.add_argument("--wall-overlap", type=float, default=1.05)
    parser.add_argument(
        "--bottom-box-side",
        type=float,
        help="Use a square bottom support plate of this side length instead of a cylinder.",
    )
    parser.add_argument(
        "--com-height-from-bottom",
        type=float,
        help="Place the pot COM this far above the proxy bottom (in meters).",
    )
    parser.add_argument("--handle-length-scale", type=float, default=1.0)
    parser.add_argument("--handle-cross-section-scale", type=float, default=1.0)
    parser.add_argument(
        "--wall-base-center-world",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Optional frame-0 world-space center of the vessel wall's bottom plane.",
    )
    parser.add_argument(
        "--wall-axis-world",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Optional frame-0 world-space vessel axis; requires --wall-base-center-world.",
    )
    parser.add_argument("--wall-height", type=float)
    parser.add_argument("--support-usda", type=Path)
    args = parser.parse_args()

    source_parquet = args.source_parquet.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    table = pq.read_table(source_parquet)
    if table.num_rows != 1:
        raise ValueError(f"expected one episode row, found {table.num_rows}")
    row = table.to_pylist()[0]
    names = list(row["object_body_names"])
    if args.pot_name not in names:
        raise ValueError(f"{args.pot_name} not found in {names}")
    pot_index = names.index(args.pot_name)
    if (args.wall_base_center_world is None) != (args.wall_axis_world is None):
        raise ValueError(
            "--wall-base-center-world and --wall-axis-world must be provided together"
        )

    wall_base_center_local = None
    wall_rotation_local = None
    if args.wall_base_center_world is not None:
        frame0_position = np.asarray(row["object_body_position"], dtype=np.float64)[0, pot_index]
        frame0_rotation = Rotation.from_quat(
            np.asarray(row["object_body_wxyz"], dtype=np.float64)[0, pot_index],
            scalar_first=True,
        )
        wall_base_center_local = frame0_rotation.inv().apply(
            np.asarray(args.wall_base_center_world, dtype=np.float64) - frame0_position
        )
        axis_world = np.asarray(args.wall_axis_world, dtype=np.float64)
        axis_world /= np.linalg.norm(axis_world)
        x_world = np.asarray([1.0, 0.0, 0.0])
        x_world -= axis_world * np.dot(x_world, axis_world)
        if np.linalg.norm(x_world) < 1.0e-8:
            x_world = np.asarray([0.0, 1.0, 0.0])
            x_world -= axis_world * np.dot(x_world, axis_world)
        x_world /= np.linalg.norm(x_world)
        y_world = np.cross(axis_world, x_world)
        wall_world_rotation = Rotation.from_matrix(
            np.column_stack((x_world, y_world, axis_world))
        )
        wall_rotation_local = frame0_rotation.inv() * wall_world_rotation

    source_report = json.loads(args.source_build_report.read_text())
    object_report = next(
        item for item in source_report["objects"] if item["semantic_name"] == args.pot_name
    )
    primitives = object_report.get("primitive_proxy", {}).get("collisions", [])
    cylinders = [item for item in primitives if item["type"] == "cylinder"]
    if len(cylinders) != 1:
        raise ValueError(f"expected exactly one solid cylinder proxy, found {len(cylinders)}")
    handles = [item for item in primitives if item["type"] != "cylinder"]

    assets = output_root / "assets"
    assets.mkdir(parents=True)
    new_mesh_paths: list[str] = []
    new_urdf_paths: list[str] = []
    hollow_report = None
    for index, name in enumerate(names):
        asset_dir = assets / name
        asset_dir.mkdir()
        source_mesh = Path(row["object_mesh_paths"][index]).resolve()
        mesh_destination = asset_dir / f"{name}{source_mesh.suffix}"
        shutil.copy2(source_mesh, mesh_destination)
        source_urdf = Path(row["object_urdf_paths"][index]).resolve()
        urdf_destination = asset_dir / f"{name}_rigid.urdf"
        if index == pot_index:
            hollow_report = write_hollow_pot_urdf(
                urdf_destination,
                source_urdf,
                name,
                cylinders[0],
                handles,
                inner_radius=args.inner_radius,
                outer_radius_override=args.outer_radius,
                bottom_thickness=args.bottom_thickness,
                wall_segments=args.wall_segments,
                overlap=args.wall_overlap,
                bottom_box_side=args.bottom_box_side,
                com_height_from_bottom=args.com_height_from_bottom,
                handle_length_scale=args.handle_length_scale,
                handle_cross_section_scale=args.handle_cross_section_scale,
                wall_base_center=wall_base_center_local,
                wall_rotation=wall_rotation_local,
                wall_height_override=args.wall_height,
            )
        else:
            copy_urdf_with_assets(source_urdf, urdf_destination, name)
        new_mesh_paths.append(str(mesh_destination))
        new_urdf_paths.append(str(urdf_destination))

    output = dict(row)
    output["object_mesh_paths"] = new_mesh_paths
    output["object_urdf_paths"] = new_urdf_paths
    sequence = partition_value(source_parquet, "sequence_id")
    robot = partition_value(source_parquet, "robot_name")
    output_parquet = (
        output_root / "selected_retarget" / f"sequence_id={sequence}" / f"robot_name={robot}" / "part-0.parquet"
    )
    output_parquet.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([output], schema=table.schema), output_parquet, compression="snappy")

    support_source = args.support_usda
    if support_source is not None:
        support_source = support_source.resolve()
        support_destination = output_root / "reconstructed_stage" / support_source.name
        support_destination.parent.mkdir(parents=True)
        shutil.copy2(support_source, support_destination)
    else:
        support_destination = None

    report = {
        "schema_version": "hollow-pot-collision-proxy-v1",
        "source_parquet": str(source_parquet),
        "source_build_report": str(args.source_build_report.resolve()),
        "output_parquet": str(output_parquet),
        "support_usda": str(support_destination) if support_destination else None,
        "pot_name": args.pot_name,
        "pot_index": pot_index,
        "trajectory_modified": False,
        "hand_data_modified": False,
        "collision_change": (
            "solid cylinder -> bottom plate + segmented wall + original handle"
            if args.wall_segments
            else "solid cylinder -> open bottom plate + original handle"
        ),
        "hollow_proxy": hollow_report,
    }
    (output_root / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
