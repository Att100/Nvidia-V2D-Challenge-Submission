#!/usr/bin/env python3
"""Create an auditable simulation-only bottom proxy and a derived parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--body-name", required=True)
    parser.add_argument("--output-urdf", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--patch-band", type=float, default=0.005)
    parser.add_argument("--thickness", type=float, default=0.004)
    parser.add_argument("--quantile", type=float, default=0.01)
    args = parser.parse_args()

    for output in (args.output_urdf, args.output_parquet, args.report):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    table = pq.read_table(args.input_parquet)
    if table.num_rows != 1:
        raise ValueError(f"expected one row, got {table.num_rows}")
    row = table.to_pylist()[0]
    names = [str(value) for value in row["object_body_names"]]
    if args.body_name not in names:
        raise ValueError(f"body {args.body_name!r} not in {names}")
    body = names.index(args.body_name)
    mesh_path = Path(row["object_mesh_paths"][body]).resolve()
    original_urdf = Path(row["object_urdf_paths"][body]).resolve()
    mesh = trimesh.load(mesh_path, force="mesh", process=False)

    position = np.asarray(row["object_body_position"][0][body], dtype=np.float64)
    quaternion = np.asarray(row["object_body_wxyz"][0][body], dtype=np.float64)
    object_rotation = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
    vertices_local = np.asarray(mesh.vertices, dtype=np.float64)
    vertices_world = vertices_local @ object_rotation.T + position
    bottom_z = float(vertices_world[:, 2].min())
    patch = vertices_world[vertices_world[:, 2] <= bottom_z + args.patch_band]
    if len(patch) < 16:
        raise ValueError(f"bottom patch has only {len(patch)} vertices")

    xy_center = np.median(patch[:, :2], axis=0)
    centered = patch[:, :2] - xy_center
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis0 = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis0 /= np.linalg.norm(axis0)
    axis1 = np.array([-axis0[1], axis0[0]], dtype=np.float64)
    coordinates = np.column_stack((centered @ axis0, centered @ axis1))
    lower = np.quantile(coordinates, args.quantile, axis=0)
    upper = np.quantile(coordinates, 1.0 - args.quantile, axis=0)
    size_xy = upper - lower
    center_shift = 0.5 * (lower + upper)
    center_xy = xy_center + center_shift[0] * axis0 + center_shift[1] * axis1
    center_world = np.array([center_xy[0], center_xy[1], bottom_z + 0.5 * args.thickness])
    world_from_box = np.array(
        [
            [axis0[0], axis1[0], 0.0],
            [axis0[1], axis1[1], 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    object_from_box = object_rotation.T @ world_from_box
    center_local = object_rotation.T @ (center_world - position)
    rpy = Rotation.from_matrix(object_from_box).as_euler("xyz")
    size = np.array([size_xy[0], size_xy[1], args.thickness], dtype=np.float64)

    tree = ET.parse(original_urdf)
    root = tree.getroot()
    link = root.find("link")
    if link is None:
        raise ValueError(f"no link in {original_urdf}")
    collision = ET.SubElement(link, "collision", {"name": "reconstructed_bottom_patch_proxy"})
    ET.SubElement(
        collision,
        "origin",
        {
            "xyz": " ".join(f"{value:.9g}" for value in center_local),
            "rpy": " ".join(f"{value:.9g}" for value in rpy),
        },
    )
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", {"size": " ".join(f"{value:.9g}" for value in size)})
    ET.indent(tree, space="   ")
    args.output_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output_urdf, encoding="unicode", xml_declaration=True)

    urdf_paths = list(row["object_urdf_paths"])
    urdf_paths[body] = str(args.output_urdf.resolve())
    index = table.schema.get_field_index("object_urdf_paths")
    array = pa.array([urdf_paths], type=table.schema.field(index).type)
    derived = table.set_column(index, table.schema.field(index), array)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(derived, args.output_parquet, compression="zstd")

    report = {
        "schema_version": "ego-flashchord-bottom-proxy-v1",
        "body_name": args.body_name,
        "source_parquet": str(args.input_parquet.resolve()),
        "derived_parquet": str(args.output_parquet.resolve()),
        "mesh": str(mesh_path),
        "mesh_sha256": sha256(mesh_path),
        "original_urdf": str(original_urdf),
        "original_urdf_sha256": sha256(original_urdf),
        "derived_urdf": str(args.output_urdf.resolve()),
        "derived_urdf_sha256": sha256(args.output_urdf),
        "frame": 0,
        "patch_band_m": args.patch_band,
        "patch_vertex_count": int(len(patch)),
        "patch_quantile": args.quantile,
        "reference_bottom_z_m": bottom_z,
        "proxy_size_m": size.tolist(),
        "proxy_origin_object_xyz_m": center_local.tolist(),
        "proxy_origin_object_rpy_rad": rpy.tolist(),
        "proxy_world_bottom_z_m": float(center_world[2] - 0.5 * args.thickness),
        "visual_mesh_modified": False,
        "original_mesh_collision_preserved": True,
        "inertia_modified": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
