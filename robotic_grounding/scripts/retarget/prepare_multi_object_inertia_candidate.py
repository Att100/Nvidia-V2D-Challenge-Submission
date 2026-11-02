#!/usr/bin/env python3
"""Create a non-destructive multi-object candidate with mesh-derived inertia."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
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


def _source_mass(path: Path) -> float:
    root = ET.parse(path).getroot()
    element = root.find(".//inertial/mass")
    return 0.3 if element is None else float(element.get("value", "0.3"))


def _urdf(name: str, mass: float, com: np.ndarray, inertia: np.ndarray) -> str:
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
    <visual><origin rpy="0 0 0" xyz="0 0 0"/><geometry><mesh filename="./visual.stl" scale="1 1 1"/></geometry></visual>
    <collision><origin rpy="0 0 0" xyz="0 0 0"/><geometry><mesh filename="./collision.stl" scale="1 1 1"/></geometry></collision>
  </link>
</robot>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    source = args.parquet.resolve()
    parquet = pq.ParquetFile(source)
    table = parquet.read()
    cells = {name: table[name][0].as_py() for name in table.column_names}
    names = list(cells["object_body_names"])
    meshes = [Path(path).resolve() for path in cells["object_mesh_paths"]]
    urdfs = [Path(path).resolve() for path in cells["object_urdf_paths"]]
    output_root = args.output_root.resolve()
    new_urdfs = []
    objects = []
    for name, mesh_path, source_urdf in zip(names, meshes, urdfs, strict=True):
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        hull = mesh.convex_hull
        mass = _source_mass(source_urdf)
        if not np.isfinite(hull.mass) or hull.mass <= 0.0:
            raise ValueError(f"invalid convex hull mass for {name}")
        com = np.asarray(hull.center_mass, dtype=np.float64)
        inertia = np.asarray(hull.moment_inertia, dtype=np.float64) * (mass / float(hull.mass))
        asset = output_root / "assets" / name
        asset.mkdir(parents=True, exist_ok=True)
        mesh.export(asset / "visual.stl")
        mesh.export(asset / "collision.stl")
        urdf = asset / f"{name}_rigid.urdf"
        urdf.write_text(_urdf(name, mass, com, inertia))
        new_urdfs.append(str(urdf))
        objects.append(
            {
                "name": name,
                "source_urdf": str(source_urdf),
                "mass_kg": mass,
                "com_local_mm": (1000.0 * com).tolist(),
                "inertia": inertia.tolist(),
                "visual_mesh_unchanged": True,
                "collision_geometry_unchanged": True,
            }
        )
    cells["object_urdf_paths"] = new_urdfs

    sequence = _partition_value(source, "sequence_id")
    robot = _partition_value(source, "robot_name")
    output = output_root / "selected_retarget" / f"sequence_id={sequence}" / f"robot_name={robot}" / "part-0.parquet"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([cells], schema=table.schema), output, compression="snappy")
    support = output_root / "reconstructed_stage" / f"{sequence}_support.usda"
    support.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.support_usda.resolve(), support)
    report = {
        "schema_version": "multi-object-inertia-candidate-v1",
        "source_parquet": str(source),
        "output_parquet": str(output),
        "object_poses_unchanged": True,
        "objects": objects,
    }
    (output_root / "generation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
