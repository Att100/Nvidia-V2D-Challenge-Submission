#!/usr/bin/env python3
"""Write a fixed tabletop collision cylinder from a retargeted reference.

This is for sequences whose scene plane was fitted from depth but whose
deliverable does not include ``support_000``.  It creates simulation-only
support geometry; it never changes or replaces a submitted object mesh.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-height", type=float, default=0.75)
    parser.add_argument("--thickness", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--minimum-radius", type=float, default=0.35)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    paths = glob.glob(str(args.partition / "*.parquet"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one parquet under {args.partition}, found {paths}")
    row = pq.read_table(paths[0]).to_pydict()
    positions = np.asarray(row["object_body_position"][0], dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"unexpected object_body_position shape: {positions.shape}")
    center_xy = np.median(positions[..., :2].reshape(-1, 2), axis=0)
    radial_motion = float(
        np.linalg.norm(positions[..., :2] - center_xy, axis=-1).max()
    )
    radii = np.asarray(row.get("object_mesh_radius", [[0.0]])[0], dtype=np.float64)
    radius = max(
        float(args.minimum_radius), radial_motion + float(radii.max()) + float(args.margin)
    )
    if not np.isfinite(radius) or radius > 2.0:
        raise ValueError(f"implausible tabletop radius: {radius}")
    z_center = float(args.table_height) - 0.5 * float(args.thickness)
    text = f'''#usda 1.0
(
    defaultPrim = "support_surfaces"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "support_surfaces"
{{
    def Cylinder "fitted_tabletop" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        uniform token axis = "Z"
        double height = {float(args.thickness):.12g}
        color3f[] primvars:displayColor = [(0.55, 0.55, 0.55)]
        double radius = {radius:.12g}
        double3 xformOp:translate = ({center_xy[0]:.12g}, {center_xy[1]:.12g}, {z_center:.12g})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
}}
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    report = {
        "method": "fitted_scene_plane_tabletop",
        "source_parquet": str(Path(paths[0]).resolve()),
        "output": str(args.output.resolve()),
        "top_height_m": float(args.table_height),
        "thickness_m": float(args.thickness),
        "center_xy_m": center_xy.tolist(),
        "radius_m": radius,
        "interaction_object_meshes_modified": False,
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
