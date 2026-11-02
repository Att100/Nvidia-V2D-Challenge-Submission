#!/usr/bin/env python3
"""Render canonical object-local views with calibrated depth for semantic contact masks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image


def camera_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--yfov-deg", type=float, default=32.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = trimesh.load(args.mesh, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected one Trimesh, got {type(loaded).__name__}")
    center = loaded.bounds.mean(axis=0)
    radius = float(np.max(np.linalg.norm(loaded.vertices - center, axis=1)))
    distance = radius / np.tan(np.radians(args.yfov_deg) / 2.0) * 1.18
    mesh = pyrender.Mesh.from_trimesh(loaded, smooth=False)
    camera = pyrender.PerspectiveCamera(yfov=np.radians(args.yfov_deg), aspectRatio=1.0)
    resolution = args.resolution
    focal = 0.5 * resolution / np.tan(0.5 * np.radians(args.yfov_deg))
    views = {
        # The lid's +Y axis is world-up in the recorded sequence.
        "top": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        "left": (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        "right": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        "front": (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
        "back": (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
    }
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    manifest: dict[str, object] = {
        "mesh": str(args.mesh.resolve()),
        "mesh_bounds_m": loaded.bounds.tolist(),
        "mesh_center_m": center.tolist(),
        "mesh_radius_m": radius,
        "resolution": resolution,
        "intrinsics": {"fx": focal, "fy": focal, "cx": resolution / 2.0, "cy": resolution / 2.0},
        "views": {},
    }
    try:
        for name, (direction, up) in views.items():
            pose = camera_pose(center + direction * distance, center, up)
            scene = pyrender.Scene(bg_color=np.array([32, 38, 48, 255], dtype=np.uint8), ambient_light=[0.45] * 3)
            scene.add(mesh)
            scene.add(camera, pose=pose)
            for light_direction in (direction, np.array([0.6, 0.8, 0.4])):
                light_pose = camera_pose(center + light_direction / np.linalg.norm(light_direction) * distance, center, up)
                scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.2), pose=light_pose)
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            Image.fromarray(color).save(args.output_dir / f"{name}.png")
            np.save(args.output_dir / f"{name}_depth.npy", depth.astype(np.float32))
            manifest["views"][name] = {"camera_to_object": pose.tolist(), "depth": f"{name}_depth.npy"}
    finally:
        renderer.delete()
    (args.output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
