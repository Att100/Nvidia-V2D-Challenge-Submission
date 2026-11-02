#!/usr/bin/env python3
"""Render actual object-only simulated poses and plot pose drift."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _offline_video import OfflineVideoRenderer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    parquet_paths = sorted(args.parquet.parent.glob("*.parquet")) if args.parquet.is_file() else sorted(args.parquet.glob("*.parquet"))
    if args.parquet.is_file():
        parquet_paths = [args.parquet]
    if len(parquet_paths) != 1:
        raise RuntimeError(f"expected one parquet, found {parquet_paths}")
    row = pq.ParquetFile(parquet_paths[0]).read().to_pylist()[0]
    sim = np.load(args.npz)
    positions = np.asarray(sim["positions"], dtype=np.float64)
    quaternions = np.asarray(sim["quaternions_xyzw"], dtype=np.float64)
    names = [str(value) for value in sim["names"].tolist()]
    fps = float(sim["fps"])
    if names != list(row["object_body_names"]):
        raise ValueError(f"simulation/parquet body mismatch: {names} vs {row['object_body_names']}")

    renderer = OfflineVideoRenderer(fps=int(round(fps)))
    for name, path in zip(names, row["object_mesh_paths"], strict=True):
        renderer.add_object(name, trimesh.load(path, force="mesh", process=False))
    support = json.loads(args.support_report.read_text())
    table = trimesh.creation.cylinder(radius=float(support["radius_m"]), height=float(support["thickness_m"]))
    table.apply_translation(
        [*support["center_xy_m"], float(support["top_height_m"]) - 0.5 * float(support["thickness_m"])]
    )
    table.visual.face_colors = np.array([140, 140, 145, 255], dtype=np.uint8)
    renderer.add_object("simulation_tabletop", table)
    all_points = np.concatenate([positions.reshape(-1, 3), np.asarray([[*support["center_xy_m"], support["top_height_m"]]])], axis=0)
    renderer.fit_camera(all_points, padding=2.0, min_extent=0.5)
    identity = np.eye(4)
    renderer.update_object("simulation_tabletop", identity)
    for frame in range(len(positions)):
        for body, name in enumerate(names):
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(quaternions[frame, body]).as_matrix()
            transform[:3, 3] = positions[frame, body]
            renderer.update_object(name, transform)
        renderer.capture()
    renderer.save(args.video)
    renderer.close()

    displacement = positions - positions[0:1]
    translation_mm = 1000.0 * np.linalg.norm(displacement, axis=-1)
    initial = quaternions[0:1] / np.linalg.norm(quaternions[0:1], axis=-1, keepdims=True)
    current = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    rotation_deg = np.degrees(2.0 * np.arccos(np.clip(np.abs(np.sum(current * initial, axis=-1)), 0.0, 1.0)))
    time = np.arange(len(positions)) / fps
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for body, name in enumerate(names):
        axes[0].plot(time, translation_mm[:, body], label=name)
        axes[1].plot(time, rotation_deg[:, body], label=name)
    axes[0].axhline(10.0, color="red", linestyle="--", linewidth=1, label="10 mm gate")
    axes[1].axhline(10.0, color="red", linestyle="--", linewidth=1, label="10° gate")
    axes[0].set_ylabel("Translation from reset (mm)")
    axes[1].set_ylabel("Rotation from reset (deg)")
    axes[1].set_xlabel("Simulation time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    fig.suptitle("Object-only free physics (robot collision disabled, VOC=0)")
    fig.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=160)
    plt.close(fig)
    print(args.video)
    print(args.plot)


if __name__ == "__main__":
    main()
