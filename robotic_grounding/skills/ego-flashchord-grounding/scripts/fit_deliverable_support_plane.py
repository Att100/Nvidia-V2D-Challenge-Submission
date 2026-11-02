#!/usr/bin/env python3
"""Fit a metric support plane from a support-less ego reconstruction delivery."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="scene", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"not a triangle mesh: {path}")
    return mesh


def motion_score(transforms: np.ndarray) -> np.ndarray:
    count = len(transforms)
    score = np.zeros(count, dtype=np.float64)
    if count < 2:
        return score
    dp = np.linalg.norm(np.diff(transforms[:, :3, 3], axis=0), axis=1)
    dr = (Rotation.from_matrix(transforms[:-1, :3, :3]).inv()
          * Rotation.from_matrix(transforms[1:, :3, :3])).magnitude()
    delta = dp + 0.1 * dr
    score[0] = delta[0]
    score[-1] = delta[-1]
    if count > 2:
        score[1:-1] = 0.5 * (delta[:-1] + delta[1:])
    return score


def select_frames(camera_to_world: np.ndarray, count: int) -> np.ndarray:
    score = motion_score(camera_to_world)
    frames = []
    for indices in np.array_split(np.arange(len(score)), min(count, len(score))):
        frames.append(int(indices[np.argmin(score[indices])]))
    return np.asarray(frames, dtype=np.int64)


def projected_object_mask(
    height: int,
    width: int,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    meshes: list[trimesh.Trimesh],
    poses: list[np.ndarray],
    frame: int,
    margin: int,
    support_roi_margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((height, width), dtype=bool)
    support_roi = np.zeros((height, width), dtype=bool)
    for mesh, timeline in zip(meshes, poses, strict=True):
        transform = world_to_camera @ timeline[frame]
        points = np.asarray(mesh.vertices, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]
        points = points[points[:, 2] > 1.0e-4]
        if not len(points):
            continue
        pixels = points[:, :2] / points[:, 2:3]
        pixels = pixels @ np.diag([intrinsics[0, 0], intrinsics[1, 1]])
        pixels += intrinsics[:2, 2]
        lower = np.floor(pixels.min(axis=0)).astype(int) - margin
        upper = np.ceil(pixels.max(axis=0)).astype(int) + margin
        x0, y0 = np.maximum(lower, [0, 0])
        x1, y1 = np.minimum(upper, [width - 1, height - 1])
        if x1 >= x0 and y1 >= y0:
            mask[y0 : y1 + 1, x0 : x1 + 1] = True
            rx0 = max(int(lower[0]) - support_roi_margin, 0)
            ry0 = max(int(lower[1]) - support_roi_margin, 0)
            rx1 = min(int(upper[0]) + support_roi_margin, width - 1)
            ry1 = min(int(upper[1]) + support_roi_margin, height - 1)
            support_roi[ry0 : ry1 + 1, rx0 : rx1 + 1] = True
    return mask, support_roi


def plane_from_points(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    normal = vh[-1]
    return normal / np.linalg.norm(normal), -float(normal @ center)


def orient_plane(normal: np.ndarray, offset: float, camera: np.ndarray) -> tuple[np.ndarray, float]:
    if float(normal @ camera + offset) < 0.0:
        return -normal, -offset
    return normal, offset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--depth-min", type=float, default=0.2)
    parser.add_argument("--depth-max", type=float, default=2.0)
    parser.add_argument("--object-mask-margin", type=int, default=12)
    parser.add_argument("--support-roi-margin", type=int, default=100)
    parser.add_argument("--ransac-iterations", type=int, default=768)
    parser.add_argument("--inlier-threshold", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    root = args.sequence_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    intrinsics = np.asarray(load_pickle(root / "cameras/intrinsics.pkl")["K_rectified_left"], float)
    cameras = load_pickle(root / "cameras/trajectory.pkl")
    camera_to_world = np.asarray(cameras["rectified_left_camera_to_world"], float)
    world_to_camera = np.asarray(cameras["rectified_left_world_to_camera"], float)
    depth = np.asarray(load_pickle(root / "depth/depth.pkl")["depth_m"], float)
    object_records = [item for item in manifest["files"]["objects"] if item["kind"] == "interaction_object"]
    object_ids = [str(item["id"]) for item in object_records]
    meshes = [load_mesh(root / "objects" / oid / "textured_mesh_metric.glb") for oid in object_ids]
    poses = [np.asarray(load_pickle(root / "objects" / oid / "poses.pkl")["object_to_world"], float)
             for oid in object_ids]
    frames = select_frames(camera_to_world, args.frames)

    collected = []
    collected_uv = []
    height, width = depth.shape[1:]
    yy, xx = np.mgrid[0:height:args.stride, 0:width:args.stride]
    for frame in frames:
        z = depth[frame, ::args.stride, ::args.stride]
        object_mask, support_roi = projected_object_mask(
            height, width, intrinsics, world_to_camera[frame], meshes, poses,
            int(frame), args.object_mask_margin, args.support_roi_margin,
        )
        object_mask = object_mask[::args.stride, ::args.stride]
        support_roi = support_roi[::args.stride, ::args.stride]
        valid = (
            np.isfinite(z)
            & (z >= args.depth_min)
            & (z <= args.depth_max)
            & support_roi
            & ~object_mask
        )
        x = (xx[valid] - intrinsics[0, 2]) / intrinsics[0, 0] * z[valid]
        y = (yy[valid] - intrinsics[1, 2]) / intrinsics[1, 1] * z[valid]
        camera_points = np.column_stack((x, y, z[valid]))
        world_points = camera_points @ camera_to_world[frame, :3, :3].T + camera_to_world[frame, :3, 3]
        collected.append(world_points)
        collected_uv.append(np.column_stack((xx[valid] / width, yy[valid] / height)))
    points = np.concatenate(collected)
    uv = np.concatenate(collected_uv)
    if len(points) < 1000:
        raise RuntimeError(f"too few valid depth samples: {len(points)}")

    generator = np.random.default_rng(args.seed)
    fit_indices = generator.choice(len(points), min(len(points), 120_000), replace=False)
    fit_points = points[fit_indices]
    object0_world = np.asarray(meshes[0].vertices) @ poses[0][0, :3, :3].T + poses[0][0, :3, 3]
    camera_anchor = camera_to_world[frames, :3, 3].mean(axis=0)
    candidates = []
    for _ in range(args.ransac_iterations):
        sample = fit_points[generator.choice(len(fit_points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1.0e-8:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        normal, offset = orient_plane(normal, offset, camera_anchor)
        distance = np.abs(fit_points @ normal + offset)
        count = int(np.count_nonzero(distance < args.inlier_threshold))
        clearance = float(np.min(object0_world @ normal + offset))
        center_distance = float(object0_world.mean(axis=0) @ normal + offset)
        candidates.append((count, abs(clearance), center_distance, normal, offset))
    plausible = [value for value in candidates if value[2] > 0.0 and value[1] < 0.08]
    if not plausible:
        raise RuntimeError("no dominant plane is physically close to the initial support object")
    _, _, _, normal, offset = max(plausible, key=lambda value: value[0])

    for threshold in (0.012, 0.008, args.inlier_threshold):
        distance = np.abs(points @ normal + offset)
        inliers = distance < threshold
        normal, offset = plane_from_points(points[inliers])
        normal, offset = orient_plane(normal, offset, camera_anchor)
    residual = np.abs(points @ normal + offset)
    inliers = residual < args.inlier_threshold
    inlier_points = points[inliers]
    axis0 = np.array([1.0, 0.0, 0.0])
    if abs(float(axis0 @ normal)) > 0.9:
        axis0 = np.array([0.0, 1.0, 0.0])
    axis0 -= normal * float(axis0 @ normal)
    axis0 /= np.linalg.norm(axis0)
    axis1 = np.cross(normal, axis0)
    coordinates = np.column_stack((inlier_points @ axis0, inlier_points @ axis1))
    extent = np.percentile(coordinates, 95, axis=0) - np.percentile(coordinates, 5, axis=0)

    object_clearance = {}
    for oid, mesh, timeline in zip(object_ids, meshes, poses, strict=True):
        values = []
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        for frame in frames:
            world = vertices @ timeline[frame, :3, :3].T + timeline[frame, :3, 3]
            values.append(float(np.min(world @ normal + offset)))
        object_clearance[oid] = {
            "sampled_frames": frames.tolist(),
            "signed_bottom_distance_m": values,
            "median_m": float(np.median(values)),
            "min_m": float(np.min(values)),
            "max_m": float(np.max(values)),
        }

    report = {
        "schema_version": "ego-recon-support-plane-fit-v2",
        "sequence_id": manifest["sequence"],
        "method": "multi_frame_low_camera_motion_depth_ransac_pca",
        "coordinate_system": manifest["world_coordinate_system"]["name"],
        "equation": "normal_world dot x + offset_m = 0",
        "normal_world": normal.tolist(),
        "offset_m": float(offset),
        "normal_orientation": "toward sampled cameras / support free space",
        "source_frames": frames.tolist(),
        "pixel_stride": args.stride,
        "depth_range_m": [args.depth_min, args.depth_max],
        "object_projection_mask_margin_px": args.object_mask_margin,
        "object_derived_support_roi_margin_px": args.support_roi_margin,
        "sample_count": int(len(points)),
        "final_inlier_threshold_m": args.inlier_threshold,
        "final_inlier_count": int(inliers.sum()),
        "final_inlier_fraction": float(inliers.mean()),
        "absolute_residual_median_m": float(np.median(residual[inliers])),
        "absolute_residual_p95_m": float(np.percentile(residual[inliers], 95)),
        "plane_coverage_p05_p95_extents_m": extent.tolist(),
        "plane_coverage_area_m2": float(np.prod(extent)),
        "object_signed_bottom_distance": object_clearance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
