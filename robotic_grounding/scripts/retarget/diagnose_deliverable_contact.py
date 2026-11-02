#!/usr/bin/env python3
"""Diagnose 2D/3D hand-object alignment in an ego-scene-HOI deliverable.

The deliverable contract is documented by ``input/.../<sequence>/README.md``:
official MANO output, including ``transl``, is multiplied by the external
per-hand scale.  This script reconstructs the right hand accordingly, compares
it with the metric object mesh, and writes reproducible visual diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
import trimesh
from scipy.spatial import cKDTree


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _project(points_world: np.ndarray, world_to_camera: np.ndarray, K: np.ndarray):
    points_camera = (
        points_world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    )
    z = points_camera[:, 2]
    uv = np.empty((len(points_camera), 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * points_camera[:, 0] / z + K[0, 2]
    uv[:, 1] = K[1, 1] * points_camera[:, 1] / z + K[1, 2]
    return uv, z


def _surface_samples(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int):
    """Sample a deformable mesh using fixed face ids and barycentric weights."""
    rng = np.random.default_rng(seed)
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    probability = area / area.sum()
    face_ids = rng.choice(len(faces), size=count, p=probability)
    uv = rng.random((count, 2))
    flip = uv.sum(axis=1) > 1.0
    uv[flip] = 1.0 - uv[flip]
    weights = np.column_stack((1.0 - uv.sum(axis=1), uv))
    return face_ids, weights


def _sample_deformable(vertices: np.ndarray, faces: np.ndarray, face_ids, weights):
    return (vertices[faces[face_ids]] * weights[:, :, None]).sum(axis=1)


def _zbuffer_residual(
    points_world: np.ndarray,
    world_to_camera: np.ndarray,
    K: np.ndarray,
    depth: np.ndarray,
):
    uv, z = _project(points_world, world_to_camera, K)
    x = np.rint(uv[:, 0]).astype(np.int32)
    y = np.rint(uv[:, 1]).astype(np.int32)
    height, width = depth.shape
    valid = (z > 0.0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, z = x[valid], y[valid], z[valid]
    linear = y * width + x
    zbuffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(zbuffer, linear, z)
    occupied = np.flatnonzero(np.isfinite(zbuffer))
    observed = depth.reshape(-1)[occupied]
    valid_depth = np.isfinite(observed) & (observed > 0.0)
    residual = zbuffer[occupied][valid_depth] - observed[valid_depth]
    # Large residuals mostly indicate occlusion/background at the same rounded
    # pixel rather than local surface alignment.
    return residual[np.abs(residual) < 0.15]


def _draw_mesh(image, vertices_world, faces, w2c, K, color, alpha):
    uv, z = _project(vertices_world, w2c, K)
    valid_faces = faces[np.all(z[faces] > 0.0, axis=1)]
    order = np.argsort(z[valid_faces].mean(axis=1))[::-1]
    layer = image.copy()
    height, width = image.shape[:2]
    for face in valid_faces[order]:
        polygon = np.rint(uv[face]).astype(np.int32)
        if (
            polygon[:, 0].max() < 0
            or polygon[:, 0].min() >= width
            or polygon[:, 1].max() < 0
            or polygon[:, 1].min() >= height
        ):
            continue
        cv2.fillConvexPoly(layer, polygon, color, lineType=cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, image, 1.0 - alpha, 0.0, dst=image)


def _longest_true_ranges(mask: np.ndarray):
    ranges = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            stop = index + 1 if value and index == len(mask) - 1 else index
            ranges.append((start, stop - 1, stop - start))
            start = None
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--mano-right-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--surface-samples", type=int, default=80_000)
    args = parser.parse_args()

    root = args.sequence_dir.resolve()
    output_dir = (args.output_dir or root / "validation").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = _load_pickle(root / "cameras/trajectory.pkl")
    intrinsics = _load_pickle(root / "cameras/intrinsics.pkl")
    depth_data = _load_pickle(root / "depth/depth.pkl")
    mano_data = _load_pickle(root / "hands/mano_parameters.pkl")
    mano_scale = _load_pickle(root / "hands/mano_scale.pkl")
    topology = _load_pickle(root / "hands/mano_topology.pkl")
    object_pose = _load_pickle(root / "objects/object_000/poses.pkl")["object_to_world"]
    support_pose = _load_pickle(root / "objects/support_000/poses.pkl")["object_to_world"]

    parameters = mano_data["parameters"]
    right_scale = float(mano_scale["scale"][1])
    model = smplx.MANO(
        str(args.mano_right_model),
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=False,
    )
    with torch.no_grad():
        result = model(
            betas=torch.from_numpy(parameters["betas"][:, 1]),
            global_orient=torch.from_numpy(parameters["global_orient"][:, 1]),
            hand_pose=torch.from_numpy(parameters["hand_pose"][:, 1]),
            transl=torch.from_numpy(parameters["transl"][:, 1]),
        )
    raw_vertices = result.vertices.numpy()
    right_vertices = right_scale * raw_vertices
    right_faces = np.asarray(model.faces, dtype=np.int32)
    if not np.array_equal(right_faces, topology["faces_right"]):
        raise ValueError("the supplied MANO_RIGHT model does not match the delivered topology")

    object_mesh = trimesh.load(
        root / "objects/object_000/textured_mesh_metric.glb", force="mesh"
    )
    support_mesh = trimesh.load(
        root / "objects/support_000/textured_mesh_metric.glb", force="mesh"
    )
    object_vertices = np.asarray(object_mesh.vertices, dtype=np.float64)
    object_faces = np.asarray(object_mesh.faces, dtype=np.int32)
    object_surface, _ = trimesh.sample.sample_surface(
        object_mesh, args.surface_samples, seed=0
    )
    object_tree = cKDTree(object_surface)

    support_surface, _ = trimesh.sample.sample_surface(
        support_mesh, args.surface_samples, seed=1
    )
    support_tree = cKDTree(support_surface)
    support_world_to_local = np.linalg.inv(support_pose[0])

    frame_count = len(right_vertices)
    gap_m = np.empty(frame_count)
    image_gap_px = np.empty(frame_count)
    nearest_depth_gap_m = np.empty(frame_count)
    support_gap_m = np.empty(frame_count)
    nearest_hand_world = np.empty((frame_count, 3))
    nearest_object_world = np.empty((frame_count, 3))
    correction_world = np.empty((frame_count, 3))

    K = intrinsics["K_rectified_left"].astype(np.float64)
    world_to_camera = camera["rectified_left_world_to_camera"].astype(np.float64)
    for frame in range(frame_count):
        transform = object_pose[frame].astype(np.float64)
        hand_local = (
            right_vertices[frame] - transform[:3, 3]
        ) @ transform[:3, :3]
        distance, indices = object_tree.query(hand_local, workers=1)
        hand_index = int(np.argmin(distance))
        object_local = object_surface[int(indices[hand_index])]
        hand_world = right_vertices[frame, hand_index]
        object_world = object_local @ transform[:3, :3].T + transform[:3, 3]
        nearest_hand_world[frame] = hand_world
        nearest_object_world[frame] = object_world
        correction_world[frame] = object_world - hand_world
        gap_m[frame] = distance[hand_index]

        pair_uv, pair_z = _project(
            np.stack((hand_world, object_world)), world_to_camera[frame], K
        )
        image_gap_px[frame] = np.linalg.norm(pair_uv[0] - pair_uv[1])
        nearest_depth_gap_m[frame] = abs(pair_z[0] - pair_z[1])

        object_world_vertices = (
            object_vertices @ transform[:3, :3].T + transform[:3, 3]
        )
        object_support_local = (
            object_world_vertices @ support_world_to_local[:3, :3].T
            + support_world_to_local[:3, 3]
        )
        support_gap_m[frame] = support_tree.query(
            object_support_local, workers=1
        )[0].min()

    # A lifted object is the most defensible interval in which the manipulating
    # hand should have physical contact.
    lifted = support_gap_m > 0.02
    interaction = lifted & (gap_m < 0.12)
    if interaction.sum() < 8:
        interaction = gap_m < np.percentile(gap_m, 50)

    # Test whether the supplied scale is the source of the discrepancy.
    scale_values = np.linspace(0.50, 0.90, 41)
    scale_scores = []
    test_frames = np.flatnonzero(interaction)[::2]
    for scale in scale_values:
        distances = []
        for frame in test_frames:
            transform = object_pose[frame].astype(np.float64)
            hand_local = (
                scale * raw_vertices[frame] - transform[:3, 3]
            ) @ transform[:3, :3]
            distances.append(object_tree.query(hand_local, workers=1)[0].min())
        scale_scores.append(float(np.median(distances)))
    best_scale_index = int(np.argmin(scale_scores))

    # Test synchronization. Negative lag means an earlier hand frame is paired
    # with the current object frame.
    lag_values = np.arange(-15, 16)
    lag_scores = []
    base_frames = np.flatnonzero(interaction)[::2]
    for lag in lag_values:
        distances = []
        for object_frame in base_frames:
            hand_frame = object_frame + int(lag)
            if hand_frame < 0 or hand_frame >= frame_count:
                continue
            transform = object_pose[object_frame].astype(np.float64)
            hand_local = (
                right_vertices[hand_frame] - transform[:3, 3]
            ) @ transform[:3, :3]
            distances.append(object_tree.query(hand_local, workers=1)[0].min())
        lag_scores.append(float(np.median(distances)))
    best_lag_index = int(np.argmin(lag_scores))

    # Dense surface-vs-depth residuals on representative interaction frames.
    hand_face_ids, hand_weights = _surface_samples(
        right_vertices[0], right_faces, 30_000, seed=2
    )
    representative = np.unique(
        np.rint(np.linspace(0, frame_count - 1, 12)).astype(np.int32)
    )
    contact_frames = np.argsort(gap_m)[:3]
    selected_frames = np.unique(np.concatenate((representative, contact_frames)))
    depth_rows = {}
    for frame in selected_frames:
        transform = object_pose[frame].astype(np.float64)
        object_world_surface = (
            object_surface @ transform[:3, :3].T + transform[:3, 3]
        )
        hand_surface = _sample_deformable(
            right_vertices[frame], right_faces, hand_face_ids, hand_weights
        )
        observed_depth = depth_data["depth_m"][frame]
        hand_residual = _zbuffer_residual(
            hand_surface, world_to_camera[frame], K, observed_depth
        )
        object_residual = _zbuffer_residual(
            object_world_surface, world_to_camera[frame], K, observed_depth
        )
        depth_rows[int(frame)] = {
            "hand_median_mm": float(np.median(hand_residual) * 1000)
            if len(hand_residual)
            else None,
            "hand_p25_mm": float(np.percentile(hand_residual, 25) * 1000)
            if len(hand_residual)
            else None,
            "hand_p75_mm": float(np.percentile(hand_residual, 75) * 1000)
            if len(hand_residual)
            else None,
            "object_median_mm": float(np.median(object_residual) * 1000)
            if len(object_residual)
            else None,
            "object_p25_mm": float(np.percentile(object_residual, 25) * 1000)
            if len(object_residual)
            else None,
            "object_p75_mm": float(np.percentile(object_residual, 75) * 1000)
            if len(object_residual)
            else None,
        }

    # Contact sheet: uniform temporal coverage plus the three closest frames.
    video = cv2.VideoCapture(str(root / "videos/rectified_left.mp4"))
    tiles = []
    for frame in selected_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = video.read()
        if not ok:
            continue
        transform = object_pose[frame].astype(np.float64)
        object_world_vertices = (
            object_vertices @ transform[:3, :3].T + transform[:3, 3]
        )
        _draw_mesh(
            image,
            object_world_vertices,
            object_faces,
            world_to_camera[frame],
            K,
            (0, 128, 255),
            0.42,
        )
        _draw_mesh(
            image,
            right_vertices[frame],
            right_faces,
            world_to_camera[frame],
            K,
            (255, 0, 255),
            0.58,
        )
        pair_uv, pair_z = _project(
            np.stack((nearest_hand_world[frame], nearest_object_world[frame])),
            world_to_camera[frame],
            K,
        )
        hand_pixel = tuple(np.rint(pair_uv[0]).astype(int))
        object_pixel = tuple(np.rint(pair_uv[1]).astype(int))
        cv2.line(image, hand_pixel, object_pixel, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(image, hand_pixel, 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, object_pixel, 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(image, (0, 0), (640, 58), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"frame {frame:03d} | 3D {gap_m[frame]*1000:.1f} mm | 2D {image_gap_px[frame]:.1f} px",
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"pair depth gap {nearest_depth_gap_m[frame]*1000:.1f} mm | orange=object magenta=right hand",
            (10, 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(image)
    video.release()
    columns = 3
    rows = int(np.ceil(len(tiles) / columns))
    sheet = np.zeros((rows * 400, columns * 640, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * 400 : (row + 1) * 400, column * 640 : (column + 1) * 640] = tile
    cv2.imwrite(str(output_dir / "diagnostic_right_contact_sheet.png"), sheet)

    frames = np.arange(frame_count)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(frames, gap_m * 1000, label="right hand-object")
    axes[0].axhline(10.0, color="red", linestyle="--", label="contact threshold")
    axes[0].set_ylabel("distance (mm)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(frames, support_gap_m * 1000, label="object-support", color="tab:blue")
    axes[1].axhline(20.0, color="gray", linestyle="--", label="lifted threshold")
    axes[1].set_ylabel("distance (mm)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[2].plot(frames, image_gap_px, label="projected nearest-pair gap", color="tab:green")
    axes[2].set_ylabel("pixels")
    axes[2].set_xlabel("frame")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "diagnostic_right_contact_curves.png", dpi=160)
    plt.close(fig)

    with (output_dir / "diagnostic_right_contact.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "hand_object_gap_mm",
                "object_support_gap_mm",
                "nearest_pair_2d_gap_px",
                "nearest_pair_depth_gap_mm",
                "correction_world_x_mm",
                "correction_world_y_mm",
                "correction_world_z_mm",
            ]
        )
        for frame in range(frame_count):
            writer.writerow(
                [
                    frame,
                    gap_m[frame] * 1000,
                    support_gap_m[frame] * 1000,
                    image_gap_px[frame],
                    nearest_depth_gap_m[frame] * 1000,
                    *(correction_world[frame] * 1000),
                ]
            )

    correction = correction_world[interaction]
    correction_norm = np.linalg.norm(correction, axis=1)
    correction_unit = correction / np.maximum(correction_norm[:, None], 1e-12)
    direction_concentration = float(np.linalg.norm(correction_unit.mean(axis=0)))

    # The observed correction is suspiciously parallel to the left-to-right
    # stereo translation.  Quantify that hypothesis and evaluate the contact
    # distance after adding the corresponding world-frame baseline term.
    stereo_baseline = np.asarray(
        intrinsics["stereo_t_right_from_left_m"], dtype=np.float64
    )
    camera_to_world = camera["rectified_left_camera_to_world"].astype(np.float64)
    baseline_world = np.stack(
        [stereo_baseline @ camera_to_world[frame, :3, :3].T for frame in np.flatnonzero(interaction)]
    )
    fitted_baseline_scale = float(
        np.sum(baseline_world * correction) / np.sum(baseline_world * baseline_world)
    )
    baseline_cosine = np.sum(baseline_world * correction, axis=1) / (
        np.linalg.norm(baseline_world, axis=1) * correction_norm
    )

    def corrected_contact_stats(multiplier: float) -> dict[str, float | int]:
        distances = []
        for frame in np.flatnonzero(interaction):
            delta_world = (
                multiplier
                * stereo_baseline
                @ camera_to_world[frame, :3, :3].T
            )
            transform = object_pose[frame].astype(np.float64)
            hand_local = (
                right_vertices[frame] + delta_world - transform[:3, 3]
            ) @ transform[:3, :3]
            distances.append(object_tree.query(hand_local, workers=1)[0].min())
        distances = np.asarray(distances)
        return {
            "min_mm": float(distances.min() * 1000),
            "median_mm": float(np.median(distances) * 1000),
            "p95_mm": float(np.percentile(distances, 95) * 1000),
            "frames_below_10mm": int((distances < 0.01).sum()),
        }

    summary = {
        "frame_count": frame_count,
        "fps": 20.0,
        "right_scale": right_scale,
        "mano_topology_matches": True,
        "hand_object_gap_mm": {
            "min": float(gap_m.min() * 1000),
            "median": float(np.median(gap_m) * 1000),
            "p95": float(np.percentile(gap_m, 95) * 1000),
            "frames_below_10mm": int((gap_m < 0.01).sum()),
        },
        "interaction_selection": {
            "frames": int(interaction.sum()),
            "ranges": _longest_true_ranges(interaction),
            "median_gap_mm": float(np.median(gap_m[interaction]) * 1000),
            "median_projected_pair_gap_px": float(
                np.median(image_gap_px[interaction])
            ),
            "median_pair_depth_gap_mm": float(
                np.median(nearest_depth_gap_m[interaction]) * 1000
            ),
        },
        "nearest_correction_during_interaction": {
            "median_world_xyz_mm": (np.median(correction, axis=0) * 1000).tolist(),
            "direction_concentration_0_to_1": direction_concentration,
        },
        "stereo_baseline_hypothesis": {
            "stereo_t_right_from_left_mm": (stereo_baseline * 1000).tolist(),
            "median_direction_cosine": float(np.median(baseline_cosine)),
            "fitted_multiplier": fitted_baseline_scale,
            "provided_hand_scale_multiplier": right_scale,
            "using_provided_hand_scale": corrected_contact_stats(right_scale),
            "using_fitted_multiplier": corrected_contact_stats(fitted_baseline_scale),
        },
        "scale_sweep": {
            "tested": [float(value) for value in scale_values],
            "median_gap_mm": [float(value * 1000) for value in scale_scores],
            "best_scale": float(scale_values[best_scale_index]),
            "best_median_gap_mm": float(scale_scores[best_scale_index] * 1000),
            "provided_scale_median_gap_mm": float(
                scale_scores[int(np.argmin(abs(scale_values - right_scale)))] * 1000
            ),
        },
        "time_lag_sweep": {
            "tested_lag_frames": [int(value) for value in lag_values],
            "median_gap_mm": [float(value * 1000) for value in lag_scores],
            "best_lag_frames": int(lag_values[best_lag_index]),
            "best_lag_seconds": float(lag_values[best_lag_index] / 20.0),
            "best_median_gap_mm": float(lag_scores[best_lag_index] * 1000),
            "zero_lag_median_gap_mm": float(lag_scores[15] * 1000),
        },
        "surface_vs_stereo_depth": depth_rows,
        "outputs": {
            "contact_sheet": "diagnostic_right_contact_sheet.png",
            "curves": "diagnostic_right_contact_curves.png",
            "per_frame_csv": "diagnostic_right_contact.csv",
        },
    }
    with (output_dir / "diagnostic_right_contact_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
