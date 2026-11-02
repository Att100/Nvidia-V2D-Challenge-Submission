#!/usr/bin/env python3
"""Build a reproducible MANO object-contact override from calibrated SAM masks.

The script fuses masked depth from canonical object views into an object-local
surface cloud, then replaces one object's active right-hand contacts with one
nearest surface contact per fingertip.  Source parquet data is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


FINGERTIP_JOINTS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
# MANO_HAND_LINKS order: thumb3, index3, middle3, ring3, pinky3.
FINGERTIP_LINK_SLOTS = np.asarray([3, 6, 9, 12, 15], dtype=np.int64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unproject(
    depth: np.ndarray,
    mask: np.ndarray,
    camera_to_object: np.ndarray,
    intrinsics: dict[str, float],
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    ys, xs = np.nonzero(valid)
    keep = ((ys % stride) == 0) & ((xs % stride) == 0)
    ys, xs = ys[keep], xs[keep]
    z = depth[ys, xs]
    camera = np.column_stack(
        (
            (xs - intrinsics["cx"]) * z / intrinsics["fx"],
            -(ys - intrinsics["cy"]) * z / intrinsics["fy"],
            -z,
        )
    )
    points = camera @ camera_to_object[:3, :3].T + camera_to_object[:3, 3]
    return points, np.column_stack((xs, ys))


def project_mask(points: np.ndarray, camera_to_object: np.ndarray, intrinsics: dict[str, float], mask: np.ndarray) -> np.ndarray:
    camera = (points - camera_to_object[:3, 3]) @ camera_to_object[:3, :3]
    depth = -camera[:, 2]
    u = np.rint(intrinsics["fx"] * camera[:, 0] / depth + intrinsics["cx"]).astype(np.int64)
    v = np.rint(intrinsics["cy"] - intrinsics["fy"] * camera[:, 1] / depth).astype(np.int64)
    inside = (depth > 0.0) & (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
    result = np.zeros(len(points), dtype=bool)
    result[inside] = mask[v[inside], u[inside]]
    return result


def runs(indices: np.ndarray) -> list[list[int]]:
    if not len(indices):
        return []
    split = np.where(np.diff(indices) != 1)[0] + 1
    return [[int(group[0]), int(group[-1])] for group in np.split(indices, split)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--object-name", default="white_pot_lid")
    parser.add_argument("--contact-part-id", type=int, default=2)
    parser.add_argument("--min-local-height", type=float, default=0.004)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--voxel-size", type=float, default=0.00075)
    args = parser.parse_args()

    manifest_path = args.views_dir / "render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    intrinsics = manifest["intrinsics"]
    mesh_path = Path(manifest["mesh"])
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    vertex_tree = cKDTree(vertices)
    top_mask = np.load(args.masks_dir / "top_mask.npy").astype(bool)

    fused_points: list[np.ndarray] = []
    fused_normals: list[np.ndarray] = []
    fused_sources: list[np.ndarray] = []
    per_view: dict[str, dict[str, int]] = {}
    for view_index, (view, view_info) in enumerate(manifest["views"].items()):
        camera_to_object = np.asarray(view_info["camera_to_object"], dtype=np.float64)
        depth = np.load(args.views_dir / view_info["depth"])
        mask = np.load(args.masks_dir / f"{view}_mask.npy").astype(bool)
        points, _pixels = unproject(depth, mask, camera_to_object, intrinsics, args.sample_stride)
        top_consistent = project_mask(
            points,
            np.asarray(manifest["views"]["top"]["camera_to_object"], dtype=np.float64),
            intrinsics,
            top_mask,
        )
        height_consistent = points[:, 1] >= args.min_local_height
        keep = top_consistent & height_consistent
        points = points[keep]
        _distance, nearest_vertices = vertex_tree.query(points, k=1)
        normals = vertex_normals[nearest_vertices].copy()
        camera_position = camera_to_object[:3, 3]
        # A visible exterior surface normal faces its source camera.
        flip = np.einsum("ij,ij->i", normals, camera_position[None] - points) < 0.0
        normals[flip] *= -1.0
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
        fused_points.append(points)
        fused_normals.append(normals)
        fused_sources.append(np.full(len(points), view_index, dtype=np.int16))
        per_view[view] = {"masked_depth_samples": int(len(keep)), "accepted_samples": int(len(points))}

    points = np.concatenate(fused_points, axis=0)
    normals = np.concatenate(fused_normals, axis=0)
    sources = np.concatenate(fused_sources, axis=0)
    voxel = np.floor(points / args.voxel_size).astype(np.int64)
    _, unique = np.unique(voxel, axis=0, return_index=True)
    unique.sort()
    points, normals, sources = points[unique], normals[unique], sources[unique]
    if len(points) < 100:
        raise RuntimeError(f"Only {len(points)} fused handle samples survived filtering")

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.output_parquet.parent / "contact_override_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cloud_colors = np.tile(np.asarray([[30, 220, 80, 255]], dtype=np.uint8), (len(points), 1))
    trimesh.points.PointCloud(points, colors=cloud_colors).export(artifact_dir / "handle_surface.ply")
    np.savez_compressed(
        artifact_dir / "handle_surface.npz",
        points_object=points.astype(np.float32),
        normals_outward_object=normals.astype(np.float32),
        source_view_index=sources,
        source_view_names=np.asarray(list(manifest["views"])),
    )

    parquet_file = pq.ParquetFile(args.source_parquet)
    table = parquet_file.read()
    if table.num_rows != 1:
        raise ValueError(f"Expected one parquet row, got {table.num_rows}")
    row = table.to_pylist()[0]
    object_names = list(row["object_body_names"])
    if args.object_name not in object_names:
        raise KeyError(f"{args.object_name!r} is not in object_body_names={object_names}")
    object_index = object_names.index(args.object_name)

    ids = np.asarray(row["mano_right_object_contact_part_ids"], dtype=np.int32)
    positions = np.asarray(row["mano_right_object_contact_positions"], dtype=np.float64)
    contact_normals = np.asarray(row["mano_right_object_contact_normals"], dtype=np.float64)
    link_positions = np.asarray(row["mano_right_link_contact_positions"], dtype=np.float64)
    joints = np.asarray(row["mano_right_joints"], dtype=np.float64)
    body_positions = np.asarray(row["object_body_position"], dtype=np.float64)
    body_wxyz = np.asarray(row["object_body_wxyz"], dtype=np.float64)
    active_frames = np.flatnonzero(np.any(ids == args.contact_part_id, axis=1))
    if not len(active_frames):
        raise RuntimeError(f"No existing right-hand contacts for part id {args.contact_part_id}")

    handle_tree = cKDTree(points)
    nearest_distances = np.zeros((len(active_frames), len(FINGERTIP_JOINTS)), dtype=np.float64)
    chosen_local = np.zeros((len(active_frames), len(FINGERTIP_JOINTS), 3), dtype=np.float64)
    for output_index, frame in enumerate(active_frames):
        rotation = Rotation.from_quat(body_wxyz[frame, object_index], scalar_first=True).as_matrix()
        translation = body_positions[frame, object_index]
        fingertip_world = joints[frame, FINGERTIP_JOINTS]
        fingertip_local = (fingertip_world - translation) @ rotation
        distance, nearest = handle_tree.query(fingertip_local, k=1)
        selected_points = points[nearest]
        selected_normals = normals[nearest]
        selected_world = selected_points @ rotation.T + translation
        inward_world = (-selected_normals) @ rotation.T

        # During a lid-contact frame, discard reconstruction-derived lower-surface
        # and incidental pot contacts, then install one lid contact per fingertip.
        ids[frame] = 0
        positions[frame] = 0.0
        contact_normals[frame] = 0.0
        ids[frame, FINGERTIP_LINK_SLOTS] = args.contact_part_id
        positions[frame, FINGERTIP_LINK_SLOTS] = selected_world
        contact_normals[frame, FINGERTIP_LINK_SLOTS] = inward_world
        link_positions[frame, FINGERTIP_LINK_SLOTS] = fingertip_world
        nearest_distances[output_index] = distance
        chosen_local[output_index] = selected_points

    row["mano_right_object_contact_part_ids"] = ids.tolist()
    row["mano_right_object_contact_positions"] = positions.tolist()
    row["mano_right_object_contact_normals"] = contact_normals.tolist()
    row["mano_right_link_contact_positions"] = link_positions.tolist()
    output_table = pa.Table.from_pylist([row], schema=table.schema)
    pq.write_table(output_table, args.output_parquet, compression="zstd")

    report = {
        "method": "sam2_multiview_depth_fusion_nearest_fingertips_v1",
        "source_parquet": str(args.source_parquet.resolve()),
        "source_parquet_sha256": sha256(args.source_parquet),
        "output_parquet": str(args.output_parquet.resolve()),
        "output_parquet_sha256": sha256(args.output_parquet),
        "mesh": str(mesh_path.resolve()),
        "mesh_sha256": sha256(mesh_path),
        "manifest": str(manifest_path.resolve()),
        "masks_dir": str(args.masks_dir.resolve()),
        "parameters": {
            "object_name": args.object_name,
            "object_index": object_index,
            "contact_part_id": args.contact_part_id,
            "min_local_height_m": args.min_local_height,
            "sample_stride_pixels": args.sample_stride,
            "voxel_size_m": args.voxel_size,
            "fingertip_joint_indices": FINGERTIP_JOINTS.tolist(),
            "contact_link_slots": FINGERTIP_LINK_SLOTS.tolist(),
        },
        "surface": {
            "num_points": int(len(points)),
            "bounds_m": np.stack((points.min(axis=0), points.max(axis=0))).tolist(),
            "per_view": per_view,
        },
        "contacts": {
            "active_frame_count": int(len(active_frames)),
            "active_frame_runs_inclusive": runs(active_frames),
            "contacts_per_active_frame": int(len(FINGERTIP_LINK_SLOTS)),
            "fingertip_to_handle_distance_m": {
                "mean": float(nearest_distances.mean()),
                "median": float(np.median(nearest_distances)),
                "p95": float(np.percentile(nearest_distances, 95)),
                "max": float(nearest_distances.max()),
                "per_finger_mean": nearest_distances.mean(axis=0).tolist(),
            },
            "selected_local_height_m": {
                "min": float(chosen_local[:, :, 1].min()),
                "median": float(np.median(chosen_local[:, :, 1])),
                "max": float(chosen_local[:, :, 1].max()),
            },
        },
    }
    (artifact_dir / "contact_override_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
