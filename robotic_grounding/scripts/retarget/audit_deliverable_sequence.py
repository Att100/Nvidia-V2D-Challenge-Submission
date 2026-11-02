#!/usr/bin/env python3
"""Audit one ego deliverable, canonicalize slots, and estimate pair penetration."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"not a triangle mesh: {path}")
    return loaded


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_transforms(name: str, transforms: np.ndarray, frame_count: int) -> dict:
    if transforms.shape != (frame_count, 4, 4):
        raise ValueError(f"{name}: expected {(frame_count, 4, 4)}, got {transforms.shape}")
    if not np.all(np.isfinite(transforms)):
        raise ValueError(f"{name}: non-finite transform")
    rotations = transforms[:, :3, :3]
    orthogonal_error = np.linalg.norm(
        np.einsum("nji,njk->nik", rotations, rotations) - np.eye(3), axis=(1, 2)
    )
    determinant_error = np.abs(np.linalg.det(rotations) - 1.0)
    if float(orthogonal_error.max()) > 2.0e-3 or float(determinant_error.max()) > 2.0e-3:
        raise ValueError(f"{name}: invalid rigid rotation")
    relative = Rotation.from_matrix(rotations[0]).inv() * Rotation.from_matrix(rotations)
    return {
        "position_min_m": transforms[:, :3, 3].min(axis=0).tolist(),
        "position_max_m": transforms[:, :3, 3].max(axis=0).tolist(),
        "max_translation_from_frame0_m": float(
            np.linalg.norm(transforms[:, :3, 3] - transforms[0, :3, 3], axis=1).max()
        ),
        "max_rotation_from_frame0_deg": float(np.degrees(relative.magnitude()).max()),
        "max_rotation_orthogonality_error": float(orthogonal_error.max()),
        "max_rotation_determinant_error": float(determinant_error.max()),
    }


def sampled_pair_penetration(
    mesh_a: trimesh.Trimesh,
    pose_a: np.ndarray,
    mesh_b: trimesh.Trimesh,
    pose_b: np.ndarray,
    sample_count: int,
    frame_stride: int,
) -> dict:
    rng_state = np.random.get_state()
    np.random.seed(0)
    samples_a, faces_a = trimesh.sample.sample_surface_even(mesh_a, sample_count)
    np.random.seed(1)
    samples_b, faces_b = trimesh.sample.sample_surface_even(mesh_b, sample_count)
    np.random.set_state(rng_state)
    normals_a = np.asarray(mesh_a.face_normals[faces_a], dtype=np.float64)
    normals_b = np.asarray(mesh_b.face_normals[faces_b], dtype=np.float64)
    tree_a = cKDTree(samples_a)
    tree_b = cKDTree(samples_b)

    sampled_frames = list(range(0, len(pose_a), frame_stride))
    if sampled_frames[-1] != len(pose_a) - 1:
        sampled_frames.append(len(pose_a) - 1)
    frame_ids = np.asarray(sampled_frames, dtype=np.int64)
    pose_a_sampled = pose_a[frame_ids]
    pose_b_sampled = pose_b[frame_ids]
    world_a = (
        np.einsum("sj,fij->fsi", samples_a, pose_a_sampled[:, :3, :3])
        + pose_a_sampled[:, None, :3, 3]
    )
    world_b = (
        np.einsum("sj,fij->fsi", samples_b, pose_b_sampled[:, :3, :3])
        + pose_b_sampled[:, None, :3, 3]
    )
    local_a_in_b = np.einsum(
        "fsi,fij->fsj",
        world_a - pose_b_sampled[:, None, :3, 3],
        pose_b_sampled[:, :3, :3],
    )
    local_b_in_a = np.einsum(
        "fsi,fij->fsj",
        world_b - pose_a_sampled[:, None, :3, 3],
        pose_a_sampled[:, :3, :3],
    )
    flat_ab = local_a_in_b.reshape(-1, 3)
    flat_ba = local_b_in_a.reshape(-1, 3)
    distance_ab, index_ab = tree_b.query(flat_ab, workers=1)
    distance_ba, index_ba = tree_a.query(flat_ba, workers=1)
    signed_ab = np.einsum(
        "ij,ij->i", flat_ab - samples_b[index_ab], normals_b[index_ab]
    ).reshape(len(frame_ids), -1)
    signed_ba = np.einsum(
        "ij,ij->i", flat_ba - samples_a[index_ba], normals_a[index_ba]
    ).reshape(len(frame_ids), -1)
    per_frame_depth = np.maximum(
        np.maximum(-signed_ab, 0.0).max(axis=1),
        np.maximum(-signed_ba, 0.0).max(axis=1),
    )
    worst_index = int(np.argmax(per_frame_depth))
    maximum_depth = float(per_frame_depth[worst_index])
    worst_frame = int(frame_ids[worst_index])
    inside_points = int(np.count_nonzero(signed_ab < 0.0) + np.count_nonzero(signed_ba < 0.0))
    evaluated_points = int(signed_ab.size + signed_ba.size)
    minimum_surface_gap = float(
        min(np.min(np.abs(signed_ab)), np.min(np.abs(signed_ba)))
    )
    return {
        "method": "bidirectional sampled surface-normal signed distance; approximate for non-watertight meshes",
        "sample_count_per_body": int(sample_count),
        "frame_stride": int(frame_stride),
        "sampled_frame_count": len(sampled_frames),
        "maximum_sampled_penetration_mm": 1000.0 * maximum_depth,
        "worst_frame": worst_frame,
        "inside_sample_fraction": inside_points / max(evaluated_points, 1),
        "minimum_absolute_sampled_surface_distance_mm": 1000.0 * minimum_surface_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--source-ids", nargs="+", required=True)
    parser.add_argument("--canonical-names", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--penetration-samples", type=int, default=1024)
    parser.add_argument("--penetration-frame-stride", type=int, default=2)
    parser.add_argument(
        "--static-invalid-object-fallback",
        action="store_true",
        help=(
            "For declared failure cases only, audit a one-valid-frame object as "
            "a static full-clip fallback. This does not recover its motion."
        ),
    )
    args = parser.parse_args()

    root = args.sequence_dir.resolve()
    output_root = args.output_root.resolve()
    if len(args.source_ids) != len(args.canonical_names):
        raise ValueError("--source-ids and --canonical-names must have equal length")
    if len(set(args.source_ids)) != len(args.source_ids):
        raise ValueError("--source-ids must be unique")
    if len(set(args.canonical_names)) != len(args.canonical_names):
        raise ValueError("--canonical-names must be unique")

    manifest = json.loads((root / "manifest.json").read_text())
    frame_count = int(manifest["frame_count"])
    fps = float(manifest["fps"])
    if fps <= 0.0:
        raise ValueError("FPS must be positive")
    records = {str(item["id"]): item for item in manifest["files"]["objects"]}
    unknown = [value for value in args.source_ids if value not in records]
    if unknown:
        raise ValueError(f"unknown source IDs {unknown}; available={sorted(records)}")

    mano = load_pickle(root / "hands/mano_parameters.pkl")
    valid_hands = np.asarray(mano["valid"], dtype=bool)
    if valid_hands.shape[:2] != (frame_count, 2):
        raise ValueError(f"invalid hand validity shape/count: {valid_hands.shape}")
    # Accept the common (T, 2, 1) representation and retain isolated invalid
    # observations as audit evidence instead of rejecting the whole sequence.
    valid_hands = valid_hands.reshape(frame_count, 2, -1).all(axis=2)
    cameras = load_pickle(root / "cameras/trajectory.pkl")
    camera_report = validate_transforms(
        "rectified_left_camera_to_world",
        np.asarray(cameras["rectified_left_camera_to_world"], dtype=np.float64),
        frame_count,
    )
    depth = load_pickle(root / "depth/depth.pkl")
    depth_m = np.asarray(depth["depth_m"])
    if depth_m.shape[0] != frame_count:
        raise ValueError(f"depth frame mismatch: {depth_m.shape}")

    meshes: list[trimesh.Trimesh] = []
    poses: list[np.ndarray] = []
    objects = []
    mapping_objects = []
    submission_dir = output_root / "submission_meshes" / str(manifest["sequence"])
    submission_dir.mkdir(parents=True, exist_ok=True)
    for slot, (source_id, canonical_name) in enumerate(
        zip(args.source_ids, args.canonical_names, strict=True)
    ):
        record = records[source_id]
        mesh_path = root / str(record["mesh"])
        pose_path = root / str(record["poses"])
        mesh = load_mesh(mesh_path)
        pose_record = load_pickle(pose_path)
        timeline = np.asarray(pose_record["object_to_world"], dtype=np.float64)
        valid = np.asarray(pose_record["valid"], dtype=bool)
        valid = valid.reshape(frame_count, -1).all(axis=1)
        valid_indices = np.flatnonzero(valid)
        fallback = None
        if valid.shape[0] != frame_count:
            raise ValueError(f"{source_id}: invalid pose validity length")
        if not valid.all():
            if not args.static_invalid_object_fallback or len(valid_indices) != 1:
                raise ValueError(f"{source_id}: invalid pose validity")
            anchor = int(valid_indices[0])
            timeline = np.repeat(timeline[anchor : anchor + 1], frame_count, axis=0)
            fallback = {
                "applied": True,
                "anchor_frame": anchor,
                "method": "repeat_only_valid_pose",
                "motion_recovered": False,
            }
        motion = validate_transforms(source_id, timeline, frame_count)
        raw_hash = sha256(mesh_path)
        declared_hash = record.get("mesh_sha256")
        if declared_hash and raw_hash != declared_hash:
            raise ValueError(f"{source_id}: mesh hash differs from manifest")
        submission = submission_dir / f"{canonical_name}.glb"
        if submission.exists():
            if sha256(submission) != raw_hash:
                raise FileExistsError(f"refusing to overwrite different mesh: {submission}")
        else:
            shutil.copy2(mesh_path, submission)
        objects.append(
            {
                "source_id": source_id,
                "kind": str(record.get("kind", "unknown")),
                "canonical_name": canonical_name,
                "canonical_slot": slot,
                "mesh": str(mesh_path),
                "mesh_sha256": raw_hash,
                "submission_mesh": str(submission),
                "submission_mesh_sha256": sha256(submission),
                "vertex_count": int(len(mesh.vertices)),
                "face_count": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
                "axis_aligned_extents_m": np.asarray(mesh.extents).tolist(),
                "motion": motion,
                "static_invalid_object_fallback": fallback,
            }
        )
        mapping_objects.append(
            {
                "source_id": source_id,
                "source_kind": str(record.get("kind", "unknown")),
                "canonical_name": canonical_name,
                "canonical_slot": slot,
                "raw_mesh": str(mesh_path),
                "submission_mesh": str(submission),
                "raw_sha256": raw_hash,
                "submission_sha256": sha256(submission),
            }
        )
        meshes.append(mesh)
        poses.append(timeline)

    pair_reports = []
    for first in range(len(meshes)):
        for second in range(first + 1, len(meshes)):
            pair_reports.append(
                {
                    "body_a": args.canonical_names[first],
                    "body_b": args.canonical_names[second],
                    **sampled_pair_penetration(
                        meshes[first],
                        poses[first],
                        meshes[second],
                        poses[second],
                        args.penetration_samples,
                        args.penetration_frame_stride,
                    ),
                }
            )

    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "episode_index": int(str(manifest["sequence"]).rsplit("_", 1)[-1]),
        "sequence_id": str(manifest["sequence"]),
        "challenge_track": 3,
        "objects": mapping_objects,
    }
    mapping_path = reports_dir / "canonical_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    report = {
        "schema_version": "ego-flashchord-preflight-v1",
        "sequence_id": str(manifest["sequence"]),
        "frame_count": frame_count,
        "fps": fps,
        "units": manifest["world_coordinate_system"]["units"],
        "gravity_aligned": bool(manifest["world_coordinate_system"]["gravity_aligned"]),
        "hand_valid_frames": int(np.count_nonzero(np.all(valid_hands, axis=1))),
        "left_hand_valid_frames": int(np.count_nonzero(valid_hands[:, 0])),
        "right_hand_valid_frames": int(np.count_nonzero(valid_hands[:, 1])),
        "invalid_hand_frame_indices": np.flatnonzero(~valid_hands.all(axis=1)).tolist(),
        "depth_shape": list(depth_m.shape),
        "depth_finite_fraction": float(np.isfinite(depth_m).mean()),
        "camera": camera_report,
        "objects": objects,
        "object_pair_penetration": pair_reports,
        "canonical_mapping_report": str(mapping_path),
    }
    report_path = reports_dir / "preflight_inventory.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
