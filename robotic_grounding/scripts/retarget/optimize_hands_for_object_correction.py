#!/usr/bin/env python3
"""Align retargeted hands to already-selected rigid-object corrections.

The object-only stage intentionally changes only the object trajectory.  This
second stage keeps that trajectory fixed and maps each hand through the exact
per-frame old-object -> new-object rigid transform while contact labels are
active.  A cosine transition outside contact intervals avoids wrist jumps.

The source and object-only parquets remain untouched.  Contact positions,
normals, MANO geometry, robot wrists, and robot FK frames are updated together.
Finger joint angles are unchanged because a world-frame rigid transform does
not alter hand articulation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

FINGERTIP_INDICES = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)


def _read_row(path: Path) -> tuple[pa.Table, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != 1:
        raise ValueError(f"expected exactly one row in {path}")
    table = parquet.read()
    return table, {name: table[name][0].as_py() for name in table.column_names}


def _partition_value(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}= partition in {path}")


def _contact_weight(
    active: np.ndarray, transition_frames: int, transition_out_frames: int
) -> np.ndarray:
    active = np.asarray(active, dtype=bool)
    weight = active.astype(np.float64)
    if not np.any(active) or (transition_frames <= 0 and transition_out_frames <= 0):
        return weight
    active_frames = np.flatnonzero(active)
    for frame in np.flatnonzero(~active):
        nearest = int(active_frames[np.argmin(np.abs(active_frames - frame))])
        distance = abs(nearest - frame)
        transition = transition_frames if nearest > frame else transition_out_frames
        if transition > 0 and distance < transition:
            weight[frame] = 0.5 * (1.0 + np.cos(np.pi * distance / transition))
    return weight


def _rotation_matrices(quaternion_wxyz: np.ndarray) -> np.ndarray:
    shape = quaternion_wxyz.shape[:-1]
    return Rotation.from_quat(quaternion_wxyz.reshape(-1, 4), scalar_first=True).as_matrix().reshape(*shape, 3, 3)


def _matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    shape = matrix.shape[:-2]
    return Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(*shape, 4)


def _map_points(
    values: np.ndarray,
    old_center: np.ndarray,
    new_center: np.ndarray,
    delta_matrix: np.ndarray,
) -> np.ndarray:
    """Apply a different object-centered rigid transform at each frame."""
    values = np.asarray(values, dtype=np.float64)
    shape = values.shape
    flat = values.reshape(len(values), -1, 3)
    relative = flat - old_center[:, None, :]
    mapped = np.einsum("tij,tnj->tni", delta_matrix, relative) + new_center[:, None, :]
    return mapped.reshape(shape)


def _map_wxyz(values: np.ndarray, delta_matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shape = values.shape
    original = _rotation_matrices(values)
    flat = original.reshape(len(values), -1, 3, 3)
    mapped = np.einsum("tij,tnjk->tnik", delta_matrix, flat)
    return _matrix_to_wxyz(mapped.reshape(*shape[:-1], 3, 3))


def _surface_distances(
    query: trimesh.proximity.ProximityQuery,
    points_world: np.ndarray,
    object_position: np.ndarray,
    object_matrix: np.ndarray,
) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    shape = points_world.shape[:-1]
    flat = points_world.reshape(len(points_world), -1, 3)
    relative = flat - object_position[:, None, :]
    local = np.einsum("tji,tnj->tni", object_matrix, relative)
    _closest, distance, _triangle = query.on_surface(local.reshape(-1, 3))
    return np.asarray(distance).reshape(shape)


def _runs(active: np.ndarray) -> list[list[int]]:
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [[int(start), int(end)] for start, end in zip(starts, ends, strict=True)]


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True, help="Pre-object-correction retarget parquet")
    parser.add_argument("--object-parquet", type=Path, required=True, help="Selected object-only parquet")
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--transition-frames", type=int, default=10)
    parser.add_argument("--transition-out-frames", type=int, default=None)
    parser.add_argument("--contact-threshold-mm", type=float, default=10.0)
    args = parser.parse_args()

    source_path = args.source_parquet.resolve()
    object_path = args.object_parquet.resolve()
    source_table, source = _read_row(source_path)
    object_table, corrected = _read_row(object_path)
    object_fields = {field.name: field for field in object_table.schema}
    incompatible = [
        field.name
        for field in source_table.schema
        if field.name not in object_fields or object_fields[field.name].type != field.type
    ]
    if incompatible:
        raise ValueError(f"source/object parquet core schema mismatch: {incompatible}")
    if source["object_body_names"] != corrected["object_body_names"]:
        raise ValueError("object names/order differ between source and object-only parquet")
    object_names = [str(value) for value in source["object_body_names"]]
    old_position_all = np.asarray(source["object_body_position"], dtype=np.float64)
    new_position_all = np.asarray(corrected["object_body_position"], dtype=np.float64)
    old_wxyz_all = np.asarray(source["object_body_wxyz"], dtype=np.float64)
    new_wxyz_all = np.asarray(corrected["object_body_wxyz"], dtype=np.float64)
    old_position = old_position_all[:, 0]
    new_position = new_position_all[:, 0]
    if old_position.shape != new_position.shape:
        raise ValueError("object trajectories have different lengths")
    frame_count = len(old_position)
    old_matrix_all = _rotation_matrices(old_wxyz_all)
    new_matrix_all = _rotation_matrices(new_wxyz_all)
    full_delta_all = np.einsum("tbij,tbkj->tbik", new_matrix_all, old_matrix_all)

    output = dict(corrected)
    threshold_m = float(args.contact_threshold_mm) / 1000.0
    transition_out_frames = (
        int(args.transition_frames)
        if args.transition_out_frames is None
        else int(args.transition_out_frames)
    )
    side_reports: dict[str, Any] = {}
    overall_pass = True

    for side in ("right", "left"):
        part_ids = np.asarray(source[f"mano_{side}_object_contact_part_ids"], dtype=np.int32)
        positive_values, positive_counts = np.unique(
            part_ids[part_ids > 0], return_counts=True
        )
        contacted = [int(value) for value in positive_values]
        if contacted:
            dominant_slot = int(np.argmax(positive_counts))
            dominant_part_id = int(positive_values[dominant_slot])
            dominant_fraction = float(
                positive_counts[dominant_slot] / np.sum(positive_counts)
            )
            if len(contacted) > 1 and dominant_fraction < 0.95:
                raise ValueError(
                    f"{side} hand has no dominant contact object: IDs {contacted}, "
                    f"counts {positive_counts.tolist()}"
                )
            object_index = dominant_part_id - 1
            if object_index < 0 or object_index >= len(object_names):
                raise ValueError(
                    f"invalid contact part ID {dominant_part_id} for {object_names}"
                )
            active = np.any(part_ids == dominant_part_id, axis=1)
        else:
            dominant_part_id = 0
            dominant_fraction = 1.0
            object_index = 0
            active = np.zeros(len(part_ids), dtype=bool)
        object_name = object_names[object_index]
        old_position = old_position_all[:, object_index]
        new_position = new_position_all[:, object_index]
        old_matrix = old_matrix_all[:, object_index]
        new_matrix = new_matrix_all[:, object_index]
        full_delta_matrix = full_delta_all[:, object_index]
        mesh_path = Path(corrected["object_mesh_paths"][object_index]).resolve()
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError(f"invalid object mesh: {mesh_path}")
        query = trimesh.proximity.ProximityQuery(mesh)
        alpha = _contact_weight(active, int(args.transition_frames), transition_out_frames)
        full_rotvec = Rotation.from_matrix(full_delta_matrix).as_rotvec()
        partial_delta = Rotation.from_rotvec(alpha[:, None] * full_rotvec).as_matrix()
        partial_center = old_position + alpha[:, None] * (new_position - old_position)

        source_joints = np.asarray(source[f"mano_{side}_joints"], dtype=np.float64)
        mapped_joints = _map_points(source_joints, old_position, partial_center, partial_delta)
        output[f"mano_{side}_joints"] = mapped_joints.tolist()
        output[f"mano_{side}_trans"] = _map_points(
            np.asarray(source[f"mano_{side}_trans"], dtype=np.float64),
            old_position,
            partial_center,
            partial_delta,
        ).tolist()

        global_matrix = Rotation.from_rotvec(
            np.asarray(source[f"mano_{side}_global_orient"], dtype=np.float64)
        ).as_matrix()
        output[f"mano_{side}_global_orient"] = Rotation.from_matrix(
            np.einsum("tij,tjk->tik", partial_delta, global_matrix)
        ).as_rotvec().tolist()
        output[f"mano_{side}_joints_wxyz"] = _map_wxyz(
            np.asarray(source[f"mano_{side}_joints_wxyz"], dtype=np.float64), partial_delta
        ).tolist()

        source_wrist = np.asarray(source[f"robot_{side}_wrist_position"], dtype=np.float64)
        mapped_wrist = _map_points(source_wrist, old_position, partial_center, partial_delta)
        output[f"robot_{side}_wrist_position"] = mapped_wrist.tolist()
        output[f"robot_{side}_wrist_wxyz"] = _map_wxyz(
            np.asarray(source[f"robot_{side}_wrist_wxyz"], dtype=np.float64), partial_delta
        ).tolist()
        output[f"robot_{side}_finger_joints"] = source[f"robot_{side}_finger_joints"]

        frames = np.asarray(source[f"robot_{side}_frames"], dtype=np.float64)
        mapped_frames = frames.copy()
        mapped_frames[..., :3] = _map_points(frames[..., :3], old_position, partial_center, partial_delta)
        mapped_frames[..., 3:] = _map_wxyz(frames[..., 3:], partial_delta)
        output[f"robot_{side}_frames"] = mapped_frames.tolist()

        # Contact points follow the object named by each slot, including rare
        # secondary-object annotations.  The hand trajectory itself follows
        # only its unambiguous dominant object.
        slot_active = part_ids > 0
        for suffix in ("link_contact_positions", "object_contact_positions"):
            key = f"mano_{side}_{suffix}"
            values = np.asarray(source[key], dtype=np.float64)
            for part_id in contacted:
                part_index = part_id - 1
                part_mask = part_ids == part_id
                mapped = _map_points(
                    values,
                    old_position_all[:, part_index],
                    new_position_all[:, part_index],
                    full_delta_all[:, part_index],
                )
                values[part_mask] = mapped[part_mask]
            output[key] = values.tolist()
        for suffix in ("link_contact_normals", "object_contact_normals"):
            key = f"mano_{side}_{suffix}"
            values = np.asarray(source[key], dtype=np.float64)
            for part_id in contacted:
                part_index = part_id - 1
                part_mask = part_ids == part_id
                mapped = np.einsum(
                    "tij,tnj->tni", full_delta_all[:, part_index], values
                )
                values[part_mask] = mapped[part_mask]
            output[key] = values.tolist()
        output[f"mano_{side}_object_contact_part_ids"] = part_ids.tolist()

        old_tips = source_joints[:, FINGERTIP_INDICES]
        new_tips = mapped_joints[:, FINGERTIP_INDICES]
        original_distance = _surface_distances(query, old_tips, old_position, old_matrix)
        mismatch_distance = _surface_distances(query, old_tips, new_position, new_matrix)
        optimized_distance = _surface_distances(query, new_tips, new_position, new_matrix)
        output[f"mano_{side}_tips_distance"] = optimized_distance.tolist()

        contact_pairs = original_distance < threshold_m
        if not np.any(contact_pairs):
            contact_pairs = np.repeat(active[:, None], len(FINGERTIP_INDICES), axis=1)
        mismatch_error = 1000.0 * np.abs(mismatch_distance[contact_pairs] - original_distance[contact_pairs])
        optimized_error = 1000.0 * np.abs(optimized_distance[contact_pairs] - original_distance[contact_pairs])
        recall_before = float(np.mean(mismatch_distance[contact_pairs] < threshold_m)) if np.any(contact_pairs) else 1.0
        recall_after = float(np.mean(optimized_distance[contact_pairs] < threshold_m)) if np.any(contact_pairs) else 1.0

        joint_correction = 1000.0 * np.linalg.norm(mapped_joints - source_joints, axis=-1)
        wrist_correction = mapped_wrist - source_wrist
        correction_step = 1000.0 * np.linalg.norm(np.diff(wrist_correction, axis=0), axis=-1)
        active_joint = joint_correction[active] if np.any(active) else np.empty(0)
        inactive_joint = joint_correction[alpha == 0.0]
        optimized_stats = _stats(optimized_error)
        side_pass = optimized_stats["mean"] < 10.0 and recall_after >= 0.70
        overall_pass &= side_pass
        side_reports[side] = {
            "status": "PASS" if side_pass else "REVIEW",
            "contact_object_index": object_index,
            "contact_object_name": object_name,
            "dominant_contact_part_id": dominant_part_id,
            "positive_part_id_slot_counts": {
                str(int(value)): int(count)
                for value, count in zip(
                    positive_values, positive_counts, strict=True
                )
            },
            "dominant_part_id_fraction": dominant_fraction,
            "contact_runs": _runs(active),
            "contact_frame_count": int(np.sum(active)),
            "transition_frames": int(args.transition_frames),
            "transition_out_frames": transition_out_frames,
            "correction_weight_nonzero_frames": int(np.sum(alpha > 0.0)),
            "contact_tip_pairs": int(np.sum(contact_pairs)),
            "contact_distance_error_before_mm": _stats(mismatch_error),
            "contact_distance_error_after_mm": optimized_stats,
            "contact_recall_before": recall_before,
            "contact_recall_after": recall_after,
            "active_hand_joint_correction_mm": _stats(active_joint),
            "fully_inactive_hand_joint_correction_mm": _stats(inactive_joint),
            "maximum_wrist_correction_step_mm_per_frame": float(np.max(correction_step)) if correction_step.size else 0.0,
        }

    sequence_id = _partition_value(object_path, "sequence_id")
    robot_name = _partition_value(object_path, "robot_name")
    output_root = args.output_root.resolve()
    output_path = output_root / "selected_retarget" / f"sequence_id={sequence_id}" / f"robot_name={robot_name}" / "part-0.parquet"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing optimized parquet: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([output], schema=object_table.schema), output_path, compression="snappy")

    support_output = output_root / "reconstructed_stage" / f"{sequence_id}_support.usda"
    support_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.support_usda.resolve(), support_output)
    report = {
        "schema_version": "multi-object-contact-aware-hand-optimization-v2",
        "status": "PASS" if overall_pass else "REVIEW",
        "source_parquet": str(source_path),
        "object_only_parquet": str(object_path),
        "output_parquet": str(output_path),
        "object_trajectory_fixed": True,
        "object_names": object_names,
        "transition_frames": int(args.transition_frames),
        "transition_out_frames": transition_out_frames,
        "contact_threshold_mm": float(args.contact_threshold_mm),
        "method": "per-hand canonical-object-relative rigid mapping on labelled contact frames with cosine transition",
        "finger_joint_angles_changed": False,
        "contact_part_ids_changed": False,
        "sides": side_reports,
        "acceptance": {"active_contact_distance_mae_mm_lt": 10.0, "active_contact_recall_gte": 0.70},
    }
    report_path = output_root / "reports" / "hand_contact_optimization.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
