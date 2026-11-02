#!/usr/bin/env python3
"""Propagate corrected object SE(3) tracks to their associated hands.

This is a deterministic coupled-transform operation, not contact optimization.
For each hand, the unique positive contact part ID selects its canonical object.
The exact per-frame transform ``T_new_object @ inv(T_old_object)`` is applied
to the full MANO and robot-hand trajectory.  Finger articulation is preserved.
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


def read_row(path: Path) -> tuple[pa.Table, dict[str, Any]]:
    table = pq.ParquetFile(path).read()
    if table.num_rows != 1:
        raise ValueError(f"expected exactly one row: {path}")
    return table, table.to_pylist()[0]


def partition(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}= partition in {path}")


def rotation_matrices(quaternion_wxyz: np.ndarray) -> np.ndarray:
    shape = quaternion_wxyz.shape[:-1]
    return Rotation.from_quat(
        quaternion_wxyz.reshape(-1, 4), scalar_first=True
    ).as_matrix().reshape(*shape, 3, 3)


def matrices_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    shape = matrix.shape[:-2]
    return Rotation.from_matrix(matrix.reshape(-1, 3, 3)).as_quat(
        scalar_first=True
    ).reshape(*shape, 4)


def map_points(
    values: np.ndarray,
    old_center: np.ndarray,
    new_center: np.ndarray,
    delta_matrix: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shape = values.shape
    flat = values.reshape(len(values), -1, 3)
    relative = flat - old_center[:, None, :]
    mapped = np.einsum("tij,tnj->tni", delta_matrix, relative) + new_center[:, None, :]
    return mapped.reshape(shape)


def map_wxyz(values: np.ndarray, delta_matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shape = values.shape
    original = rotation_matrices(values).reshape(len(values), -1, 3, 3)
    mapped = np.einsum("tij,tnjk->tnik", delta_matrix, original)
    return matrices_to_wxyz(mapped.reshape(*shape[:-1], 3, 3))


def surface_distances(
    mesh: trimesh.Trimesh,
    points_world: np.ndarray,
    object_position: np.ndarray,
    object_matrix: np.ndarray,
) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    shape = points_world.shape[:-1]
    flat = points_world.reshape(len(points_world), -1, 3)
    local = np.einsum(
        "tji,tnj->tni", object_matrix, flat - object_position[:, None, :]
    )
    query = trimesh.proximity.ProximityQuery(mesh)
    _closest, distance, _triangle = query.on_surface(local.reshape(-1, 3))
    return np.asarray(distance).reshape(shape)


def max_local_error(
    source_points: np.ndarray,
    output_points: np.ndarray,
    old_position: np.ndarray,
    new_position: np.ndarray,
    old_matrix: np.ndarray,
    new_matrix: np.ndarray,
) -> float:
    source_local = np.einsum(
        "tji,tnj->tni",
        old_matrix,
        source_points.reshape(len(source_points), -1, 3) - old_position[:, None, :],
    )
    output_local = np.einsum(
        "tji,tnj->tni",
        new_matrix,
        output_points.reshape(len(output_points), -1, 3) - new_position[:, None, :],
    )
    return float(np.max(np.linalg.norm(source_local - output_local, axis=-1)))


def frame_object_association(
    part_ids: np.ndarray,
    object_count: int,
    global_counts: dict[int, int],
) -> np.ndarray:
    """Choose one contacted object per frame, filling non-contact gaps by proximity.

    Contact part IDs are one-based.  A frame with several active object IDs uses
    the largest slot count, then the globally more frequent ID, then the smaller
    ID as deterministic tie breakers.  Frames without contact inherit the
    association of the nearest contact frame, so the hand transform stays
    piecewise constant through approach/release intervals.
    """
    frame_count = len(part_ids)
    association = np.full(frame_count, -1, dtype=np.int32)
    for frame, ids in enumerate(part_ids):
        values, counts = np.unique(ids[ids > 0], return_counts=True)
        if not len(values):
            continue
        ranked = sorted(
            zip(values.tolist(), counts.tolist(), strict=True),
            key=lambda item: (-item[1], -global_counts[int(item[0])], int(item[0])),
        )
        association[frame] = int(ranked[0][0]) - 1
    contact_frames = np.flatnonzero(association >= 0)
    if not len(contact_frames):
        raise ValueError("hand has no contact frames")
    for frame in np.flatnonzero(association < 0):
        nearest = contact_frames[np.argmin(np.abs(contact_frames - frame))]
        association[frame] = association[nearest]
    if np.any((association < 0) | (association >= object_count)):
        raise ValueError(f"invalid frame/object association: {association.tolist()}")
    return association


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--object-parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--allow-multi-object-hand",
        action="store_true",
        help="Allow one hand to interact with different objects over time. The "
        "per-frame dominant contact part selects the SE(3), and non-contact gaps "
        "inherit the nearest contact frame's association.",
    )
    args = parser.parse_args()

    source_path = args.source_parquet.resolve()
    object_path = args.object_parquet.resolve()
    source_table, source = read_row(source_path)
    object_table, corrected = read_row(object_path)
    object_fields = {field.name: field for field in object_table.schema}
    incompatible = [
        field.name
        for field in source_table.schema
        if field.name not in object_fields or object_fields[field.name].type != field.type
    ]
    if incompatible:
        raise ValueError(f"source/object core schema mismatch: {incompatible}")
    if source["object_body_names"] != corrected["object_body_names"]:
        raise ValueError("canonical object names/order differ")

    names = [str(value) for value in source["object_body_names"]]
    old_position = np.asarray(source["object_body_position"], dtype=np.float64)
    new_position = np.asarray(corrected["object_body_position"], dtype=np.float64)
    old_wxyz = np.asarray(source["object_body_wxyz"], dtype=np.float64)
    new_wxyz = np.asarray(corrected["object_body_wxyz"], dtype=np.float64)
    if old_position.shape != new_position.shape or old_wxyz.shape != new_wxyz.shape:
        raise ValueError("source/corrected object trajectory shapes differ")
    old_matrix = rotation_matrices(old_wxyz)
    new_matrix = rotation_matrices(new_wxyz)
    delta_matrix = np.einsum("tbij,tbkj->tbik", new_matrix, old_matrix)

    output = dict(corrected)
    reports: dict[str, Any] = {}
    for side in ("left", "right"):
        part_ids = np.asarray(
            source[f"mano_{side}_object_contact_part_ids"], dtype=np.int32
        )
        positive_values, positive_counts = np.unique(
            part_ids[part_ids > 0], return_counts=True
        )
        positive = [int(value) for value in positive_values]
        if not positive:
            raise ValueError(f"{side} hand has no positive contact part IDs")
        dominant = int(positive_values[int(np.argmax(positive_counts))])
        dominant_fraction = float(np.max(positive_counts) / np.sum(positive_counts))
        multi_object = len(positive) > 1 and dominant_fraction < 0.95
        if multi_object and not args.allow_multi_object_hand:
            raise ValueError(
                f"{side} hand has no unambiguous dominant object: IDs {positive}, "
                f"counts {positive_counts.tolist()}"
            )
        if dominant < 1 or dominant > len(names):
            raise ValueError(f"invalid {side} contact part ID {positive[0]}")
        global_counts = {
            int(value): int(count)
            for value, count in zip(positive_values, positive_counts, strict=True)
        }
        if multi_object:
            association = frame_object_association(part_ids, len(names), global_counts)
            association_method = "per-frame dominant contact; nearest-contact gap fill"
        else:
            association = np.full(len(part_ids), dominant - 1, dtype=np.int32)
            association_method = "dominant positive contact slot count"
        frames_index = np.arange(len(part_ids))
        old_center = old_position[frames_index, association]
        new_center = new_position[frames_index, association]
        old_rotation = old_matrix[frames_index, association]
        new_rotation = new_matrix[frames_index, association]
        delta = delta_matrix[frames_index, association]

        source_joints = np.asarray(source[f"mano_{side}_joints"], dtype=np.float64)
        mapped_joints = map_points(source_joints, old_center, new_center, delta)
        output[f"mano_{side}_joints"] = mapped_joints.tolist()
        output[f"mano_{side}_trans"] = map_points(
            np.asarray(source[f"mano_{side}_trans"], dtype=np.float64),
            old_center,
            new_center,
            delta,
        ).tolist()
        source_global = Rotation.from_rotvec(
            np.asarray(source[f"mano_{side}_global_orient"], dtype=np.float64)
        ).as_matrix()
        output[f"mano_{side}_global_orient"] = Rotation.from_matrix(
            np.einsum("tij,tjk->tik", delta, source_global)
        ).as_rotvec().tolist()
        output[f"mano_{side}_joints_wxyz"] = map_wxyz(
            np.asarray(source[f"mano_{side}_joints_wxyz"], dtype=np.float64), delta
        ).tolist()

        source_wrist = np.asarray(
            source[f"robot_{side}_wrist_position"], dtype=np.float64
        )
        mapped_wrist = map_points(source_wrist, old_center, new_center, delta)
        output[f"robot_{side}_wrist_position"] = mapped_wrist.tolist()
        output[f"robot_{side}_wrist_wxyz"] = map_wxyz(
            np.asarray(source[f"robot_{side}_wrist_wxyz"], dtype=np.float64), delta
        ).tolist()
        output[f"robot_{side}_finger_joints"] = source[
            f"robot_{side}_finger_joints"
        ]
        frames = np.asarray(source[f"robot_{side}_frames"], dtype=np.float64)
        mapped_frames = frames.copy()
        mapped_frames[..., :3] = map_points(
            frames[..., :3], old_center, new_center, delta
        )
        mapped_frames[..., 3:] = map_wxyz(frames[..., 3:], delta)
        output[f"robot_{side}_frames"] = mapped_frames.tolist()

        slot_active = part_ids > 0
        for suffix in ("link_contact_positions", "object_contact_positions"):
            key = f"mano_{side}_{suffix}"
            values = np.asarray(source[key], dtype=np.float64)
            for part_id in positive:
                slot_mask = part_ids == part_id
                part_index = part_id - 1
                mapped = map_points(
                    values,
                    old_position[:, part_index],
                    new_position[:, part_index],
                    delta_matrix[:, part_index],
                )
                values[slot_mask] = mapped[slot_mask]
            output[key] = values.tolist()
        for suffix in ("link_contact_normals", "object_contact_normals"):
            key = f"mano_{side}_{suffix}"
            values = np.asarray(source[key], dtype=np.float64)
            for part_id in positive:
                slot_mask = part_ids == part_id
                part_index = part_id - 1
                mapped = np.einsum(
                    "tij,tnj->tni", delta_matrix[:, part_index], values
                )
                values[slot_mask] = mapped[slot_mask]
            output[key] = values.tolist()
        output[f"mano_{side}_object_contact_part_ids"] = part_ids.tolist()

        tip_distance = np.empty((len(part_ids), len(FINGERTIP_INDICES)), dtype=np.float64)
        for object_index in np.unique(association):
            frame_mask = association == object_index
            mesh_path = Path(corrected["object_mesh_paths"][object_index]).resolve()
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            tip_distance[frame_mask] = surface_distances(
                mesh,
                mapped_joints[frame_mask][:, FINGERTIP_INDICES],
                new_center[frame_mask],
                new_rotation[frame_mask],
            )
        output[f"mano_{side}_tips_distance"] = tip_distance.tolist()

        wrist_correction = mapped_wrist - source_wrist
        wrist_step = 1000.0 * np.linalg.norm(np.diff(wrist_correction, axis=0), axis=-1)
        association_counts = {
            names[index]: int(np.sum(association == index))
            for index in np.unique(association)
        }
        reports[side] = {
            "object_index": None if multi_object else int(dominant - 1),
            "object_name": None if multi_object else names[dominant - 1],
            "association_method": association_method,
            "positive_part_id_slot_counts": {
                str(value): count for value, count in global_counts.items()
            },
            "dominant_part_id_fraction": dominant_fraction,
            "associated_frame_counts": association_counts,
            "association_switch_count": int(np.sum(np.diff(association) != 0)),
            "transform_applied_to_all_frames": True,
            "contact_frame_count": int(np.sum(np.any(slot_active, axis=1))),
            "maximum_hand_object_local_coordinate_error_m": max_local_error(
                source_joints,
                mapped_joints,
                old_center,
                new_center,
                old_rotation,
                new_rotation,
            ),
            "wrist_correction_mm": {
                "mean": float(1000.0 * np.mean(np.linalg.norm(wrist_correction, axis=-1))),
                "maximum": float(1000.0 * np.max(np.linalg.norm(wrist_correction, axis=-1))),
                "maximum_step_per_frame": float(np.max(wrist_step)) if wrist_step.size else 0.0,
            },
        }

    sequence = partition(object_path, "sequence_id")
    robot = partition(object_path, "robot_name")
    root = args.output_root.resolve()
    output_path = root / "selected_retarget" / f"sequence_id={sequence}" / f"robot_name={robot}" / "part-0.parquet"
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    output_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([output], schema=object_table.schema),
        output_path,
        compression="snappy",
    )
    support_output = root / "reconstructed_stage" / f"{sequence}_support.usda"
    support_output.parent.mkdir(parents=True)
    shutil.copy2(args.support_usda.resolve(), support_output)
    report = {
        "schema_version": "full-trajectory-object-se3-to-hands-v2",
        "status": "GENERATED",
        "source_parquet": str(source_path),
        "object_parquet": str(object_path),
        "output_parquet": str(output_path),
        "object_trajectory_fixed": True,
        "method": "exact per-frame T_new_object @ inverse(T_old_object) applied to the contact-associated hand over all frames",
        "contact_optimization_used": False,
        "contact_weighting_used": False,
        "transition_frames_used": False,
        "finger_joint_angles_changed": False,
        "contact_part_ids_changed": False,
        "sides": reports,
    }
    report_path = root / "reports" / "object_se3_hand_propagation.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
