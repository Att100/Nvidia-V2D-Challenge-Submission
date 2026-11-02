#!/usr/bin/env python3
"""Improve labelled hand-object contact with a bounded wrist translation.

The reconstructed MANO contacts and the retargeted robot can disagree because
the two hands have different link geometry.  This tool estimates, per frame, a
robust translation that brings the retargeted robot link centres toward the
labelled object surface.  The translation is smoothed, bounded, and faded in
and out around the labelled contact interval.  Object poses and finger joint
angles remain fixed.

This is intentionally a conservative contact refinement, not a replacement for
IK: the complete robot hand is translated rigidly through its free wrist.  A
surface clearance approximating the robot link radius prevents the objective
from placing link centres inside the object.
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
from scipy.signal import savgol_filter


def _read_row(path: Path) -> tuple[pa.Table, dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != 1:
        raise ValueError(f"expected one row in {path}")
    table = parquet.read()
    return table, {name: table[name][0].as_py() for name in table.column_names}


def _partition_value(path: Path, key: str) -> str:
    prefix = f"{key}="
    for parent in path.parents:
        if parent.name.startswith(prefix):
            return parent.name[len(prefix) :]
    raise ValueError(f"missing {key}= partition in {path}")


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _contact_weight(active: np.ndarray, transition_frames: int) -> np.ndarray:
    weight = active.astype(np.float64)
    if not np.any(active) or transition_frames <= 0:
        return weight
    contact_frames = np.flatnonzero(active)
    for frame in np.flatnonzero(~active):
        distance = int(np.min(np.abs(contact_frames - frame)))
        if distance < transition_frames:
            weight[frame] = 0.5 * (
                1.0 + np.cos(np.pi * distance / transition_frames)
            )
    return weight


def _smooth_active_runs(
    raw: np.ndarray, active: np.ndarray, window: int, polyorder: int
) -> np.ndarray:
    result = np.zeros_like(raw)
    for start, end in _runs(active):
        values = raw[start : end + 1]
        length = len(values)
        local_window = min(window if window % 2 else window + 1, length)
        if local_window % 2 == 0:
            local_window -= 1
        if local_window >= max(3, polyorder + 2):
            values = savgol_filter(
                values,
                window_length=local_window,
                polyorder=min(polyorder, local_window - 1),
                axis=0,
                mode="interp",
            )
        result[start : end + 1] = values
    # Extend the nearest active correction into the transition region.  The
    # cosine weight later brings it continuously back to zero.
    if np.any(active):
        indices = np.flatnonzero(active)
        for frame in np.flatnonzero(~active):
            nearest = int(indices[np.argmin(np.abs(indices - frame))])
            result[frame] = result[nearest]
    return result


def _link_frame_groups(side: str) -> list[list[str]]:
    p = side
    return [
        [f"{p}_hand_C_MC"],
        [f"{p}_thumb_CMC_VL_site", f"{p}_thumb_MCP_VL_site"],
        [f"{p}_thumb_MCP_VL_site", f"{p}_thumb_DP_site"],
        [f"{p}_thumb_DP_site", f"{p}_thumb_tip_site"],
        [f"{p}_index_MCP_VL_site", f"{p}_index_MP_site"],
        [f"{p}_index_MP_site", f"{p}_index_DP_site"],
        [f"{p}_index_DP_site", f"{p}_index_tip_site"],
        [f"{p}_middle_MCP_VL_site", f"{p}_middle_MP_site"],
        [f"{p}_middle_MP_site", f"{p}_middle_DP_site"],
        [f"{p}_middle_DP_site", f"{p}_middle_tip_site"],
        [f"{p}_ring_MCP_VL_site", f"{p}_ring_MP_site"],
        [f"{p}_ring_MP_site", f"{p}_ring_DP_site"],
        [f"{p}_ring_DP_site", f"{p}_ring_tip_site"],
        [f"{p}_pinky_MC_site", f"{p}_pinky_MP_site"],
        [f"{p}_pinky_MP_site", f"{p}_pinky_DP_site"],
        [f"{p}_pinky_DP_site", f"{p}_pinky_tip_site"],
    ]


def _stats_mm(values_m: np.ndarray) -> dict[str, float]:
    values = 1000.0 * np.asarray(values_m, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--support-usda", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--contact-part-id", type=int, required=True)
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--surface-clearance-mm", type=float, default=6.0)
    parser.add_argument("--max-offset-mm", type=float, default=12.0)
    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--smooth-polyorder", type=int, default=2)
    parser.add_argument("--transition-frames", type=int, default=10)
    parser.add_argument(
        "--contact-lead-frames",
        type=int,
        default=0,
        help="Reach full correction this many frames before each labelled contact run.",
    )
    parser.add_argument(
        "--include-palm", action="store_true", help="Include MANO link slot 0 in the robust fit"
    )
    args = parser.parse_args()

    source_path = args.source_parquet.resolve()
    table, source = _read_row(source_path)
    object_names = [str(value) for value in source["object_body_names"]]
    if args.object_name not in object_names:
        raise ValueError(f"{args.object_name!r} not found in {object_names}")
    object_index = object_names.index(args.object_name)
    expected_part_id = object_index + 1
    if args.contact_part_id != expected_part_id:
        raise ValueError(
            f"contact part id {args.contact_part_id} disagrees with object index "
            f"{object_index} ({args.object_name}); expected {expected_part_id}"
        )

    side = args.side
    frames_key = f"robot_{side}_frames"
    wrist_key = f"robot_{side}_wrist_position"
    frames = np.asarray(source[frames_key], dtype=np.float64)
    frame_names = [str(value) for value in source[f"{side}_robot_frame_names"]]
    name_to_index = {name: index for index, name in enumerate(frame_names)}
    groups = _link_frame_groups(side)
    missing = sorted({name for group in groups for name in group if name not in name_to_index})
    if missing:
        raise ValueError(f"robot frames required for contact fitting are missing: {missing}")
    link_centres = np.stack(
        [
            frames[:, [name_to_index[name] for name in group], :3].mean(axis=1)
            for group in groups
        ],
        axis=1,
    )

    part_ids = np.asarray(
        source[f"mano_{side}_object_contact_part_ids"], dtype=np.int32
    )
    object_points = np.asarray(
        source[f"mano_{side}_object_contact_positions"], dtype=np.float64
    )
    object_normals = np.asarray(
        source[f"mano_{side}_object_contact_normals"], dtype=np.float64
    )
    slot_active = part_ids == int(args.contact_part_id)
    active = np.any(slot_active, axis=1)
    if not np.any(active):
        raise ValueError(
            f"no {side} contact slots use part id {args.contact_part_id}"
        )

    clearance = float(args.surface_clearance_mm) / 1000.0
    raw = np.zeros((len(frames), 3), dtype=np.float64)
    fit_before: list[float] = []
    for frame in np.flatnonzero(active):
        mask = slot_active[frame].copy()
        if not args.include_palm:
            mask[0] = False
        if not np.any(mask):
            mask = slot_active[frame]
        # Stored object normals use the inward convention.  Subtracting the
        # normal therefore places the robot link centre outside the surface.
        desired = object_points[frame, mask] - clearance * object_normals[frame, mask]
        errors = desired - link_centres[frame, mask]
        raw[frame] = np.median(errors, axis=0)
        fit_before.extend(np.linalg.norm(errors, axis=1).tolist())

    smoothed = _smooth_active_runs(
        raw, active, int(args.smooth_window), int(args.smooth_polyorder)
    )
    correction = float(args.gain) * smoothed
    max_offset = float(args.max_offset_mm) / 1000.0
    norms = np.linalg.norm(correction, axis=1)
    clip = np.minimum(1.0, max_offset / np.maximum(norms, 1.0e-12))
    correction *= clip[:, None]
    application_active = active.copy()
    lead_frames = int(args.contact_lead_frames)
    if lead_frames < 0:
        raise ValueError("--contact-lead-frames must be nonnegative")
    if lead_frames:
        for start, end in _runs(active):
            application_active[max(0, start - lead_frames) : end + 1] = True
    correction *= _contact_weight(
        application_active, int(args.transition_frames)
    )[:, None]

    output = dict(source)
    output_frames = frames.copy()
    output_frames[..., :3] += correction[:, None, :]
    output[frames_key] = output_frames.tolist()
    wrist = np.asarray(source[wrist_key], dtype=np.float64)
    output[wrist_key] = (wrist + correction).tolist()

    # Hand-side label points follow the translated robot hand; object-side
    # surface points remain fixed.  Normals, part IDs, orientations, and finger
    # articulation do not change.
    link_key = f"mano_{side}_link_contact_positions"
    link_points = np.asarray(source[link_key], dtype=np.float64)
    link_points[slot_active] += np.broadcast_to(
        correction[:, None, :], link_points.shape
    )[slot_active]
    output[link_key] = link_points.tolist()

    fit_after: list[float] = []
    translated_centres = link_centres + correction[:, None, :]
    for frame in np.flatnonzero(active):
        mask = slot_active[frame].copy()
        if not args.include_palm:
            mask[0] = False
        if not np.any(mask):
            mask = slot_active[frame]
        desired = object_points[frame, mask] - clearance * object_normals[frame, mask]
        fit_after.extend(
            np.linalg.norm(desired - translated_centres[frame, mask], axis=1).tolist()
        )

    output_root = args.output_root.resolve()
    sequence_id = _partition_value(source_path, "sequence_id")
    robot_name = _partition_value(source_path, "robot_name")
    output_path = (
        output_root
        / "selected_retarget"
        / f"sequence_id={sequence_id}"
        / f"robot_name={robot_name}"
        / "part-0.parquet"
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([output], schema=table.schema),
        output_path,
        compression="snappy",
    )
    support_output = output_root / "reconstructed_stage" / f"{sequence_id}_support.usda"
    support_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.support_usda.resolve(), support_output)

    step = np.linalg.norm(np.diff(correction, axis=0), axis=1)
    correction_norm = np.linalg.norm(correction, axis=1)
    report = {
        "schema_version": "bounded-hand-object-contact-translation-v1",
        "source_parquet": str(source_path),
        "output_parquet": str(output_path),
        "side": side,
        "object_name": args.object_name,
        "object_index": object_index,
        "contact_part_id": int(args.contact_part_id),
        "contact_runs": [list(run) for run in _runs(active)],
        "contact_frame_count": int(np.sum(active)),
        "gain": float(args.gain),
        "surface_clearance_mm": float(args.surface_clearance_mm),
        "maximum_offset_mm": float(args.max_offset_mm),
        "smooth_window": int(args.smooth_window),
        "transition_frames": int(args.transition_frames),
        "contact_lead_frames": lead_frames,
        "include_palm": bool(args.include_palm),
        "object_trajectory_fixed": True,
        "finger_joint_angles_changed": False,
        "hand_orientation_changed": False,
        "mano_geometry_changed": False,
        "object_contact_points_changed": False,
        "robot_link_fit_error_before_mm": _stats_mm(np.asarray(fit_before)),
        "robot_link_fit_error_after_mm": _stats_mm(np.asarray(fit_after)),
        "applied_translation_mm": _stats_mm(correction_norm[active]),
        "maximum_translation_step_mm_per_frame": float(
            1000.0 * np.max(step) if step.size else 0.0
        ),
    }
    report_path = output_root / "reports" / "hand_object_contact_translation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
