#!/usr/bin/env python3
"""Build a non-destructive release-timing fix for episode_000008 Sharpa data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pinocchio as pin
import trimesh
from scipy.spatial.transform import Rotation

from robotic_grounding.retarget.data_logger import ManoSharpaData
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics


PROBE_SUFFIXES = (
    "hand_C_MC",
    "thumb_MCP_VL_site",
    "thumb_tip_site",
    "index_MP_site",
    "index_tip_site",
    "middle_MP_site",
    "middle_tip_site",
    "ring_MP_site",
    "ring_tip_site",
    "pinky_MP_site",
    "pinky_tip_site",
)
CONTACT_SLOTS = 16


def smoothstep(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def nlerp(values: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Hemisphere-safe normalized quaternion interpolation on the first axis."""
    lo = np.floor(sample).astype(int)
    hi = np.minimum(lo + 1, values.shape[0] - 1)
    alpha = (sample - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    q0 = values[lo]
    q1 = values[hi].copy()
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    out = (1.0 - alpha) * q0 + alpha * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1.0e-12)


def linear(values: np.ndarray, sample: np.ndarray) -> np.ndarray:
    lo = np.floor(sample).astype(int)
    hi = np.minimum(lo + 1, values.shape[0] - 1)
    alpha = (sample - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    return (1.0 - alpha) * values[lo] + alpha * values[hi]


def nearest(values: np.ndarray, sample: np.ndarray) -> np.ndarray:
    return values[np.floor(sample + 0.5).astype(int)]


def arc_length_time_map(wrist: np.ndarray, start: int, end: int) -> np.ndarray:
    """Map output frames to original fractional frames at uniform wrist arc speed."""
    if not 0 <= start < end < len(wrist):
        raise ValueError(f"invalid release interval {start}:{end} for {len(wrist)} frames")
    segment = wrist[start : end + 1]
    distance = np.linalg.norm(np.diff(segment, axis=0), axis=-1)
    cumulative = np.r_[0.0, np.cumsum(distance)]
    if cumulative[-1] <= 0.0:
        raise ValueError("release segment has zero wrist path length")
    # Remove zero-length intervals so np.interp has a strictly increasing x axis.
    keep = np.r_[True, np.diff(cumulative) > 1.0e-12]
    target_distance = np.linspace(0.0, cumulative[-1], end - start + 1)
    original_local = np.interp(target_distance, cumulative[keep], np.arange(len(segment))[keep])
    sample = np.arange(len(wrist), dtype=np.float64)
    sample[start : end + 1] = start + original_local
    return sample


def resample_time_series(data: ManoSharpaData, sample: np.ndarray) -> None:
    quaternion_fields = {
        "mano_right_joints_wxyz",
        "mano_left_joints_wxyz",
        "robot_right_wrist_wxyz",
        "robot_left_wrist_wxyz",
        "object_body_wxyz",
    }
    robot_frame_fields = {"robot_right_frames", "robot_left_frames"}
    for name, _, _, is_time_series in data._field_specs:
        if not is_time_series:
            continue
        raw = getattr(data, name)
        if raw is None or len(raw) != len(sample):
            continue
        values = np.asarray(raw)
        if values.dtype.kind not in "biuf":
            continue
        if name in robot_frame_fields:
            out = np.empty_like(values, dtype=np.float64)
            out[..., :3] = linear(values[..., :3].astype(np.float64), sample)
            out[..., 3:7] = nlerp(values[..., 3:7].astype(np.float64), sample)
        elif name in quaternion_fields:
            out = nlerp(values.astype(np.float64), sample)
        elif values.dtype.kind in "biu":
            out = nearest(values, sample)
        else:
            out = linear(values.astype(np.float64), sample)
        setattr(data, name, out.tolist())


def advance_left_finger_release(
    data: ManoSharpaData,
    lead_frames: float,
    blend_start: int,
    blend_end: int,
) -> np.ndarray:
    finger = np.asarray(data.robot_left_finger_joints, dtype=np.float64)
    frame = np.arange(len(finger), dtype=np.float64)
    fraction = smoothstep((frame - blend_start) / float(blend_end - blend_start))
    sample = np.minimum(frame + lead_frames * fraction, len(finger) - 1)
    data.robot_left_finger_joints = linear(finger, sample).tolist()
    return sample


def adjust_right_contact_clearance(
    data: ManoSharpaData,
    inset_m: float,
    contact_start: int,
    contact_end: int,
    fade_frames: int,
    clearance_m: float,
    clearance_start: int,
    clearance_peak: int,
    clearance_end: int,
) -> dict:
    """Move the right wrist slightly toward object-1, except for one penetration-risk bump."""
    wrist = np.asarray(data.robot_right_wrist_position, dtype=np.float64)
    object_position = np.asarray(data.object_body_position, dtype=np.float64)[:, 1]
    toward = object_position - wrist
    toward[:, 2] = 0.0
    toward /= np.linalg.norm(toward, axis=-1, keepdims=True).clip(min=1.0e-12)
    frames = np.arange(len(wrist), dtype=np.float64)
    contact_weight = np.zeros(len(wrist), dtype=np.float64)
    inside = (frames >= contact_start) & (frames <= contact_end)
    contact_weight[inside] = 1.0
    if fade_frames > 0:
        lead = (frames >= contact_start - fade_frames) & (frames < contact_start)
        trail = (frames > contact_end) & (frames <= contact_end + fade_frames)
        contact_weight[lead] = smoothstep(
            (frames[lead] - (contact_start - fade_frames)) / fade_frames
        )
        contact_weight[trail] = smoothstep(
            ((contact_end + fade_frames) - frames[trail]) / fade_frames
        )

    clearance_weight = np.zeros(len(wrist), dtype=np.float64)
    rising = (frames >= clearance_start) & (frames <= clearance_peak)
    falling = (frames > clearance_peak) & (frames <= clearance_end)
    clearance_weight[rising] = smoothstep(
        (frames[rising] - clearance_start) / max(clearance_peak - clearance_start, 1)
    )
    clearance_weight[falling] = smoothstep(
        (clearance_end - frames[falling]) / max(clearance_end - clearance_peak, 1)
    )
    signed_offset = inset_m * contact_weight - clearance_m * clearance_weight
    data.robot_right_wrist_position = (wrist + toward * signed_offset[:, None]).tolist()
    return {
        "max_inward_mm": float(np.maximum(signed_offset, 0.0).max() * 1000.0),
        "max_outward_mm": float(np.maximum(-signed_offset, 0.0).max() * 1000.0),
        "offset_at_clearance_peak_mm": float(signed_offset[clearance_peak] * 1000.0),
    }


def adjust_left_release_clearance(
    data: ManoSharpaData,
    clearance_m: float,
    clearance_start: int,
    clearance_peak: int,
    clearance_end: int,
) -> dict:
    """Add a smooth, local wrist displacement away from object-0 during release."""
    wrist = np.asarray(data.robot_left_wrist_position, dtype=np.float64)
    object_position = np.asarray(data.object_body_position, dtype=np.float64)[:, 0]
    away = wrist - object_position
    away /= np.linalg.norm(away, axis=-1, keepdims=True).clip(min=1.0e-12)
    frames = np.arange(len(wrist), dtype=np.float64)
    weight = np.zeros(len(wrist), dtype=np.float64)
    rising = (frames >= clearance_start) & (frames <= clearance_peak)
    falling = (frames > clearance_peak) & (frames <= clearance_end)
    weight[rising] = smoothstep(
        (frames[rising] - clearance_start) / max(clearance_peak - clearance_start, 1)
    )
    weight[falling] = smoothstep(
        (clearance_end - frames[falling]) / max(clearance_end - clearance_peak, 1)
    )
    offset = clearance_m * weight
    data.robot_left_wrist_position = (wrist + away * offset[:, None]).tolist()
    return {
        "max_clearance_mm": float(offset.max() * 1000.0),
        "peak_frame": clearance_peak,
        "interval": [clearance_start, clearance_end],
    }


def adjust_left_stability_clearance(
    data: ManoSharpaData,
    clearance_m: float,
    intervals: tuple[tuple[int, int, int], ...],
) -> dict:
    """Reduce object torque with small outward bumps at diagnosed instability intervals."""
    wrist = np.asarray(data.robot_left_wrist_position, dtype=np.float64)
    object_position = np.asarray(data.object_body_position, dtype=np.float64)[:, 0]
    away = wrist - object_position
    away /= np.linalg.norm(away, axis=-1, keepdims=True).clip(min=1.0e-12)
    frames = np.arange(len(wrist), dtype=np.float64)
    weight = np.zeros(len(wrist), dtype=np.float64)
    for start, peak, end in intervals:
        bump = np.zeros(len(wrist), dtype=np.float64)
        rising = (frames >= start) & (frames <= peak)
        falling = (frames > peak) & (frames <= end)
        bump[rising] = smoothstep((frames[rising] - start) / max(peak - start, 1))
        bump[falling] = smoothstep((end - frames[falling]) / max(end - peak, 1))
        weight += bump
    offset = clearance_m * weight
    data.robot_left_wrist_position = (wrist + away * offset[:, None]).tolist()
    return {
        "clearance_mm": float(clearance_m * 1000.0),
        "max_combined_clearance_mm": float(offset.max() * 1000.0),
        "intervals": [list(interval) for interval in intervals],
    }


def recompute_robot_frames(data: ManoSharpaData, side: str) -> np.ndarray:
    kin = setup_sharpa_kinematics(side)
    stored_names = list(getattr(data, f"{side}_robot_frame_names"))
    model_names = list(kin.robot_frame_names.values())
    if stored_names != model_names:
        raise ValueError(f"{side} frame-name order differs from the Sharpa FK model")
    stored_joints = list(getattr(data, f"{side}_robot_finger_joint_names"))
    model_joints = list(kin.robot_finger_joint_names.values())
    if stored_joints != model_joints:
        raise ValueError(f"{side} finger-joint order differs from the Sharpa FK model")

    wrist = np.asarray(getattr(data, f"robot_{side}_wrist_position"), dtype=np.float64)
    wrist_wxyz = np.asarray(getattr(data, f"robot_{side}_wrist_wxyz"), dtype=np.float64)
    finger = np.asarray(getattr(data, f"robot_{side}_finger_joints"), dtype=np.float64)
    frames = np.zeros((len(wrist), len(model_names), 7), dtype=np.float64)
    q = kin.robot.q0.copy()
    for frame in range(len(wrist)):
        q[:3] = wrist[frame]
        q[3:7] = wrist_wxyz[frame, [1, 2, 3, 0]]  # wxyz -> Pinocchio xyzw
        q[7:] = finger[frame]
        pin.forwardKinematics(kin.robot.model, kin.robot.data, q)
        pin.updateFramePlacements(kin.robot.model, kin.robot.data)
        for index, placement in enumerate(kin.robot.data.oMf):
            frames[frame, index, :3] = placement.translation
            frames[frame, index, 3:7] = Rotation.from_matrix(placement.rotation).as_quat(scalar_first=True)
    setattr(data, f"robot_{side}_frames", frames.tolist())
    return frames


def deflicker(active: np.ndarray, min_consecutive: int) -> np.ndarray:
    output = np.zeros_like(active)
    for slot in range(active.shape[1]):
        begin = 0
        while begin < active.shape[0]:
            if not active[begin, slot]:
                begin += 1
                continue
            end = begin + 1
            while end < active.shape[0] and active[end, slot]:
                end += 1
            if end - begin >= min_consecutive:
                output[begin:end, slot] = True
            begin = end
    return output


def mesh_probe_contacts(
    mesh_path: str,
    probes_w: np.ndarray,
    object_position: np.ndarray,
    object_wxyz: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    query = trimesh.proximity.ProximityQuery(mesh)
    rotation = Rotation.from_quat(object_wxyz, scalar_first=True).as_matrix()
    relative = probes_w - object_position[:, None, :]
    probes_object = np.einsum("tji,tkj->tki", rotation, relative)
    closest_object, distance, triangle = query.on_surface(probes_object.reshape(-1, 3))
    closest_object = closest_object.reshape(probes_w.shape)
    distance = distance.reshape(probes_w.shape[:2])
    triangle = triangle.reshape(probes_w.shape[:2])
    active = deflicker(distance < threshold, min_consecutive=2)
    closest_w = np.einsum("tij,tkj->tki", rotation, closest_object) + object_position[:, None, :]
    inward_object = -np.asarray(mesh.face_normals, dtype=np.float64)[triangle]
    inward_w = np.einsum("tij,tkj->tki", rotation, inward_object)
    inward_w /= np.linalg.norm(inward_w, axis=-1, keepdims=True).clip(min=1.0e-12)
    return closest_w, inward_w, np.where(active, distance, np.inf)


def recompute_contacts(
    data: ManoSharpaData,
    frames_by_side: dict[str, np.ndarray],
    threshold: float,
) -> dict:
    num_frames = len(data.robot_left_wrist_position)
    num_probes = len(PROBE_SUFFIXES)
    object_position = np.asarray(data.object_body_position, dtype=np.float64)
    object_wxyz = np.asarray(data.object_body_wxyz, dtype=np.float64)
    mesh_paths = list(data.object_mesh_paths)
    result = {}
    for side in ("left", "right"):
        names = list(getattr(data, f"{side}_robot_frame_names"))
        probe_indices = [names.index(f"{side}_{suffix}") for suffix in PROBE_SUFFIXES]
        probes = frames_by_side[side][:, probe_indices, :3]
        best_distance = np.full((num_frames, num_probes), np.inf, dtype=np.float64)
        best_position = np.zeros((num_frames, num_probes, 3), dtype=np.float64)
        best_normal = np.zeros_like(best_position)
        best_part = np.zeros((num_frames, num_probes), dtype=np.int32)
        for body, mesh_path in enumerate(mesh_paths):
            position, normal, distance = mesh_probe_contacts(
                mesh_path,
                probes,
                object_position[:, body],
                object_wxyz[:, body],
                threshold,
            )
            better = distance < best_distance
            best_distance[better] = distance[better]
            best_position[better] = position[better]
            best_normal[better] = normal[better]
            best_part[better] = body + 1

        active = best_part > 0
        object_contact_position = np.zeros((num_frames, CONTACT_SLOTS, 3), dtype=np.float64)
        object_contact_normal = np.zeros_like(object_contact_position)
        part_ids = np.zeros((num_frames, CONTACT_SLOTS), dtype=np.int32)
        link_contact_position = np.zeros_like(object_contact_position)
        link_contact_normal = np.zeros_like(object_contact_position)
        object_contact_position[:, :num_probes][active] = best_position[active]
        object_contact_normal[:, :num_probes][active] = best_normal[active]
        part_ids[:, :num_probes] = best_part
        link_contact_position[:, :num_probes][active] = probes[active]
        link_contact_normal[:, :num_probes][active] = -best_normal[active]

        setattr(data, f"mano_{side}_object_contact_positions", object_contact_position.tolist())
        setattr(data, f"mano_{side}_object_contact_normals", object_contact_normal.tolist())
        setattr(data, f"mano_{side}_object_contact_part_ids", part_ids.tolist())
        setattr(data, f"mano_{side}_link_contact_positions", link_contact_position.tolist())
        setattr(data, f"mano_{side}_link_contact_normals", link_contact_normal.tolist())
        active_frame = np.any(active, axis=1)
        indices = np.flatnonzero(active_frame)
        result[side] = {
            "active_frames": int(active_frame.sum()),
            "first_active_frame": int(indices[0]) if indices.size else None,
            "last_active_frame": int(indices[-1]) if indices.size else None,
            "active_slots": int(active.sum()),
            "part_counts": {
                str(part): int(np.count_nonzero(best_part == part))
                for part in range(1, len(mesh_paths) + 1)
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--support-source", required=True)
    parser.add_argument("--support-output-dir", required=True)
    parser.add_argument("--release-start", type=int, default=190)
    parser.add_argument("--release-end", type=int, default=218)
    parser.add_argument("--finger-lead", type=float, default=6.0)
    parser.add_argument("--finger-blend-start", type=int, default=176)
    parser.add_argument("--finger-blend-end", type=int, default=188)
    parser.add_argument("--contact-threshold", type=float, default=0.018)
    parser.add_argument("--left-release-clearance", type=float, default=0.0)
    parser.add_argument("--left-clearance-start", type=int, default=194)
    parser.add_argument("--left-clearance-peak", type=int, default=202)
    parser.add_argument("--left-clearance-end", type=int, default=214)
    parser.add_argument("--left-stability-clearance", type=float, default=0.0)
    parser.add_argument("--left-stability1-start", type=int, default=115)
    parser.add_argument("--left-stability1-peak", type=int, default=124)
    parser.add_argument("--left-stability1-end", type=int, default=135)
    parser.add_argument("--left-stability2-start", type=int, default=180)
    parser.add_argument("--left-stability2-peak", type=int, default=188)
    parser.add_argument("--left-stability2-end", type=int, default=198)
    parser.add_argument("--right-contact-inset", type=float, default=0.0)
    parser.add_argument("--right-contact-start", type=int, default=32)
    parser.add_argument("--right-contact-end", type=int, default=200)
    parser.add_argument("--right-contact-fade", type=int, default=8)
    parser.add_argument("--right-clearance", type=float, default=0.0)
    parser.add_argument("--right-clearance-start", type=int, default=65)
    parser.add_argument("--right-clearance-peak", type=int, default=73)
    parser.add_argument("--right-clearance-end", type=int, default=82)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    output_partition = Path(args.output_root) / f"sequence_id={args.sequence_id}" / "robot_name=sharpa_wave"
    support_output = Path(args.support_output_dir) / f"{args.sequence_id}_support.usda"
    if output_partition.exists() or support_output.exists():
        raise FileExistsError("release-fix output already exists; refusing to overwrite")

    data = ManoSharpaData.from_parquet(args.input)
    old_sequence_id = data.sequence_id
    original_wrist = np.asarray(data.robot_left_wrist_position, dtype=np.float64)
    sample = arc_length_time_map(original_wrist, args.release_start, args.release_end)
    resample_time_series(data, sample)
    finger_sample = advance_left_finger_release(
        data,
        lead_frames=args.finger_lead,
        blend_start=args.finger_blend_start,
        blend_end=args.finger_blend_end,
    )
    right_adjustment = adjust_right_contact_clearance(
        data,
        inset_m=args.right_contact_inset,
        contact_start=args.right_contact_start,
        contact_end=args.right_contact_end,
        fade_frames=args.right_contact_fade,
        clearance_m=args.right_clearance,
        clearance_start=args.right_clearance_start,
        clearance_peak=args.right_clearance_peak,
        clearance_end=args.right_clearance_end,
    )
    left_adjustment = adjust_left_release_clearance(
        data,
        clearance_m=args.left_release_clearance,
        clearance_start=args.left_clearance_start,
        clearance_peak=args.left_clearance_peak,
        clearance_end=args.left_clearance_end,
    )
    left_stability_adjustment = adjust_left_stability_clearance(
        data,
        clearance_m=args.left_stability_clearance,
        intervals=(
            (
                args.left_stability1_start,
                args.left_stability1_peak,
                args.left_stability1_end,
            ),
            (
                args.left_stability2_start,
                args.left_stability2_peak,
                args.left_stability2_end,
            ),
        ),
    )
    data.sequence_id = args.sequence_id

    frames = {
        side: recompute_robot_frames(data, side)
        for side in ("left", "right")
    }
    contact_report = recompute_contacts(data, frames, args.contact_threshold)

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    data.save_to_parquet(args.output_root, partition_cols=["sequence_id", "robot_name"])
    Path(args.support_output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.support_source, support_output)
    local_support_dir = Path(args.output_root) / "reconstructed_stage"
    local_support_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.support_source, local_support_dir / support_output.name)

    fixed_wrist = np.asarray(data.robot_left_wrist_position, dtype=np.float64)
    original_step = np.linalg.norm(np.diff(original_wrist, axis=0), axis=-1)
    fixed_step = np.linalg.norm(np.diff(fixed_wrist, axis=0), axis=-1)
    fixed_finger = np.asarray(data.robot_left_finger_joints, dtype=np.float64)
    finger_step = np.linalg.norm(np.diff(fixed_finger, axis=0), axis=-1)
    report = {
        "input_sequence_id": old_sequence_id,
        "output_sequence_id": args.sequence_id,
        "output_partition": str(output_partition),
        "support": str(support_output),
        "release_interval_source_frames": [args.release_start, args.release_end],
        "finger_lead_source_frames": args.finger_lead,
        "finger_blend_source_frames": [args.finger_blend_start, args.finger_blend_end],
        "contact_threshold_m": args.contact_threshold,
        "right_contact_adjustment": right_adjustment,
        "left_release_adjustment": left_adjustment,
        "left_stability_adjustment": left_stability_adjustment,
        "original_left_wrist_max_step_mm": float(original_step.max() * 1000.0),
        "fixed_left_wrist_max_step_mm": float(fixed_step.max() * 1000.0),
        "fixed_left_wrist_release_max_step_mm": float(
            fixed_step[args.release_start : args.release_end].max() * 1000.0
        ),
        "fixed_left_finger_max_step_rad_l2": float(finger_step.max()),
        "time_map": sample.tolist(),
        "finger_time_map": finger_sample.tolist(),
        "contacts": contact_report,
    }
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
