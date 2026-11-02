#!/usr/bin/env python3
"""Convert an ego-scene-HOI delivery into loaded ``ManoSharpaData``.

The delivery contract is the one documented by ``input/my_input/<id>/README.md``:
the official MANO output (including ``transl``) is multiplied by the external
per-hand scale.  This converter reconstructs that geometry with the same MANO
models used by the validation renderer, gravity-aligns the complete hand/object
scene, computes the standard V2D contact fields, and writes a loaded parquet for
``ego_recon_to_sharpa.py``.

``support_000`` is used to estimate the table normal when present.  It remains
gravity-only by default, but ``--object-ids`` can explicitly export it as a
tracked challenge object when the support itself is part of the scored object
roster.  For deliveries without a reconstructed support,
``--support-up-world`` accepts the normal of an independently fitted scene
plane.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation, Slerp
from trimesh.exchange.obj import export_obj

RG_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = RG_ROOT.parent
RG_SOURCE = RG_ROOT / "source" / "robotic_grounding"
LOADER_LIB = REPO_ROOT / "reconstruction" / "modules" / "v2d_task_library_loader" / "lib"
sys.path.insert(0, str(RG_SOURCE))
sys.path.insert(0, str(LOADER_LIB))

from contact_utils import (  # noqa: E402
    approximate_contact_with_id,
    compute_hand_link_contact_positions,
    find_object_contact_positions,
)
from distance_utils import compute_tip_to_object_surface_distance  # noqa: E402
from read_mano import MANO  # noqa: E402
from robotic_grounding.retarget.data_logger import (  # noqa: E402
    ManoSharpaData,
)
from robotic_grounding.retarget.naming import make_usd_safe  # noqa: E402
from robotic_grounding.retarget.params import (  # noqa: E402
    MANO_HAND_LINKS,
    NUM_MANO_LINKS,
)


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected dictionary in {path}")
    return value


def _fill_invalid_hand_parameters(
    parameters: dict[str, Any], valid: np.ndarray, max_gap: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Linearly fill short invalid MANO spans, independently per hand.

    The source pickle remains untouched.  This is deliberately opt-in and the
    exact filled frame indices are recorded in the conversion report.
    """
    frame_count = len(valid)
    filled = {key: np.asarray(value).copy() for key, value in parameters.items()}
    report: dict[str, Any] = {"enabled": True, "method": "linear_parameter_interpolation"}
    frames = np.arange(frame_count)
    for hand_id, side in enumerate(("left", "right")):
        invalid = ~valid[:, hand_id]
        invalid_indices = np.flatnonzero(invalid)
        if len(invalid_indices) == 0:
            report[side] = {"filled_frames": [], "maximum_contiguous_gap": 0}
            continue
        padded = np.r_[False, invalid, False]
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        longest = int(np.max(ends - starts))
        if longest > max_gap:
            raise ValueError(
                f"{side} hand has an invalid span of {longest} frames, exceeding "
                f"--max-invalid-hand-gap={max_gap}"
            )
        good = np.flatnonzero(~invalid)
        if len(good) < 2:
            raise ValueError(f"{side} hand has fewer than two valid frames")
        for key, array in filled.items():
            if array.shape[:2] != (frame_count, 2):
                raise ValueError(f"Unexpected MANO parameter shape for {key}: {array.shape}")
            values = array[:, hand_id].reshape(frame_count, -1)
            for component in range(values.shape[1]):
                values[invalid, component] = np.interp(
                    frames[invalid], good, values[good, component]
                )
            array[:, hand_id] = values.reshape(array[:, hand_id].shape)
        report[side] = {
            "filled_frames": invalid_indices.tolist(),
            "maximum_contiguous_gap": longest,
        }
    return filled, report


def _reinterpolate_hand_segment(
    parameters: dict[str, Any], side: str, start: int, end: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Replace a bad internal MANO span from its two trusted neighbors.

    Translation and shape use linear interpolation. Global and per-joint MANO
    axis-angle rotations use SO(3) Slerp, avoiding the discontinuities produced
    by component-wise interpolation of rotation vectors.
    """
    hand_id = {"left": 0, "right": 1}[side]
    frame_count = len(np.asarray(parameters["transl"]))
    before, after = start - 1, end + 1
    if not (0 <= before < start <= end < after < frame_count):
        raise ValueError(
            f"invalid {side} repair span [{start}, {end}] for {frame_count} frames"
        )
    output = {key: np.asarray(value).copy() for key, value in parameters.items()}
    query = np.arange(start, end + 1, dtype=np.float64)
    alpha = ((query - before) / float(after - before)).astype(np.float64)
    for key in ("transl", "betas"):
        values = output[key]
        left = values[before, hand_id].astype(np.float64)
        right = values[after, hand_id].astype(np.float64)
        values[start : end + 1, hand_id] = (
            left[None] * (1.0 - alpha[:, None]) + right[None] * alpha[:, None]
        ).astype(values.dtype)
    for key in ("global_orient", "hand_pose"):
        values = output[key]
        joint_count = values.shape[-1] // 3
        anchors = values[[before, after], hand_id].reshape(2, joint_count, 3)
        repaired = np.empty((len(query), joint_count, 3), dtype=np.float64)
        for joint in range(joint_count):
            repaired[:, joint] = Slerp(
                [float(before), float(after)],
                Rotation.from_rotvec(anchors[:, joint]),
            )(query).as_rotvec()
        values[start : end + 1, hand_id] = repaired.reshape(len(query), -1).astype(
            values.dtype
        )
    return output, {
        "side": side,
        "replaced_frames": list(range(start, end + 1)),
        "trusted_anchor_frames": [before, after],
        "translation_method": "linear",
        "rotation_method": "SO3_Slerp_per_MANO_joint",
    }


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = (
            loaded.to_geometry()
            if hasattr(loaded, "to_geometry")
            else loaded.dump(concatenate=True)
        )
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected triangle mesh: {path}")
    return loaded


def _write_textured_obj(mesh: trimesh.Trimesh, output: Path) -> None:
    """Export OBJ plus any MTL/texture payload returned by trimesh."""
    output.parent.mkdir(parents=True, exist_ok=True)
    obj_text, payload = export_obj(mesh, include_texture=True, return_texture=True)
    renamed: dict[str, str] = {}
    for name in payload:
        source = Path(name)
        renamed[name] = f"{output.stem}_{source.name}"
    for old, new in renamed.items():
        obj_text = obj_text.replace(old, new)
    output.write_text(obj_text)
    for name, data in payload.items():
        raw = data if isinstance(data, bytes) else data.encode()
        # MTL files can themselves refer to a texture from the payload.
        if Path(name).suffix.lower() == ".mtl":
            material = raw.decode()
            for old, new in renamed.items():
                material = material.replace(old, new)
            raw = material.encode()
        (output.parent / renamed[name]).write_bytes(raw)


def _vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    triangles = vertices[faces.long()]
    face_normals = torch.linalg.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(0, faces[:, corner].long(), face_normals)
    return torch.nn.functional.normalize(normals, dim=1, eps=1.0e-8)


def _support_up_world(
    support_mesh: trimesh.Trimesh,
    support_to_world: np.ndarray,
    camera_position_world: np.ndarray,
) -> np.ndarray:
    """Estimate the support normal from its thinnest mesh-coordinate axis.

    The delivered support is a densely reconstructed slab.  Vertex PCA is not
    appropriate here because reconstruction density is non-uniform and rotates
    the inferred normal away from the actual top faces.  Its local AABB thin
    axis is stable and, for this asset, agrees with the dominant face normals.
    """
    vertices = np.asarray(support_mesh.vertices, dtype=np.float64)
    extents = np.ptp(vertices, axis=0)
    normal_local = np.eye(3, dtype=np.float64)[int(np.argmin(extents))]
    normal_world = support_to_world[:3, :3] @ normal_local
    center_world = (
        support_to_world[:3, :3] @ vertices.mean(axis=0)
        + support_to_world[:3, 3]
    )
    # The visible table normal points from the table toward the frame-0 camera.
    if np.dot(normal_world, camera_position_world - center_world) < 0.0:
        normal_world *= -1.0
    return normal_world / np.linalg.norm(normal_world)


def _alignment_transform(
    up_world: np.ndarray,
    object_mesh: trimesh.Trimesh,
    object_to_world_0: np.ndarray,
    table_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate ``up_world`` to +Z and seat frame-0 object bottom at table height."""
    rotation = Rotation.align_vectors(
        np.array([[0.0, 0.0, 1.0]]), up_world[None]
    )[0].as_matrix()
    vertices_world = (
        np.asarray(object_mesh.vertices, dtype=np.float64)
        @ object_to_world_0[:3, :3].T
        + object_to_world_0[:3, 3]
    )
    rotated = vertices_world @ rotation.T
    translation = np.array(
        [
            -float(rotated[:, 0].mean()),
            -float(rotated[:, 1].mean()),
            float(table_height) - float(rotated[:, 2].min()),
        ],
        dtype=np.float64,
    )
    return rotation, translation


def _transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return points @ rotation.T + translation


def _transform_wxyz(wxyz: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    flat = np.asarray(wxyz, dtype=np.float64).reshape(-1, 4)
    matrices = Rotation.from_quat(flat, scalar_first=True).as_matrix()
    aligned = np.einsum("ij,njk->nik", rotation, matrices)
    return Rotation.from_matrix(aligned).as_quat(scalar_first=True).reshape(wxyz.shape)


def _align_poses(poses: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    aligned = np.asarray(poses, dtype=np.float64).copy()
    aligned[:, :3, :3] = np.einsum("ij,njk->nik", rotation, poses[:, :3, :3])
    aligned[:, :3, 3] = _transform_points(poses[:, :3, 3], rotation, translation)
    return aligned


def _fix_left_shapedirs(mano: MANO) -> bool:
    left = mano.left_mano_layer.th_shapedirs
    right = mano.right_mano_layer.th_shapedirs
    if float(torch.sum(torch.abs(left[:, 0] - right[:, 0]))) < 1.0:
        with torch.no_grad():
            left[:, 0] *= -1.0
        return True
    return False


def _effective_parameters(
    mano: MANO,
    side: str,
    betas: torch.Tensor,
    hand_pose: torch.Tensor,
    aligned_global_orient: np.ndarray,
    desired_wrist: np.ndarray,
) -> np.ndarray:
    """Fit the legacy transl field so re-FK reproduces the aligned wrist exactly.

    The parquet schema has no per-hand external-scale field.  Authoritative
    retarget inputs are therefore the explicitly stored scaled joints; this
    effective ``transl`` keeps the redundant parameter representation anchored
    at the same wrist while the conversion report retains the original scales.
    """
    zeros = torch.zeros((len(hand_pose), 3), dtype=torch.float32, device=betas.device)
    with torch.inference_mode():
        zero = mano.forward(
            side=side,
            betas=betas,
            global_orient=torch.as_tensor(
                aligned_global_orient, dtype=torch.float32, device=betas.device
            ),
            transl=zeros,
            finger_pose=hand_pose,
        )["joints"][:, 0]
    return desired_wrist - zero.cpu().numpy()


def _true_segments(mask: np.ndarray, min_frames: int) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    segments = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [segment for segment in segments if len(segment) >= min_frames]


def _vertical_drift_correction(
    object_poses: np.ndarray,
    object_mesh: trimesh.Trimesh,
    surface_local: np.ndarray,
    hand_data: dict[str, dict[str, Any]],
    table_height: float,
    device: torch.device,
    pos_threshold_m: float,
    angle_threshold_rad: float,
    release_threshold_m: float,
    min_segment_frames: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate a smooth common Z correction from released, still intervals.

    The correction is shared by both hands and the object, so hand-object
    relative geometry is exactly preserved.  A cubic smoothstep connects the
    first and last rest anchors and is constant outside their centers.
    """
    frame_count = len(object_poses)
    held = np.zeros(frame_count, dtype=bool)
    for frame_id, pose in enumerate(object_poses):
        surface_world = torch.as_tensor(
            surface_local @ pose[:3, :3].T + pose[:3, 3],
            dtype=torch.float32,
            device=device,
        )
        for side in ("left", "right"):
            tips_distance = compute_tip_to_object_surface_distance(
                hand_data[side]["joints"][frame_id], surface_world
            )
            held[frame_id] |= float(np.min(tips_distance)) < release_threshold_m

    pos_delta = np.linalg.norm(np.diff(object_poses[:, :3, 3], axis=0), axis=1)
    rotations = Rotation.from_matrix(object_poses[:, :3, :3])
    angle_delta = (rotations[:-1].inv() * rotations[1:]).magnitude()
    locally_still = np.zeros(frame_count, dtype=bool)
    locally_still[0] = (
        pos_delta[0] < pos_threshold_m and angle_delta[0] < angle_threshold_rad
    )
    locally_still[1:] = (pos_delta < pos_threshold_m) & (
        angle_delta < angle_threshold_rad
    )
    segments = _true_segments(
        locally_still & ~held,
        min_frames=min_segment_frames,
    )

    vertices = np.asarray(object_mesh.vertices, dtype=np.float64)
    bottom_z = np.empty(frame_count, dtype=np.float64)
    for frame_id, pose in enumerate(object_poses):
        bottom_z[frame_id] = np.min(
            vertices @ pose[:3, :3].T + pose[:3, 3], axis=0
        )[2]

    report: dict[str, Any] = {
        "enabled": True,
        "held_frame_count": int(held.sum()),
        "released_still_segments": [
            [int(segment[0]), int(segment[-1])] for segment in segments
        ],
        "bottom_z_before_m": {
            "min": float(bottom_z.min()),
            "max": float(bottom_z.max()),
        },
    }
    if len(segments) < 2:
        report["applied"] = False
        report["reason"] = "fewer than two released-still anchor segments"
        return np.zeros(frame_count, dtype=np.float64), report

    # The first and last released rest intervals anchor the table at both ends.
    anchor_segments = (segments[0], segments[-1])
    anchor_frames = np.array(
        [float(np.median(segment)) for segment in anchor_segments], dtype=np.float64
    )
    anchor_bottom_z = np.array(
        [float(np.median(bottom_z[segment])) for segment in anchor_segments],
        dtype=np.float64,
    )
    anchor_corrections = float(table_height) - anchor_bottom_z
    denominator = max(float(anchor_frames[1] - anchor_frames[0]), 1.0)
    phase = np.clip(
        (np.arange(frame_count, dtype=np.float64) - anchor_frames[0]) / denominator,
        0.0,
        1.0,
    )
    smooth_phase = phase * phase * (3.0 - 2.0 * phase)
    correction = anchor_corrections[0] + (
        anchor_corrections[1] - anchor_corrections[0]
    ) * smooth_phase
    report.update(
        {
            "applied": True,
            "anchor_frames": anchor_frames.tolist(),
            "anchor_bottom_z_m": anchor_bottom_z.tolist(),
            "anchor_correction_m": anchor_corrections.tolist(),
            "correction_range_m": [float(correction.min()), float(correction.max())],
        }
    )
    return correction, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--mano-model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--sequence-id", default="episode_000038")
    parser.add_argument("--object-name", default="episode_000038_object")
    parser.add_argument(
        "--object-names",
        nargs="+",
        default=None,
        help=(
            "Canonical per-body names in manifest interaction-object/slot order. "
            "When omitted, retain the historical single/group-name behavior."
        ),
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--table-height", type=float, default=0.75)
    parser.add_argument("--mano-to-robot-scale", type=float, default=1.2)
    parser.add_argument("--surface-points", type=int, default=4096)
    parser.add_argument(
        "--support-up-world",
        type=float,
        nargs=3,
        default=None,
        metavar=("NX", "NY", "NZ"),
        help=(
            "World-space support-plane normal pointing toward the camera/scene "
            "free space. Required when objects/support_000 is absent."
        ),
    )
    parser.add_argument(
        "--all-interaction-objects",
        action="store_true",
        help=(
            "Export every manifest object whose kind is interaction_object. "
            "Without this flag, preserve the historical object_000-only behavior."
        ),
    )
    parser.add_argument(
        "--object-ids",
        nargs="+",
        default=None,
        help=(
            "Explicit source object IDs to export in canonical slot order, for "
            "example '--object-ids support_000 object_000'. This may include a "
            "support object when it is part of the scored challenge roster and "
            "takes precedence over --all-interaction-objects."
        ),
    )
    parser.add_argument(
        "--object-z-offsets",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional constant Z offsets, one per exported interaction object. "
            "Useful for reconstructions whose independently estimated object "
            "bottoms disagree with one fitted physical support plane."
        ),
    )
    parser.add_argument(
        "--alignment-object-id",
        default=None,
        help=(
            "Source object ID whose frame-0 mesh bottom is seated at table height. "
            "Defaults to the first exported object. Use a reconstructed static support "
            "when canonical submission order places the manipulated object first."
        ),
    )
    parser.add_argument(
        "--hand-z-offsets",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("LEFT", "RIGHT"),
        help=(
            "Constant left/right hand Z offsets applied with object corrections "
            "to preserve the corresponding hand-object relative geometry."
        ),
    )
    parser.add_argument(
        "--post-align-transform",
        type=Path,
        default=None,
        help=(
            "Optional JSON 4x4 transform applied after gravity alignment to the "
            "complete hand-object trajectory (for physics-derived support stabilization)."
        ),
    )
    parser.add_argument(
        "--correct-vertical-drift",
        action="store_true",
        help=(
            "Estimate table-height drift from the first/last released-still "
            "object intervals and apply one smooth Z correction to both hands and object."
        ),
    )
    parser.add_argument("--drift-still-pos-threshold", type=float, default=0.005)
    parser.add_argument("--drift-still-angle-threshold", type=float, default=0.05)
    parser.add_argument("--drift-release-threshold", type=float, default=0.02)
    parser.add_argument("--drift-min-segment-frames", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--fill-invalid-hands",
        action="store_true",
        help=(
            "Linearly interpolate short MANO parameter gaps marked invalid. "
            "The source remains unchanged and filled frames are reported."
        ),
    )
    parser.add_argument("--max-invalid-hand-gap", type=int, default=10)
    parser.add_argument(
        "--reinterpolate-hand-segment",
        nargs=3,
        action="append",
        default=[],
        metavar=("SIDE", "START", "END"),
        help=(
            "Replace an internal bad hand span from adjacent trusted frames. "
            "May be repeated; SIDE is left/right and frame bounds are inclusive."
        ),
    )
    parser.add_argument(
        "--static-invalid-object-fallback",
        action="store_true",
        help=(
            "For a declared failure sequence only: if an exported object has exactly "
            "one valid pose, repeat that pose over the full clip. This creates an "
            "explicit static fallback reference; it does not recover object motion."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.sequence_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    asset_dir = args.asset_dir.expanduser().resolve()
    device = torch.device(
        args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    )

    manifest = json.loads((root / "manifest.json").read_text())
    frame_count = int(manifest["frame_count"])
    if abs(float(manifest["fps"]) - args.fps) > 1.0e-6:
        raise ValueError(f"FPS mismatch: manifest={manifest['fps']} CLI={args.fps}")

    parameters_data = _load_pickle(root / "hands/mano_parameters.pkl")
    parameters = parameters_data["parameters"]
    valid = np.asarray(parameters_data["valid"], dtype=bool)[..., 0]
    scales = np.asarray(_load_pickle(root / "hands/mano_scale.pkl")["scale"], dtype=np.float64)
    if valid.shape != (frame_count, 2):
        raise ValueError(f"Unexpected hand validity shape: {valid.shape}")
    hand_fill_report: dict[str, Any] = {"enabled": False}
    if not valid.all():
        if not args.fill_invalid_hands:
            invalid = np.argwhere(~valid).tolist()
            raise ValueError(
                "Invalid hand frames require explicit --fill-invalid-hands; "
                f"(frame, hand)={invalid}"
            )
        parameters, hand_fill_report = _fill_invalid_hand_parameters(
            parameters, valid, int(args.max_invalid_hand_gap)
        )
    hand_segment_repairs: list[dict[str, Any]] = []
    for side_text, start_text, end_text in args.reinterpolate_hand_segment:
        if side_text not in {"left", "right"}:
            raise ValueError(f"invalid repair side {side_text!r}; expected left/right")
        parameters, repair = _reinterpolate_hand_segment(
            parameters, side_text, int(start_text), int(end_text)
        )
        hand_segment_repairs.append(repair)
    if scales.shape != (2,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError(f"Invalid MANO scales: {scales}")

    manifest_objects = manifest.get("files", {}).get("objects", [])
    manifest_object_ids = {str(item["id"]) for item in manifest_objects}
    if args.object_ids is not None:
        object_ids = [str(value) for value in args.object_ids]
        if not object_ids or len(set(object_ids)) != len(object_ids):
            raise ValueError("--object-ids must be non-empty and unique")
        unknown = [value for value in object_ids if value not in manifest_object_ids]
        if unknown:
            raise ValueError(
                f"--object-ids contains IDs absent from the manifest: {unknown}; "
                f"available={sorted(manifest_object_ids)}"
            )
    elif args.all_interaction_objects:
        object_ids = [
            str(item["id"])
            for item in manifest_objects
            if item.get("kind") == "interaction_object"
        ]
        if not object_ids:
            raise ValueError("Manifest contains no interaction_object entries")
    else:
        object_ids = ["object_000"]

    object_meshes = [
        _load_mesh(root / f"objects/{object_id}/textured_mesh_metric.glb")
        for object_id in object_ids
    ]
    object_pose_records = [
        _load_pickle(root / f"objects/{object_id}/poses.pkl")
        for object_id in object_ids
    ]
    trajectory = _load_pickle(root / "cameras/trajectory.pkl")
    object_pose_timelines = [
        np.asarray(record["object_to_world"], dtype=np.float64)
        for record in object_pose_records
    ]
    static_object_fallbacks: dict[str, dict[str, Any]] = {}
    for object_index, (object_id, poses, record) in enumerate(zip(
        object_ids, object_pose_timelines, object_pose_records, strict=True
    )):
        valid_object = np.asarray(record["valid"], dtype=bool)[..., 0]
        if len(poses) != frame_count or len(valid_object) != frame_count:
            raise ValueError(f"Invalid {object_id} timeline length")
        if valid_object.all():
            continue
        valid_indices = np.flatnonzero(valid_object)
        if not args.static_invalid_object_fallback or len(valid_indices) != 1:
            raise ValueError(
                f"Invalid {object_id} timeline: valid frames={valid_indices.tolist()}; "
                "the static fallback is opt-in and requires exactly one valid pose"
            )
        anchor = int(valid_indices[0])
        object_pose_timelines[object_index] = np.repeat(
            poses[anchor : anchor + 1], frame_count, axis=0
        )
        static_object_fallbacks[object_id] = {
            "applied": True,
            "anchor_frame": anchor,
            "valid_source_frame_count": 1,
            "method": "repeat_only_valid_pose",
            "motion_recovered": False,
        }

    camera_position = np.asarray(
        trajectory["rectified_left_camera_to_world"][0, :3, 3], dtype=np.float64
    )
    support_path = root / "objects/support_000/textured_mesh_metric.glb"
    support_pose_path = root / "objects/support_000/poses.pkl"
    if args.support_up_world is not None:
        up_world = np.asarray(args.support_up_world, dtype=np.float64)
        norm = float(np.linalg.norm(up_world))
        if not np.isfinite(norm) or norm < 1.0e-8:
            raise ValueError("--support-up-world must be a finite non-zero vector")
        up_world /= norm
        support_normal_source = "cli_fitted_plane"
    elif support_path.is_file() and support_pose_path.is_file():
        support_mesh = _load_mesh(support_path)
        support_pose_data = _load_pickle(support_pose_path)
        support_poses = np.asarray(
            support_pose_data["object_to_world"], dtype=np.float64
        )
        if not np.asarray(support_pose_data["valid"])[..., 0].all():
            raise ValueError("Invalid support_000 timeline")
        up_world = _support_up_world(support_mesh, support_poses[0], camera_position)
        support_normal_source = "support_000_mesh"
    else:
        raise FileNotFoundError(
            "objects/support_000 is absent; fit the scene support plane and pass "
            "its free-space-facing normal with --support-up-world NX NY NZ"
        )
    alignment_object_id = (
        object_ids[0]
        if args.alignment_object_id is None
        else str(args.alignment_object_id)
    )
    if alignment_object_id not in object_ids:
        raise ValueError(
            f"--alignment-object-id {alignment_object_id!r} is not among exported "
            f"object IDs {object_ids}"
        )
    alignment_object_index = object_ids.index(alignment_object_id)
    align_rotation, align_translation = _alignment_transform(
        up_world,
        object_meshes[alignment_object_index],
        object_pose_timelines[alignment_object_index][0],
        args.table_height,
    )
    post_align_transform = np.eye(4, dtype=np.float64)
    if args.post_align_transform is not None:
        transform_path = args.post_align_transform.expanduser().resolve()
        post_align_transform = np.asarray(
            json.loads(transform_path.read_text()), dtype=np.float64
        )
        if post_align_transform.shape != (4, 4):
            raise ValueError(
                f"post-align transform must be 4x4, got {post_align_transform.shape}"
            )
        if not np.all(np.isfinite(post_align_transform)):
            raise ValueError("post-align transform contains non-finite values")
        if not np.allclose(post_align_transform[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("post-align transform must be an affine rigid transform")
        post_rotation = post_align_transform[:3, :3]
        if not np.allclose(post_rotation.T @ post_rotation, np.eye(3), atol=1e-5):
            raise ValueError("post-align rotation is not orthonormal")
        if not np.isclose(np.linalg.det(post_rotation), 1.0, atol=1e-5):
            raise ValueError("post-align rotation must have determinant +1")
        align_translation = (
            post_rotation @ align_translation + post_align_transform[:3, 3]
        )
        align_rotation = post_rotation @ align_rotation
    aligned_object_pose_timelines = [
        _align_poses(poses, align_rotation, align_translation)
        for poses in object_pose_timelines
    ]
    object_z_offsets = (
        np.zeros(len(object_ids), dtype=np.float64)
        if args.object_z_offsets is None
        else np.asarray(args.object_z_offsets, dtype=np.float64)
    )
    if object_z_offsets.shape != (len(object_ids),):
        raise ValueError(
            f"--object-z-offsets must provide {len(object_ids)} values, "
            f"got {object_z_offsets.shape}"
        )
    if not np.all(np.isfinite(object_z_offsets)):
        raise ValueError("--object-z-offsets contains non-finite values")
    for poses, z_offset in zip(
        aligned_object_pose_timelines, object_z_offsets, strict=True
    ):
        poses[:, 2, 3] += float(z_offset)

    mano = MANO(
        mano_assets_root=str(args.mano_model_root.expanduser().resolve()),
        gender="neutral",
        device=device,
        flat_hand_mean=False,
        center_idx=None,
    )
    left_shapedirs_fixed = _fix_left_shapedirs(mano)

    hand_data: dict[str, dict[str, Any]] = {}
    for hand_id, side in enumerate(("left", "right")):
        betas_np = np.asarray(parameters["betas"][:, hand_id], dtype=np.float32)
        if float(np.max(np.abs(betas_np - betas_np[0]))) > 1.0e-5:
            raise ValueError(f"{side} betas vary over time; ManoSharpaData stores one shape")
        betas = torch.as_tensor(betas_np, device=device)
        global_orient_np = np.asarray(
            parameters["global_orient"][:, hand_id], dtype=np.float32
        )
        hand_pose_np = np.asarray(parameters["hand_pose"][:, hand_id], dtype=np.float32)
        transl_np = np.asarray(parameters["transl"][:, hand_id], dtype=np.float32)
        hand_pose = torch.as_tensor(hand_pose_np, device=device)
        with torch.inference_mode():
            result = mano.forward(
                side=side,
                betas=betas,
                global_orient=torch.as_tensor(global_orient_np, device=device),
                transl=torch.as_tensor(transl_np, device=device),
                finger_pose=hand_pose,
            )
        vertices = float(scales[hand_id]) * result["vertices"].cpu().numpy()
        joints = float(scales[hand_id]) * result["joints"].cpu().numpy()
        vertices = _transform_points(vertices, align_rotation, align_translation)
        joints = _transform_points(joints, align_rotation, align_translation)
        joints_wxyz = _transform_wxyz(
            result["joints_wxyz"].cpu().numpy(), align_rotation
        )
        aligned_global = Rotation.from_matrix(
            np.einsum(
                "ij,njk->nik",
                align_rotation,
                Rotation.from_rotvec(global_orient_np).as_matrix(),
            )
        ).as_rotvec()
        effective_transl = _effective_parameters(
            mano,
            side,
            betas,
            hand_pose,
            aligned_global,
            joints[:, 0],
        )
        faces = (
            mano.left_faces if side == "left" else mano.right_faces
        ).detach().to(device=device, dtype=torch.long)
        hand_data[side] = {
            "betas": betas_np[0].tolist(),
            "global_orient": aligned_global.astype(np.float32),
            "hand_pose": hand_pose_np,
            "transl": effective_transl.astype(np.float32),
            "vertices": torch.as_tensor(vertices, dtype=torch.float32, device=device),
            "joints": torch.as_tensor(joints, dtype=torch.float32, device=device),
            "joints_wxyz": joints_wxyz.astype(np.float32),
            "faces": faces,
        }

    hand_z_offsets = {
        "left": float(args.hand_z_offsets[0]),
        "right": float(args.hand_z_offsets[1]),
    }
    if not all(np.isfinite(value) for value in hand_z_offsets.values()):
        raise ValueError("--hand-z-offsets contains non-finite values")
    for side, z_offset in hand_z_offsets.items():
        hand_data[side]["vertices"][:, :, 2] += z_offset
        hand_data[side]["joints"][:, :, 2] += z_offset
        hand_data[side]["transl"][:, 2] += np.float32(z_offset)

    surface_samples: list[tuple[np.ndarray, np.ndarray]] = []
    rng_state = np.random.get_state()
    for body_index, object_mesh in enumerate(object_meshes):
        np.random.seed(body_index)
        surface_local, face_ids = trimesh.sample.sample_surface_even(
            object_mesh, args.surface_points
        )
        normals_local = np.asarray(
            object_mesh.face_normals[face_ids], dtype=np.float64
        )
        normals_local /= np.linalg.norm(normals_local, axis=1, keepdims=True).clip(
            min=1e-9
        )
        surface_samples.append((surface_local, normals_local))
    np.random.set_state(rng_state)

    drift_report: dict[str, Any] = {"enabled": False, "applied": False}
    if args.correct_vertical_drift:
        correction, drift_report = _vertical_drift_correction(
            aligned_object_pose_timelines[0],
            object_meshes[0],
            surface_samples[0][0],
            hand_data,
            float(args.table_height),
            device,
            float(args.drift_still_pos_threshold),
            float(args.drift_still_angle_threshold),
            float(args.drift_release_threshold),
            int(args.drift_min_segment_frames),
        )
        for aligned_object_poses in aligned_object_pose_timelines:
            aligned_object_poses[:, 2, 3] += correction
        correction_tensor = torch.as_tensor(
            correction, dtype=torch.float32, device=device
        )
        for side in ("left", "right"):
            hand_data[side]["vertices"][:, :, 2] += correction_tensor[:, None]
            hand_data[side]["joints"][:, :, 2] += correction_tensor[:, None]
            hand_data[side]["transl"][:, 2] += correction.astype(np.float32)
        print(
            "[deliverable-convert] vertical drift correction: "
            f"{1000.0 * correction[0]:.2f} mm -> "
            f"{1000.0 * correction[-1]:.2f} mm",
            flush=True,
        )

    safe_object = make_usd_safe(args.object_name)
    if args.object_names is not None:
        body_names = [str(name) for name in args.object_names]
        if len(body_names) != len(object_ids):
            raise ValueError(
                f"--object-names must provide {len(object_ids)} names in slot order, "
                f"got {len(body_names)}"
            )
        if any(not name.strip() for name in body_names) or len(set(body_names)) != len(body_names):
            raise ValueError("--object-names must be non-empty and unique")
    elif len(object_ids) == 1:
        body_names = [args.object_name]
    else:
        body_names = [f"{args.object_name}_{object_id}" for object_id in object_ids]
    safe_body_names = [make_usd_safe(name) for name in body_names]
    asset_meshes = [asset_dir / f"{name}.obj" for name in safe_body_names]
    asset_urdfs = [asset_dir / f"{name}_rigid.urdf" for name in safe_body_names]
    for object_mesh, asset_mesh in zip(object_meshes, asset_meshes, strict=True):
        _write_textured_obj(object_mesh, asset_mesh)

    logger = ManoSharpaData(
        sequence_id=args.sequence_id,
        raw_motion_file=str(root / "hands/mano_parameters.pkl"),
        robot_name="sharpa_wave",
        fps=float(args.fps),
        mano_flat_hand_mean=False,
        mano_center_idx=-1,
        mano_to_robot_scale=float(args.mano_to_robot_scale),
        mano_right_betas=hand_data["right"]["betas"],
        mano_left_betas=hand_data["left"]["betas"],
        mano_link_names=list(MANO_HAND_LINKS.keys()),
        right_robot_finger_joint_names=[],
        right_robot_frame_names=[],
        right_robot_frame_task_names=[],
        left_robot_finger_joint_names=[],
        left_robot_frame_names=[],
        left_robot_frame_task_names=[],
        object_name=args.object_name,
        safe_object_name=safe_object,
        object_body_names=safe_body_names,
        safe_object_body_names=safe_body_names,
        object_mesh_paths=[str(path) for path in asset_meshes],
        object_urdf_paths=[str(path) for path in asset_urdfs],
        object_mesh_radius=[
            float(
                np.linalg.norm(
                    np.asarray(object_mesh.vertices)
                    - np.asarray(object_mesh.vertices).mean(axis=0),
                    axis=1,
                ).max()
            )
            for object_mesh in object_meshes
        ],
    )

    contact_frame_counts = {"left": 0, "right": 0}
    for frame_id in range(frame_count):
        frame_poses = [poses[frame_id] for poses in aligned_object_pose_timelines]
        frame_surfaces: list[np.ndarray] = []
        frame_normals: list[np.ndarray] = []
        frame_part_ids: list[np.ndarray] = []
        for body_index, (pose, (surface_local, normals_local)) in enumerate(
            zip(frame_poses, surface_samples, strict=True)
        ):
            frame_surfaces.append(
                surface_local @ pose[:3, :3].T + pose[:3, 3]
            )
            frame_normals.append(normals_local @ pose[:3, :3].T)
            frame_part_ids.append(
                np.full(len(surface_local), body_index + 1, dtype=np.int32)
            )
        surface_world_np = np.concatenate(frame_surfaces, axis=0)
        normals_world_np = np.concatenate(frame_normals, axis=0)
        surface_world = torch.as_tensor(
            surface_world_np, dtype=torch.float32, device=device
        )
        # Official contact utility expects inward-pointing object normals.
        normals_world = -torch.nn.functional.normalize(
            torch.as_tensor(normals_world_np, dtype=torch.float32, device=device),
            dim=1,
            eps=1.0e-8,
        )
        part_ids = torch.as_tensor(
            np.concatenate(frame_part_ids), dtype=torch.int32, device=device
        )

        per_hand: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            vertices = hand_data[side]["vertices"][frame_id]
            joints = hand_data[side]["joints"][frame_id]
            hand_normals = _vertex_normals(vertices, hand_data[side]["faces"])
            tips_distance = compute_tip_to_object_surface_distance(
                joints, surface_world
            )
            (
                _object_points,
                _object_normals,
                contact_part_ids,
                hand_points,
                hand_normals_at_contact,
                contact_distances,
            ) = approximate_contact_with_id(
                surface_world,
                normals_world,
                part_ids,
                vertices,
                hand_normals,
                threshold=0.01,
            )
            link_positions = torch.zeros((NUM_MANO_LINKS, 3), device=device)
            link_normals = torch.zeros((NUM_MANO_LINKS, 3), device=device)
            link_part_ids = torch.zeros((NUM_MANO_LINKS,), device=device)
            object_positions = torch.zeros((NUM_MANO_LINKS, 3), device=device)
            object_normals = torch.zeros((NUM_MANO_LINKS, 3), device=device)
            if len(contact_distances):
                contact_frame_counts[side] += 1
                link_positions, link_normals, link_part_ids = (
                    compute_hand_link_contact_positions(
                        joints,
                        contact_part_ids,
                        hand_points,
                        hand_normals_at_contact,
                        contact_distances,
                    )
                )
                object_positions, object_normals = find_object_contact_positions(
                    link_positions, surface_world, normals_world
                )
            per_hand[side] = {
                "tips": tips_distance,
                "link_positions": link_positions.cpu().tolist(),
                "link_normals": link_normals.cpu().tolist(),
                "object_positions": object_positions.cpu().tolist(),
                "object_normals": object_normals.cpu().tolist(),
                "part_ids": link_part_ids.to(torch.int32).cpu().tolist(),
            }

        body_wxyz = [
            Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
            for pose in frame_poses
        ]
        root_pose = frame_poses[0]
        obj_axis_angle = Rotation.from_matrix(root_pose[:3, :3]).as_rotvec()
        logger.log_timestep(
            mano_right_trans=hand_data["right"]["transl"][frame_id].tolist(),
            mano_right_global_orient=hand_data["right"]["global_orient"][frame_id].tolist(),
            mano_right_finger_pose=hand_data["right"]["hand_pose"][frame_id].tolist(),
            mano_right_joints=hand_data["right"]["joints"][frame_id].cpu().tolist(),
            mano_right_joints_wxyz=hand_data["right"]["joints_wxyz"][frame_id].tolist(),
            mano_right_fitting_err=0.0,
            mano_right_tips_distance=per_hand["right"]["tips"],
            mano_right_link_contact_positions=per_hand["right"]["link_positions"],
            mano_right_link_contact_normals=per_hand["right"]["link_normals"],
            mano_right_object_contact_positions=per_hand["right"]["object_positions"],
            mano_right_object_contact_normals=per_hand["right"]["object_normals"],
            mano_right_object_contact_part_ids=per_hand["right"]["part_ids"],
            mano_left_trans=hand_data["left"]["transl"][frame_id].tolist(),
            mano_left_global_orient=hand_data["left"]["global_orient"][frame_id].tolist(),
            mano_left_finger_pose=hand_data["left"]["hand_pose"][frame_id].tolist(),
            mano_left_joints=hand_data["left"]["joints"][frame_id].cpu().tolist(),
            mano_left_joints_wxyz=hand_data["left"]["joints_wxyz"][frame_id].tolist(),
            mano_left_fitting_err=0.0,
            mano_left_tips_distance=per_hand["left"]["tips"],
            mano_left_link_contact_positions=per_hand["left"]["link_positions"],
            mano_left_link_contact_normals=per_hand["left"]["link_normals"],
            mano_left_object_contact_positions=per_hand["left"]["object_positions"],
            mano_left_object_contact_normals=per_hand["left"]["object_normals"],
            mano_left_object_contact_part_ids=per_hand["left"]["part_ids"],
            object_articulation=0.0,
            object_root_axis_angle=obj_axis_angle.tolist(),
            object_root_position=root_pose[:3, 3].tolist(),
            object_body_position=[pose[:3, 3].tolist() for pose in frame_poses],
            object_body_wxyz=[quat.tolist() for quat in body_wxyz],
        )

        if frame_id == 0 or (frame_id + 1) % 50 == 0:
            print(f"[deliverable-convert] {frame_id + 1}/{frame_count}", flush=True)

    logger.save_to_parquet(
        root_path=str(output_dir), partition_cols=["sequence_id", "robot_name"]
    )
    report = {
        "sequence_id": args.sequence_id,
        "frame_count": frame_count,
        "fps": float(args.fps),
        "source": str(root),
        "output_dir": str(output_dir),
        "object_ids": object_ids,
        "object_body_names": body_names,
        "object_meshes": [str(path) for path in asset_meshes],
        "object_urdfs": [str(path) for path in asset_urdfs],
        "support_mesh_exported": "support_000" in object_ids,
        "support_used_for_gravity_only": "support_000" not in object_ids,
        "support_normal_source": support_normal_source,
        "support_up_world": up_world.tolist(),
        "alignment_object_id": alignment_object_id,
        "alignment_object_index": alignment_object_index,
        "alignment_rotation": align_rotation.tolist(),
        "alignment_translation": align_translation.tolist(),
        "post_align_transform": post_align_transform.tolist(),
        "post_align_transform_path": (
            None
            if args.post_align_transform is None
            else str(args.post_align_transform.expanduser().resolve())
        ),
        "table_height_m": float(args.table_height),
        "object_z_offsets_m": object_z_offsets.tolist(),
        "hand_z_offsets_m": hand_z_offsets,
        "mano_scales": {"left": float(scales[0]), "right": float(scales[1])},
        "invalid_hand_fill": hand_fill_report,
        "hand_segment_repairs": hand_segment_repairs,
        "static_invalid_object_fallbacks": static_object_fallbacks,
        "left_shapedirs_fixed": left_shapedirs_fixed,
        "contact_frames": contact_frame_counts,
        "vertical_drift_correction": drift_report,
    }
    report_path = output_dir / f"{args.sequence_id}_conversion_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[deliverable-convert] wrote {report_path}")


if __name__ == "__main__":
    main()
