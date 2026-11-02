#!/usr/bin/env python3
"""Export one stereo HOI sequence into a self-contained delivery directory.

The exporter uses the rectified-left VIPE world as the shared metric frame.  It
converts UmeTrack-fitted MANO representation to the official MANO_LEFT model,
fits the final global orientation, and folds every downstream rigid correction
into official MANO ``global_orient`` and ``transl`` parameters.  The external
per-hand scalar remains in a separate pickle as requested.

MANO reconstruction follows this explicit convention::

    vertices_world = scale * MANO(
        betas=betas,
        global_orient=global_orient,
        hand_pose=hand_pose,
        transl=transl,
    ).vertices

All numeric delivery artifacts are pickle files containing NumPy arrays.
Sequence-valued records include a trailing singleton validity-mask dimension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import trimesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


HAND_LABELS = ("left", "right")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mano-root",
        type=Path,
        default=REPOSITORY_ROOT / "thirdparty/MANO",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--interaction-object-source-root",
        type=Path,
        help=(
            "Optional root containing objects/object_*/scale_primary and "
            "pose_track. This selects an earlier self-reconstructed object "
            "version without changing the active pipeline run."
        ),
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Run numeric round-trip validation without producing the overlay video.",
    )
    return parser.parse_args()


def _dump_numpy_pickle(path: Path, value: dict[str, Any]) -> None:
    """Write a dictionary whose leaves are NumPy arrays using pickle protocol 5."""

    arrays: dict[str, np.ndarray | dict[str, np.ndarray]] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            arrays[key] = {name: np.asarray(array) for name, array in item.items()}
        else:
            arrays[key] = np.asarray(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(arrays, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_numpy_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dictionary in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_properties(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    try:
        return {
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()


def _copy(source: Path, destination: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _resolve_hand_motion(run_root: Path) -> Path:
    candidates = (
        # A joint HACO result owns the final MANO geometry whenever present.
        run_root / "hands/haco_joint_refined/hand_motion.npz",
        run_root / "hands/umetrack_mano_surface/hand_motion.npz",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("No final stereo hand-motion archive was found")


def _load_official_mano_models(
    mano_root: Path, device: torch.device
) -> tuple[Any, Any]:
    """Load axis-angle MANO models used by the UmeTrack fitting stage."""

    import smplx

    model_root = mano_root / "mano_data"
    left = smplx.MANO(
        model_path=str(model_root / "MANO_LEFT.pkl"),
        is_rhand=False,
        use_pca=False,
        flat_hand_mean=False,
    ).to(device).eval()
    right = smplx.MANO(
        model_path=str(model_root / "MANO_RIGHT.pkl"),
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=False,
    ).to(device).eval()
    shapedirs_difference = torch.sum(
        torch.abs(left.shapedirs[:, 0, :] - right.shapedirs[:, 0, :])
    )
    if float(shapedirs_difference.detach().cpu()) < 1.0:
        with torch.no_grad():
            left.shapedirs[:, 0, :] *= -1.0
    return left, right


def _rotation_matrix_to_axis_angle(matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(matrices, dtype=np.float64)
    result = np.full(values.shape[:-2] + (3,), np.nan, dtype=np.float32)
    for index in np.ndindex(values.shape[:-2]):
        matrix = values[index]
        if np.isfinite(matrix).all():
            vector, _ = cv2.Rodrigues(matrix)
            result[index] = vector[:, 0].astype(np.float32)
    return result


def _kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return R for row-vector points satisfying target ~= source @ R.T."""

    covariance = np.asarray(source, dtype=np.float64).T @ np.asarray(
        target, dtype=np.float64
    )
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    return rotation.astype(np.float32)


def _mano_zero_global_geometry(
    layer: Any,
    betas: np.ndarray,
    hand_pose_matrices: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(betas)
    identity = np.broadcast_to(
        np.eye(3, dtype=np.float32), (count, 1, 3, 3)
    ).copy()
    with torch.inference_mode():
        output = layer(
            betas=torch.as_tensor(betas, device=device),
            global_orient=torch.as_tensor(identity, device=device),
            hand_pose=torch.as_tensor(hand_pose_matrices, device=device),
            transl=torch.zeros((count, 3), dtype=torch.float32, device=device),
            pose2rot=False,
        )
    return (
        output.vertices.detach().cpu().numpy().astype(np.float32),
        output.joints.detach().cpu().numpy().astype(np.float32),
    )


def _mano_zero_global_geometry_axis_angle(
    layer: Any,
    betas: np.ndarray,
    hand_pose_axis_angle: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate official MANO parameters while preserving its mean pose."""

    count = len(betas)
    with torch.inference_mode():
        output = layer(
            betas=torch.as_tensor(betas, device=device),
            global_orient=torch.zeros((count, 3), dtype=torch.float32, device=device),
            hand_pose=torch.as_tensor(hand_pose_axis_angle, device=device),
            transl=torch.zeros((count, 3), dtype=torch.float32, device=device),
            pose2rot=True,
        )
    return (
        output.vertices.detach().cpu().numpy().astype(np.float32),
        output.joints.detach().cpu().numpy().astype(np.float32),
    )


def _mano_oriented_geometry_axis_angle(
    layer: Any,
    betas: np.ndarray,
    global_orient_axis_angle: np.ndarray,
    hand_pose_axis_angle: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Evaluate MANO global orientation with zero official translation."""

    count = len(betas)
    with torch.inference_mode():
        output = layer(
            betas=torch.as_tensor(betas, device=device),
            global_orient=torch.as_tensor(global_orient_axis_angle, device=device),
            hand_pose=torch.as_tensor(hand_pose_axis_angle, device=device),
            transl=torch.zeros((count, 3), dtype=torch.float32, device=device),
            pose2rot=True,
        )
    return output.vertices.detach().cpu().numpy().astype(np.float32)


def _mano_oriented_geometry_matrices(
    layer: Any,
    betas: np.ndarray,
    global_orient_matrices: np.ndarray,
    hand_pose_matrices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Evaluate matrix-parameterized MANO with zero official translation."""

    count = len(betas)
    with torch.inference_mode():
        output = layer(
            betas=torch.as_tensor(betas, device=device),
            global_orient=torch.as_tensor(
                global_orient_matrices[:, None], device=device
            ),
            hand_pose=torch.as_tensor(hand_pose_matrices, device=device),
            transl=torch.zeros((count, 3), dtype=torch.float32, device=device),
            pose2rot=False,
        )
    return output.vertices.detach().cpu().numpy().astype(np.float32)


def _export_mano(
    run_root: Path,
    output_root: Path,
    mano_root: Path,
    device: torch.device,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Create official world-space MANO parameters and a separate scale file."""

    source_path = _resolve_hand_motion(run_root)
    with np.load(source_path, allow_pickle=True) as archive:
        motion = {key: np.asarray(archive[key]) for key in archive.files}
    labels = tuple(str(item) for item in motion["hand_labels"].tolist())
    if labels != HAND_LABELS:
        raise ValueError(f"Expected hand labels {HAND_LABELS}, got {labels}")
    valid = np.asarray(motion["valid"], dtype=bool)
    target_vertices = np.asarray(motion["vertices_world"], dtype=np.float32)
    source_model = str(np.asarray(motion.get("source_model", "")).item())
    if not source_model.startswith("umetrack"):
        raise ValueError(
            f"The cleaned Track-3 exporter requires UmeTrack/MANO input, got {source_model!r}"
        )
    betas = np.asarray(motion["mano_betas"], dtype=np.float32)
    hand_pose_matrices = np.asarray(motion["mano_hand_pose"], dtype=np.float32).copy()
    # ``mano_hand_pose`` is the authoritative post-optimization value and may
    # include a later HACO articulation correction.
    hand_pose_axis_angle = _rotation_matrix_to_axis_angle(
        hand_pose_matrices
    ).reshape(*valid.shape, 45)
    if hand_pose_axis_angle.shape != valid.shape + (45,):
        raise ValueError(
            f"Unexpected UmeTrack MANO hand-pose shape: {hand_pose_axis_angle.shape}"
        )
    if "hand_scene_scale" in motion:
        scale = np.asarray(motion["hand_scene_scale"], dtype=np.float32)
    elif "mano_scale" in motion:
        scale = np.repeat(
            np.asarray(motion["mano_scale"], dtype=np.float32).reshape(1), 2
        )
    else:
        scale = np.repeat(
            np.asarray(motion["scene_scale"], dtype=np.float32).reshape(1), 2
        )
    if "render_refine_hand_scale" in motion:
        scale = scale * np.asarray(
            motion["render_refine_hand_scale"], dtype=np.float32
        )
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"Invalid per-hand MANO scale: {scale}")

    layers = _load_official_mano_models(mano_root, device)
    frame_count, hand_count = valid.shape
    global_matrices = np.full(
        (frame_count, hand_count, 3, 3), np.nan, dtype=np.float32
    )
    translation_unscaled = np.full(
        (frame_count, hand_count, 3), np.nan, dtype=np.float32
    )
    reconstructed_vertices = np.full_like(target_vertices, np.nan)
    faces: list[np.ndarray] = []

    for hand_id, layer in enumerate(layers):
        frame_ids = np.flatnonzero(valid[:, hand_id])
        pose = hand_pose_matrices[frame_ids, hand_id]
        vertices_zero, joints_zero = _mano_zero_global_geometry_axis_angle(
            layer,
            betas[frame_ids, hand_id],
            hand_pose_axis_angle[frame_ids, hand_id],
            device,
        )
        hand_scale = float(scale[hand_id])
        for local_id, frame_id in enumerate(frame_ids):
            source_centered = hand_scale * (
                vertices_zero[local_id] - vertices_zero[local_id].mean(axis=0)
            )
            target_centered = target_vertices[frame_id, hand_id] - target_vertices[
                frame_id, hand_id
            ].mean(axis=0)
            rotation = _kabsch_rotation(source_centered, target_centered)
            global_matrices[frame_id, hand_id] = rotation

        selected_global_matrices = global_matrices[frame_ids, hand_id]
        selected_global_axis_angle = _rotation_matrix_to_axis_angle(
            selected_global_matrices
        )
        oriented_vertices = _mano_oriented_geometry_axis_angle(
            layer,
            betas[frame_ids, hand_id],
            selected_global_axis_angle,
            hand_pose_axis_angle[frame_ids, hand_id],
            device,
        )
        for local_id, frame_id in enumerate(frame_ids):
            # MANO applies global orientation about its kinematic root, not the
            # coordinate origin.  Derive official transl after evaluating that
            # exact operation; otherwise an R-dependent systematic offset is
            # introduced when consumers call the official MANO forward pass.
            translation = (
                target_vertices[frame_id, hand_id].mean(axis=0) / hand_scale
                - oriented_vertices[local_id].mean(axis=0)
            )
            translation_unscaled[frame_id, hand_id] = translation
            reconstructed_vertices[frame_id, hand_id] = hand_scale * (
                oriented_vertices[local_id] + translation
            )
        faces.append(np.asarray(layer.faces, dtype=np.int32))

    official_hand_pose = (
        hand_pose_axis_angle
        if hand_pose_axis_angle is not None
        else _rotation_matrix_to_axis_angle(hand_pose_matrices).reshape(
            frame_count, hand_count, 45
        )
    )
    official = {
        "parameters": {
            "betas": betas.astype(np.float32),
            "global_orient": _rotation_matrix_to_axis_angle(global_matrices),
            "hand_pose": official_hand_pose.astype(np.float32),
            "transl": translation_unscaled.astype(np.float32),
        },
        "valid": valid[..., None],
    }
    scale_record = {
        "scale": scale.astype(np.float32),
        "valid": valid.any(axis=0)[:, None],
    }
    topology = {
        "faces_left": faces[0],
        "faces_right": faces[1],
        "hand_labels": np.asarray(labels),
    }
    parameter_path = output_root / "hands/mano_parameters.pkl"
    scale_path = output_root / "hands/mano_scale.pkl"
    topology_path = output_root / "hands/mano_topology.pkl"
    _dump_numpy_pickle(parameter_path, official)
    _dump_numpy_pickle(scale_path, scale_record)
    _dump_numpy_pickle(topology_path, topology)

    validation_motion = output_root / "validation/reconstructed_hand_motion.npz"
    validation_motion.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        validation_motion,
        vertices_world=reconstructed_vertices,
        valid=valid,
        hand_labels=np.asarray(labels),
        faces_left=faces[0],
        faces_right=faces[1],
    )
    errors = np.linalg.norm(reconstructed_vertices - target_vertices, axis=-1)
    report: dict[str, Any] = {
        "source": str(source_path),
        "reconstruction_convention": (
            "vertices_world = scale * MANO(official_parameters).vertices"
        ),
        "coordinate_system": "VIPE rectified-left world",
        "per_hand": {},
    }
    for hand_id, label in enumerate(labels):
        vertex_values = errors[:, hand_id][valid[:, hand_id]] * 1000.0
        report["per_hand"][label] = {
            "valid_frames": int(valid[:, hand_id].sum()),
            "scale": float(scale[hand_id]),
            "vertex_error_mm": {
                "median": float(np.median(vertex_values)),
                "p95": float(np.quantile(vertex_values, 0.95)),
                "max": float(np.max(vertex_values)),
            },
        }
    return parameter_path, scale_path, validation_motion, report


def _export_cameras(run_root: Path, output_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Export rectified-stereo calibration and the rectified-left trajectory.

    Native RGB is a synchronized reference stream only. The cleaned Track-3
    pipeline does not estimate RGB-to-left extrinsics, so no RGB intrinsics,
    cross-camera transform, or synthetic RGB trajectory is exported.
    """

    calibration_path = run_root / "calibration/calibration.npz"
    pose_path = run_root / "vipe/pose/rectified_left.npz"
    with np.load(calibration_path) as archive:
        k_left = np.asarray(archive["K_working"], dtype=np.float32)
        baseline = np.asarray(archive["baseline_m"], dtype=np.float32)
        stereo_rotation = np.asarray(archive["stereo_R"], dtype=np.float32)
        stereo_translation = np.asarray(archive["stereo_t"], dtype=np.float32)
    with np.load(pose_path) as archive:
        left_to_world = np.asarray(archive["data"], dtype=np.float32)
        frame_ids = np.asarray(archive["inds"], dtype=np.int32)
    frame_count = len(left_to_world)
    if not np.array_equal(frame_ids, np.arange(frame_count)):
        raise ValueError("VIPE trajectory does not contain a dense frame timeline")
    left_valid = np.isfinite(left_to_world).all(axis=(1, 2))
    with np.load(run_root / "4d/cameras.npz") as archive:
        source_indices = np.asarray(archive["source_indices"], dtype=np.int32)
        timestamps_seconds = np.asarray(
            archive["timestamps_seconds"], dtype=np.float64
        )
    intrinsics_path = output_root / "cameras/intrinsics.pkl"
    trajectory_path = output_root / "cameras/trajectory.pkl"
    _dump_numpy_pickle(
        intrinsics_path,
        {
            "K_rectified_left": k_left,
            "stereo_baseline_m": baseline.reshape(1),
            "stereo_R_right_from_left": stereo_rotation,
            "stereo_t_right_from_left_m": stereo_translation,
        },
    )
    _dump_numpy_pickle(
        trajectory_path,
        {
            "rectified_left_camera_to_world": left_to_world,
            "rectified_left_world_to_camera": np.linalg.inv(left_to_world).astype(
                np.float32
            ),
            "source_indices": source_indices,
            "timestamps_seconds": timestamps_seconds,
            "valid": left_valid[:, None],
        },
    )
    report = {
        "frame_count": frame_count,
        "valid_frames": int(left_valid.sum()),
        "first_left_camera_to_world": left_to_world[0].tolist(),
        "coordinate_system": "VIPE rectified-left world, meters",
        "native_rgb_geometry": "not estimated or exported",
    }
    return intrinsics_path, trajectory_path, report


def _export_depth(run_root: Path, output_root: Path, frame_count: int) -> tuple[Path, dict[str, Any]]:
    frames = []
    frame_valid = np.zeros(frame_count, dtype=bool)
    for frame_id in range(frame_count):
        path = run_root / "depth/frames" / f"{frame_id:06d}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        depth = np.asarray(np.load(path), dtype=np.float32)
        if frames and depth.shape != frames[0].shape:
            raise ValueError(f"Depth shape changed at frame {frame_id}: {depth.shape}")
        frame_valid[frame_id] = bool(
            np.isfinite(depth).any() and np.any(np.isfinite(depth) & (depth > 0.0))
        )
        frames.append(depth)
    depth_sequence = np.stack(frames)
    path = output_root / "depth/depth.pkl"
    _dump_numpy_pickle(
        path,
        {
            "depth_m": depth_sequence,
            "valid": frame_valid[:, None],
        },
    )
    return path, {
        "shape": list(depth_sequence.shape),
        "valid_frames": int(frame_valid.sum()),
        "finite_positive_fraction": float(
            np.mean(np.isfinite(depth_sequence) & (depth_sequence > 0.0))
        ),
        "units": "meters along rectified-left camera z",
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
        raise TypeError(f"Expected a triangle mesh: {path}")
    return loaded


def _export_objects(
    run_root: Path,
    interaction_root: Path,
    output_root: Path,
    camera_to_world: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    for object_dir in sorted((interaction_root / "objects").glob("object_*")):
        if not object_dir.is_dir():
            continue
        object_id = object_dir.name
        scale_directory = object_dir / "scale_primary"
        meshes = sorted(scale_directory.glob("*_metric_centered.glb"))
        # Some experiment branches intentionally replace the reconstructed
        # mesh with an already metric GLB (for example ``*_public_metric.glb``).
        # The active scale directory is still the source of truth, so export
        # its sole GLB when the historical ``*_metric_centered`` suffix is not
        # present.
        if not meshes:
            meshes = sorted(scale_directory.glob("*.glb"))
        if not meshes:
            continue
        source_mesh = meshes[0]
        joint_pose = (
            run_root / "optim/contact_haco_joint_v2" / object_id / "poses.npz"
        )
        base_pose = object_dir / "pose_track/poses.npz"
        source_pose = (
            joint_pose if joint_pose.is_file() else base_pose
        )
        with np.load(source_pose) as archive:
            object_to_world = np.asarray(archive["object_to_world"], dtype=np.float32)
            source_indices = np.asarray(archive["source_indices"], dtype=np.int32)
            contact_optimization_valid = (
                np.asarray(archive["valid"], dtype=bool)
                if source_pose == joint_pose and "valid" in archive.files
                else np.zeros(len(object_to_world), dtype=bool)
            )
            object_to_camera = (
                np.asarray(archive["object_to_camera"], dtype=np.float32)
                if "object_to_camera" in archive.files
                else np.einsum(
                    "fij,fjk->fik",
                    np.linalg.inv(camera_to_world).astype(np.float32),
                    object_to_world,
                ).astype(np.float32)
            )
        # Joint HACO's ``valid`` means "participated in contact
        # optimization", not "has a renderable pose". Preserve object
        # visibility from the upstream tracker as the delivery validity.
        with np.load(base_pose) as archive:
            observed = np.asarray(archive["observed"], dtype=bool)
        if len(observed) != len(object_to_world):
            raise ValueError(
                f"Visibility and joint-pose timelines differ for {object_id}"
            )
        valid = (
            observed
            & np.isfinite(object_to_world).all(axis=(1, 2))
            & np.isfinite(object_to_camera).all(axis=(1, 2))
        )
        destination = output_root / "objects" / object_id
        mesh_path = _copy(source_mesh, destination / "textured_mesh_metric.glb")
        pose_path = destination / "poses.pkl"
        _dump_numpy_pickle(
            pose_path,
            {
                "object_to_world": object_to_world,
                "object_to_rectified_left_camera": object_to_camera,
                "source_indices": source_indices,
                "valid": valid[:, None],
                "contact_optimization_valid": contact_optimization_valid[:, None],
            },
        )
        composition = np.einsum(
            "fij,fjk->fik", camera_to_world, object_to_camera
        )
        error = float(np.nanmax(np.abs(composition - object_to_world)))
        mesh = _load_mesh(mesh_path)
        records.append(
            {
                "id": object_id,
                "kind": "interaction_object",
                "mesh": str(mesh_path.relative_to(output_root)),
                "poses": str(pose_path.relative_to(output_root)),
                "valid_frames": int(valid.sum()),
                "contact_optimized_frames": int(contact_optimization_valid.sum()),
                "mesh_extents_m": np.asarray(mesh.extents).tolist(),
                "mesh_sha256": _sha256(mesh_path),
                "source": "active_scale_primary",
                "pose_source": str(source_pose),
            }
        )
        validation[object_id] = {"world_composition_max_abs": error}

    support_manifest = run_root / "support_objects/pose/manifest.json"
    if support_manifest.is_file():
        manifest = json.loads(support_manifest.read_text(encoding="utf-8"))
        for record in manifest.get("objects", []):
            support_id = str(record["global_support_id"])
            source_mesh = Path(record["metric_mesh"])
            source_pose = Path(record["poses"])
            if not source_mesh.is_file():
                source_mesh = support_manifest.parent / support_id / source_mesh.name
            if not source_pose.is_file():
                source_pose = support_manifest.parent / support_id / source_pose.name
            with np.load(source_pose) as archive:
                object_to_world = np.asarray(archive["ob_in_world"], dtype=np.float32)
                object_to_camera = np.asarray(archive["ob_in_cam"], dtype=np.float32)
                source_indices = np.asarray(archive["source_indices"], dtype=np.int32)
            valid = (
                np.isfinite(object_to_world).all(axis=(1, 2))
                & np.isfinite(object_to_camera).all(axis=(1, 2))
            )
            destination = output_root / "objects" / support_id
            mesh_path = _copy(source_mesh, destination / "textured_mesh_metric.glb")
            pose_path = destination / "poses.pkl"
            _dump_numpy_pickle(
                pose_path,
                {
                    "object_to_world": object_to_world,
                    "object_to_rectified_left_camera": object_to_camera,
                    "source_indices": source_indices,
                    "valid": valid[:, None],
                },
            )
            composition = np.einsum(
                "fij,fjk->fik", camera_to_world, object_to_camera
            )
            error = float(np.nanmax(np.abs(composition - object_to_world)))
            mesh = _load_mesh(mesh_path)
            records.append(
                {
                    "id": support_id,
                    "kind": "support_object",
                    "name": str(record.get("name", support_id)),
                    "mesh": str(mesh_path.relative_to(output_root)),
                    "poses": str(pose_path.relative_to(output_root)),
                    "valid_frames": int(valid.sum()),
                    "mesh_extents_m": np.asarray(mesh.extents).tolist(),
                    "mesh_sha256": _sha256(mesh_path),
                }
            )
            validation[support_id] = {"world_composition_max_abs": error}
    return records, validation


def _render_validation(
    output_root: Path,
    mano_model_dir: Path,
    destination: Path,
    device: str,
) -> Path:
    """Copy and execute the self-contained delivery renderer."""

    renderer_path = _copy(
        REPOSITORY_ROOT / "tools/render_exported_deliverable.py",
        output_root / "render_hand_object.py",
    )
    subprocess.run(
        [
            sys.executable,
            str(renderer_path),
            "--root",
            str(output_root),
            "--mano-model-dir",
            str(mano_model_dir),
            "--output",
            str(destination),
            "--device",
            device,
        ],
        check=True,
    )
    return renderer_path


def _dimension_names(relative_path: str, key: str, shape: tuple[int, ...]) -> list[str]:
    """Return semantic names for every array dimension in the delivery schema."""

    exact: dict[tuple[str, str], list[str]] = {
        ("cameras/intrinsics.pkl", "K_rectified_left"): ["matrix_row", "matrix_column"],
        ("cameras/intrinsics.pkl", "stereo_baseline_m"): ["scalar"],
        ("cameras/intrinsics.pkl", "stereo_R_right_from_left"): ["matrix_row", "matrix_column"],
        ("cameras/intrinsics.pkl", "stereo_t_right_from_left_m"): ["xyz"],
        ("cameras/trajectory.pkl", "rectified_left_camera_to_world"): ["frame", "matrix_row", "matrix_column"],
        ("cameras/trajectory.pkl", "rectified_left_world_to_camera"): ["frame", "matrix_row", "matrix_column"],
        ("cameras/trajectory.pkl", "source_indices"): ["frame"],
        ("cameras/trajectory.pkl", "timestamps_seconds"): ["frame"],
        ("cameras/trajectory.pkl", "valid"): ["frame", "validity_mask"],
        ("depth/depth.pkl", "depth_m"): ["frame", "image_y", "image_x"],
        ("depth/depth.pkl", "valid"): ["frame", "validity_mask"],
        ("hands/mano_parameters.pkl", "parameters/betas"): ["frame", "hand", "shape_coefficient"],
        ("hands/mano_parameters.pkl", "parameters/global_orient"): ["frame", "hand", "axis_angle_xyz"],
        ("hands/mano_parameters.pkl", "parameters/hand_pose"): ["frame", "hand", "15_joints_x_axis_angle_xyz"],
        ("hands/mano_parameters.pkl", "parameters/transl"): ["frame", "hand", "xyz"],
        ("hands/mano_parameters.pkl", "valid"): ["frame", "hand", "validity_mask"],
        ("hands/mano_scale.pkl", "scale"): ["hand"],
        ("hands/mano_scale.pkl", "valid"): ["hand", "validity_mask"],
        ("hands/mano_topology.pkl", "faces_left"): ["triangle", "vertex_index"],
        ("hands/mano_topology.pkl", "faces_right"): ["triangle", "vertex_index"],
        ("hands/mano_topology.pkl", "hand_labels"): ["hand"],
    }
    if (relative_path, key) in exact:
        return exact[(relative_path, key)]
    if relative_path.startswith("objects/") and relative_path.endswith("/poses.pkl"):
        object_dimensions = {
            "object_to_world": ["frame", "matrix_row", "matrix_column"],
            "object_to_rectified_left_camera": ["frame", "matrix_row", "matrix_column"],
            "source_indices": ["frame"],
            "valid": ["frame", "validity_mask"],
            "contact_optimization_valid": ["frame", "validity_mask"],
        }
        if key in object_dimensions:
            return object_dimensions[key]
    return [f"dimension_{index}" for index in range(len(shape))]


def _write_array_schema(output_root: Path) -> Path:
    """Inspect every delivered PKL and describe its arrays in a JSON schema."""

    files: dict[str, Any] = {}
    for path in sorted(output_root.rglob("*.pkl")):
        relative = path.relative_to(output_root).as_posix()
        value = _load_numpy_pickle(path)
        arrays: dict[str, Any] = {}

        def visit(node: dict[str, Any], prefix: str = "") -> None:
            for name, item in node.items():
                key = f"{prefix}/{name}" if prefix else name
                if isinstance(item, dict):
                    visit(item, key)
                    continue
                array = np.asarray(item)
                arrays[key] = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "dimensions": _dimension_names(relative, key, array.shape),
                }

        visit(value)
        files[relative] = {"format": "pickle dictionary of NumPy arrays", "arrays": arrays}
    schema = {
        "schema_version": "ego-scene-hoi-deliverable-array-schema-v1",
        "dimension_conventions": {
            "frame": "shared zero-based video frame index",
            "hand": "index 0 is left and index 1 is right",
            "validity_mask": "singleton dimension of size 1; false means the sequence value must not be consumed",
            "matrix_row_matrix_column": "row-major homogeneous transform matrix",
            "xyz": "OpenCV axes: x right, y down, z forward at world frame 0",
        },
        "files": files,
    }
    path = output_root / "array_schema.json"
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return path


def _write_readme(output_root: Path, sequence_name: str) -> Path:
    """Write concise English loading and MANO reconstruction instructions."""

    text = f"""# Ego-Scene-HOI deliverable: {sequence_name}

This directory contains a synchronized native RGB reference video,
rectified-left stereo geometry, camera motion, metric object meshes/poses, and
MANO hand motion. See
`array_schema.json` for the exact shape, dtype, and named dimensions of every
NumPy array.

## Coordinate system

All 3D deliverables use the metric **VIPE rectified-left world**. Its gauge is
approximately the rectified-left camera at frame 0, with OpenCV axes (`+x`
right, `+y` down, `+z` forward). Distances are meters. The world is not gravity
aligned and does not use dataset ground-truth extrinsics.

The native RGB video is provided only as synchronized visual evidence. This
delivery intentionally contains no RGB-to-left registration, RGB intrinsics, or
RGB camera trajectory. All geometric arrays belong to the rectified-left camera
and world frames.

Transforms use column-vector notation in their names. For example,
`rectified_left_camera_to_world[t]` is `T_world_from_left(t)`, and

```text
p_world = R_world_from_left @ p_left + t_world_from_left
T_world_from_object = T_world_from_left @ T_left_from_object
```

For NumPy row-vector point arrays, use `points @ R.T + t`.

## Reading PKL arrays

All `.pkl` files contain dictionaries whose numeric leaves are NumPy arrays:

```python
import pickle
from pathlib import Path

root = Path("{sequence_name}")
with (root / "cameras/trajectory.pkl").open("rb") as f:
    cameras = pickle.load(f)

c2w = cameras["rectified_left_camera_to_world"]  # [F, 4, 4]
camera_valid = cameras["valid"][..., 0]           # [F]
```

Every sequence-valued record has a `valid` array. Its final dimension is the
singleton `validity_mask` dimension. Never consume an invalid frame/hand pose.

## Depth

`depth/depth.pkl["depth_m"]` has shape `[F,H,W]` and stores rectified-left
camera-z depth in meters. With `K_rectified_left`, a pixel `(u,v)` back-projects
as

```python
x = (u - cx) / fx * z
y = (v - cy) / fy * z
p_left = [x, y, z]
```

## Objects

Each `objects/<id>/textured_mesh_metric.glb` is already centered and scaled in
meters. `poses.pkl` contains both `object_to_world` and
`object_to_rectified_left_camera`, each `[F,4,4]`, plus `[F,1]` validity.

## MANO parameters and vertices

`hands/mano_parameters.pkl` contains only the official MANO parameter groups:
`betas`, `global_orient`, `hand_pose`, and `transl`, plus the validity mask.
Rotations are axis-angle radians. `hand_pose[...,45]` is 15 joint rotations in
joint-major XYZ axis-angle order. Use MANO_LEFT for hand index 0 and MANO_RIGHT
for hand index 1, with `use_pca=False` and `flat_hand_mean=False`.

Our estimated scalar is deliberately separate in `hands/mano_scale.pkl`.
Reconstruction must scale the **complete official MANO output**, including the
official `transl` parameter:

```python
import pickle
import torch
import smplx

with open(root / "hands/mano_parameters.pkl", "rb") as f:
    mano_data = pickle.load(f)
with open(root / "hands/mano_scale.pkl", "rb") as f:
    scale_data = pickle.load(f)

p = mano_data["parameters"]
valid = mano_data["valid"][..., 0]  # [frame, hand]
frame, hand = 0, 0                  # hand: 0=left, 1=right

def create_mano(is_rhand):
    return smplx.create(
        "PATH_TO_MANO_MODELS",
        model_type="mano",
        is_rhand=is_rhand,
        use_pca=False,
        flat_hand_mean=False,
    )

mano_left = create_mano(False)
mano_right = create_mano(True)
# Correct the known MANO_LEFT shapedirs issue when present in the model files.
if torch.sum(torch.abs(mano_left.shapedirs[:, 0] - mano_right.shapedirs[:, 0])) < 1:
    with torch.no_grad():
        mano_left.shapedirs[:, 0] *= -1
mano = mano_left if hand == 0 else mano_right
out = mano(
    betas=torch.from_numpy(p["betas"][frame, hand][None]),
    global_orient=torch.from_numpy(p["global_orient"][frame, hand][None]),
    hand_pose=torch.from_numpy(p["hand_pose"][frame, hand][None]),
    transl=torch.from_numpy(p["transl"][frame, hand][None]),
)
scale = torch.tensor(scale_data["scale"][hand], dtype=out.vertices.dtype)
vertices_world = scale * out.vertices
joints_world = scale * out.joints
```

The shapedirs check above is required only for public MANO packages containing
the known MANO_LEFT first-shapedirs sign issue.

To obtain left-camera vertices for frame `t`:

```python
w2c = cameras["rectified_left_world_to_camera"][frame]
vertices_left = vertices_world.numpy()[0] @ w2c[:3, :3].T + w2c[:3, 3]
```

The exported `transl` already includes all downstream rigid translation
corrections. Likewise, `global_orient` includes MANO fitting, joint-projection,
surface-alignment, and final HACO rotation corrections. Do not apply those
corrections again.

## Complete hand/object rendering example

`render_hand_object.py` is a self-contained reader and renderer for this
delivery. It reconstructs the two official MANO meshes from the parameter and
scale PKLs, loads every exported GLB and object pose, transforms everything to
the rectified-left camera, depth-tests the combined geometry, applies simple
lighting, and writes a two-column validation video:

```bash
python render_hand_object.py \
  --root . \
  --mano-model-dir /path/to/mano/model/files \
  --output validation/rendered_from_export.mp4
```

The MANO directory must contain `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`. Runtime
dependencies are NumPy, OpenCV, PyTorch, smplx, trimesh, and nvdiffrast. The
renderer imports no code from the ego-scene-hoi repository.

## Validation

`validation/report.json` records numerical MANO round-trip and object-transform
checks. `validation/rendered_overlay.mp4` shows the rectified-left image beside
the geometry reconstructed from the delivered parameters.
"""
    path = output_root / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    run_root = args.run_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    mano_root = args.mano_root.expanduser().resolve()
    interaction_root = (
        args.interaction_object_source_root.expanduser().resolve()
        if args.interaction_object_source_root is not None
        else run_root
    )
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    if not interaction_root.is_dir():
        raise FileNotFoundError(interaction_root)
    output_root.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, Any] = {}
    failure_source = run_root / "failure_case.json"
    if failure_source.is_file():
        failure_output = _copy(
            failure_source, output_root / "quality/failure_case.json"
        )
        provenance["quality_status"] = "failure_case"
        provenance["failure_case_report"] = str(
            failure_output.relative_to(output_root)
        )
    else:
        provenance["quality_status"] = "reconstructed"
    device = torch.device(
        args.device
        if not args.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )

    # Native RGB is retained as synchronized appearance evidence. Geometry is
    # deliberately not assigned to this camera because the current pipeline
    # does not estimate RGB-to-left extrinsics.
    (output_root / "videos/rgb_undistorted.mp4").unlink(missing_ok=True)
    rgb_video = _copy(
        run_root / "input/rgb_upright.mp4",
        output_root / "videos/rgb_native.mp4",
    )
    left_video = _copy(
        run_root / "calibration/left_rectified_working.mp4",
        output_root / "videos/rectified_left.mp4",
    )
    rgb_properties = _video_properties(rgb_video)
    left_properties = _video_properties(left_video)
    if rgb_properties["frame_count"] != left_properties["frame_count"]:
        raise ValueError("RGB and rectified-left videos have different frame counts")
    frame_count = int(left_properties["frame_count"])

    intrinsics_path, trajectory_path, camera_report = _export_cameras(
        run_root, output_root
    )
    trajectory = _load_numpy_pickle(trajectory_path)
    camera_to_world = np.asarray(
        trajectory["rectified_left_camera_to_world"], dtype=np.float32
    )
    if len(camera_to_world) != frame_count:
        raise ValueError("Camera trajectory and video timelines differ")
    depth_path, depth_report = _export_depth(run_root, output_root, frame_count)
    object_records, object_validation = _export_objects(
        run_root, interaction_root, output_root, camera_to_world
    )
    mano_parameters, mano_scale, validation_motion, mano_report = _export_mano(
        run_root, output_root, mano_root, device
    )

    # The self-contained renderer discovers objects through ``manifest.json``.
    # Write the current object list before rendering so a repeated export can
    # never consume a stale manifest from an earlier, partial export.
    manifest_path = output_root / "manifest.json"
    render_manifest = {
        "schema_version": "ego-scene-hoi-sequence-deliverable-render-v1",
        "files": {
            "rectified_left_video": str(left_video.relative_to(output_root)),
            "objects": object_records,
        },
    }
    manifest_path.write_text(
        json.dumps(render_manifest, indent=2) + "\n", encoding="utf-8"
    )

    render_path = output_root / "validation/rendered_overlay.mp4"
    renderer_path = output_root / "render_hand_object.py"
    if not args.skip_render:
        renderer_path = _render_validation(
            output_root, mano_root / "mano_data", render_path, args.device
        )
    else:
        renderer_path = _copy(
            REPOSITORY_ROOT / "tools/render_exported_deliverable.py",
            renderer_path,
        )
    # The NPZ is only an adapter for the existing renderer.  It is deliberately
    # not part of the deliverable because all delivered numeric arrays/scales
    # must use the NumPy-array pickle contract.
    validation_motion.unlink(missing_ok=True)

    maximum_mano_error = max(
        record["vertex_error_mm"]["max"]
        for record in mano_report["per_hand"].values()
    )
    maximum_pose_error = max(
        (record["world_composition_max_abs"] for record in object_validation.values()),
        default=0.0,
    )
    validation_status = (
        "PASS"
        if maximum_mano_error <= 0.1
        and maximum_pose_error <= 1.0e-5
        and (args.skip_render or render_path.is_file())
        else "FAIL"
    )
    validation_report = {
        "schema_version": "ego-scene-hoi-deliverable-validation-v1",
        "status": validation_status,
        "mano_round_trip": mano_report,
        "object_pose_consistency": object_validation,
        "camera": camera_report,
        "depth": depth_report,
        "render": (
            str(render_path.relative_to(output_root))
            if render_path.is_file()
            else None
        ),
        "thresholds": {
            "maximum_mano_vertex_error_mm": 0.1,
            "maximum_object_world_composition_abs": 1.0e-5,
        },
    }
    validation_path = output_root / "validation/report.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(validation_report, indent=2) + "\n", encoding="utf-8"
    )
    array_schema_path = _write_array_schema(output_root)
    readme_path = _write_readme(output_root, output_root.name)
    if provenance["quality_status"] == "failure_case":
        with readme_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n## Quality warning\n\n"
                "This sequence is a declared failure case. Interaction-object "
                "poses are valid only where their exported validity masks are "
                "true (currently the independently registered first frame). "
                "See `quality/failure_case.json`.\n"
            )

    manifest = {
        "schema_version": "ego-scene-hoi-sequence-deliverable-v1",
        "sequence": run_root.name,
        "source_run": str(run_root),
        "interaction_object_source": str(interaction_root),
        "provenance": provenance,
        "frame_count": frame_count,
        "fps": float(left_properties["fps"]),
        "world_coordinate_system": {
            "name": "VIPE rectified-left world",
            "units": "meters",
            "gauge": "approximately the rectified-left camera at frame 0",
            "axes_at_frame_0": "OpenCV x-right, y-down, z-forward",
            "gravity_aligned": False,
            "uses_dataset_gt_extrinsics": False,
        },
        "mano_reconstruction": {
            "formula": "vertices_world = scale * MANO(official_parameters).vertices",
            "official_parameters": str(mano_parameters.relative_to(output_root)),
            "external_scale": str(mano_scale.relative_to(output_root)),
            "topology": "hands/mano_topology.pkl",
            "model": "MANO_LEFT for left and MANO_RIGHT for right",
            "pose_representation": "axis-angle radians",
            "translation": (
                "official pre-scale MANO transl; all downstream rigid translations "
                "are baked into this parameter"
            ),
        },
        "files": {
            "readme": str(readme_path.relative_to(output_root)),
            "array_schema": str(array_schema_path.relative_to(output_root)),
            "rgb_native_video": str(rgb_video.relative_to(output_root)),
            "rectified_left_video": str(left_video.relative_to(output_root)),
            "depth": str(depth_path.relative_to(output_root)),
            "camera_intrinsics": str(intrinsics_path.relative_to(output_root)),
            "camera_trajectory": str(trajectory_path.relative_to(output_root)),
            "mano_parameters": str(mano_parameters.relative_to(output_root)),
            "mano_scale": str(mano_scale.relative_to(output_root)),
            "objects": object_records,
            "hand_object_renderer": str(renderer_path.relative_to(output_root)),
            "validation_report": str(validation_path.relative_to(output_root)),
            "validation_render": (
                str(render_path.relative_to(output_root))
                if render_path.is_file()
                else None
            ),
        },
        "video_properties": {
            "rgb_native": rgb_properties,
            "rectified_left": left_properties,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": validation_status, "output": str(output_root)}, indent=2))
    if validation_status != "PASS":
        raise RuntimeError(f"Deliverable validation failed: {validation_path}")


if __name__ == "__main__":
    main()
