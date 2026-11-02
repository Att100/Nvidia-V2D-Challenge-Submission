"""Rectified-left 2D joint optimization for MANO fitted from UmeTrack."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.utils import PipelineRequest, StageResult, StageStatus, all_artifacts_exist

from .geometry import apply_framewise_hand_pose, optimize_hand_sequence


HAND_NAMES = ("left", "right")
WRIST_CANONICAL_INDEX = 5


@dataclass(frozen=True)
class UmeTrackJointProjectionConfig:
    """Settings for fixed-articulation per-frame MANO R/t optimization."""

    iterations: int = 300
    learning_rate: float = 0.02
    rotation_prior_weight: float = 0.30
    translation_prior_weight: float = 0.01
    translation_acceleration_weight: float = 0.10
    rotation_acceleration_weight: float = 0.10
    device: str = "cuda"


def _segments(selected: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(selected)
    if not len(indices):
        return []
    return [
        part
        for part in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        if len(part)
    ]


def _project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    homogeneous = points @ intrinsic.T
    return homogeneous[..., :2] / homogeneous[..., 2:3]


def _stats(
    joints: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    intrinsic: np.ndarray,
) -> dict[str, float | int | None]:
    errors = np.linalg.norm(_project(joints, intrinsic) - targets, axis=-1)
    selected = valid[..., None] & np.isfinite(errors)
    values = errors[selected]
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def _build_motion(run_root: Path) -> dict[str, np.ndarray]:
    """Convert fitted camera-space MANO into the pipeline hand-motion schema."""

    with np.load(
        run_root / "hands/umetrack_mano/mano_fitted_results.npz"
    ) as archive:
        fit = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(run_root / "hands/umetrack/umetrack_results.npz") as archive:
        ume = {key: np.asarray(archive[key]) for key in archive.files}
    vertices_camera = np.asarray(fit["vertices_left_camera_m"], np.float32)
    joints_camera = np.asarray(fit["joints_left_camera_m"], np.float32)
    valid = np.asarray(fit["valid"], bool) & np.asarray(ume["valid"], bool)
    intrinsic = np.asarray(fit["K"], np.float32)
    frame_count = len(vertices_camera)
    target_joints = (
        np.asarray(ume["keypoints_left_camera_mm"], np.float32)[..., :20, :] / 1000.0
    )
    target_pixels = _project(target_joints, intrinsic)
    wrist = joints_camera[:, :, WRIST_CANONICAL_INDEX].copy()
    local_vertices = vertices_camera - wrist[:, :, None]
    local_joints = joints_camera - wrist[:, :, None]

    with np.load(run_root / "tracks/sam3_hands_left/hand_masks.npz") as archive:
        boxes = np.asarray(archive["bbox_xyxy"], np.float32)
        scores = np.asarray(archive["scores"], np.float32)
        mask_valid = np.asarray(archive["valid"], bool)
        native_size = np.asarray(archive["native_size"], np.int32)
    valid &= mask_valid
    with np.load(run_root / "4d/cameras.npz") as cameras:
        camera_to_world = np.asarray(
            cameras["camera_to_world"]
            if "camera_to_world" in cameras.files
            else cameras["camera_poses"],
            np.float32,
        )
        source_indices = np.asarray(cameras["source_indices"], np.int32)
        timestamps = np.asarray(cameras["timestamps_seconds"], np.float64)
    if len(camera_to_world) != frame_count:
        raise ValueError("UmeTrack and camera timelines differ")

    global_rotvec = np.asarray(fit["global_orient"], np.float32)
    pose_rotvec = np.asarray(fit["hand_pose"], np.float32).reshape(
        frame_count, 2, 15, 3
    )
    global_matrix = Rotation.from_rotvec(global_rotvec.reshape(-1, 3)).as_matrix().reshape(
        frame_count, 2, 1, 3, 3
    ).astype(np.float32)
    pose_matrix = Rotation.from_rotvec(pose_rotvec.reshape(-1, 3)).as_matrix().reshape(
        frame_count, 2, 15, 3, 3
    ).astype(np.float32)
    betas = np.broadcast_to(
        np.asarray(fit["betas"], np.float32)[None], (frame_count, 2, 10)
    ).copy()
    vertices_world = np.full_like(vertices_camera, np.nan)
    joints_world = np.full_like(joints_camera, np.nan)
    for frame_id, hand_id in np.argwhere(valid):
        rotation = camera_to_world[frame_id, :3, :3]
        translation = camera_to_world[frame_id, :3, 3]
        vertices_world[frame_id, hand_id] = (
            vertices_camera[frame_id, hand_id] @ rotation.T + translation
        )
        joints_world[frame_id, hand_id] = (
            joints_camera[frame_id, hand_id] @ rotation.T + translation
        )
    return {
        "vertices_local": local_vertices,
        "joints_local": local_joints,
        "joints_2d_native": target_pixels.astype(np.float32),
        "vertices_camera": vertices_camera,
        "joints_camera": joints_camera,
        "vertices_world": vertices_world,
        "joints_world": joints_world,
        "camera_translation": wrist,
        "wrist_joint_index": np.asarray(WRIST_CANONICAL_INDEX, np.int32),
        "projection_rotation": np.broadcast_to(
            np.eye(3, dtype=np.float32), (frame_count, 2, 3, 3)
        ).copy(),
        "projection_translation": wrist.copy(),
        "valid": valid,
        "detection_valid": valid.copy(),
        "bbox_xyxy": boxes,
        "detection_score": scores,
        "mano_betas": betas,
        "mano_global_orient": global_matrix,
        "mano_hand_pose": pose_matrix,
        "mano_scale": np.asarray(fit["scale"], np.float32),
        "faces_left": np.asarray(fit["faces_right"], np.int32)[:, (0, 2, 1)],
        "faces_right": np.asarray(fit["faces_right"], np.int32),
        "hand_labels": np.asarray(HAND_NAMES),
        "intrinsics_video": intrinsic,
        "intrinsics_geometry": intrinsic,
        "camera_to_world": camera_to_world,
        "source_indices": source_indices,
        "timestamps_seconds": timestamps,
        "video_size": native_size,
        "pointmap_size": native_size,
        "scene_scale": np.asarray(1.0, np.float32),
        "source_model": np.asarray("umetrack_stereo_mano_fit"),
    }


def optimize_umetrack_joint_projection(
    run_root: Path,
    output_dir: Path,
    config: UmeTrackJointProjectionConfig,
) -> None:
    """Optimize per-frame rigid MANO pose against UmeTrack 2D joints."""

    motion = _build_motion(run_root)
    local_joints = motion["joints_local"]
    local_vertices = motion["vertices_local"]
    targets = motion["joints_2d_native"]
    valid = motion["valid"].copy()
    intrinsic = motion["intrinsics_video"]
    initial_vertices = motion["vertices_camera"].copy()
    initial_joints = motion["joints_camera"].copy()
    rotations = motion["projection_rotation"].copy()
    translations = motion["projection_translation"].copy()
    histories: dict[str, list[float]] = {}
    for hand_id, hand_name in enumerate(HAND_NAMES):
        for segment_index, ids in enumerate(_segments(valid[:, hand_id])):
            if len(ids) < 2:
                continue
            bbox = motion["bbox_xyxy"][ids, hand_id]
            bbox_size = np.maximum.reduce(
                (
                    bbox[:, 2] - bbox[:, 0],
                    bbox[:, 3] - bbox[:, 1],
                    np.full(len(ids), 8.0),
                )
            )
            result = optimize_hand_sequence(
                local_joints[ids, hand_id],
                targets[ids, hand_id],
                np.full(targets[ids, hand_id].shape[:-1], np.nan, np.float32),
                intrinsic,
                bbox_size,
                rotations[ids, hand_id],
                translations[ids, hand_id],
                iterations=config.iterations,
                learning_rate=config.learning_rate,
                depth_weight=0.0,
                rotation_prior_weight=config.rotation_prior_weight,
                translation_prior_weight=config.translation_prior_weight,
                translation_acceleration_weight=config.translation_acceleration_weight,
                rotation_acceleration_weight=config.rotation_acceleration_weight,
                fixed_scale=1.0,
                temporal_group_ids=np.zeros(len(ids), np.int64),
                device=config.device,
            )
            rotations[ids, hand_id] = result.rotations
            translations[ids, hand_id] = result.translations
            histories[f"{hand_name}_{segment_index}"] = result.loss_history.tolist()

    vertices = np.full_like(initial_vertices, np.nan)
    joints = np.full_like(initial_joints, np.nan)
    for hand_id in range(2):
        ids = np.flatnonzero(valid[:, hand_id])
        vertices[ids, hand_id] = apply_framewise_hand_pose(
            local_vertices[ids, hand_id], 1.0, rotations[ids, hand_id], translations[ids, hand_id]
        )
        joints[ids, hand_id] = apply_framewise_hand_pose(
            local_joints[ids, hand_id], 1.0, rotations[ids, hand_id], translations[ids, hand_id]
        )
    camera_to_world = motion["camera_to_world"]
    vertices_world = np.full_like(vertices, np.nan)
    joints_world = np.full_like(joints, np.nan)
    for frame_id, hand_id in np.argwhere(valid):
        world_rotation = camera_to_world[frame_id, :3, :3]
        world_translation = camera_to_world[frame_id, :3, 3]
        vertices_world[frame_id, hand_id] = (
            vertices[frame_id, hand_id] @ world_rotation.T + world_translation
        )
        joints_world[frame_id, hand_id] = (
            joints[frame_id, hand_id] @ world_rotation.T + world_translation
        )
    orient = motion["mano_global_orient"].copy()
    orient[valid] = rotations[valid][:, None] @ orient[valid]
    motion.update(
        vertices_camera=vertices,
        joints_camera=joints,
        vertices_world=vertices_world,
        joints_world=joints_world,
        camera_translation=translations,
        projection_rotation=rotations,
        projection_translation=translations,
        mano_global_orient=orient,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "hand_motion.npz", **motion)
    report = {
        "schema_version": "umetrack-mano-left-joint-projection-v1",
        "source": "stereo UmeTrack followed by GPU MANO fitting",
        "optimized_variables": "per-frame rigid R/t; MANO scale, shape, and articulation frozen",
        "target": "UmeTrack 2D joints in rectified-left view",
        "reprojection_before_px": _stats(initial_joints, targets, valid, intrinsic),
        "reprojection_after_px": _stats(joints, targets, valid, intrinsic),
        "valid_frames": {
            label: int(valid[:, index].sum()) for index, label in enumerate(HAND_NAMES)
        },
        "loss_histories": histories,
    }
    (output_dir / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")


class UmeTrackJointProjectionOptimizer:
    """Resumable pipeline stage for rectified-left UmeTrack/MANO projection."""

    name = "umetrack_mano_joint_projection"

    def __init__(self, config: UmeTrackJointProjectionConfig | None = None) -> None:
        self.config = config or UmeTrackJointProjectionConfig()

    def run(self, request: PipelineRequest) -> StageResult:
        output = request.run_root / "hands/umetrack_mano_projection"
        artifacts = output / "hand_motion.npz", output / "diagnostics.json"
        if request.resume and not request.force and all_artifacts_exist(artifacts):
            return StageResult(self.name, StageStatus.REUSED, artifacts)
        if request.dry_run:
            return StageResult(self.name, StageStatus.SKIPPED, artifacts)
        started = time.perf_counter()
        optimize_umetrack_joint_projection(request.run_root, output, self.config)
        if not all_artifacts_exist(artifacts):
            raise RuntimeError("UmeTrack MANO projection did not produce artifacts")
        return StageResult(
            self.name,
            StageStatus.COMPLETED,
            artifacts,
            elapsed_seconds=time.perf_counter() - started,
        )
