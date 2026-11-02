"""Visibility-aware rigid object alignment against metric stereo surfaces.

The optimizer keeps mesh scale fixed.  It estimates a small per-frame SE(3)
correction from z-buffer-visible mesh points to FoundationStereo points at the
same pixels, smooths that correction in the VIPE world frame, and rejects any
final correction that degrades surface, depth, or silhouette agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh

from pipeline.modules.foundation_pose import runtime as foundationpose_runtime
from pipeline.optim.hand_surface.rigid_alignment import (
    _robust_rigid_alignment,
    _rotation_matrix,
    _rotation_vector,
)
from pipeline.optim.hand_surface.surface_geometry import BatchedSurfaceRasterizer
from pipeline.utils import PipelineRequest, StageResult, StageStatus, all_artifacts_exist


@dataclass(frozen=True)
class ObjectSurfaceAlignmentConfig:
    """Settings for bounded per-frame rigid stereo-surface refinement."""

    enabled: bool = True
    minimum_mask_pixels: int = 128
    minimum_correspondences: int = 96
    maximum_correspondences: int = 12000
    trim_iterations: int = 5
    raster_batch_size: int = 8
    raster_device: str = "auto"
    maximum_rotation_degrees: float = 5.0
    maximum_translation_meters: float = 0.02
    data_weight_cap: float = 4.0
    prior_weight: float = 0.20
    velocity_weight: float = 8.0
    acceleration_weight: float = 3.0
    maximum_surface_error_increase_meters: float = 0.001
    maximum_depth_error_increase_meters: float = 0.003
    maximum_silhouette_iou_drop: float = 0.02
    write_overlay: bool = False


def _depth_to_pointmap(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Back-project a metric depth map into OpenCV-camera XYZ."""

    height, width = depth.shape
    y, x = np.mgrid[:height, :width]
    result = np.stack(
        (
            (x - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (y - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
        ),
        axis=-1,
    ).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0.10) & (depth < 10.0)
    result[~valid] = np.nan
    return result


def _transform_vertices(vertices: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Transform one object mesh by a batch of object-to-camera poses."""

    return np.einsum("bij,vj->bvi", poses[:, :3, :3], vertices) + poses[:, None, :3, 3]


def _clip_vector(vector: np.ndarray, maximum: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(value))
    if maximum > 0 and length > maximum:
        value *= maximum / length
    return value


def _smooth_vectors(
    raw: np.ndarray,
    weights: np.ndarray,
    config: ObjectSurfaceAlignmentConfig,
) -> np.ndarray:
    """Solve a quadratic data/prior/velocity/acceleration trajectory objective."""

    values = np.asarray(raw, dtype=np.float64)
    data_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    frame_count = len(values)
    if frame_count == 0:
        return values.copy()
    finite = np.isfinite(values).all(axis=1) & (data_weights > 0)
    data = np.zeros_like(values)
    data[finite] = values[finite]
    weight = np.zeros(frame_count, dtype=np.float64)
    weight[finite] = np.minimum(data_weights[finite], config.data_weight_cap)
    matrix = np.diag(weight + float(config.prior_weight))
    if frame_count > 1 and config.velocity_weight > 0:
        first = np.zeros((frame_count - 1, frame_count), dtype=np.float64)
        ids = np.arange(frame_count - 1)
        first[ids, ids] = -1.0
        first[ids, ids + 1] = 1.0
        matrix += float(config.velocity_weight) * (first.T @ first)
    if frame_count > 2 and config.acceleration_weight > 0:
        second = np.zeros((frame_count - 2, frame_count), dtype=np.float64)
        ids = np.arange(frame_count - 2)
        second[ids, ids] = 1.0
        second[ids, ids + 1] = -2.0
        second[ids, ids + 2] = 1.0
        matrix += float(config.acceleration_weight) * (second.T @ second)
    rhs = weight[:, None] * data
    return np.linalg.solve(matrix, rhs)


def _load_sequence_inputs(dataset: Path, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    masks, depths = [], []
    for frame_id in range(frame_count):
        stem = f"{frame_id:06d}"
        mask = cv2.imread(str(dataset / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(dataset / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if mask is None or depth is None:
            raise FileNotFoundError(f"Missing object RGB-D input at {dataset}/{stem}")
        masks.append(mask > 0)
        depths.append(depth.astype(np.float32) / 1000.0)
    return np.stack(masks), np.stack(depths)


def _surface_correspondences(
    vertices: np.ndarray,
    faces: np.ndarray,
    poses: np.ndarray,
    depths: np.ndarray,
    masks: np.ndarray,
    intrinsic: np.ndarray,
    config: ObjectSurfaceAlignmentConfig,
):
    """Rasterize a complete pose sequence and return visible projective pairs."""

    rasterizer = BatchedSurfaceRasterizer(
        intrinsic,
        masks.shape[1:],
        device=config.raster_device,
        # SAM3D meshes can concentrate thousands of small faces in one image
        # tile.  PyTorch3D's coarse binning may overflow silently in that case;
        # the naive path is deterministic and complete at this sequence size.
        bin_size=0,
    )
    result = [None] * len(poses)
    eligible = np.flatnonzero(
        masks.reshape(len(masks), -1).sum(axis=1) >= config.minimum_mask_pixels
    )
    for start in range(0, len(eligible), config.raster_batch_size):
        ids = eligible[start : start + config.raster_batch_size]
        pointmaps = np.stack([_depth_to_pointmap(depths[index], intrinsic) for index in ids])
        pairs = rasterizer(
            _transform_vertices(vertices, poses[ids]),
            faces,
            pointmaps,
            np.ones(masks[ids].shape, dtype=np.float32),
            confidence_threshold=0.5,
            sample_masks=masks[ids],
        )
        for frame_id, pair in zip(ids, pairs, strict=True):
            result[int(frame_id)] = pair
    return result


def _pair_error(pair) -> float:
    if pair is None or not len(pair.predicted):
        return float("nan")
    return float(np.median(np.linalg.norm(pair.predicted - pair.observed, axis=1)))


def _render_metrics(
    mesh: trimesh.Trimesh,
    poses: np.ndarray,
    depths: np.ndarray,
    masks: np.ndarray,
    intrinsic: np.ndarray,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure full rendered silhouette IoU and visible-region depth residual."""

    runtime = foundationpose_runtime.load_foundationpose_runtime()
    context = runtime.rasterizer.RasterizeCudaContext()
    tensors = runtime.make_mesh_tensors(mesh)
    iou = np.full(len(poses), np.nan, dtype=np.float32)
    depth_error = np.full(len(poses), np.nan, dtype=np.float32)
    height, width = masks.shape[1:]
    for start in range(0, len(poses), batch_size):
        stop = min(start + batch_size, len(poses))
        _, rendered, _ = runtime.render_mesh(
            K=intrinsic,
            H=height,
            W=width,
            ob_in_cams=torch.as_tensor(
                poses[start:stop], device="cuda", dtype=torch.float32
            ),
            glctx=context,
            mesh_tensors=tensors,
            use_light=False,
        )
        rendered = rendered.detach().cpu().numpy()
        for local, frame_id in enumerate(range(start, stop)):
            predicted = np.isfinite(rendered[local]) & (rendered[local] > 0)
            observed = masks[frame_id]
            intersection = predicted & observed
            union = predicted | observed
            iou[frame_id] = float(intersection.sum() / max(int(union.sum()), 1))
            valid = intersection & np.isfinite(depths[frame_id]) & (depths[frame_id] > 0)
            if valid.any():
                depth_error[frame_id] = float(
                    np.median(np.abs(rendered[local][valid] - depths[frame_id][valid]))
                )
    return iou, depth_error


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "median": None, "p90": None, "mean": None}
    return {
        "count": int(len(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
        "mean": float(np.mean(finite)),
    }


def align_object_pose_sequence(
    dataset_dir: Path,
    mesh_path: Path,
    pose_path: Path,
    output_dir: Path,
    config: ObjectSurfaceAlignmentConfig | None = None,
) -> Path:
    """Align one rigid object sequence and return the refined pose archive."""

    settings = config or ObjectSurfaceAlignmentConfig()
    dataset = Path(dataset_dir).resolve()
    mesh_file = Path(mesh_path).resolve()
    source_pose = Path(pose_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source_pose) as archive:
        pose_archive = {key: np.asarray(archive[key]) for key in archive.files}
    camera_poses = np.asarray(pose_archive["ob_in_cam"], dtype=np.float64)
    camera_to_world = np.asarray(pose_archive["camera_to_world"], dtype=np.float64)
    world_poses = np.asarray(pose_archive["ob_in_world"], dtype=np.float64)
    frame_count = len(camera_poses)
    with np.load(dataset / "camera_K.npz") as archive:
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float64)
    if float(np.max(np.abs(intrinsics - intrinsics[0]))) > 1e-6:
        raise ValueError("Object surface alignment currently requires shared intrinsics")
    intrinsic = intrinsics[0]
    masks, depths = _load_sequence_inputs(dataset, frame_count)
    mesh = foundationpose_runtime.load_metric_mesh(mesh_file)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    initial_pairs = _surface_correspondences(
        vertices, faces, camera_poses, depths, masks, intrinsic, settings
    )
    pre_surface = np.asarray([_pair_error(pair) for pair in initial_pairs], dtype=np.float32)
    raw_world_rotation = np.full((frame_count, 3), np.nan, dtype=np.float64)
    raw_world_translation = np.full((frame_count, 3), np.nan, dtype=np.float64)
    support = np.zeros(frame_count, dtype=np.int32)
    raw_surface = np.full(frame_count, np.nan, dtype=np.float32)
    mesh_diameter = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    maximum_translation = min(
        float(settings.maximum_translation_meters), max(0.003, 0.10 * mesh_diameter)
    )

    for frame_id, pair in enumerate(initial_pairs):
        if pair is None or len(pair.predicted) < settings.minimum_correspondences:
            continue
        pivot = camera_poses[frame_id, :3, 3]
        rotation, pivot_translation, count, error = _robust_rigid_alignment(
            pair.predicted,
            pair.observed,
            pivot,
            minimum_count=settings.minimum_correspondences,
            trim_iterations=settings.trim_iterations,
            maximum_correspondences=settings.maximum_correspondences,
        )
        if not np.isfinite(pivot_translation).all():
            continue
        camera_vector = _clip_vector(
            _rotation_vector(rotation), np.deg2rad(settings.maximum_rotation_degrees)
        )
        camera_rotation = _rotation_matrix(camera_vector)
        pivot_translation = _clip_vector(pivot_translation, maximum_translation)
        standard_translation = pivot + pivot_translation - camera_rotation @ pivot
        delta_camera = np.eye(4, dtype=np.float64)
        delta_camera[:3, :3] = camera_rotation
        delta_camera[:3, 3] = standard_translation
        candidate_camera = delta_camera @ camera_poses[frame_id]
        candidate_world = camera_to_world[frame_id] @ candidate_camera
        delta_world_rotation = (
            candidate_world[:3, :3] @ world_poses[frame_id, :3, :3].T
        )
        transformed = pair.predicted @ camera_rotation.T + standard_translation
        after_error = float(
            np.median(np.linalg.norm(transformed - pair.observed, axis=1))
        )
        if not np.isfinite(after_error) or after_error >= float(pre_surface[frame_id]):
            continue
        raw_world_rotation[frame_id] = _rotation_vector(delta_world_rotation)
        # Smooth displacement of the object origin, not the affine translation
        # of a left-multiplied transform about the arbitrary world origin.
        raw_world_translation[frame_id] = (
            candidate_world[:3, 3] - world_poses[frame_id, :3, 3]
        )
        support[frame_id] = count
        raw_surface[frame_id] = after_error

    weights = support.astype(np.float64) / max(settings.minimum_correspondences, 1)
    smooth_rotation = _smooth_vectors(raw_world_rotation, weights, settings)
    smooth_translation = _smooth_vectors(raw_world_translation, weights, settings)
    for frame_id in range(frame_count):
        smooth_rotation[frame_id] = _clip_vector(
            smooth_rotation[frame_id], np.deg2rad(settings.maximum_rotation_degrees)
        )
        smooth_translation[frame_id] = _clip_vector(
            smooth_translation[frame_id], maximum_translation
        )

    candidate_world = world_poses.copy()
    for frame_id in range(frame_count):
        correction = _rotation_matrix(smooth_rotation[frame_id])
        candidate_world[frame_id, :3, :3] = (
            correction @ world_poses[frame_id, :3, :3]
        )
        candidate_world[frame_id, :3, 3] = (
            world_poses[frame_id, :3, 3] + smooth_translation[frame_id]
        )
    candidate_camera = np.einsum(
        "bij,bjk->bik", np.linalg.inv(camera_to_world), candidate_world
    )

    candidate_pairs = _surface_correspondences(
        vertices, faces, candidate_camera, depths, masks, intrinsic, settings
    )
    post_surface_candidate = np.asarray(
        [_pair_error(pair) for pair in candidate_pairs], dtype=np.float32
    )
    pre_iou, pre_depth = _render_metrics(
        mesh, camera_poses, depths, masks, intrinsic
    )
    post_iou_candidate, post_depth_candidate = _render_metrics(
        mesh, candidate_camera, depths, masks, intrinsic
    )
    measured = support >= settings.minimum_correspondences
    accepted = (
        measured
        & np.isfinite(post_surface_candidate)
        & (
            post_surface_candidate
            <= pre_surface + settings.maximum_surface_error_increase_meters
        )
        & (
            ~np.isfinite(pre_depth)
            | (
                post_depth_candidate
                <= pre_depth + settings.maximum_depth_error_increase_meters
            )
        )
        & (
            ~np.isfinite(pre_iou)
            | (post_iou_candidate >= pre_iou - settings.maximum_silhouette_iou_drop)
        )
    )
    refined_world = world_poses.copy()
    refined_camera = camera_poses.copy()
    refined_world[accepted] = candidate_world[accepted]
    refined_camera[accepted] = candidate_camera[accepted]
    final_surface = pre_surface.copy()
    final_depth = pre_depth.copy()
    final_iou = pre_iou.copy()
    final_surface[accepted] = post_surface_candidate[accepted]
    final_depth[accepted] = post_depth_candidate[accepted]
    final_iou[accepted] = post_iou_candidate[accepted]

    pose_archive.update(
        ob_in_cam=refined_camera.astype(np.float32),
        ob_in_world=refined_world.astype(np.float32),
        surface_alignment_rotation_vector_world=smooth_rotation.astype(np.float32),
        surface_alignment_translation_world=smooth_translation.astype(np.float32),
        surface_alignment_support=support,
        surface_alignment_accepted=accepted,
        surface_alignment_pre_error=pre_surface,
        surface_alignment_post_error=final_surface,
    )
    result = output / "poses.npz"
    np.savez_compressed(result, **pose_archive)
    report = {
        "format": "ego-scene-hoi-object-stereo-surface-alignment-v1",
        "source_poses": str(source_pose),
        "dataset": str(dataset),
        "mesh": str(mesh_file),
        "scale_optimized": False,
        "optimized_variables": ["per-frame rigid rotation", "per-frame translation"],
        "coordinate_system": "VIPE world with rectified-left camera observations",
        "objective": {
            "data": "projective visible-surface point-to-point robust alignment",
            "pose_prior_weight": settings.prior_weight,
            "velocity_weight": settings.velocity_weight,
            "acceleration_weight": settings.acceleration_weight,
            "final_guards": ["surface", "rendered depth", "silhouette IoU"],
        },
        "limits": {
            "rotation_degrees": settings.maximum_rotation_degrees,
            "translation_meters": maximum_translation,
        },
        "frames": {
            "total": frame_count,
            "measured": int(measured.sum()),
            "accepted": int(accepted.sum()),
        },
        "surface_error_before_m": _stats(pre_surface[accepted]),
        "surface_error_after_m": _stats(final_surface[accepted]),
        "depth_error_before_m": _stats(pre_depth[accepted]),
        "depth_error_after_m": _stats(final_depth[accepted]),
        "silhouette_iou_before": _stats(pre_iou[accepted]),
        "silhouette_iou_after": _stats(final_iou[accepted]),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    if settings.write_overlay:
        runtime = foundationpose_runtime.load_foundationpose_runtime()
        context = runtime.rasterizer.RasterizeCudaContext()
        tensors = runtime.make_mesh_tensors(mesh)
        rgb_files = sorted((dataset / "rgb").glob("*.png"))
        first = imageio.imread(rgb_files[0])[..., :3]
        writer = cv2.VideoWriter(
            str(output / "surface_alignment_overlay.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(json.loads((dataset / "dataset_report.json").read_text())["fps"]),
            (first.shape[1] * 2, first.shape[0]),
        )
        for frame_id, rgb_file in enumerate(rgb_files):
            rgb = imageio.imread(rgb_file)[..., :3]
            before = foundationpose_runtime.render_mesh_overlay(
                rgb, camera_poses[frame_id], intrinsic, tensors, context, runtime.render_mesh
            )
            after = foundationpose_runtime.render_mesh_overlay(
                rgb, refined_camera[frame_id], intrinsic, tensors, context, runtime.render_mesh
            )
            cv2.putText(before, "before", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(after, "surface aligned", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            panel = np.concatenate((before, after), axis=1)
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        writer.release()
    return result


class StereoObjectSurfaceAlignmentOptimizer:
    """Pipeline stage refining stereo FoundationPose tracks before contact."""

    name = "stereo_object_surface_alignment"

    def __init__(self, config: ObjectSurfaceAlignmentConfig | None = None) -> None:
        self.config = config or ObjectSurfaceAlignmentConfig()

    def run(self, request: PipelineRequest) -> StageResult:
        manifest_path = request.run_root / "optim/object_surface_v1/manifest.json"
        source_candidates = (
            request.run_root / "foundationpose_left_adaptive/manifest.json",
            request.run_root / "foundationpose_standard_no_reregister/manifest.json",
        )
        source_manifest = next(
            (path for path in source_candidates if path.is_file()),
            source_candidates[0],
        )
        if not source_manifest.is_file():
            raise FileNotFoundError(
                "No stereo FoundationPose manifest found; checked: "
                + ", ".join(str(path) for path in source_candidates)
            )
        source_root = source_manifest.parent
        records = json.loads(source_manifest.read_text(encoding="utf-8")).get("objects", [])
        artifacts = tuple(
            request.run_root / "optim/object_surface_v1" / str(item["object_id"]) / "poses.npz"
            for item in records
        ) + (manifest_path,)
        if request.resume and not request.force and all_artifacts_exist(artifacts):
            return StageResult(self.name, StageStatus.REUSED, artifacts)
        if request.dry_run or not self.config.enabled:
            return StageResult(self.name, StageStatus.SKIPPED, artifacts)
        started = time.perf_counter()
        outputs = []
        for item in records:
            object_id = str(item["object_id"])
            object_output = request.run_root / "optim/object_surface_v1" / object_id
            result = align_object_pose_sequence(
                source_root / "objects" / object_id / "dataset",
                Path(item["metric_mesh"]),
                Path(item["poses"]),
                object_output,
                self.config,
            )
            pose_track = request.run_root / "objects" / object_id / "pose_track" / "poses.npz"
            pose_track.parent.mkdir(parents=True, exist_ok=True)
            if pose_track.is_file():
                backup = pose_track.with_name("foundationpose_poses.npz")
                if not backup.exists():
                    shutil.copy2(pose_track, backup)
                with np.load(pose_track) as archive:
                    adapter_values = {key: np.asarray(archive[key]) for key in archive.files}
                with np.load(result) as archive:
                    aligned = {key: np.asarray(archive[key]) for key in archive.files}
                adapter_values["object_to_world"] = aligned["ob_in_world"]
                adapter_values["object_to_camera"] = aligned["ob_in_cam"]
                adapter_values["surface_alignment_accepted"] = aligned[
                    "surface_alignment_accepted"
                ]
                np.savez_compressed(pose_track, **adapter_values)
            outputs.append(
                {
                    "object_id": object_id,
                    "poses": str(result),
                    "report": str(object_output / "report.json"),
                    "optimizer_pose_track": str(pose_track),
                }
            )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "ego-scene-hoi-object-surface-manifest-v1",
                    "source": str(source_manifest),
                    "objects": outputs,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return StageResult(
            self.name,
            StageStatus.COMPLETED,
            artifacts,
            elapsed_seconds=time.perf_counter() - started,
            metrics={"object_count": len(outputs)},
        )
