#!/usr/bin/env python3
"""Estimate one world-fixed pose for each reconstructed static support object.

Eight temporally distributed, high-visibility observations are registered
independently with FoundationPose.  The selected registration minimizes
rendered RGB-D/mask disagreement and disagreement with the static-world
candidate consensus.  The selected object-to-world transform is then expressed
in every moving camera frame; FoundationPose ``track_one`` is never used.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh

from pipeline.modules.foundation_pose import runtime as fp_runtime
from pipeline.modules.foundation_pose.tracker import (
    prepare_observed_appearance,
    prepare_rendered_appearance,
)
from pipeline.modules.object_scale.fitting import (
    _axis_scales,
    _mesh_frame,
    _robust_frame,
    _scale_transform,
)
from pipeline.utils import parse_worker_args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument(
        "--appearance-mode",
        choices=("rgb", "grayscale", "depth_only"),
        default="grayscale",
    )
    parser.add_argument("--minimum-mask-area", type=int, default=256)
    parser.add_argument(
        "--scale-mode",
        choices=("isotropic", "anisotropic"),
        default="isotropic",
    )
    parser.add_argument("--max-anisotropy", type=float, default=2.0)
    parser.add_argument("--thickness-weight", type=float, default=0.15)
    parser.add_argument("--write-overlay", action=argparse.BooleanOptionalAction, default=True)
    return parse_worker_args(parser)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_-") or "support"


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.dump())
        if not geometries:
            raise ValueError(f"Mesh scene is empty: {path}")
        loaded = geometries[0] if len(geometries) == 1 else trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError(f"Mesh contains no triangles: {path}")
    return loaded


def _load_masks(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        width = int(np.asarray(archive["mask_width"]).item())
        masks = np.unpackbits(
            np.asarray(archive["packed_masks"], dtype=np.uint8), axis=2
        )[..., :width].astype(bool)
        scores = np.asarray(archive["scores"], dtype=np.float32)
        source_indices = np.asarray(archive["source_indices"], dtype=np.int64)
    return masks, scores, source_indices


def _read_frame(video: Path, frame_id: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame_id} from {video}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _candidate_frames(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    count: int,
    minimum_area: int,
    canonical_frame: int,
) -> list[int]:
    """Choose one strong visible observation per temporal bin and keep canonical."""

    areas = masks.sum(axis=(1, 2)).astype(np.float64)
    finite_scores = np.nan_to_num(scores.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    visible = areas >= minimum_area
    quality = np.log1p(areas) * np.maximum(finite_scores, 0.05)
    selected: list[int] = []
    for indices in np.array_split(np.arange(len(masks)), max(1, count)):
        valid = indices[visible[indices]]
        if len(valid):
            selected.append(int(valid[np.argmax(quality[valid])]))
    if visible[int(canonical_frame)]:
        selected.append(int(canonical_frame))
    selected = list(dict.fromkeys(selected))
    if len(selected) > count:
        keep = sorted(selected, key=lambda frame: quality[frame], reverse=True)[:count]
        if visible[int(canonical_frame)] and canonical_frame not in keep:
            keep[-1] = int(canonical_frame)
        selected = sorted(set(keep))
    if not selected:
        raise RuntimeError("No visible support-object observations pass the mask-area gate")
    return selected


def _rotation_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3] @ second[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class StaticSupportPoseEstimator:
    """Run independent multi-frame registration and emit a static world pose."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments

    @classmethod
    def from_worker_job(cls) -> "StaticSupportPoseEstimator":
        return cls(_parse_args())

    def run(self) -> Path:
        args = self.arguments
        run_root = args.run_root.expanduser().resolve()
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        reasoning_path = run_root / "semantics/reasoning/contact_events.json"
        mesh_manifest_path = (
            run_root / "support_objects/sam3d_objects/reconstruction_manifest.json"
        )
        mask_manifest_path = run_root / "tracks/sam3_support/global_masks/manifest.json"
        video = run_root / "calibration/left_rectified_working.mp4"
        depth_dir = run_root / "depth/frames"
        calibration_path = run_root / "calibration/calibration.npz"
        camera_path = run_root / "vipe/pose/rectified_left.npz"
        for path in (
            reasoning_path,
            mesh_manifest_path,
            mask_manifest_path,
            video,
            calibration_path,
            camera_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        reasoning = json.loads(reasoning_path.read_text(encoding="utf-8"))
        if not reasoning.get("support_objects"):
            manifest = output / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "ego-scene-hoi-static-support-pose-v1",
                        "status": "skipped_no_support_objects",
                        "registration_candidates_per_object": int(args.candidate_count),
                        "tracking_used": False,
                        "objects": [],
                        "overlay": None,
                        "elapsed_seconds": 0.0,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(manifest)
            return manifest
        reconstruction = json.loads(mesh_manifest_path.read_text(encoding="utf-8"))
        mesh_jobs = {
            str(item["global_object_id"]): item
            for item in reconstruction["jobs"]
            if item.get("role") == "primary" and item.get("status") == "pass"
        }
        mask_manifest = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
        mask_paths = {
            str(item["global_object_id"]): Path(item["mask_file"])
            for item in mask_manifest["objects"]
        }
        with np.load(calibration_path) as archive:
            K = np.asarray(archive["K_working"], dtype=np.float64)
            left_valid = np.asarray(archive["left_valid"], dtype=bool)
        with np.load(camera_path) as archive:
            camera_to_world = np.asarray(archive["data"], dtype=np.float64)
            camera_frames = np.asarray(archive["inds"], dtype=np.int64)
        if not np.array_equal(camera_frames, np.arange(len(camera_to_world))):
            pose_by_frame = dict(zip(camera_frames.tolist(), camera_to_world, strict=True))
            camera_to_world = np.stack(
                [pose_by_frame.get(frame, np.eye(4)) for frame in range(len(camera_to_world))]
            )

        runtime = fp_runtime.load_foundationpose_runtime()
        runtime.set_logging_format()
        runtime.set_seed(0)
        glctx = runtime.rasterizer.RasterizeCudaContext()
        started = time.perf_counter()
        records: list[dict[str, Any]] = []
        overlay_inputs: list[tuple[str, trimesh.Trimesh, np.ndarray]] = []
        for support in reasoning.get("support_objects", []):
            support_id = str(support["global_object_id"])
            if support_id not in mesh_jobs or support_id not in mask_paths:
                continue
            masks, scores, source_indices = _load_masks(mask_paths[support_id])
            if not np.array_equal(source_indices, np.arange(len(masks))):
                raise ValueError(f"Support masks are not dense frame aligned: {support_id}")
            canonical = int(support["canonical"]["frame"])
            candidates = _candidate_frames(
                masks,
                scores,
                count=int(args.candidate_count),
                minimum_area=int(args.minimum_mask_area),
                canonical_frame=canonical,
            )

            depth = np.asarray(np.load(depth_dir / f"{canonical:06d}.npy"), dtype=np.float32)
            yy, xx = np.indices(depth.shape, dtype=np.float64)
            valid = (
                masks[canonical]
                & left_valid
                & np.isfinite(depth)
                & (depth > 0.10)
                & (depth < 10.0)
            )
            z = depth[valid].astype(np.float64)
            points = np.column_stack(
                (
                    (xx[valid] - K[0, 2]) * z / K[0, 0],
                    (yy[valid] - K[1, 2]) * z / K[1, 1],
                    z,
                )
            )
            cloud_frame = _robust_frame(
                points,
                np.ones(len(points), dtype=np.float64),
                confidence_quantile=0.0,
                extent_quantile=0.01,
            )
            # SAM3D commonly exports PBRMaterial.baseColorTexture, while
            # FoundationPose's tensor adapter expects SimpleMaterial.image.
            # Reuse the established compatibility loader used by the dynamic
            # object tracker instead of maintaining a divergent conversion.
            source_mesh = fp_runtime.load_metric_mesh(
                Path(mesh_jobs[support_id]["output_glb_path"])
            )
            mesh_center, mesh_axes, mesh_extents = _mesh_frame(source_mesh)
            released_metric_mesh = bool(mesh_jobs[support_id].get("metric_mesh", False))
            if released_metric_mesh:
                # Public challenge assets are already expressed in meters.  A
                # second mask/depth scale fit would corrupt their released
                # geometry, especially under partial support-object occlusion.
                scales = np.ones(3, dtype=np.float64)
                metric_mesh = source_mesh.copy()
            else:
                scales = _axis_scales(
                    mesh_extents,
                    cloud_frame.extents,
                    mode=str(args.scale_mode),
                    thickness_weight=float(args.thickness_weight),
                    max_anisotropy=float(args.max_anisotropy),
                )
                metric_mesh = source_mesh.copy()
                metric_mesh.apply_transform(
                    _scale_transform(mesh_center, mesh_axes, scales)
                )
            support_dir = output / support_id
            support_dir.mkdir(parents=True, exist_ok=True)
            metric_mesh_path = support_dir / f"{_safe_name(support['name'])}_metric.glb"
            trimesh.Scene(metric_mesh).export(metric_mesh_path)

            scorer = runtime.score_predictor_class()
            refiner = runtime.refine_predictor_class()
            estimator = runtime.estimator_class(
                model_pts=metric_mesh.vertices,
                model_normals=metric_mesh.vertex_normals,
                mesh=metric_mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir=str(support_dir / "foundationpose_debug"),
                debug=1,
                glctx=glctx,
            )
            prepare_rendered_appearance(estimator.mesh_tensors, args.appearance_mode)
            mesh_tensors = runtime.make_mesh_tensors(metric_mesh)
            candidate_records: list[dict[str, Any]] = []
            world_candidates: list[np.ndarray] = []
            camera_candidates: list[np.ndarray] = []
            for frame_id in candidates:
                rgb = _read_frame(video, frame_id)
                frame_depth = np.asarray(
                    np.load(depth_dir / f"{frame_id:06d}.npy"), dtype=np.float32
                )
                pose = np.asarray(
                    estimator.register(
                        K=K,
                        rgb=prepare_observed_appearance(rgb, args.appearance_mode),
                        depth=frame_depth,
                        ob_mask=masks[frame_id],
                        iteration=int(args.est_refine_iter),
                    ),
                    dtype=np.float64,
                ).reshape(4, 4)
                rendered_depth = fp_runtime.render_candidate_depth(
                    rgb, pose, K, mesh_tensors, glctx, runtime.render_mesh
                )
                confidence = fp_runtime.evaluate_rendered_candidate(
                    masks[frame_id],
                    frame_depth,
                    rendered_depth,
                    depth_scale_meters=0.02,
                )
                camera_candidates.append(pose)
                world_candidates.append(camera_to_world[frame_id] @ pose)
                candidate_records.append(
                    {
                        "frame": int(frame_id),
                        "sam3_score": float(scores[frame_id]),
                        "mask_pixels": int(masks[frame_id].sum()),
                        "foundationpose": confidence.to_dict(),
                    }
                )

            diameter = max(float(np.linalg.norm(metric_mesh.extents)), 1e-4)
            for index, world_pose in enumerate(world_candidates):
                pair_costs = []
                for other in world_candidates:
                    translation = np.linalg.norm(world_pose[:3, 3] - other[:3, 3]) / diameter
                    rotation = _rotation_degrees(world_pose, other) / 45.0
                    pair_costs.append(float(translation + 0.25 * rotation))
                consensus = float(np.median(pair_costs))
                quality = float(candidate_records[index]["foundationpose"]["quality_cost"])
                candidate_records[index]["world_consensus_cost"] = consensus
                candidate_records[index]["selection_cost"] = quality + 0.25 * consensus
            selected_index = int(
                np.argmin([item["selection_cost"] for item in candidate_records])
            )
            selected_frame = int(candidate_records[selected_index]["frame"])
            selected_world = world_candidates[selected_index]
            poses_camera = np.linalg.inv(camera_to_world) @ selected_world
            poses_world = np.repeat(selected_world[None], len(camera_to_world), axis=0)
            fixed_pose_validation: list[dict[str, Any]] = []
            for frame_id in candidates:
                rgb = _read_frame(video, frame_id)
                frame_depth = np.asarray(
                    np.load(depth_dir / f"{frame_id:06d}.npy"), dtype=np.float32
                )
                rendered_depth = fp_runtime.render_candidate_depth(
                    rgb,
                    poses_camera[frame_id],
                    K,
                    mesh_tensors,
                    glctx,
                    runtime.render_mesh,
                )
                confidence = fp_runtime.evaluate_rendered_candidate(
                    masks[frame_id],
                    frame_depth,
                    rendered_depth,
                    depth_scale_meters=0.02,
                )
                fixed_pose_validation.append(
                    {"frame": int(frame_id), **confidence.to_dict()}
                )
            finite_depth = [
                item["depth_median_meters"]
                for item in fixed_pose_validation
                if np.isfinite(item["depth_median_meters"])
            ]
            fixed_pose_aggregate = {
                "frame_count": len(fixed_pose_validation),
                "median_silhouette_iou": float(
                    np.median([item["silhouette_iou"] for item in fixed_pose_validation])
                ),
                "median_visible_mask_coverage": float(
                    np.median(
                        [item["visible_mask_coverage"] for item in fixed_pose_validation]
                    )
                ),
                "median_normalized_center_error": float(
                    np.median(
                        [item["normalized_center_error"] for item in fixed_pose_validation]
                    )
                ),
                "median_depth_residual_meters": (
                    float(np.median(finite_depth)) if finite_depth else None
                ),
            }
            np.savez_compressed(
                support_dir / "poses.npz",
                ob_in_cam=poses_camera.astype(np.float32),
                ob_in_world=poses_world.astype(np.float32),
                camera_to_world=camera_to_world.astype(np.float32),
                intrinsics=np.repeat(K[None], len(camera_to_world), axis=0).astype(np.float32),
                source_indices=np.arange(len(camera_to_world), dtype=np.int32),
                candidate_frames=np.asarray(candidates, dtype=np.int32),
                selected_frame=np.asarray(selected_frame, dtype=np.int32),
            )
            report = {
                "schema_version": "ego-scene-hoi-static-support-pose-v1",
                "global_support_id": support_id,
                "name": support["name"],
                "pose_semantics": "one_selected_object_to_world_transform_expressed_per_camera_frame",
                "tracking_used": False,
                "candidate_policy": "eight_temporal_bins_max_mask_area_times_sam_score",
                "candidate_count_requested": int(args.candidate_count),
                "candidate_frames": candidates,
                "selected_frame": selected_frame,
                "candidates": candidate_records,
                "fixed_pose_validation": fixed_pose_validation,
                "fixed_pose_aggregate": fixed_pose_aggregate,
                "metric_mesh": str(metric_mesh_path),
                "poses": str(support_dir / "poses.npz"),
                "uniform_scale": (
                    float(scales[0])
                    if released_metric_mesh or args.scale_mode == "isotropic"
                    else None
                ),
                "axis_scales_m_per_unit": scales.tolist(),
                "max_anisotropy": float(args.max_anisotropy),
                "scale_mode": (
                    "released_metric_mesh"
                    if released_metric_mesh
                    else f"{args.scale_mode}_depth_fit"
                ),
                "source_mesh": str(mesh_jobs[support_id]["output_glb_path"]),
                "pointcloud_extents_m": cloud_frame.extents.tolist(),
                "mesh_extents_input": mesh_extents.tolist(),
            }
            (support_dir / "report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            records.append(report)
            overlay_inputs.append((support_id, metric_mesh, poses_camera))
            del estimator, scorer, refiner, mesh_tensors
            gc.collect()
            torch.cuda.empty_cache()

        if not records:
            raise RuntimeError("No support object had both a primary mesh and SAM3 masks")
        overlay_path = None
        if args.write_overlay:
            overlay_path = self._write_overlay(
                video, K, runtime, glctx, overlay_inputs, output
            )
        manifest = output / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "ego-scene-hoi-static-support-pose-v1",
                    "status": "pass",
                    "registration_candidates_per_object": int(args.candidate_count),
                    "tracking_used": False,
                    "objects": records,
                    "overlay": str(overlay_path) if overlay_path else None,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(manifest)
        return manifest

    @staticmethod
    def _write_overlay(
        video: Path,
        K: np.ndarray,
        runtime: Any,
        glctx: Any,
        items: list[tuple[str, trimesh.Trimesh, np.ndarray]],
        output: Path,
    ) -> Path:
        capture = cv2.VideoCapture(str(video))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        path = output / "static_support_pose_overlay.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        tensors = [(name, runtime.make_mesh_tensors(mesh), poses) for name, mesh, poses in items]
        try:
            for frame_id in range(count):
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"Support overlay ended at frame {frame_id}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                for item_index, (name, mesh_tensors, poses) in enumerate(tensors):
                    rgb = fp_runtime.render_mesh_overlay(
                        rgb, poses[frame_id], K, mesh_tensors, glctx, runtime.render_mesh
                    )
                    cv2.putText(
                        rgb,
                        f"{name}: static world pose",
                        (12, 30 + 28 * item_index),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            capture.release()
            writer.release()
        return path


def main() -> None:
    StaticSupportPoseEstimator.from_worker_job().run()


if __name__ == "__main__":
    main()
