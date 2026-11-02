#!/usr/bin/env python3
"""Camera-motion-compensated bidirectional FoundationPose tracking.

Registration starts at a selected high-visibility frame.  Tracking then runs
forward and backward.  Before each ``track_one`` call, the previous object pose
is transported through the VIPE camera motion, which is essential for an
egocentric moving camera.  No dataset camera extrinsics are read.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import trimesh
from pipeline.utils import parse_worker_args


from pipeline.modules.foundation_pose import runtime as foundationpose_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-label", required=True)
    parser.add_argument("--seed-frame", type=int, required=True, help="Dataset-local seed frame")
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument(
        "--appearance-mode",
        choices=("rgb", "grayscale", "depth_only"),
        default="rgb",
        help=(
            "Appearance ablation used by the learned FoundationPose scorer/refiner. "
            "grayscale converts both the observation and rendered mesh to replicated "
            "luminance; depth_only zeros both appearance inputs while retaining XYZ/depth."
        ),
    )
    parser.add_argument(
        "--adaptive-reregister",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--reregister-interval", type=int, default=20)
    parser.add_argument("--minimum-silhouette-iou", type=float, default=0.30)
    parser.add_argument("--minimum-mask-coverage", type=float, default=0.50)
    parser.add_argument("--reregister-accept-margin", type=float, default=0.02)
    parser.add_argument("--maximum-reregister-rotation-degrees", type=float, default=25.0)
    parser.add_argument("--maximum-reregister-translation-meters", type=float, default=0.03)
    parser.add_argument("--maximum-reregister-translation-fraction", type=float, default=0.15)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--render-batch-size",
        type=int,
        default=512,
        help="Maximum number of mesh pose hypotheses rendered concurrently.",
    )
    parser.add_argument("--write-overlay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--registration-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Register only the canonical frame. Other frames receive the same "
            "object-to-world initialization expressed in their camera frame; "
            "they are initialization values, not tracking results."
        ),
    )
    return parse_worker_args(parser)


def load_frame(dataset: Path, frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stem = f"{frame_id:06d}"
    rgb = imageio.imread(dataset / "rgb" / f"{stem}.png")[..., :3]
    raw_depth = cv2.imread(str(dataset / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(dataset / "masks" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    if raw_depth is None or raw_depth.dtype != np.uint16 or mask is None:
        raise ValueError(f"Invalid RGB-D-mask input at frame {stem}")
    return rgb, raw_depth.astype(np.float32) / 1000.0, mask > 0


def prepare_observed_appearance(rgb: np.ndarray, mode: str) -> np.ndarray:
    """Prepare the observed three-channel appearance for a controlled ablation."""

    if mode == "rgb":
        return rgb
    if mode == "depth_only":
        return np.zeros_like(rgb)
    if mode == "grayscale":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return np.repeat(gray[..., None], 3, axis=2)
    raise ValueError(f"Unsupported FoundationPose appearance mode: {mode}")


def prepare_rendered_appearance(mesh_tensors: dict, mode: str) -> None:
    """Modify renderer colors in place to match the observation appearance domain."""

    if mode == "rgb":
        return
    key = "tex" if "tex" in mesh_tensors else "vertex_color"
    colors = mesh_tensors[key]
    if mode == "depth_only":
        mesh_tensors[key] = torch.zeros_like(colors)
        return
    if mode == "grayscale":
        weights = torch.as_tensor(
            [0.299, 0.587, 0.114], dtype=colors.dtype, device=colors.device
        )
        luminance = (colors[..., :3] * weights).sum(dim=-1, keepdim=True)
        mesh_tensors[key] = luminance.expand_as(colors).contiguous()
        return
    raise ValueError(f"Unsupported FoundationPose appearance mode: {mode}")


def set_camera_compensated_prior(
    estimator,
    previous_pose_camera: np.ndarray,
    previous_camera_to_world: np.ndarray,
    current_camera_to_world: np.ndarray,
) -> None:
    object_to_world = previous_camera_to_world @ previous_pose_camera
    prior_camera = np.linalg.inv(current_camera_to_world) @ object_to_world
    to_center = estimator.get_tf_to_centered_mesh().detach().cpu().numpy().reshape(4, 4)
    internal = prior_camera @ np.linalg.inv(to_center)
    estimator.pose_last = torch.as_tensor(
        internal[None], dtype=torch.float32, device="cuda"
    )


def temporal_world_cost(
    previous_camera_to_world: np.ndarray,
    previous_pose: np.ndarray,
    current_camera_to_world: np.ndarray,
    candidate_pose: np.ndarray,
    diameter: float,
) -> float:
    previous_world = previous_camera_to_world @ previous_pose
    candidate_world = current_camera_to_world @ candidate_pose
    translation = np.linalg.norm(candidate_world[:3, 3] - previous_world[:3, 3])
    delta = candidate_world[:3, :3] @ previous_world[:3, :3].T
    angle = np.degrees(np.arccos(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)))
    return float(0.08 * translation / max(diameter, 1e-4) + 0.04 * angle / 45.0)


def world_pose_delta(
    previous_camera_to_world: np.ndarray,
    previous_pose: np.ndarray,
    current_camera_to_world: np.ndarray,
    candidate_pose: np.ndarray,
) -> tuple[float, float]:
    """Return candidate motion relative to the camera-compensated world prior."""

    previous_world = previous_camera_to_world @ previous_pose
    candidate_world = current_camera_to_world @ candidate_pose
    relative_rotation = candidate_world[:3, :3] @ previous_world[:3, :3].T
    cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    rotation_degrees = float(np.degrees(np.arccos(cosine)))
    translation_meters = float(
        np.linalg.norm(candidate_world[:3, 3] - previous_world[:3, 3])
    )
    return rotation_degrees, translation_meters


def run_bidirectional_tracking(args: argparse.Namespace) -> None:
    """Run camera-compensated registration and bidirectional tracking."""
    dataset = args.dataset_dir.expanduser().resolve()
    mesh_path = args.mesh.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pose_dir = output / "ob_in_cam"
    vis_dir = output / "track_vis"
    debug_dir = output / "foundationpose_debug"
    for directory in (pose_dir, vis_dir, debug_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted((dataset / "rgb").glob("*.png"))
    frame_count = len(rgb_files)
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    if not 0 <= args.seed_frame < frame_count:
        raise ValueError(f"Seed frame {args.seed_frame} outside [0,{frame_count})")
    with np.load(dataset / "camera_K.npz") as archive:
        intrinsics = np.asarray(archive["intrinsics"][:frame_count], dtype=np.float64)
    with np.load(dataset / "camera_poses_world.npz") as archive:
        camera_to_world = np.asarray(
            archive["rigid_camera_to_world"][:frame_count], dtype=np.float64
        )
        source_indices = np.asarray(archive["source_indices"][:frame_count], dtype=np.int32)

    appearance_mode = str(getattr(args, "appearance_mode", "rgb"))
    maximum_reregister_rotation_degrees = float(
        getattr(args, "maximum_reregister_rotation_degrees", 25.0)
    )
    maximum_reregister_translation_meters = float(
        getattr(args, "maximum_reregister_translation_meters", 0.03)
    )
    maximum_reregister_translation_fraction = float(
        getattr(args, "maximum_reregister_translation_fraction", 0.15)
    )
    render_batch_size = int(getattr(args, "render_batch_size", 512))
    if render_batch_size < 1:
        raise ValueError("render_batch_size must be positive")
    os.environ["FOUNDATIONPOSE_RENDER_BATCH_SIZE"] = str(render_batch_size)
    reference = foundationpose_runtime
    runtime = reference.load_foundationpose_runtime()
    mesh = reference.load_metric_mesh(mesh_path)
    runtime.set_logging_format()
    runtime.set_seed(0)
    scorer = runtime.score_predictor_class()
    refiner = runtime.refine_predictor_class()
    glctx = runtime.rasterizer.RasterizeCudaContext()
    estimator = runtime.estimator_class(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=1,
        glctx=glctx,
    )
    prepare_rendered_appearance(estimator.mesh_tensors, appearance_mode)
    mesh_tensors = runtime.make_mesh_tensors(mesh)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    poses = np.full((frame_count, 4, 4), np.nan, dtype=np.float64)
    frame_seconds = np.full(frame_count, np.nan, dtype=np.float64)
    confidence_records = []
    reregister_attempts = 0
    reregister_accepts = 0
    started = time.perf_counter()

    seed_rgb, seed_depth, seed_mask = load_frame(dataset, args.seed_frame)
    seed_model_rgb = prepare_observed_appearance(seed_rgb, appearance_mode)
    poses[args.seed_frame] = np.asarray(
        estimator.register(
            K=intrinsics[args.seed_frame],
            rgb=seed_model_rgb,
            depth=seed_depth,
            ob_mask=seed_mask,
            iteration=args.est_refine_iter,
        ),
        dtype=np.float64,
    ).reshape(4, 4)

    registration_only = bool(getattr(args, "registration_only", False))
    if registration_only:
        canonical_object_to_world = (
            camera_to_world[args.seed_frame] @ poses[args.seed_frame]
        )
        poses[:] = np.linalg.inv(camera_to_world) @ canonical_object_to_world
        frame_seconds[:] = 0.0
        orders = []
    else:
        orders = [
            range(args.seed_frame + 1, frame_count),
            range(args.seed_frame - 1, -1, -1),
        ]
    for order in orders:
        previous = args.seed_frame
        # Restore the seed state before changing temporal direction.
        set_camera_compensated_prior(
            estimator,
            poses[previous],
            camera_to_world[previous],
            camera_to_world[previous],
        )
        for frame_id in order:
            frame_start = time.perf_counter()
            rgb, depth, mask = load_frame(dataset, frame_id)
            model_rgb = prepare_observed_appearance(rgb, appearance_mode)
            set_camera_compensated_prior(
                estimator,
                poses[previous],
                camera_to_world[previous],
                camera_to_world[frame_id],
            )
            # An absent object mask means that the object is not observable in
            # this frame.  Preserve its last world pose and express it in the
            # current camera frame; never send an empty mask to registration.
            # This also prevents RGB-D refinement from drifting an invisible
            # object onto background geometry.
            if args.adaptive_reregister and int(mask.sum()) < 64:
                object_to_world = camera_to_world[previous] @ poses[previous]
                pose = np.linalg.inv(camera_to_world[frame_id]) @ object_to_world
                poses[frame_id] = pose
                frame_seconds[frame_id] = time.perf_counter() - frame_start
                confidence_records.append(
                    {
                        "frame": frame_id,
                        "direction": "forward" if frame_id > args.seed_frame else "backward",
                        "reasons": ["object_mask_missing"],
                        "accepted": False,
                        "visible": False,
                        "observed_mask_pixels": int(mask.sum()),
                    }
                )
                previous = frame_id
                completed = int(np.isfinite(poses[:, 0, 0]).sum())
                if completed % 20 == 0 or completed == frame_count:
                    print(
                        f"[FoundationPose] {args.object_label}: {completed}/{frame_count}",
                        flush=True,
                    )
                continue
            pose = np.asarray(
                estimator.track_one(
                    rgb=model_rgb,
                    depth=depth,
                    K=intrinsics[frame_id],
                    iteration=args.track_refine_iter,
                ),
                dtype=np.float64,
            ).reshape(4, 4)
            if not np.isfinite(pose).all():
                pose = np.asarray(
                    estimator.register(
                        K=intrinsics[frame_id],
                        rgb=model_rgb,
                        depth=depth,
                        ob_mask=mask,
                        iteration=args.est_refine_iter,
                    ),
                    dtype=np.float64,
                ).reshape(4, 4)
            elif args.adaptive_reregister:
                tracked_state = estimator.pose_last.clone()
                tracked_depth = reference.render_candidate_depth(
                    rgb,
                    pose,
                    intrinsics[frame_id],
                    mesh_tensors,
                    glctx,
                    runtime.render_mesh,
                )
                tracked_confidence = reference.evaluate_rendered_candidate(
                    mask,
                    depth,
                    tracked_depth,
                    depth_scale_meters=0.015,
                )
                distance_from_seed = abs(frame_id - args.seed_frame)
                reasons = []
                if tracked_confidence.silhouette_iou < args.minimum_silhouette_iou:
                    reasons.append("low_iou")
                if tracked_confidence.visible_mask_coverage < args.minimum_mask_coverage:
                    reasons.append("low_coverage")
                if args.reregister_interval > 0 and distance_from_seed % args.reregister_interval == 0:
                    reasons.append("periodic")
                accepted = False
                registered_confidence = None
                if reasons:
                    reregister_attempts += 1
                    registered_pose = np.asarray(
                        estimator.register(
                            K=intrinsics[frame_id],
                            rgb=model_rgb,
                            depth=depth,
                            ob_mask=mask,
                            iteration=args.est_refine_iter,
                        ),
                        dtype=np.float64,
                    ).reshape(4, 4)
                    registered_state = estimator.pose_last.clone()
                    registered_depth = reference.render_candidate_depth(
                        rgb,
                        registered_pose,
                        intrinsics[frame_id],
                        mesh_tensors,
                        glctx,
                        runtime.render_mesh,
                    )
                    registered_confidence = reference.evaluate_rendered_candidate(
                        mask,
                        depth,
                        registered_depth,
                        depth_scale_meters=0.015,
                    )
                    tracked_cost = tracked_confidence.quality_cost + temporal_world_cost(
                        camera_to_world[previous],
                        poses[previous],
                        camera_to_world[frame_id],
                        pose,
                        float(estimator.diameter),
                    )
                    registered_cost = registered_confidence.quality_cost + temporal_world_cost(
                        camera_to_world[previous],
                        poses[previous],
                        camera_to_world[frame_id],
                        registered_pose,
                        float(estimator.diameter),
                    )
                    candidate_rotation_degrees, candidate_translation_meters = world_pose_delta(
                        camera_to_world[previous],
                        poses[previous],
                        camera_to_world[frame_id],
                        registered_pose,
                    )
                    translation_gate_meters = max(
                        maximum_reregister_translation_meters,
                        maximum_reregister_translation_fraction * float(estimator.diameter),
                    )
                    rotation_gate_passed = (
                        candidate_rotation_degrees
                        <= maximum_reregister_rotation_degrees
                    )
                    translation_gate_passed = (
                        candidate_translation_meters <= translation_gate_meters
                    )
                    gate_passed = rotation_gate_passed and translation_gate_passed
                    quality_passed = (
                        registered_cost + args.reregister_accept_margin < tracked_cost
                        or (
                            tracked_confidence.silhouette_iou < 0.10
                            and registered_confidence.silhouette_iou > 0.25
                        )
                    )
                    accepted = gate_passed and quality_passed
                    if accepted:
                        pose = registered_pose
                        estimator.pose_last = registered_state
                        reregister_accepts += 1
                    else:
                        estimator.pose_last = tracked_state
                    confidence_records.append(
                        {
                            "frame": frame_id,
                            "direction": "forward" if frame_id > args.seed_frame else "backward",
                            "reasons": reasons,
                            "accepted": accepted,
                            "tracked": tracked_confidence.to_dict(),
                            "registered": registered_confidence.to_dict(),
                            "tracked_cost": float(tracked_cost),
                            "registered_cost": float(registered_cost),
                            "candidate_rotation_degrees_from_world_prior": (
                                candidate_rotation_degrees
                            ),
                            "candidate_translation_meters_from_world_prior": (
                                candidate_translation_meters
                            ),
                            "rotation_gate_degrees": float(
                                maximum_reregister_rotation_degrees
                            ),
                            "translation_gate_meters": translation_gate_meters,
                            "rotation_gate_passed": rotation_gate_passed,
                            "translation_gate_passed": translation_gate_passed,
                            "quality_gate_passed": quality_passed,
                        }
                    )
            poses[frame_id] = pose
            frame_seconds[frame_id] = time.perf_counter() - frame_start
            previous = frame_id
            completed = int(np.isfinite(poses[:, 0, 0]).sum())
            if completed % 20 == 0 or completed == frame_count:
                print(f"[FoundationPose] {args.object_label}: {completed}/{frame_count}", flush=True)

    object_to_world = camera_to_world @ poses
    for frame_id, pose in enumerate(poses):
        np.savetxt(pose_dir / f"{frame_id:06d}.txt", pose, fmt="%.10f")

    dataset_report = json.loads((dataset / "dataset_report.json").read_text())
    if args.write_overlay:
        height, width = seed_rgb.shape[:2]
        fps = float(dataset_report["fps"])
        writer = cv2.VideoWriter(
            str(output / "foundationpose_tracking_vis.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for frame_id in range(frame_count):
            rgb, _, _ = load_frame(dataset, frame_id)
            vis = reference.render_mesh_overlay(
                rgb, poses[frame_id], intrinsics[frame_id], mesh_tensors, glctx, runtime.render_mesh
            )
            center_pose = poses[frame_id] @ np.linalg.inv(to_origin)
            vis = runtime.draw_box(intrinsics[frame_id], img=vis, ob_in_cam=center_pose, bbox=bbox)
            vis = runtime.draw_axis(
                vis,
                ob_in_cam=center_pose,
                scale=0.08,
                K=intrinsics[frame_id],
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            imageio.imwrite(vis_dir / f"{frame_id:06d}.png", vis)
            writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        writer.release()

    np.savez_compressed(
        output / "poses.npz",
        ob_in_cam=poses.astype(np.float32),
        ob_in_world=object_to_world.astype(np.float32),
        intrinsics=intrinsics.astype(np.float32),
        camera_to_world=camera_to_world.astype(np.float32),
        source_indices=source_indices,
        seed_frame=np.asarray(args.seed_frame, dtype=np.int32),
    )
    report = {
        "format": "foundationpose-camera-compensated-bidirectional-v2",
        "object_label": args.object_label,
        "mesh": str(mesh_path),
        "frame_count": frame_count,
        "seed_dataset_frame": args.seed_frame,
        "seed_source_frame": int(source_indices[args.seed_frame]),
        "directions": ["forward", "backward"],
        "registration_only": registration_only,
        "noncanonical_pose_semantics": (
            "camera-compensated static-world initialization"
            if registration_only
            else "FoundationPose bidirectional track_one"
        ),
        "camera_motion_compensation": True,
        "camera_trajectory_source": dataset_report.get(
            "camera_trajectory_source",
            dataset_report.get("depth_source", "pipeline camera estimate"),
        ),
        "uses_gt_camera_extrinsics": False,
        "elapsed_seconds": time.perf_counter() - started,
        "median_track_seconds": float(np.nanmedian(frame_seconds)),
        "adaptive_reregistration": {
            "enabled": args.adaptive_reregister,
            "interval": args.reregister_interval,
            "attempts": reregister_attempts,
            "accepted": reregister_accepts,
            "maximum_rotation_degrees": float(
                maximum_reregister_rotation_degrees
            ),
            "maximum_translation_meters": float(
                maximum_reregister_translation_meters
            ),
            "maximum_translation_fraction_of_diameter": float(
                maximum_reregister_translation_fraction
            ),
        },
        "write_overlay": bool(args.write_overlay),
        "appearance_mode": appearance_mode,
        "render_batch_size": render_batch_size,
    }
    confidence_path = output / "tracking_confidence.json"
    if args.adaptive_reregister:
        confidence_path.write_text(
            json.dumps(confidence_records, indent=2) + "\n", encoding="utf-8"
        )
    else:
        # Avoid leaving diagnostics from a previous re-registration-enabled
        # run in an output directory that is being overwritten.
        confidence_path.unlink(missing_ok=True)
    result = output / "tracking_report.json"
    result.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(result)


class FoundationPoseTracker:
    """Worker-facing class for egocentric FoundationPose tracking."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments

    @classmethod
    def from_worker_job(cls) -> "FoundationPoseTracker":
        return cls(parse_args())

    def run(self) -> None:
        run_bidirectional_tracking(self.arguments)


def main() -> None:
    FoundationPoseTracker.from_worker_job().run()


if __name__ == "__main__":
    main()
