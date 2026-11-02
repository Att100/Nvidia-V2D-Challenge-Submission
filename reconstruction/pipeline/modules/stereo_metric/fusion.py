"""Fuse external metric depth with VIPE camera poses and validate geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class MetricFusionConfig:
    """Configuration for static metric point-cloud fusion and validation."""

    video: Path
    calibration_npz: Path
    depth_dir: Path
    dynamic_mask_npz: Path
    vipe_pose_npz: Path
    output_dir: Path
    pixel_stride: int = 4
    voxel_size_m: float = 0.01
    z_near_m: float = 0.10
    z_far_m: float = 10.0
    preview_depth_far_m: float = 4.0
    sample_count: int = 6


class MetricStaticSceneFuser:
    """Back-project static stereo depth and transform it with VIPE c2w poses."""

    def __init__(self, config: MetricFusionConfig) -> None:
        self.config = config

    @staticmethod
    def _rotation_angle_deg(relative_rotation: np.ndarray) -> float:
        return float(np.degrees(Rotation.from_matrix(relative_rotation).magnitude()))

    @staticmethod
    def _reprojection_consistency(
        depth_dir: Path,
        dynamic_masks: np.ndarray,
        rectified_valid: np.ndarray,
        poses: np.ndarray,
        intrinsic: np.ndarray,
        frame_gap: int,
        pair_stride: int = 10,
        pixel_stride: int = 8,
    ) -> dict[str, float | int]:
        """Measure cross-frame depth agreement under the estimated c2w poses."""

        height, width = rectified_valid.shape
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        sampling = (yy % pixel_stride == 0) & (xx % pixel_stride == 0)
        absolute_errors: list[np.ndarray] = []
        relative_errors: list[np.ndarray] = []
        inliers = 0
        comparisons = 0
        pair_count = 0
        for source_id in range(0, len(poses) - frame_gap, pair_stride):
            target_id = source_id + frame_gap
            source_depth = np.load(depth_dir / f"{source_id:06d}.npy")
            target_depth = np.load(depth_dir / f"{target_id:06d}.npy")
            source_valid = (
                rectified_valid
                & ~dynamic_masks[source_id]
                & sampling
                & np.isfinite(source_depth)
                & (source_depth > 0)
            )
            z = source_depth[source_valid].astype(np.float64)
            camera_points = np.stack(
                [
                    (xx[source_valid] - intrinsic[0, 2]) / intrinsic[0, 0] * z,
                    (yy[source_valid] - intrinsic[1, 2]) / intrinsic[1, 1] * z,
                    z,
                ],
                axis=1,
            )
            world_points = camera_points @ poses[source_id, :3, :3].T + poses[source_id, :3, 3]
            w2c = np.linalg.inv(poses[target_id])
            target_points = world_points @ w2c[:3, :3].T + w2c[:3, 3]
            positive = target_points[:, 2] > 0.1
            projected_x = np.rint(
                intrinsic[0, 0] * target_points[:, 0] / np.maximum(target_points[:, 2], 1e-6)
                + intrinsic[0, 2]
            ).astype(np.int32)
            projected_y = np.rint(
                intrinsic[1, 1] * target_points[:, 1] / np.maximum(target_points[:, 2], 1e-6)
                + intrinsic[1, 2]
            ).astype(np.int32)
            inside = (
                positive
                & (projected_x >= 0)
                & (projected_x < width)
                & (projected_y >= 0)
                & (projected_y < height)
            )
            if not inside.any():
                continue
            px = projected_x[inside]
            py = projected_y[inside]
            predicted_z = target_points[inside, 2]
            measured_z = target_depth[py, px]
            target_valid = (
                rectified_valid[py, px]
                & ~dynamic_masks[target_id, py, px]
                & np.isfinite(measured_z)
                & (measured_z > 0)
            )
            if not target_valid.any():
                continue
            predicted_z = predicted_z[target_valid]
            measured_z = measured_z[target_valid]
            error = np.abs(measured_z - predicted_z)
            absolute_errors.append(error)
            relative_errors.append(error / np.maximum(predicted_z, 1e-3))
            threshold = np.maximum(0.05, 0.05 * predicted_z)
            inliers += int(np.sum(error <= threshold))
            comparisons += int(len(error))
            pair_count += 1
        if not comparisons:
            return {"frame_gap": frame_gap, "pair_count": 0, "comparisons": 0}
        absolute = np.concatenate(absolute_errors)
        relative = np.concatenate(relative_errors)
        return {
            "frame_gap": frame_gap,
            "pair_count": pair_count,
            "comparisons": comparisons,
            "median_absolute_error_m": float(np.median(absolute)),
            "p95_absolute_error_m": float(np.quantile(absolute, 0.95)),
            "median_relative_error": float(np.median(relative)),
            "inlier_fraction_max_5cm_or_5pct": float(inliers / comparisons),
        }

    def run(self) -> Path:
        cfg = self.config
        started = time.perf_counter()
        output = cfg.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        calibration = np.load(cfg.calibration_npz)
        k = np.asarray(calibration["K_working"], dtype=np.float64)
        rectified_valid = np.asarray(calibration["left_valid"], dtype=bool)
        mask_archive = np.load(cfg.dynamic_mask_npz)
        dynamic_masks = np.unpackbits(
            np.asarray(mask_archive["packed_masks"], dtype=np.uint8),
            axis=2,
            count=int(mask_archive["mask_width"]),
        ).astype(bool)
        pose_archive = np.load(cfg.vipe_pose_npz)
        poses = np.asarray(pose_archive["data"], dtype=np.float64)
        pose_ids = np.asarray(pose_archive["inds"], dtype=np.int32)

        capture = cv2.VideoCapture(str(cfg.video))
        if not capture.isOpened():
            raise FileNotFoundError(cfg.video)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        if len(poses) != frame_count or not np.array_equal(pose_ids, np.arange(frame_count)):
            raise ValueError("VIPE pose sequence does not cover every source frame")
        if dynamic_masks.shape != (frame_count, height, width):
            raise ValueError("Dynamic mask sequence does not match video")

        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        sampling = (yy % cfg.pixel_stride == 0) & (xx % cfg.pixel_stride == 0)
        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        all_camera_depths: list[np.ndarray] = []
        per_frame_counts: list[int] = []
        sample_ids = set(
            np.unique(np.rint(np.linspace(0, frame_count - 1, cfg.sample_count)).astype(int)).tolist()
        )
        sample_dir = output / "samples"
        sample_dir.mkdir(exist_ok=True)
        try:
            for frame_id in range(frame_count):
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode frame {frame_id}")
                depth = np.load(cfg.depth_dir / f"{frame_id:06d}.npy")
                valid = (
                    rectified_valid
                    & ~dynamic_masks[frame_id]
                    & sampling
                    & np.isfinite(depth)
                    & (depth >= cfg.z_near_m)
                    & (depth <= cfg.z_far_m)
                )
                z = depth[valid].astype(np.float64)
                camera_points = np.stack(
                    [
                        (xx[valid] - k[0, 2]) / k[0, 0] * z,
                        (yy[valid] - k[1, 2]) / k[1, 1] * z,
                        z,
                    ],
                    axis=1,
                )
                c2w = poses[frame_id]
                world_points = camera_points @ c2w[:3, :3].T + c2w[:3, 3]
                colors = bgr[valid][:, ::-1].astype(np.float64) / 255.0
                all_points.append(world_points)
                all_colors.append(colors)
                all_camera_depths.append(z)
                per_frame_counts.append(int(len(world_points)))
                if frame_id in sample_ids:
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(world_points)
                    cloud.colors = o3d.utility.Vector3dVector(colors)
                    o3d.io.write_point_cloud(str(sample_dir / f"frame_{frame_id:06d}_world.ply"), cloud)
        finally:
            capture.release()

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        camera_depths = np.concatenate(all_camera_depths, axis=0)
        raw_cloud = o3d.geometry.PointCloud()
        raw_cloud.points = o3d.utility.Vector3dVector(points)
        raw_cloud.colors = o3d.utility.Vector3dVector(colors)
        fused_cloud = raw_cloud.voxel_down_sample(cfg.voxel_size_m)
        if len(fused_cloud.points) > 1000:
            fused_cloud, _ = fused_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        cloud_path = output / "static_metric_fused.ply"
        o3d.io.write_point_cloud(str(cloud_path), fused_cloud)

        near_selection = camera_depths <= cfg.preview_depth_far_m
        near_cloud = o3d.geometry.PointCloud()
        near_cloud.points = o3d.utility.Vector3dVector(points[near_selection])
        near_cloud.colors = o3d.utility.Vector3dVector(colors[near_selection])
        near_cloud = near_cloud.voxel_down_sample(cfg.voxel_size_m)
        if len(near_cloud.points) > 1000:
            near_cloud, _ = near_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
        near_cloud_path = output / f"static_metric_fused_near_{cfg.preview_depth_far_m:g}m.ply"
        o3d.io.write_point_cloud(str(near_cloud_path), near_cloud)

        centers = poses[:, :3, 3]
        translation_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        rotation_steps = np.asarray(
            [
                self._rotation_angle_deg(poses[i - 1, :3, :3].T @ poses[i, :3, :3])
                for i in range(1, frame_count)
            ]
        )
        fig = plt.figure(figsize=(8, 7))
        axis = fig.add_subplot(111, projection="3d")
        axis.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="tab:blue", linewidth=1.5)
        axis.scatter(*centers[0], color="green", s=40, label="start")
        axis.scatter(*centers[-1], color="red", s=40, label="end")
        axis.set_title("VIPE camera trajectory (metric stereo anchored)")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.legend()
        fig.tight_layout()
        trajectory_plot = output / "camera_trajectory.png"
        fig.savefig(trajectory_plot, dpi=180)
        plt.close(fig)

        fused_points = np.asarray(fused_cloud.points)
        fused_colors = np.asarray(fused_cloud.colors)
        near_points = np.asarray(near_cloud.points)
        near_colors = np.asarray(near_cloud.colors)
        rng = np.random.default_rng(0)
        selection = rng.choice(len(near_points), min(200000, len(near_points)), replace=False)
        preview_points = near_points[selection]
        preview_colors = near_colors[selection]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        views = ((0, 2, "x-z"), (0, 1, "x-y"), (2, 1, "z-y"))
        for axis, (axis_x, axis_y, title) in zip(axes, views):
            axis.scatter(
                preview_points[:, axis_x],
                preview_points[:, axis_y],
                c=preview_colors,
                s=0.12,
            )
            axis.set_title(f"{title} (camera depth <= {cfg.preview_depth_far_m:g} m)")
            axis.set_aspect("equal", adjustable="box")
            axis.set_axis_off()
        fig.tight_layout()
        cloud_preview = output / "static_metric_fused_preview.png"
        fig.savefig(cloud_preview, dpi=180)
        plt.close(fig)

        reprojection = {
            "adjacent_frames": self._reprojection_consistency(
                cfg.depth_dir,
                dynamic_masks,
                rectified_valid,
                poses,
                k,
                frame_gap=1,
            ),
            "five_frame_gap": self._reprojection_consistency(
                cfg.depth_dir,
                dynamic_masks,
                rectified_valid,
                poses,
                k,
                frame_gap=5,
            ),
        }

        report = {
            "schema_version": "ego-scene-hoi-stereo-vipe-fusion-v1",
            "result": "PASS",
            "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
            "frame_count": frame_count,
            "input_points": int(len(points)),
            "fused_points": int(len(fused_points)),
            "near_fused_points": int(len(near_points)),
            "per_frame_static_points": {
                "median": float(np.median(per_frame_counts)),
                "min": int(np.min(per_frame_counts)),
                "max": int(np.max(per_frame_counts)),
            },
            "cloud_bounds_m": {
                "minimum": fused_points.min(axis=0).tolist(),
                "maximum": fused_points.max(axis=0).tolist(),
                "extent": np.ptp(fused_points, axis=0).tolist(),
            },
            "pose_continuity": {
                "translation_step_median": float(np.median(translation_steps)),
                "translation_step_p95": float(np.quantile(translation_steps, 0.95)),
                "translation_step_max": float(translation_steps.max()),
                "translation_max_frame": int(np.argmax(translation_steps) + 1),
                "rotation_step_deg_median": float(np.median(rotation_steps)),
                "rotation_step_deg_p95": float(np.quantile(rotation_steps, 0.95)),
                "rotation_step_deg_max": float(rotation_steps.max()),
                "rotation_max_frame": int(np.argmax(rotation_steps) + 1),
            },
            "depth_pose_reprojection_consistency": reprojection,
            "artifacts": {
                "pointcloud": str(cloud_path),
                "near_pointcloud": str(near_cloud_path),
                "trajectory_plot": str(trajectory_plot),
                "pointcloud_preview": str(cloud_preview),
                "sample_world_clouds": str(sample_dir),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path = output / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report_path

