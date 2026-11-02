"""Formal pipeline stages for native RGB plus synchronized grayscale stereo."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import numpy as np

from pipeline.config import get_config, repository_root, resolve_path
from pipeline.utils import PipelineRequest, StageResult, StageStatus, all_artifacts_exist

from .foundation_pose.stereo_sequence import (
    StereoFoundationPoseConfig,
    StereoFoundationPoseExperiment,
)
from .stereo_metric.calibration import FixedStereoCalibrator, StereoCalibrationConfig
from .stereo_metric.public_calibration import (
    PublicStereoCalibrator,
    PublicStereoCalibrationConfig,
)
from .stereo_metric.fusion import MetricFusionConfig, MetricStaticSceneFuser
from .stereo_metric.inference import (
    DynamicMaskConfig,
    FoundationStereoSequenceConfig,
    FoundationStereoSequenceEstimator,
    SAM3DynamicMaskBuilder,
)
from .stereo_metric.vipe_pose import ViPEMetricPoseEstimator, ViPEPoseConfig


class _DirectStage:
    """Small resumable adapter for an already class-based in-process module."""

    name: str

    def artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        raise NotImplementedError

    def execute(self, request: PipelineRequest) -> None:
        raise NotImplementedError

    def run(self, request: PipelineRequest) -> StageResult:
        artifacts = self.artifacts(request)
        if request.resume and not request.force and all_artifacts_exist(artifacts):
            return StageResult(self.name, StageStatus.REUSED, artifacts)
        if request.dry_run:
            return StageResult(self.name, StageStatus.SKIPPED, artifacts)
        started = time.perf_counter()
        self.execute(request)
        if not all_artifacts_exist(artifacts):
            raise RuntimeError(f"{self.name} did not produce {artifacts}")
        metrics = {}
        report = next((path for path in artifacts if path.name == "report.json"), None)
        if report is not None:
            payload = json.loads(report.read_text(encoding="utf-8"))
            metrics = {
                key: payload[key]
                for key in ("result", "status", "elapsed_seconds")
                if key in payload
            }
        return StageResult(
            self.name,
            StageStatus.COMPLETED,
            artifacts,
            elapsed_seconds=time.perf_counter() - started,
            metrics=metrics,
        )


def _calibration_dir(request: PipelineRequest) -> Path:
    return request.run_root / "calibration"


class StereoCalibrationModule(_DirectStage):
    """Estimate shared intrinsics/distortion and rectify the complete stereo pair."""

    name = "stereo_calibration"

    def artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        root = _calibration_dir(request)
        return (
            root / "calibration.npz",
            root / "left_rectified_working.mp4",
            root / "right_rectified_working.mp4",
            root / "report.json",
        )

    def execute(self, request: PipelineRequest) -> None:
        calibration_source = str(
            get_config(request.config, "pipeline.stereo.calibration_source", "anycalib")
        )
        if calibration_source == "public":
            calibration_path = resolve_path(
                request.config, "pipeline.stereo.public_calibration_path"
            )
            if calibration_path is None:
                raise ValueError(
                    "pipeline.stereo.public_calibration_path is required for public calibration"
                )
            PublicStereoCalibrator(
                PublicStereoCalibrationConfig(
                    # For the released, upright ego videos the public metadata
                    # explicitly defines C as left and B as right.
                    left_video=request.run_root / "input/stereo_left_raw.mp4",
                    right_video=request.run_root / "input/stereo_right_raw.mp4",
                    calibration_json=calibration_path,
                    output_dir=_calibration_dir(request),
                    work_scale=float(
                        get_config(request.config, "pipeline.stereo.work_scale", 0.5)
                    ),
                    sample_count=int(
                        get_config(request.config, "pipeline.stereo.sample_count", 12)
                    ),
                    rectification_alpha=float(
                        get_config(
                            request.config,
                            "pipeline.stereo.public_rectification_alpha",
                            0.0,
                        )
                    ),
                )
            ).run()
            return
        if calibration_source != "anycalib":
            raise ValueError(f"Unsupported stereo calibration source: {calibration_source}")
        FixedStereoCalibrator(
            StereoCalibrationConfig(
                camera_b_video=request.run_root / "input/stereo_left_raw.mp4",
                camera_c_video=request.run_root / "input/stereo_right_raw.mp4",
                output_dir=_calibration_dir(request),
                baseline_m=float(get_config(request.config, "dataset.stereo.baseline_m", 0.075)),
                work_scale=float(get_config(request.config, "pipeline.stereo.work_scale", 0.5)),
                sample_count=int(get_config(request.config, "pipeline.stereo.sample_count", 12)),
                selection_mode=str(
                    get_config(
                        request.config,
                        "pipeline.stereo.calibration_selection_mode",
                        "multi_frame_candidates",
                    )
                ),
                max_rectified_center_offset_fraction=float(
                    get_config(
                        request.config,
                        "pipeline.stereo.calibration_max_center_offset_fraction",
                        0.15,
                    )
                ),
                min_rectified_valid_fraction=float(
                    get_config(
                        request.config,
                        "pipeline.stereo.calibration_min_valid_fraction",
                        0.70,
                    )
                ),
                candidate_consensus_weight=float(
                    get_config(
                        request.config,
                        "pipeline.stereo.calibration_candidate_consensus_weight",
                        0.10,
                    )
                ),
            )
        ).run()


class PublicObjectMeshModule(_DirectStage):
    """Expose released metric meshes through the standard SAM3D manifest."""

    name = "public_object_meshes"

    def artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root / "objects/sam3d_objects/reconstruction_manifest.json",
        )

    def execute(self, request: PipelineRequest) -> None:
        configured = get_config(
            request.config, "pipeline.stereo.public_object_meshes", {}
        )
        if not isinstance(configured, dict) or not configured:
            raise ValueError(
                "pipeline.stereo.public_object_meshes must map object IDs to metric meshes"
            )
        reasoning_path = request.run_root / "semantics/reasoning/contact_events.json"
        reasoning = json.loads(reasoning_path.read_text(encoding="utf-8"))
        known = {
            str(record["global_object_id"]): str(record.get("name", "object"))
            for record in reasoning.get("objects", [])
        }
        if set(configured) != set(known):
            raise ValueError(
                "Released mesh mapping must cover exactly the discovered interaction objects; "
                f"configured={sorted(configured)}, discovered={sorted(known)}"
            )
        output = request.run_root / "objects/sam3d_objects"
        jobs: list[dict[str, object]] = []
        for object_id, value in sorted(configured.items()):
            source = Path(str(value)).expanduser()
            if not source.is_absolute():
                source = repository_root(request.config) / source
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            target_dir = output / object_id / "primary"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source.name
            shutil.copy2(source, target)
            jobs.append(
                {
                    "global_object_id": object_id,
                    "object_name": known[object_id],
                    "role": "primary",
                    "status": "pass",
                    "output_glb_path": str(target.resolve()),
                    "source_glb_path": str(source),
                    "metric_mesh": True,
                }
            )
        manifest = self.artifacts(request)[0]
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "ego-scene-hoi-public-object-meshes-v1",
                    "jobs": jobs,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


class StereoDynamicMaskModule(_DirectStage):
    """Compose object tracks, second-pass hand masks, and body exclusions."""

    name = "stereo_dynamic_mask"

    def artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        root = request.run_root / "dynamic_mask"
        return (
            root / "dynamic_mask.npz",
            root / "body_masks.npz",
            root / "object_masks.npz",
            root / "report.json",
        )

    def execute(self, request: PipelineRequest) -> None:
        SAM3DynamicMaskBuilder(
            DynamicMaskConfig(
                video=_calibration_dir(request) / "left_rectified_working.mp4",
                reasoning_json=request.run_root / "semantics/reasoning/contact_events.json",
                output_dir=request.run_root / "dynamic_mask",
                repository_root=repository_root(request.config),
                object_tracks_dir=request.run_root / "tracks/sam3",
                hand_masks_npz=(
                    request.run_root / "tracks/sam3_hands_left/hand_masks.npz"
                ),
                checkpoint=resolve_path(request.config, "models.sam3.checkpoint"),
                confidence_threshold=float(
                    get_config(request.config, "models.sam3.confidence_threshold", 0.3)
                ),
                dilation_pixels=int(
                    get_config(request.config, "pipeline.stereo.mask_dilation_pixels", 4)
                ),
            )
        ).run()


class FoundationStereoModule(_DirectStage):
    """Infer metric rectified-left depth for every synchronized frame."""

    name = "foundation_stereo_sequence"

    def artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (request.run_root / "depth/report.json",)

    def execute(self, request: PipelineRequest) -> None:
        checkpoint = resolve_path(request.config, "models.foundation_stereo.checkpoint")
        if checkpoint is None:
            raise ValueError("models.foundation_stereo.checkpoint is required")
        FoundationStereoSequenceEstimator(
            FoundationStereoSequenceConfig(
                left_video=_calibration_dir(request) / "left_rectified_working.mp4",
                right_video=_calibration_dir(request) / "right_rectified_working.mp4",
                calibration_npz=_calibration_dir(request) / "calibration.npz",
                repository=repository_root(request.config) / "thirdparty/FoundationStereo",
                checkpoint=checkpoint,
                output_dir=request.run_root / "depth",
                valid_iters=int(
                    get_config(request.config, "pipeline.stereo.foundation_stereo_iters", 32)
                ),
                sample_count=int(get_config(request.config, "pipeline.stereo.sample_count", 12)),
            )
        ).run()


class ViPEMetricCameraModule(_DirectStage):
    """Estimate metric c2w poses using fixed intrinsics, stereo depth, and dynamic masks."""

    name = "vipe_metric_camera"

    def artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root / "vipe/report.json",
            request.run_root / "vipe/pose/rectified_left.npz",
        )

    def execute(self, request: PipelineRequest) -> None:
        ViPEMetricPoseEstimator(
            ViPEPoseConfig(
                video=_calibration_dir(request) / "left_rectified_working.mp4",
                calibration_npz=_calibration_dir(request) / "calibration.npz",
                depth_dir=request.run_root / "depth/frames",
                dynamic_mask_npz=request.run_root / "dynamic_mask/dynamic_mask.npz",
                vipe_repository=repository_root(request.config) / "thirdparty/vipe",
                output_dir=request.run_root / "vipe",
                stream_name="rectified_left",
            )
        ).run()


class StereoStaticFusionModule(_DirectStage):
    """Fuse dynamic-masked metric depth in the VIPE world coordinate system."""

    name = "stereo_static_fusion"

    def artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root / "fusion/static_metric_fused.ply",
            request.run_root / "fusion/report.json",
        )

    def execute(self, request: PipelineRequest) -> None:
        MetricStaticSceneFuser(
            MetricFusionConfig(
                video=_calibration_dir(request) / "left_rectified_working.mp4",
                calibration_npz=_calibration_dir(request) / "calibration.npz",
                depth_dir=request.run_root / "depth/frames",
                dynamic_mask_npz=request.run_root / "dynamic_mask/dynamic_mask.npz",
                vipe_pose_npz=request.run_root / "vipe/pose/rectified_left.npz",
                output_dir=request.run_root / "fusion",
                pixel_stride=int(
                    get_config(request.config, "pipeline.stereo.fusion_pixel_stride", 4)
                ),
                voxel_size_m=float(
                    get_config(request.config, "pipeline.stereo.fusion_voxel_size_m", 0.01)
                ),
                sample_count=int(get_config(request.config, "pipeline.stereo.sample_count", 12)),
            )
        ).run()


class StereoFoundationPoseModule(_DirectStage):
    """Fit configured mesh scale and track 6-DoF poses in rectified-left RGB-D."""

    name = "stereo_foundation_pose"

    def artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (request.run_root / "foundationpose_left_adaptive/manifest.json",)

    def execute(self, request: PipelineRequest) -> None:
        StereoFoundationPoseExperiment(
            StereoFoundationPoseConfig(
                stereo_root=request.run_root,
                source_run=request.run_root,
                output_dir=request.run_root / "foundationpose_left_adaptive",
                reasoning_path=request.run_root / "semantics/reasoning/contact_events.json",
                object_tracks_dir=request.run_root / "tracks/sam3/refined_masks",
                scale_mode=str(
                    get_config(
                        request.config,
                        "pipeline.object_scale.mode",
                        "isotropic",
                    )
                ),
                max_anisotropy=float(
                    get_config(
                        request.config,
                        "pipeline.object_scale.max_anisotropy",
                        2.0,
                    )
                ),
                adaptive_reregister=bool(
                    get_config(
                        request.config,
                        "pipeline.stereo.foundation_pose_adaptive_reregister",
                        True,
                    )
                ),
                appearance_mode=str(
                    get_config(
                        request.config,
                        "pipeline.stereo.foundation_pose_appearance_mode",
                        "grayscale",
                    )
                ),
                maximum_reregister_rotation_degrees=float(
                    get_config(
                        request.config,
                        "pipeline.stereo.foundation_pose_maximum_reregister_rotation_degrees",
                        25.0,
                    )
                ),
                maximum_reregister_translation_meters=float(
                    get_config(
                        request.config,
                        "pipeline.stereo.foundation_pose_maximum_reregister_translation_meters",
                        0.03,
                    )
                ),
                maximum_reregister_translation_fraction=float(
                    get_config(
                        request.config,
                        "pipeline.stereo.foundation_pose_maximum_reregister_translation_fraction",
                        0.15,
                    )
                ),
                registration_only=bool(
                    get_config(
                        request.config,
                        "pipeline.stereo.foundation_pose_registration_only",
                        False,
                    )
                ),
                write_overlay=bool(
                    get_config(
                        request.config,
                        "pipeline.stereo.foundation_pose_write_overlay",
                        True,
                    )
                ),
                render_batch_size=int(
                    get_config(
                        request.config,
                        "pipeline.stereo.foundation_pose_render_batch_size",
                        8,
                    )
                ),
            )
        ).run()


class StereoOptimizationAdapterModule(_DirectStage):
    """Expose stereo object/camera results through optimizer-neutral paths."""

    name = "stereo_optimization_adapter"

    def artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root / "4d/cameras.npz",
            request.run_root / "foundationpose_left_adaptive/optimizer_adapter.json",
        )

    def execute(self, request: PipelineRequest) -> None:
        root = request.run_root
        with np.load(root / "vipe/pose/rectified_left.npz") as archive:
            values = np.asarray(archive["data"], dtype=np.float32)
            indices = np.asarray(
                archive["inds"] if "inds" in archive else np.arange(len(values)),
                dtype=np.int32,
            )
        with np.load(root / "input/cameras.npz") as archive:
            source_indices = np.asarray(archive["source_indices"], dtype=np.int32)
            timestamps_ns = np.asarray(archive["timestamps_ns"], dtype=np.int64)
        pose_by_frame = {int(index): pose for index, pose in zip(indices, values)}
        camera_to_world = np.stack(
            [pose_by_frame.get(int(index), np.eye(4, dtype=np.float32)) for index in source_indices]
        ).astype(np.float32)
        with np.load(root / "calibration/calibration.npz") as archive:
            intrinsic = np.asarray(archive["K_working"], dtype=np.float32)
        camera_dir = root / "4d"
        camera_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            camera_dir / "cameras.npz",
            camera_poses=camera_to_world,
            camera_to_world=camera_to_world,
            intrinsics=np.repeat(intrinsic[None], len(source_indices), axis=0),
            source_indices=source_indices,
            timestamps_seconds=timestamps_ns.astype(np.float64) * 1.0e-9,
        )

        # Prefer the regular pipeline output, but accept the explicitly audited
        # no-re-registration batch result.  This lets downstream hand/contact
        # optimization consume a verified pose-only experiment without silently
        # rerunning FoundationPose with a different registration policy.
        manifest_candidates = (
            root / "foundationpose_left_adaptive/manifest.json",
            root / "foundationpose_standard_no_reregister/manifest.json",
        )
        manifest_path = next(
            (path for path in manifest_candidates if path.is_file()),
            manifest_candidates[0],
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "No stereo FoundationPose manifest found; checked: "
                + ", ".join(str(path) for path in manifest_candidates)
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = []
        for record in manifest.get("objects", []):
            object_id = str(record["object_id"])
            source_pose = Path(record["poses"])
            with np.load(source_pose) as archive:
                object_to_world = np.asarray(archive["ob_in_world"], dtype=np.float32)
                ob_in_cam = np.asarray(archive["ob_in_cam"], dtype=np.float32)
                pose_source_indices = np.asarray(archive["source_indices"], dtype=np.int32)
            tracklet = root / "tracks/sam3/refined_masks" / f"{object_id}.npz"
            with np.load(tracklet) as archive:
                width = int(np.asarray(archive["mask_width"]).item())
                masks = np.unpackbits(
                    np.asarray(archive["packed_masks"], dtype=np.uint8), axis=2
                )[..., :width]
            observed = masks.any(axis=(1, 2)) & np.isfinite(object_to_world).all(axis=(1, 2))
            object_dir = root / "objects" / object_id
            pose_dir = object_dir / "pose_track"
            pose_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pose_dir / "poses.npz",
                object_to_world=object_to_world,
                object_to_camera=ob_in_cam,
                observed=observed,
                source_indices=pose_source_indices,
            )
            source_mesh = Path(record["metric_mesh"])
            scale_dir = object_dir / "scale_primary"
            scale_dir.mkdir(parents=True, exist_ok=True)
            target_mesh = scale_dir / source_mesh.name
            # The upstream mesh may have been regenerated with corrected
            # geometry/material handling while keeping the same filename.
            # Always refresh this adapter copy so resume cannot retain a stale
            # simplified or textureless mesh.
            shutil.copy2(source_mesh, target_mesh)
            records.append(
                {
                    "object_id": object_id,
                    "source_poses": str(source_pose),
                    "optimizer_poses": str(pose_dir / "poses.npz"),
                    "optimizer_mesh": str(target_mesh),
                }
            )
        adapter = self.artifacts(request)[1]
        adapter.parent.mkdir(parents=True, exist_ok=True)
        adapter.write_text(
            json.dumps(
                {
                    "schema_version": "ego-scene-hoi-stereo-optimizer-adapter-v1",
                    "camera_coordinate_system": "rectified-left VIPE world",
                    "source_manifest": str(manifest_path),
                    "objects": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
