"""Pipeline v2 orchestration with resumable class-based stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pipeline.config import get_config
from pipeline.modules import (
    CodexCanonicalBBoxTransferModule,
    Flux2CompletionModule,
    HACOContactPredictionModule,
    InputPreparationModule,
    ObjectReferenceModule,
    ObjectMaskRefinementModule,
    ObjectMaskRefinementConfig,
    CodexHOIReasoningModule,
    SAM3DObjectsModule,
    SAM3CanonicalLeftHandTracker,
    SAM3TrackedBBoxLeftHandSegmenter,
    SAM3ObjectTracker,
    SAM3RGBCanonicalSegmenter,
    SAM3SupportTracker,
    SAM3RGBSupportCanonicalSegmenter,
    SupportObjectReferenceModule,
    SupportFlux2CompletionModule,
    SupportSAM3DObjectsModule,
    StaticSupportPoseModule,
    FoundationStereoModule,
    PublicObjectMeshModule,
    StereoCalibrationModule,
    StereoDynamicMaskModule,
    StereoFoundationPoseModule,
    StereoOptimizationAdapterModule,
    StereoStaticFusionModule,
    ViPEMetricCameraModule,
    UmeTrackInferenceModule,
    UmeTrackMANOFittingModule,
)
from pipeline.optim import (
    HACOContactOptimizationStage,
    HACOJointContactOptimizationStage,
    StereoHandSurfaceConfig,
    UmeTrackHandSurfaceAlignmentOptimizer,
    UmeTrackJointProjectionConfig,
    UmeTrackJointProjectionOptimizer,
    JointHandObjectConfig,
    ObjectSurfaceAlignmentConfig,
    StereoObjectSurfaceAlignmentOptimizer,
)
from pipeline.utils import PipelineRequest, StageResult, write_json


class PipelineStage(Protocol):
    """Structural interface implemented by model and optimization classes."""

    name: str

    def run(self, request: PipelineRequest) -> StageResult:
        """Execute or resume one stage."""


@dataclass(frozen=True)
class PipelineRunResult:
    """End-to-end stage results and the generated execution report."""

    stages: tuple[StageResult, ...]
    report_path: Path


def _joint_haco_config(
    config: Mapping[str, Any],
    *,
    contact_iterations: int,
    device: str,
) -> JointHandObjectConfig:
    """Build the shared mono/stereo joint-HACO configuration."""

    return JointHandObjectConfig(
        iterations=int(
            get_config(config, "pipeline.haco_joint.iterations", contact_iterations)
        ),
        device=device,
        silhouette_weight=2.0,
        hand_learning_rate=float(
            get_config(config, "pipeline.haco_joint.hand_learning_rate", 0.001)
        ),
        articulation_prior_weight=float(
            get_config(config, "pipeline.haco_joint.articulation_prior_weight", 2.0)
        ),
        articulation_velocity_weight=float(
            get_config(config, "pipeline.haco_joint.articulation_velocity_weight", 0.5)
        ),
        articulation_acceleration_weight=float(
            get_config(config, "pipeline.haco_joint.articulation_acceleration_weight", 1.0)
        ),
        hand_reprojection_weight=float(
            get_config(config, "pipeline.haco_joint.hand_reprojection_weight", 2.0)
        ),
        optimize_hand_se3=bool(
            get_config(config, "pipeline.haco_joint.optimize_hand_se3", True)
        ),
        hand_se3_learning_rate=float(
            get_config(config, "pipeline.haco_joint.hand_se3_learning_rate", 0.001)
        ),
        object_object_penetration_weight=float(
            get_config(
                config,
                "pipeline.haco_joint.object_object_penetration_weight",
                4.0,
            )
        ),
        object_object_penetration_sigma=float(
            get_config(
                config,
                "pipeline.haco_joint.object_object_penetration_sigma",
                0.003,
            )
        ),
        object_object_penetration_tolerance=float(
            get_config(
                config,
                "pipeline.haco_joint.object_object_penetration_tolerance",
                0.001,
            )
        ),
        object_object_surface_samples=int(
            get_config(
                config,
                "pipeline.haco_joint.object_object_surface_samples",
                768,
            )
        ),
        maximum_articulation_radians=float(
            np.deg2rad(
                get_config(
                    config,
                    "pipeline.haco_joint.maximum_articulation_degrees",
                    12.0,
                )
            )
        ),
        silhouette_guard_median_tolerance=float(
            get_config(
                config,
                "pipeline.haco_joint.silhouette_guard_median_tolerance",
                0.05,
            )
        ),
        silhouette_guard_p90_tolerance=float(
            get_config(
                config,
                "pipeline.haco_joint.silhouette_guard_p90_tolerance",
                0.15,
            )
        ),
        silhouette_guard_steps=int(
            get_config(config, "pipeline.haco_joint.silhouette_guard_steps", 20)
        ),
    )


class EgoSceneHOIPipeline:
    """Run the Track-3 RGB plus grayscale-stereo pipeline in dependency order."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        contact_iterations: int = 800,
        device: str = "cuda",
    ) -> None:
        self.config = dict(config or {})
        if (
            str(get_config(self.config, "dataset.input_mode", ""))
            != "rgb_plus_stereo_grayscale"
        ):
            raise ValueError(
                "This cleaned bundle supports only dataset=rgb_stereo_gray"
            )
        joint_config = _joint_haco_config(
            self.config,
            contact_iterations=contact_iterations,
            device=device,
        )
        if (
            str(get_config(self.config, "dataset.input_mode", ""))
            == "rgb_plus_stereo_grayscale"
        ):
            joint_projection = UmeTrackJointProjectionConfig(
                iterations=int(
                    get_config(self.config, "pipeline.stereo.joint_projection.iterations", 300)
                ),
                learning_rate=float(
                    get_config(self.config, "pipeline.stereo.joint_projection.learning_rate", 0.02)
                ),
                rotation_prior_weight=float(
                    get_config(
                        self.config,
                        "pipeline.stereo.joint_projection.rotation_prior_weight",
                        0.30,
                    )
                ),
                translation_prior_weight=float(
                    get_config(
                        self.config,
                        "pipeline.stereo.joint_projection.translation_prior_weight",
                        0.01,
                    )
                ),
                translation_acceleration_weight=float(
                    get_config(
                        self.config,
                        "pipeline.stereo.joint_projection.translation_acceleration_weight",
                        0.10,
                    )
                ),
                rotation_acceleration_weight=float(
                    get_config(
                        self.config,
                        "pipeline.stereo.joint_projection.rotation_acceleration_weight",
                        0.10,
                    )
                ),
                device=device,
            )
            canonical_left_hand_masks = Path("tracks/sam3_hands_left/hand_masks.npz")
            stereo_surface = StereoHandSurfaceConfig(
                scale_mode=str(
                    get_config(self.config, "pipeline.stereo.hand_surface.scale_mode", "global")
                ),
                minimum_correspondences=int(
                    get_config(self.config, "pipeline.stereo.hand_surface.minimum_correspondences", 96)
                ),
                maximum_reprojection_increase_pixels=float(
                    get_config(
                        self.config,
                        "pipeline.stereo.hand_surface.maximum_reprojection_increase_pixels",
                        2.0,
                    )
                ),
                maximum_depth_median_increase_m=float(
                    get_config(
                        self.config,
                        "pipeline.stereo.hand_surface.maximum_depth_median_increase_m",
                        0.002,
                    )
                ),
                maximum_depth_p90_increase_m=float(
                    get_config(
                        self.config,
                        "pipeline.stereo.hand_surface.maximum_depth_p90_increase_m",
                        0.005,
                    )
                ),
                minimum_depth_target_support=int(
                    get_config(
                        self.config,
                        "pipeline.stereo.hand_surface.minimum_depth_target_support",
                        5,
                    )
                ),
                hand_masks_path=canonical_left_hand_masks,
                se3_scope=str(
                    get_config(
                        self.config,
                        "pipeline.stereo.hand_surface.se3_scope",
                        "per_frame",
                    )
                ),
            )
            object_surface = ObjectSurfaceAlignmentConfig(
                enabled=bool(
                    get_config(
                        self.config,
                        "pipeline.stereo.object_surface.enabled",
                        True,
                    )
                ),
                minimum_correspondences=int(
                    get_config(
                        self.config,
                        "pipeline.stereo.object_surface.minimum_correspondences",
                        96,
                    )
                ),
                maximum_rotation_degrees=float(
                    get_config(
                        self.config,
                        "pipeline.stereo.object_surface.maximum_rotation_degrees",
                        5.0,
                    )
                ),
                maximum_translation_meters=float(
                    get_config(
                        self.config,
                        "pipeline.stereo.object_surface.maximum_translation_meters",
                        0.02,
                    )
                ),
                prior_weight=float(
                    get_config(
                        self.config,
                        "pipeline.stereo.object_surface.prior_weight",
                        0.20,
                    )
                ),
                velocity_weight=float(
                    get_config(
                        self.config,
                        "pipeline.stereo.object_surface.velocity_weight",
                        8.0,
                    )
                ),
                acceleration_weight=float(
                    get_config(
                        self.config,
                        "pipeline.stereo.object_surface.acceleration_weight",
                        3.0,
                    )
                ),
                write_overlay=bool(
                    get_config(
                        self.config,
                        "pipeline.stereo.object_surface.write_overlay",
                        False,
                    )
                ),
            )
            public_object_meshes = get_config(
                self.config, "pipeline.stereo.public_object_meshes", {}
            )
            object_mesh_stages: list[PipelineStage]
            if isinstance(public_object_meshes, dict) and public_object_meshes:
                object_mesh_stages = [PublicObjectMeshModule()]
            else:
                object_mesh_stages = [
                    ObjectReferenceModule(),
                    Flux2CompletionModule(),
                    SAM3DObjectsModule(),
                ]
            stereo_stages: list[PipelineStage] = [
                InputPreparationModule(),
                StereoCalibrationModule(),
                CodexHOIReasoningModule(),
                SAM3ObjectTracker(),
                SAM3SupportTracker(),
                CodexCanonicalBBoxTransferModule(),
                SAM3CanonicalLeftHandTracker(),
                SAM3TrackedBBoxLeftHandSegmenter(),
                SAM3RGBCanonicalSegmenter(),
                SAM3RGBSupportCanonicalSegmenter(),
                StereoDynamicMaskModule(),
                ObjectMaskRefinementModule(
                    ObjectMaskRefinementConfig(
                        minimum_area=int(
                            get_config(self.config, "pipeline.object_mask_refinement.minimum_area", 64)
                        ),
                        containment_threshold=float(
                            get_config(
                                self.config,
                                "pipeline.object_mask_refinement.containment_threshold",
                                0.50,
                            )
                        ),
                    )
                ),
                FoundationStereoModule(),
                UmeTrackInferenceModule(),
                UmeTrackMANOFittingModule(),
                ViPEMetricCameraModule(),
                StereoStaticFusionModule(),
                *object_mesh_stages,
                SupportObjectReferenceModule(),
                SupportFlux2CompletionModule(),
                SupportSAM3DObjectsModule(),
                StaticSupportPoseModule(),
                StereoFoundationPoseModule(),
                StereoOptimizationAdapterModule(),
                StereoObjectSurfaceAlignmentOptimizer(object_surface),
                UmeTrackJointProjectionOptimizer(joint_projection),
                UmeTrackHandSurfaceAlignmentOptimizer(stereo_surface),
            ]
            if bool(
                get_config(
                    self.config,
                    "pipeline.stereo.haco.enabled",
                    False,
                )
            ):
                stereo_stages.extend(
                    [
                        HACOContactPredictionModule(),
                        HACOContactOptimizationStage(
                            iterations=contact_iterations,
                            device=device,
                            contact_weight=float(
                                get_config(
                                    self.config,
                                    "pipeline.haco.contact_weight",
                                    0.1,
                                )
                            ),
                        ),
                    ]
                )
                if bool(
                    get_config(self.config, "pipeline.haco_joint.enabled", True)
                ):
                    stereo_stages.append(
                        HACOJointContactOptimizationStage(joint_config, enabled=True)
                    )
            self.stages = tuple(stereo_stages)
            return

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Return stable CLI stage identifiers."""

        return tuple(stage.name for stage in self.stages)

    def run(
        self,
        request: PipelineRequest,
        *,
        selected_stages: Sequence[str] | None = None,
    ) -> PipelineRunResult:
        """Execute selected stages and write a machine-readable run report."""

        selected = set(selected_stages or self.stage_names)
        unknown = selected.difference(self.stage_names)
        if unknown:
            raise ValueError(f"Unknown pipeline stages: {sorted(unknown)}")
        results: list[StageResult] = []
        for stage in self.stages:
            if stage.name not in selected:
                continue
            print(f"[pipeline-v2] {stage.name}", flush=True)
            results.append(stage.run(request))
        report_path = request.run_root / "pipeline_v2_run.json"
        write_json(
            report_path,
            {
                "schema_version": "ego-scene-hoi-pipeline-v2",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "sequence_root": str(request.sequence_root),
                "run_root": str(request.run_root),
                "frame_count": request.frame_count,
                "resume": request.resume,
                "force": request.force,
                "dry_run": request.dry_run,
                "stages": [result.to_json() for result in results],
            },
        )
        return PipelineRunResult(tuple(results), report_path)
