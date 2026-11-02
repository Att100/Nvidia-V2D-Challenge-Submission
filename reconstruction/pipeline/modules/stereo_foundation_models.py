"""Class wrappers for the v2 foundation-model stages."""

from __future__ import annotations

import json
import time
from pathlib import Path

from pipeline.config import get_config, python_executable, repository_root, resolve_path
from pipeline.utils import (
    PipelineRequest,
    StageResult,
    StageStatus,
    all_artifacts_exist,
)

from .base import WORKER_DISPATCHER, FoundationModule


def _is_rgb_stereo(request: PipelineRequest) -> bool:
    return (
        str(get_config(request.config, "dataset.input_mode", ""))
        == "rgb_plus_stereo_grayscale"
    )


def _reasoning_video(request: PipelineRequest) -> Path:
    return request.run_root / "calibration/left_rectified_working.mp4"


def _reference_reasoning(request: PipelineRequest) -> Path:
    return request.run_root / "semantics/reasoning_rgb/contact_events_rgb.json"


def _python(request: PipelineRequest, stage: str) -> str:
    return python_executable(request.config, stage)


def _configured_path(request: PipelineRequest, key: str) -> str:
    path = resolve_path(request.config, key)
    if path is None:
        raise ValueError(f"Missing configured path: {key}")
    return str(path)


class CodexHOIReasoningModule(FoundationModule):
    """Run whole-video Codex HOI discovery with strict structured output."""

    name = "codex_hoi_reasoning"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root / "semantics/reasoning/contact_events.json",
            request.run_root / "semantics/reasoning/report.json",
        )

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        return (
            _python(request, self.name),
            "-m",
            WORKER_DISPATCHER,
            "reason",
            "--video",
            str(_reasoning_video(request)),
            "--output-dir",
            str(request.run_root / "semantics/reasoning"),
            "--repository-root",
            str(repository_root(request.config)),
            "--codex-binary",
            str(get_config(request.config, "models.codex.binary", "codex")),
            "--model",
            str(get_config(request.config, "models.codex.model", "gpt-6-astra")),
            "--reasoning-effort",
            str(get_config(request.config, "models.codex.reasoning_effort", "high")),
            "--output-schema",
            str(
                resolve_path(
                    request.config,
                    "pipeline.codex_reasoning.output_schema",
                )
            ),
            "--schema-version",
            str(get_config(request.config, "schemas.contact_events")),
            "--prompt-template",
            str(get_config(request.config, "prompts.hoi_reasoning.template")),
            "--timeout-seconds",
            str(get_config(request.config, "pipeline.codex_reasoning.timeout_seconds", 3600)),
        )


class HACOContactPredictionModule(FoundationModule):
    """Predict dense MANO contacts and associate them with tracked objects."""

    name = "haco_contact_prediction"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root / "hands/haco/contact_predictions.npz",
            request.run_root / "hands/haco/object_contacts/manifest.json",
            request.run_root / "hands/haco/report.json",
        )

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        hand_motion = request.run_root / "hands/umetrack_mano_surface/hand_motion.npz"
        return (
            _python(request, self.name),
            "-m",
            WORKER_DISPATCHER,
            "infer",
            "--run-root",
            str(request.run_root),
            "--haco-root",
            _configured_path(request, "models.haco.root"),
            "--checkpoint",
            _configured_path(request, "models.haco.checkpoint"),
            "--hand-motion",
            str(hand_motion),
            "--backbone",
            str(get_config(request.config, "models.haco.backbone", "vit-b-16")),
            "--image-source",
            str(get_config(request.config, "pipeline.haco.image_source", "rgb")),
            "--device",
            request.device,
            "--batch-size",
            str(get_config(request.config, "pipeline.haco.batch_size", 32)),
            "--threshold",
            str(get_config(request.config, "pipeline.haco.threshold", 0.5)),
            "--temporal-alpha",
            str(get_config(request.config, "pipeline.haco.temporal_alpha", 0.8)),
            "--minimum-component-vertices",
            str(get_config(request.config, "pipeline.haco.minimum_component_vertices", 20)),
            "--bbox-expand-ratio",
            str(get_config(request.config, "pipeline.haco.bbox_expand_ratio", 2.0)),
            "--object-mask-dilation",
            str(get_config(request.config, "pipeline.haco.object_mask_dilation", 24)),
            "--geometry-soft-radius",
            str(get_config(request.config, "pipeline.haco.geometry_soft_radius", 0.03)),
            "--geometry-soft-sigma",
            str(get_config(request.config, "pipeline.haco.geometry_soft_sigma", 0.04)),
            "--geometry-maximum-distance",
            str(get_config(request.config, "pipeline.haco.geometry_maximum_distance", 0.15)),
            "--contact-region",
            str(get_config(request.config, "pipeline.haco.contact_region", "palmar")),
            "--palmar-maximum-normal-y",
            str(
                get_config(
                    request.config,
                    "pipeline.haco.palmar_maximum_normal_y",
                    -0.10,
                )
            ),
        )


class SAM3ObjectTracker(FoundationModule):
    """Track each Codex physical object from its canonical frame and box."""

    name = "sam3_object_tracking"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root / "tracks/sam3/contact_tracks.json",
            request.run_root / "tracks/sam3/global_masks/manifest.json",
        )

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        command = [
            _python(request, self.name),
            "-m",
            WORKER_DISPATCHER,
            "track",
            "--reasoning",
            str(request.run_root / "semantics/reasoning/contact_events.json"),
            "--video",
            str(_reasoning_video(request)),
            "--output-dir",
            str(request.run_root / "tracks/sam3"),
            "--min-mask-area",
            str(get_config(request.config, "pipeline.sam3_object.min_mask_area", 64)),
            "--min-mask-area-fraction",
            str(get_config(request.config, "pipeline.sam3_object.min_mask_area_fraction", 0.0)),
            "--prompt-mode",
            str(get_config(request.config, "pipeline.sam3_object.prompt_mode", "bbox_short_text")),
            "--schema-version",
            str(get_config(request.config, "schemas.sam3_object_tracks")),
        ]
        return tuple(command)

    def build_commands(self, request: PipelineRequest) -> tuple[tuple[str, ...], ...]:
        tracking = self.build_command(request)
        consolidate = (
            _python(request, self.name),
            "-m",
            WORKER_DISPATCHER,
            "consolidate_masks",
            "--tracks",
            str(request.run_root / "tracks/sam3/contact_tracks.json"),
            "--camera-data",
            str(request.run_root / "input/cameras.npz"),
            "--output-dir",
            str(request.run_root / "tracks/sam3/global_masks"),
            "--compat-tracking-dir",
            str(request.run_root / "4d/tracking"),
            "--schema-version",
            str(get_config(request.config, "schemas.global_object_masks")),
        )
        return tracking, consolidate


class CodexCanonicalBBoxTransferModule(FoundationModule):
    """Relocalize rectified-left canonical objects in synchronized native RGB."""

    name = "codex_canonical_bbox_transfer"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        root = request.run_root / "semantics/reasoning_rgb"
        return (root / "contact_events_rgb.json", root / "report.json")

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        command = [
            _python(request, self.name),
            "-m",
            WORKER_DISPATCHER,
            "transfer",
            "--source-video",
            str(_reasoning_video(request)),
            "--rgb-video",
            str(request.run_root / "input/rgb_upright.mp4"),
            "--source-reasoning",
            str(request.run_root / "semantics/reasoning/contact_events.json"),
            "--output-dir",
            str(request.run_root / "semantics/reasoning_rgb"),
            "--repository-root",
            str(repository_root(request.config)),
            "--codex-binary",
            str(get_config(request.config, "models.codex.binary", "codex")),
            "--model",
            str(get_config(request.config, "models.codex.model", "gpt-6-astra")),
            "--reasoning-effort",
            str(get_config(request.config, "models.codex.reasoning_effort", "high")),
            "--output-schema",
            str(
                resolve_path(
                    request.config,
                    "pipeline.codex_bbox_transfer.output_schema",
                )
            ),
            "--prompt-template",
            str(get_config(request.config, "prompts.canonical_bbox_transfer.template")),
            "--timeout-seconds",
            str(
                get_config(
                    request.config,
                    "pipeline.codex_bbox_transfer.timeout_seconds",
                    1800,
                )
            ),
        ]
        validation = resolve_path(request.config, "pipeline.codex_bbox_transfer.validation_masks")
        if validation is not None:
            command.extend(("--rgb-validation-masks", str(validation)))
        if _is_rgb_stereo(request):
            command.append("--exclude-hands")
        return tuple(command)


class SAM3RGBCanonicalSegmenter(FoundationModule):
    """Segment only transferred canonical boxes in the native RGB camera."""

    name = "sam3_rgb_canonical_segmentation"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root / "tracks/sam3_rgb_canonical/global_masks/manifest.json",
        )

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        return ()

    def build_commands(self, request: PipelineRequest) -> tuple[tuple[str, ...], ...]:
        root = request.run_root / "tracks/sam3_rgb_canonical"
        return (
            (
                _python(request, self.name),
                "-m",
                WORKER_DISPATCHER,
                "track",
                "--reasoning",
                str(_reference_reasoning(request)),
                "--video",
                str(request.run_root / "input/rgb_upright.mp4"),
                "--output-dir",
                str(root),
                "--min-mask-area",
                str(get_config(request.config, "pipeline.sam3_object.min_mask_area", 64)),
                "--prompt-mode",
                str(get_config(request.config, "pipeline.sam3_object.prompt_mode", "bbox_short_text")),
                "--schema-version",
                str(get_config(request.config, "schemas.sam3_object_tracks")),
                "--seed-only",
                "--no-overlay",
            ),
            (
                _python(request, self.name),
                "-m",
                WORKER_DISPATCHER,
                "consolidate_masks",
                "--tracks",
                str(root / "contact_tracks.json"),
                "--camera-data",
                str(request.run_root / "input/cameras.npz"),
                "--output-dir",
                str(root / "global_masks"),
                "--compat-tracking-dir",
                str(root / "compat"),
                "--schema-version",
                str(get_config(request.config, "schemas.global_object_masks")),
            ),
        )


class SAM3SupportTracker(FoundationModule):
    """Track static support objects in rectified left solely for pose observations."""

    name = "sam3_support_tracking"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root / "tracks/sam3_support/global_masks/manifest.json",
        )

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        return ()

    def build_commands(self, request: PipelineRequest) -> tuple[tuple[str, ...], ...]:
        root = request.run_root / "tracks/sam3_support"
        return (
            (
                _python(request, self.name),
                "-m",
                WORKER_DISPATCHER,
                "track",
                "--reasoning",
                str(request.run_root / "semantics/reasoning/contact_events.json"),
                "--video",
                str(_reasoning_video(request)),
                "--output-dir",
                str(root),
                "--registry-key",
                "support_objects",
                "--min-mask-area",
                str(get_config(request.config, "pipeline.sam3_object.min_mask_area", 64)),
                "--min-mask-area-fraction",
                str(
                    get_config(
                        request.config,
                        "pipeline.sam3_object.min_mask_area_fraction",
                        0.0,
                    )
                ),
                "--prompt-mode",
                str(get_config(request.config, "pipeline.sam3_object.prompt_mode", "bbox_short_text")),
                "--schema-version",
                str(get_config(request.config, "schemas.sam3_object_tracks")),
            ),
            (
                _python(request, self.name),
                "-m",
                WORKER_DISPATCHER,
                "consolidate_masks",
                "--tracks",
                str(root / "contact_tracks.json"),
                "--camera-data",
                str(request.run_root / "input/cameras.npz"),
                "--output-dir",
                str(root / "global_masks"),
                "--compat-tracking-dir",
                str(root / "compat"),
                "--schema-version",
                str(get_config(request.config, "schemas.global_object_masks")),
            ),
        )


class SAM3RGBSupportCanonicalSegmenter(FoundationModule):
    """Segment each support object's transferred canonical box in native RGB."""

    name = "sam3_rgb_support_segmentation"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root
            / "tracks/sam3_support_rgb_canonical/global_masks/manifest.json",
        )

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        return ()

    def build_commands(self, request: PipelineRequest) -> tuple[tuple[str, ...], ...]:
        root = request.run_root / "tracks/sam3_support_rgb_canonical"
        return (
            (
                _python(request, self.name),
                "-m",
                WORKER_DISPATCHER,
                "track",
                "--reasoning",
                str(_reference_reasoning(request)),
                "--video",
                str(request.run_root / "input/rgb_upright.mp4"),
                "--output-dir",
                str(root),
                "--registry-key",
                "support_objects",
                "--min-mask-area",
                str(get_config(request.config, "pipeline.sam3_object.min_mask_area", 64)),
                "--prompt-mode",
                str(get_config(request.config, "pipeline.sam3_object.prompt_mode", "bbox_short_text")),
                "--schema-version",
                str(get_config(request.config, "schemas.sam3_object_tracks")),
                "--seed-only",
                "--no-overlay",
            ),
            (
                _python(request, self.name),
                "-m",
                WORKER_DISPATCHER,
                "consolidate_masks",
                "--tracks",
                str(root / "contact_tracks.json"),
                "--camera-data",
                str(request.run_root / "input/cameras.npz"),
                "--output-dir",
                str(root / "global_masks"),
                "--compat-tracking-dir",
                str(root / "compat"),
                "--schema-version",
                str(get_config(request.config, "schemas.global_object_masks")),
            ),
        )

class _SAM3CanonicalHandTracker(FoundationModule):
    """Shared worker adapter for Codex-canonical wearer-hand tracking."""

    video_relative_path: str
    reasoning_relative_path: str
    output_relative_path: str

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        root = request.run_root / self.output_relative_path
        return root / "hand_masks.npz", root / "report.json", root / "overlay.mp4"

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        command = [
            _python(request, self.name),
            "-m",
            WORKER_DISPATCHER,
            "track_hands",
            "--video",
            str(request.run_root / self.video_relative_path),
            "--reasoning",
            str(request.run_root / self.reasoning_relative_path),
            "--output-dir",
            str(request.run_root / self.output_relative_path),
            "--minimum-area",
            str(get_config(request.config, "masks.hands.min_mask_area", 64)),
            "--maximum-area-fraction",
            str(get_config(request.config, "masks.hands.maximum_area_fraction", 0.15)),
            "--prompt-mode",
            str(get_config(request.config, "pipeline.sam3_object.prompt_mode", "bbox_short_text")),
        ]
        checkpoint = resolve_path(request.config, "models.sam3.checkpoint")
        if checkpoint is not None:
            command.extend(("--checkpoint", str(checkpoint)))
        return tuple(command)


class SAM3CanonicalLeftHandTracker(_SAM3CanonicalHandTracker):
    """Track wearer hands directly in rectified left from first-stage canonicals."""

    name = "sam3_left_hand_tracking_coarse"
    video_relative_path = "calibration/left_rectified_working.mp4"
    reasoning_relative_path = "semantics/reasoning/contact_events.json"
    output_relative_path = "tracks/sam3_hands_left_coarse"


class _SAM3TrackedBBoxHandSegmenter(FoundationModule):
    """Refine a temporal hand track by independent per-frame box segmentation."""

    video_relative_path: str
    coarse_relative_path: str
    output_relative_path: str
    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        root = request.run_root / self.output_relative_path
        return root / "hand_masks.npz", root / "manifest.json"

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        command = [
            _python(request, self.name),
            "-m",
            WORKER_DISPATCHER,
            "segment_frames",
            "--video",
            str(request.run_root / self.video_relative_path),
            "--hand-motion",
            str(request.run_root / self.coarse_relative_path / "hand_masks.npz"),
            "--output-dir",
            str(request.run_root / self.output_relative_path),
            "--repository-root",
            str(repository_root(request.config)),
            "--confidence-threshold",
            str(get_config(request.config, "masks.hands.confidence_threshold", 0.3)),
            "--min-mask-area",
            str(get_config(request.config, "masks.hands.min_mask_area", 64)),
            "--box-expand-ratio",
            str(get_config(request.config, "masks.hands.second_pass_box_expand_ratio", 1.05)),
            "--duplicate-box-iou",
            str(get_config(request.config, "masks.hands.duplicate_box_iou", 0.65)),
            "--maximum-flip-run",
            str(get_config(request.config, "masks.hands.maximum_flip_run", 12)),
            "--identity-gap",
            str(get_config(request.config, "masks.hands.identity_gap", 5)),
            "--device",
            request.device,
            "--schema-version",
            str(get_config(request.config, "schemas.global_object_masks")),
        ]
        checkpoint = resolve_path(request.config, "models.sam3.checkpoint")
        if checkpoint is not None:
            command.extend(("--checkpoint", str(checkpoint)))
        return tuple(command)


class SAM3TrackedBBoxLeftHandSegmenter(_SAM3TrackedBBoxHandSegmenter):
    """Create independent second-pass left-view hand masks."""

    name = "sam3_left_hand_segmentation"
    video_relative_path = "calibration/left_rectified_working.mp4"
    coarse_relative_path = "tracks/sam3_hands_left_coarse"
    output_relative_path = "tracks/sam3_hands_left"


class ObjectReferenceModule(FoundationModule):
    """Export Codex canonical frames as primary reconstruction references."""

    name = "object_reference_selection"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (request.run_root / "objects/reference_selection/reference_frames.json",)

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        masks = (
            request.run_root / "tracks/sam3_rgb_canonical/global_masks"
            if _is_rgb_stereo(request)
            else request.run_root / "tracks/sam3/refined_masks"
        )
        return (
            _python(request, self.name),
            "-m",
            WORKER_DISPATCHER,
            "export",
            "--reasoning",
            str(_reference_reasoning(request)),
            "--global-masks",
            str(masks),
            "--source-video",
            str(request.run_root / "input/rgb_upright.mp4"),
            "--output-dir",
            str(request.run_root / "objects/reference_selection"),
            "--crop-size",
            str(get_config(request.config, "pipeline.object_reference.crop_size", 768)),
            "--crop-context",
            str(get_config(request.config, "pipeline.object_reference.crop_context", 3.0)),
            "--schema-version",
            str(get_config(request.config, "schemas.object_reference")),
        )


class Flux2CompletionModule(FoundationModule):
    """FLUX.2 Klein 9B occlusion completion."""

    name = "flux2_completion"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (request.run_root / "objects/flux2/completion_manifest.json",)

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        return (
            _python(request, self.name),
            "-m",
            WORKER_DISPATCHER,
            "complete",
            "--selection-dir",
            str(request.run_root / "objects/reference_selection"),
            "--output-dir",
            str(request.run_root / "objects/flux2"),
            "--model-id",
            str(get_config(request.config, "models.flux2.model_id")),
            "--schema-version",
            str(get_config(request.config, "schemas.flux2_completion")),
            "--roles",
            "primary",
        )


class SAM3DObjectsModule(FoundationModule):
    """SAM 3D Objects canonical textured-mesh reconstruction."""

    name = "sam3d_objects"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (request.run_root / "objects/sam3d_objects/reconstruction_manifest.json",)

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        # ``build_commands`` contains the actual two-worker workflow.
        return ()

    def build_commands(self, request: PipelineRequest) -> tuple[tuple[str, ...], ...]:
        flux_dir = request.run_root / "objects/flux2"
        return (
            (
                _python(request, self.name),
                "-m",
                WORKER_DISPATCHER,
                "segment_completed_images",
                "--selection-dir",
                str(request.run_root / "objects/reference_selection"),
                "--flux-dir",
                str(flux_dir),
                "--confidence-threshold",
                str(get_config(request.config, "models.sam3.confidence_threshold", 0.3)),
                "--schema-version",
                str(get_config(request.config, "schemas.flux2_segmentation")),
            ),
            (
                _python(request, self.name),
                "-m",
                WORKER_DISPATCHER,
                "reconstruct",
                "--segmentation-manifest",
                str(flux_dir / "segmentation_manifest.json"),
                "--output-dir",
                str(request.run_root / "objects/sam3d_objects"),
                "--config",
                str(resolve_path(request.config, "models.sam3d_objects.root") / "checkpoints/hf/pipeline.yaml"),
                "--schema-version",
                str(get_config(request.config, "schemas.sam3d_objects")),
                "--roles",
                *(
                    str(role)
                    for role in get_config(
                        request.config,
                        "pipeline.sam3d_objects.roles",
                        ["primary"],
                    )
                ),
            ),
        )


class SupportObjectReferenceModule(FoundationModule):
    """Export one native-RGB canonical reference per static support object."""

    name = "support_reference_selection"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root
            / "support_objects/reference_selection/reference_frames.json",
        )

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        return (
            _python(request, self.name),
            "-m",
            WORKER_DISPATCHER,
            "export",
            "--reasoning",
            str(_reference_reasoning(request)),
            "--registry-key",
            "support_objects",
            "--global-masks",
            str(
                request.run_root
                / "tracks/sam3_support_rgb_canonical/global_masks"
            ),
            "--source-video",
            str(request.run_root / "input/rgb_upright.mp4"),
            "--output-dir",
            str(request.run_root / "support_objects/reference_selection"),
            "--crop-size",
            str(get_config(request.config, "pipeline.object_reference.crop_size", 768)),
            "--crop-context",
            str(get_config(request.config, "pipeline.object_reference.crop_context", 3.0)),
            "--schema-version",
            str(get_config(request.config, "schemas.object_reference")),
        )


class SupportFlux2CompletionModule(FoundationModule):
    """Complete support-object canonical references with FLUX.2."""

    name = "support_flux2_completion"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root / "support_objects/flux2/completion_manifest.json",
        )

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        return (
            _python(request, self.name),
            "-m",
            WORKER_DISPATCHER,
            "complete",
            "--selection-dir",
            str(request.run_root / "support_objects/reference_selection"),
            "--output-dir",
            str(request.run_root / "support_objects/flux2"),
            "--model-id",
            str(get_config(request.config, "models.flux2.model_id")),
            "--schema-version",
            str(get_config(request.config, "schemas.flux2_completion")),
            "--roles",
            "primary",
            "--allow-empty",
        )


class SupportSAM3DObjectsModule(FoundationModule):
    """Reconstruct primary textured meshes for static support objects."""

    name = "support_sam3d_objects"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root
            / "support_objects/sam3d_objects/reconstruction_manifest.json",
        )

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        return ()

    def build_commands(self, request: PipelineRequest) -> tuple[tuple[str, ...], ...]:
        root = request.run_root / "support_objects"
        flux_dir = root / "flux2"
        return (
            (
                _python(request, self.name),
                "-m",
                WORKER_DISPATCHER,
                "segment_completed_images",
                "--selection-dir",
                str(root / "reference_selection"),
                "--flux-dir",
                str(flux_dir),
                "--confidence-threshold",
                str(get_config(request.config, "models.sam3.confidence_threshold", 0.3)),
                "--schema-version",
                str(get_config(request.config, "schemas.flux2_segmentation")),
                "--allow-empty",
            ),
            (
                _python(request, self.name),
                "-m",
                WORKER_DISPATCHER,
                "reconstruct",
                "--segmentation-manifest",
                str(flux_dir / "segmentation_manifest.json"),
                "--output-dir",
                str(root / "sam3d_objects"),
                "--config",
                str(
                    resolve_path(request.config, "models.sam3d_objects.root")
                    / "checkpoints/hf/pipeline.yaml"
                ),
                "--schema-version",
                str(get_config(request.config, "schemas.sam3d_objects")),
                "--roles",
                "primary",
                "--cpu-offload",
                "--allow-empty",
            ),
        )


class StaticSupportPoseModule(FoundationModule):
    """Register eight observations and keep one support pose fixed in world."""

    name = "static_support_pose"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        return (request.run_root / "support_objects/pose/manifest.json",)

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        command = [
            _python(request, self.name),
            "-m",
            WORKER_DISPATCHER,
            "register",
            "--run-root",
            str(request.run_root),
            "--output-dir",
            str(request.run_root / "support_objects/pose"),
            "--candidate-count",
            str(get_config(request.config, "pipeline.support_objects.pose_candidates", 8)),
            "--est-refine-iter",
            str(get_config(request.config, "pipeline.support_objects.est_refine_iter", 5)),
            "--appearance-mode",
            str(get_config(request.config, "pipeline.support_objects.appearance_mode", "grayscale")),
            "--minimum-mask-area",
            str(get_config(request.config, "pipeline.support_objects.minimum_mask_area", 256)),
            "--scale-mode",
            str(
                get_config(
                    request.config,
                    "pipeline.support_objects.scale_mode",
                    "isotropic",
                )
            ),
            "--max-anisotropy",
            str(
                get_config(
                    request.config,
                    "pipeline.support_objects.max_anisotropy",
                    2.0,
                )
            ),
            "--thickness-weight",
            str(
                get_config(
                    request.config,
                    "pipeline.support_objects.thickness_weight",
                    0.15,
                )
            ),
        ]
        if bool(get_config(request.config, "pipeline.support_objects.write_overlay", True)):
            command.append("--write-overlay")
        else:
            command.append("--no-write-overlay")
        return tuple(command)


