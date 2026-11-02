"""Lazy worker dispatcher for the Track-3 stereo pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from pipeline.utils import worker_route


@dataclass(frozen=True)
class HandlerSpec:
    """Import path for one whitelisted stage/task worker."""

    module: str
    attribute: str


HANDLERS: dict[tuple[str, str], HandlerSpec] = {
    ("codex_hoi_reasoning", "reason"): HandlerSpec(
        "pipeline.modules.codex_reasoning.reasoner", "CodexHOIReasoner"
    ),
    ("codex_canonical_bbox_transfer", "transfer"): HandlerSpec(
        "pipeline.modules.codex_reasoning.cross_view_bbox", "CrossViewCanonicalBBoxMapper"
    ),
    ("haco_contact_prediction", "infer"): HandlerSpec(
        "pipeline.modules.haco.inference", "HACOContactPredictor"
    ),
    ("umetrack_hand_reconstruction", "infer"): HandlerSpec(
        "pipeline.modules.umetrack.inference", "UmeTrackHandReconstructor"
    ),
    ("umetrack_mano_fitting", "fit_mano"): HandlerSpec(
        "pipeline.modules.umetrack.fitting", "UmeTrackMANOFitter"
    ),
    ("object_reference_selection", "export"): HandlerSpec(
        "pipeline.modules.object_reference", "ObjectReferenceExporter"
    ),
    ("support_reference_selection", "export"): HandlerSpec(
        "pipeline.modules.object_reference", "ObjectReferenceExporter"
    ),
    ("sam3_object_tracking", "track"): HandlerSpec(
        "pipeline.modules.sam3.object_tracker", "SAM3ObjectTracker"
    ),
    ("sam3_object_tracking", "consolidate_masks"): HandlerSpec(
        "pipeline.modules.sam3.masks", "main"
    ),
    ("sam3_support_tracking", "track"): HandlerSpec(
        "pipeline.modules.sam3.object_tracker", "SAM3ObjectTracker"
    ),
    ("sam3_support_tracking", "consolidate_masks"): HandlerSpec(
        "pipeline.modules.sam3.masks", "main"
    ),
    ("sam3_rgb_canonical_segmentation", "track"): HandlerSpec(
        "pipeline.modules.sam3.object_tracker", "SAM3ObjectTracker"
    ),
    ("sam3_rgb_canonical_segmentation", "consolidate_masks"): HandlerSpec(
        "pipeline.modules.sam3.masks", "main"
    ),
    ("sam3_rgb_support_segmentation", "track"): HandlerSpec(
        "pipeline.modules.sam3.object_tracker", "SAM3ObjectTracker"
    ),
    ("sam3_rgb_support_segmentation", "consolidate_masks"): HandlerSpec(
        "pipeline.modules.sam3.masks", "main"
    ),
    ("sam3_left_hand_tracking_coarse", "track_hands"): HandlerSpec(
        "pipeline.modules.sam3.canonical_hand_tracker", "CanonicalHandTracker"
    ),
    ("sam3_left_hand_segmentation", "segment_frames"): HandlerSpec(
        "pipeline.modules.sam3.hand_frame_worker", "main"
    ),
    ("flux2_completion", "complete"): HandlerSpec(
        "pipeline.modules.flux2.completion", "Flux2ObjectCompleter"
    ),
    ("support_flux2_completion", "complete"): HandlerSpec(
        "pipeline.modules.flux2.completion", "Flux2ObjectCompleter"
    ),
    ("sam3d_objects", "segment_completed_images"): HandlerSpec(
        "pipeline.modules.sam3d_objects.segmentation", "main"
    ),
    ("sam3d_objects", "reconstruct"): HandlerSpec(
        "pipeline.modules.sam3d_objects.reconstruction", "SAM3DObjectReconstructor"
    ),
    ("support_sam3d_objects", "segment_completed_images"): HandlerSpec(
        "pipeline.modules.sam3d_objects.segmentation", "main"
    ),
    ("support_sam3d_objects", "reconstruct"): HandlerSpec(
        "pipeline.modules.sam3d_objects.reconstruction", "SAM3DObjectReconstructor"
    ),
    ("static_support_pose", "register"): HandlerSpec(
        "pipeline.modules.support_objects.pose", "StaticSupportPoseEstimator"
    ),
}


def registered_routes() -> tuple[tuple[str, str], ...]:
    return tuple(sorted(HANDLERS))


def _run_handler(target: Any) -> None:
    factory = getattr(target, "from_worker_job", None)
    if callable(factory):
        worker = factory()
        run = getattr(worker, "run", None)
        if not callable(run):
            raise TypeError(f"Worker instance has no callable run(): {target!r}")
        run()
        return
    if not callable(target):
        raise TypeError(f"Registered worker handler is not callable: {target!r}")
    target()


def main() -> None:
    route = worker_route()
    try:
        spec = HANDLERS[route]
    except KeyError as error:
        available = ", ".join(f"{stage}/{task}" for stage, task in registered_routes())
        raise ValueError(f"Unknown worker route {route}; available: {available}") from error
    module = import_module(spec.module)
    _run_handler(getattr(module, spec.attribute))


if __name__ == "__main__":
    main()
