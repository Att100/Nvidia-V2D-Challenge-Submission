"""Lazy public API for the Track-3 stereo foundation-model stages."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_FOUNDATION = "pipeline.modules.stereo_foundation_models"
_STEREO = "pipeline.modules.stereo_pipeline"
_EXPORTS = {
    "FoundationModule": ("pipeline.modules.base", "FoundationModule"),
    **{
        name: (_FOUNDATION, name)
        for name in (
            "CodexCanonicalBBoxTransferModule",
            "CodexHOIReasoningModule",
            "Flux2CompletionModule",
            "HACOContactPredictionModule",
            "ObjectReferenceModule",
            "SAM3CanonicalLeftHandTracker",
            "SAM3DObjectsModule",
            "SAM3ObjectTracker",
            "SAM3RGBCanonicalSegmenter",
            "SAM3RGBSupportCanonicalSegmenter",
            "SAM3SupportTracker",
            "SAM3TrackedBBoxLeftHandSegmenter",
            "StaticSupportPoseModule",
            "SupportFlux2CompletionModule",
            "SupportObjectReferenceModule",
            "SupportSAM3DObjectsModule",
        )
    },
    "UmeTrackInferenceModule": ("pipeline.modules.umetrack", "UmeTrackInferenceModule"),
    "UmeTrackMANOFittingModule": ("pipeline.modules.umetrack", "UmeTrackMANOFittingModule"),
    **{
        name: (_STEREO, name)
        for name in (
            "FoundationStereoModule",
            "PublicObjectMeshModule",
            "StereoCalibrationModule",
            "StereoDynamicMaskModule",
            "StereoFoundationPoseModule",
            "StereoOptimizationAdapterModule",
            "StereoStaticFusionModule",
            "ViPEMetricCameraModule",
        )
    },
    "InputPreparationModule": ("pipeline.modules.input_preparation", "InputPreparationModule"),
    "ObjectMaskRefinementConfig": (
        "pipeline.modules.object_masks",
        "ObjectMaskRefinementConfig",
    ),
    "ObjectMaskRefinementModule": (
        "pipeline.modules.object_masks",
        "ObjectMaskRefinementModule",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import only the module that owns the requested stereo symbol."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
