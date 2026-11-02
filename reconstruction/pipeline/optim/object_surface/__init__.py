"""Rigid object-to-stereo-surface alignment."""

from .optimizer import (
    ObjectSurfaceAlignmentConfig,
    StereoObjectSurfaceAlignmentOptimizer,
    align_object_pose_sequence,
)

__all__ = [
    "ObjectSurfaceAlignmentConfig",
    "StereoObjectSurfaceAlignmentOptimizer",
    "align_object_pose_sequence",
]
