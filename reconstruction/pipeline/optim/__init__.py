"""Class-based hand and object optimization stages."""

from .contact import (
    ContactObjectConfig,
    ContactPoseOptimizer,
    HACOJointHandObjectOptimizer,
    JointHandObjectConfig,
    ObjectCollisionObstacle,
)
from .hand_surface import (
    StereoHandSurfaceConfig,
    UmeTrackHandSurfaceAlignmentOptimizer,
)
from .hand_proj import (
    UmeTrackJointProjectionConfig,
    UmeTrackJointProjectionOptimizer,
)
from .object_surface import (
    ObjectSurfaceAlignmentConfig,
    StereoObjectSurfaceAlignmentOptimizer,
)
from .stages import (
    HACOContactOptimizationStage,
    HACOJointContactOptimizationStage,
)

__all__ = [
    "ContactObjectConfig",
    "HACOContactOptimizationStage",
    "HACOJointContactOptimizationStage",
    "HACOJointHandObjectOptimizer",
    "JointHandObjectConfig",
    "ObjectCollisionObstacle",
    "ContactPoseOptimizer",
    "StereoHandSurfaceConfig",
    "UmeTrackHandSurfaceAlignmentOptimizer",
    "UmeTrackJointProjectionConfig",
    "UmeTrackJointProjectionOptimizer",
    "ObjectSurfaceAlignmentConfig",
    "StereoObjectSurfaceAlignmentOptimizer",
]
