"""Typed inputs and outputs for fixed-hand object contact optimization.

The containers keep the optimizer independent of dataset file layouts and give
all optimization stages a stable interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ContactObjectConfig:
    """Weights, robust scales, solver settings, and pose-update bounds."""

    iterations: int = 800
    learning_rate: float = 0.0015
    contact_weight: float = 1.0
    penetration_weight: float = 8.0
    rotation_prior_weight: float = 2.0
    translation_prior_weight: float = 2.0
    rotation_velocity_weight: float = 0.5
    translation_velocity_weight: float = 0.5
    rotation_acceleration_weight: float = 2.0
    translation_acceleration_weight: float = 2.0
    object_rotation_acceleration_weight: float = 0.5
    object_translation_acceleration_weight: float = 0.5
    # Pipeline v2 deliberately enables the requested 2D silhouette term.
    silhouette_weight: float = 2.0
    silhouette_containment_weight: float = 0.5
    silhouette_coverage_weight: float = 0.5
    contact_sigma: float = 0.005
    penetration_sigma: float = 0.003
    penetration_tolerance: float = 0.002
    rotation_prior_sigma_radians: float = np.deg2rad(4.0)
    translation_prior_sigma: float = 0.015
    rotation_velocity_sigma_radians: float = np.deg2rad(1.0)
    translation_velocity_sigma: float = 0.004
    rotation_acceleration_sigma_radians: float = np.deg2rad(0.75)
    translation_acceleration_sigma: float = 0.0025
    object_rotation_acceleration_sigma_radians: float = np.deg2rad(2.0)
    object_translation_acceleration_sigma: float = 0.005
    silhouette_sigma_pixels: float = 2.0
    contact_fraction: float = 0.20
    maximum_rotation_radians: float = np.deg2rad(10.0)
    maximum_translation: float = 0.040
    device: str = "cuda"


@dataclass(frozen=True)
class JointHandObjectConfig(ContactObjectConfig):
    """Additional constraints for joint MANO/object SE(3) refinement.

    Each active hand/frame receives a bounded rigid residual about its wrist
    in addition to the MANO articulation residual by default.
    """

    hand_learning_rate: float = 0.001
    articulation_prior_weight: float = 2.0
    articulation_velocity_weight: float = 0.5
    articulation_acceleration_weight: float = 1.0
    hand_reprojection_weight: float = 2.0
    hand_mask_weight: float = 2.0
    optimize_hand_se3: bool = True
    hand_se3_learning_rate: float = 0.001
    hand_rotation_prior_weight: float = 1.0
    hand_translation_prior_weight: float = 1.0
    hand_rotation_velocity_weight: float = 0.5
    hand_translation_velocity_weight: float = 0.5
    hand_rotation_acceleration_weight: float = 1.0
    hand_translation_acceleration_weight: float = 1.0
    articulation_prior_sigma_radians: float = np.deg2rad(4.0)
    articulation_velocity_sigma_radians: float = np.deg2rad(2.0)
    articulation_acceleration_sigma_radians: float = np.deg2rad(1.5)
    hand_reprojection_sigma_pixels: float = 4.0
    hand_rotation_prior_sigma_radians: float = np.deg2rad(4.0)
    hand_translation_prior_sigma: float = 0.010
    hand_rotation_velocity_sigma_radians: float = np.deg2rad(2.0)
    hand_translation_velocity_sigma: float = 0.004
    hand_rotation_acceleration_sigma_radians: float = np.deg2rad(1.5)
    hand_translation_acceleration_sigma: float = 0.003
    maximum_articulation_radians: float = np.deg2rad(12.0)
    maximum_hand_rotation_radians: float = np.deg2rad(10.0)
    maximum_hand_translation: float = 0.025
    hand_mask_downsample: int = 4
    hand_mask_surface_samples: int = 160
    hand_mask_boundary_samples: int = 96
    hand_mask_sigma_pixels: float = 2.0
    silhouette_guard_median_tolerance: float = 0.05
    silhouette_guard_p90_tolerance: float = 0.15
    silhouette_guard_steps: int = 20
    object_object_penetration_weight: float = 4.0
    object_object_penetration_sigma: float = 0.003
    object_object_penetration_tolerance: float = 0.001
    object_object_surface_samples: int = 768
    mano_model_root: str = "thirdparty/HACO_RELEASE/data/base_data/human_models"


@dataclass(frozen=True)
class ObjectCollisionObstacle:
    """Another rigid object used by the inter-object penetration constraint.

    The obstacle trajectory is treated as fixed during one object's update.
    Running the same constraint for every object gives a symmetric
    Gauss--Seidel-style approximation without duplicating the hand variables.
    """

    object_id: str
    mesh_path: Path
    object_to_world: np.ndarray
    valid_frames: np.ndarray


@dataclass(frozen=True)
class ProjectedSilhouetteConstraints:
    """Low-resolution occlusion-aware projected-silhouette observations."""

    surface_points_object: np.ndarray
    world_to_camera: np.ndarray
    intrinsics: np.ndarray
    allowed_distance_fields: np.ndarray
    boundary_points_xy: np.ndarray
    boundary_valid: np.ndarray
    frame_active: np.ndarray


@dataclass(frozen=True)
class ContactOptimizationInput:
    """One object's complete sequence in the shared world coordinate system.

    Arrays use a leading frame dimension ``T``. Object poses are object-to-world
    transforms, camera poses are world-to-camera transforms, and masks use
    values 0 (background), 1 (visible object), and 255 (hand occluder).
    """

    object_id: str
    object_name: str
    mesh_path: Path
    output_dir: Path
    hand_side: str
    hand_vertices_world: np.ndarray
    hand_joints_world: np.ndarray
    hand_joints_2d: np.ndarray | None
    initial_object_to_world: np.ndarray
    world_to_camera: np.ndarray
    intrinsics: np.ndarray
    masks: np.ndarray
    contact_active: np.ndarray
    valid_frames: np.ndarray
    source_indices: np.ndarray
    # Optional independent masks in anatomical ``hand_labels`` order.  Shape
    # is [T,H,image_height,image_width].  ``masks`` above remains the combined
    # object/occluder image consumed by the object silhouette term.
    hand_masks: np.ndarray | None = None
    contact_vertex_weights: np.ndarray | None = None
    hand_vertex_active: np.ndarray | None = None
    # Optional canonical object-surface region used by part-aware contact.
    # When supplied, HACO-positive hand vertices are attracted only to these
    # points; penetration is still evaluated against the complete object SDF.
    object_contact_vertices: np.ndarray | None = None
    hand_labels: tuple[str, ...] = ()
    mano_betas: np.ndarray | None = None
    mano_global_orient: np.ndarray | None = None
    mano_hand_pose: np.ndarray | None = None
    hand_motion_path: Path | None = None
    collision_obstacles: tuple[ObjectCollisionObstacle, ...] = ()


@dataclass(frozen=True)
class ContactOptimizationResult:
    """Saved contact-refinement artifacts and concise sequence diagnostics."""

    object_id: str
    poses_path: Path
    metrics_path: Path
    report_path: Path
    active_frame_count: int
    optimized_frame_count: int
    contact_distance_before_median: float | None
    contact_distance_after_median: float | None
    silhouette_error_before_median: float | None
    silhouette_error_after_median: float | None


@dataclass(frozen=True)
class JointHandObjectOptimizationResult(ContactOptimizationResult):
    """Artifacts and diagnostics from joint MANO/object optimization."""

    hand_articulation_path: Path
    articulation_delta_degrees_median: float | None
    hand_reprojection_error_median_pixels: float | None
