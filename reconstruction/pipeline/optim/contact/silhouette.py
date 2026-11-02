"""Build occlusion-aware 2D silhouette constraints from the supplied HOI masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
import trimesh

from .types import ProjectedSilhouetteConstraints


@dataclass(frozen=True)
class SilhouetteBuildReport:
    active_frames: int
    total_frames: int
    source_resolution: tuple[int, int]
    optimization_resolution: tuple[int, int]
    surface_samples: int
    boundary_samples: int
    occluder_dilation: int


def build_constraints(
    *,
    mesh: trimesh.Trimesh,
    masks: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray | None = None,
    valid_frames: np.ndarray,
    downsample: int = 4,
    surface_sample_count: int = 768,
    boundary_sample_count: int = 128,
    occluder_dilation: int = 2,
) -> tuple[ProjectedSilhouetteConstraints, SilhouetteBuildReport]:
    """Create sampled symmetric silhouette constraints in camera coordinates.

    Mask value 1 is the visible object and 255 is the hand. The dilated hand is
    accepted by the containment term because object surface may legitimately
    project behind it. Coverage targets only the visible object boundary.
    """
    masks = np.asarray(masks)
    frame_count, source_height, source_width = masks.shape
    factor = max(1, int(downsample))
    height = max(8, int(round(source_height / factor)))
    width = max(8, int(round(source_width / factor)))
    boundary_count = max(8, int(boundary_sample_count))
    distance_fields = np.zeros((frame_count, height, width), dtype=np.float32)
    boundary_points = np.zeros((frame_count, boundary_count, 2), dtype=np.float32)
    boundary_valid = np.zeros((frame_count, boundary_count), dtype=bool)
    active = np.asarray(valid_frames, dtype=bool).copy()

    for frame_index in range(frame_count):
        reduced = cv2.resize(
            masks[frame_index], (width, height), interpolation=cv2.INTER_NEAREST
        )
        object_mask = reduced == 1
        hand_mask = reduced == 255
        if occluder_dilation > 0:
            hand_mask = binary_dilation(hand_mask, iterations=int(occluder_dilation))
        allowed = object_mask | hand_mask
        distance_fields[frame_index] = distance_transform_edt(~allowed).astype(
            np.float32
        )
        boundary = object_mask & ~binary_erosion(object_mask)
        ys, xs = np.nonzero(boundary)
        if len(xs) < 3:
            active[frame_index] = False
            continue
        if len(xs) > boundary_count:
            selected = np.linspace(0, len(xs) - 1, boundary_count).round().astype(int)
            xs, ys = xs[selected], ys[selected]
        count = len(xs)
        boundary_points[frame_index, :count] = np.column_stack((xs, ys))
        boundary_valid[frame_index, :count] = True

    scaled_intrinsics = np.asarray(intrinsics, dtype=np.float32).copy()
    scaled_intrinsics[:, 0, :] *= float(width) / float(source_width)
    scaled_intrinsics[:, 1, :] *= float(height) / float(source_height)
    surface, _ = trimesh.sample.sample_surface(
        mesh, max(64, int(surface_sample_count)), seed=73
    )
    # The original standalone experiment used camera-space poses and therefore
    # an identity extrinsic. Track-3 operates in the VIPE shared world frame,
    # so its dataset adapter supplies the actual world-to-camera sequence.
    if world_to_camera is None:
        world_to_camera_array = np.repeat(
            np.eye(4, dtype=np.float32)[None], frame_count, axis=0
        )
    else:
        world_to_camera_array = np.asarray(world_to_camera, dtype=np.float32)
        if world_to_camera_array.shape != (frame_count, 4, 4):
            raise ValueError("world_to_camera must have shape frames,4,4")
    constraints = ProjectedSilhouetteConstraints(
        surface_points_object=np.asarray(surface, dtype=np.float32),
        world_to_camera=world_to_camera_array,
        intrinsics=scaled_intrinsics,
        allowed_distance_fields=distance_fields,
        boundary_points_xy=boundary_points,
        boundary_valid=boundary_valid,
        frame_active=active,
    )
    report = SilhouetteBuildReport(
        active_frames=int(active.sum()),
        total_frames=frame_count,
        source_resolution=(source_width, source_height),
        optimization_resolution=(width, height),
        surface_samples=len(surface),
        boundary_samples=boundary_count,
        occluder_dilation=int(occluder_dilation),
    )
    return constraints, report
