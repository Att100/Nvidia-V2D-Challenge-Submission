"""Reusable PyTorch loss modules for fixed-hand object-pose refinement.

The modules in this file contain no optimization loop and perform no file I/O.
They can therefore be imported by another project, used in a training loss, or
combined with a project-specific optimizer. Sequence tensors use a leading
frame dimension ``T`` and OpenCV camera coordinates unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import axis_angle_to_matrix


def smooth_l1_to_zero(values: torch.Tensor) -> torch.Tensor:
    """Return mean Smooth-L1 distance to zero, including an empty-input guard."""

    if values.numel() == 0:
        return values.new_zeros(())
    return F.smooth_l1_loss(values, torch.zeros_like(values), reduction="mean")


def masked_smooth_l1(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply mean Smooth-L1 only where the leading-dimension mask is true."""

    if values.numel() == 0 or not bool(mask.any()):
        return values.new_zeros(())
    return smooth_l1_to_zero(values[mask])


def rotation_matrix_to_axis_angle(rotation: torch.Tensor) -> torch.Tensor:
    """Differentiable SO(3) logarithm used for temporal angular velocity."""

    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    cosine = (
        (torch.diagonal(rotation, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    ).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cosine)
    sine = torch.sin(angle)
    regular_scale = angle / (2.0 * sine.clamp_min(1e-6))
    scale = torch.where(angle < 1e-3, torch.full_like(angle, 0.5), regular_scale)
    return vee * scale[..., None]


class VoxelSDFSampler(nn.Module):
    """Differentiably sample an object-local signed-distance voxel grid."""

    def __init__(
        self,
        values_zyx: torch.Tensor | Any,
        object_to_index: torch.Tensor | Any,
        pitch: float,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "values_zyx", torch.as_tensor(values_zyx, dtype=torch.float32).contiguous()
        )
        self.register_buffer(
            "object_to_index",
            torch.as_tensor(object_to_index, dtype=torch.float32).contiguous(),
        )
        self.pitch = float(pitch)

    @classmethod
    def from_sdf(cls, sdf: Any) -> "VoxelSDFSampler":
        """Build from any object exposing values_zyx, object_to_index, and pitch."""

        return cls(sdf.values_zyx, sdf.object_to_index, sdf.pitch)

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        depth, height, width = self.values_zyx.shape
        return width, height, depth

    def forward(self, points_object: torch.Tensor) -> torch.Tensor:
        if points_object.shape[-1] != 3:
            raise ValueError("points_object must end in XYZ")
        transform = self.object_to_index.to(
            device=points_object.device, dtype=points_object.dtype
        )
        indices = points_object @ transform[:3, :3].T + transform[:3, 3]
        nx, ny, nz = self.shape_xyz
        upper = points_object.new_tensor([nx - 1, ny - 1, nz - 1])
        clamped_indices = torch.minimum(
            torch.maximum(indices, torch.zeros_like(indices)), upper
        )

        # Preserve a useful gradient outside the finite SDF volume instead of
        # allowing border-clamped sampling to make large gaps invisible.
        outside_distance = (
            torch.linalg.norm(indices - clamped_indices, dim=-1) * self.pitch
        )
        normalized = torch.empty_like(clamped_indices)
        normalized[..., 0] = 2.0 * clamped_indices[..., 0] / max(nx - 1, 1) - 1.0
        normalized[..., 1] = 2.0 * clamped_indices[..., 1] / max(ny - 1, 1) - 1.0
        normalized[..., 2] = 2.0 * clamped_indices[..., 2] / max(nz - 1, 1) - 1.0
        original_shape = normalized.shape[:-1]
        grid = normalized.reshape(1, -1, 1, 1, 3)
        volume = self.values_zyx.to(
            device=points_object.device, dtype=points_object.dtype
        )[None, None]
        sampled = F.grid_sample(
            volume,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(original_shape) + outside_distance


class ContactLoss(nn.Module):
    """Encourage the closest fraction of each hand patch to touch the object."""

    def __init__(self, sigma: float = 0.005, contact_fraction: float = 0.20) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.contact_fraction = float(contact_fraction)

    def forward(
        self,
        sdf_values: torch.Tensor,
        contact_vertex_indices: torch.Tensor,
        contact_group_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, vertices_per_group = contact_vertex_indices.shape
        expanded_sdf = sdf_values[:, None].expand(
            -1, contact_vertex_indices.shape[1], -1
        )
        contact_sdf = torch.gather(expanded_sdf, 2, contact_vertex_indices)
        contact_distance = contact_sdf.abs()
        selected_count = max(
            1,
            min(
                vertices_per_group,
                int(round(self.contact_fraction * vertices_per_group)),
            ),
        )
        nearest = torch.topk(
            contact_distance, k=selected_count, dim=-1, largest=False
        ).values
        active = contact_group_active[..., None].expand_as(nearest)
        return masked_smooth_l1(nearest / self.sigma, active), nearest


class WeightedVertexContactLoss(nn.Module):
    """Attract only HACO-positive vertices, weighted by calibrated confidence."""

    def __init__(self, sigma: float = 0.005) -> None:
        super().__init__()
        self.sigma = float(sigma)

    def forward(
        self,
        sdf_values: torch.Tensor,
        contact_vertex_weights: torch.Tensor,
        frame_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sdf_values.shape != contact_vertex_weights.shape:
            raise ValueError("HACO contact weights must match the flattened hand vertices")
        weights = contact_vertex_weights.clamp_min(0.0)
        active = frame_active[:, None] & (weights > 0.0)
        distances = sdf_values.abs()
        if not bool(active.any()):
            return distances.new_zeros(()), torch.full_like(distances, float("nan"))
        residual = F.smooth_l1_loss(
            distances / self.sigma,
            torch.zeros_like(distances),
            reduction="none",
        )
        effective = torch.where(active, weights, torch.zeros_like(weights))
        loss = (residual * effective).sum() / effective.sum().clamp_min(1e-8)
        diagnostic = torch.where(active, distances, torch.full_like(distances, float("nan")))
        return loss, diagnostic


class ObjectRegionWeightedContactLoss(nn.Module):
    """Attract HACO hand vertices only to an allowed canonical object region."""

    def __init__(self, sigma: float = 0.005, maximum_hand_vertices: int = 128) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.maximum_hand_vertices = int(maximum_hand_vertices)

    def forward(
        self,
        hand_vertices_world: torch.Tensor,
        object_rotation: torch.Tensor,
        object_translation: torch.Tensor,
        object_contact_vertices: torch.Tensor,
        contact_vertex_weights: torch.Tensor,
        frame_active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights_all = contact_vertex_weights.clamp_min(0.0)
        count = min(self.maximum_hand_vertices, weights_all.shape[1])
        weights, indices = torch.topk(weights_all, k=count, dim=1)
        hand = torch.gather(
            hand_vertices_world,
            1,
            indices[..., None].expand(-1, -1, 3),
        )
        active = frame_active[:, None] & (weights > 0.0)
        if not bool(active.any()):
            empty = torch.full_like(weights, float("nan"))
            return weights.new_zeros(()), empty
        region_world = (
            torch.einsum("fij,kj->fki", object_rotation, object_contact_vertices)
            + object_translation[:, None]
        )
        # Keep all HACO vertices in the tensor for stable batching; inactive
        # values receive zero weight after their diagnostic distance is found.
        distance = torch.cdist(hand, region_world).min(dim=-1).values
        residual = F.smooth_l1_loss(
            distance / self.sigma,
            torch.zeros_like(distance),
            reduction="none",
        )
        effective = torch.where(active, weights, torch.zeros_like(weights))
        loss = (residual * effective).sum() / effective.sum().clamp_min(1.0e-8)
        diagnostic = torch.where(
            active, distance, torch.full_like(distance, float("nan"))
        )
        return loss, diagnostic


class GroupedWeightedVertexContactLoss(nn.Module):
    """Robust HACO contact over disjoint anatomical hand-surface groups.

    HACO often labels a broad contact region, while only a subset of each
    finger or palm patch can physically touch a rigid surface.  This loss keeps
    the closest fraction independently in every active anatomical group.  The
    hard selection is detached; gradients flow through the selected SDF
    distances and confidence weights.
    """

    def __init__(self, sigma: float = 0.005, closest_fraction: float = 0.30) -> None:
        super().__init__()
        if not 0.0 < closest_fraction <= 1.0:
            raise ValueError("closest_fraction must be in (0, 1]")
        self.sigma = float(sigma)
        self.closest_fraction = float(closest_fraction)

    def forward(
        self,
        sdf_values: torch.Tensor,
        contact_vertex_weights: torch.Tensor,
        frame_active: torch.Tensor,
        group_ids: torch.Tensor,
        group_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return robust contact loss, all diagnostics, and selected vertices."""

        if sdf_values.shape != contact_vertex_weights.shape:
            raise ValueError("HACO weights must match flattened hand vertices")
        if group_ids.shape != sdf_values.shape:
            raise ValueError("Anatomical group IDs must match flattened vertices")
        weights = contact_vertex_weights.clamp_min(0.0)
        distances = sdf_values.abs()
        base_active = frame_active[:, None] & (weights > 0.0)
        selected = torch.zeros_like(base_active)
        detached_distance = distances.detach()
        infinity = torch.full_like(detached_distance, float("inf"))
        for group_id in range(int(group_count)):
            candidates = base_active & (group_ids == group_id)
            counts = candidates.sum(dim=1)
            active_rows = counts > 0
            if not bool(active_rows.any()):
                continue
            masked_distance = torch.where(candidates, detached_distance, infinity)
            sorted_distance = torch.sort(masked_distance, dim=1).values
            keep = torch.ceil(
                counts.to(torch.float32) * self.closest_fraction
            ).to(torch.long).clamp_min(1)
            threshold = sorted_distance.gather(1, (keep - 1)[:, None])[:, 0]
            selected |= candidates & (detached_distance <= threshold[:, None])
        if not bool(selected.any()):
            diagnostic = torch.full_like(distances, float("nan"))
            return distances.new_zeros(()), diagnostic, selected
        residual = F.smooth_l1_loss(
            distances / self.sigma,
            torch.zeros_like(distances),
            reduction="none",
        )
        effective = torch.where(selected, weights, torch.zeros_like(weights))
        loss = (residual * effective).sum() / effective.sum().clamp_min(1.0e-8)
        diagnostic = torch.where(
            selected, distances, torch.full_like(distances, float("nan"))
        )
        return loss, diagnostic, selected


class PenetrationLoss(nn.Module):
    """Penalize hand vertices farther than a tolerance inside the object."""

    def __init__(
        self,
        sigma: float = 0.003,
        tolerance: float = 0.002,
    ) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.tolerance = float(tolerance)

    def forward(
        self,
        sdf_values: torch.Tensor,
        frame_active: torch.Tensor,
        vertex_active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        penetration = torch.relu(-sdf_values - self.tolerance)
        active = frame_active[:, None].expand_as(penetration)
        if vertex_active is not None:
            if vertex_active.shape != penetration.shape:
                raise ValueError("hand vertex activity must match sampled SDF values")
            active = active & vertex_active
        return masked_smooth_l1(penetration / self.sigma, active), penetration


class PosePriorLoss(nn.Module):
    """Penalize a pose residual relative to its zero initialization."""

    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.sigma = float(sigma)

    def forward(
        self, residual: torch.Tensor, frame_active: torch.Tensor
    ) -> torch.Tensor:
        return masked_smooth_l1(residual / self.sigma, frame_active)


class FirstOrderSmoothnessLoss(nn.Module):
    """Penalize first-order differences on adjacent valid frames."""

    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, values: torch.Tensor, frame_active: torch.Tensor) -> torch.Tensor:
        adjacent = frame_active[1:] & frame_active[:-1]
        return masked_smooth_l1((values[1:] - values[:-1]) / self.sigma, adjacent)


class SecondOrderSmoothnessLoss(nn.Module):
    """Penalize second-order differences on valid frame triplets."""

    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.sigma = float(sigma)

    def forward(self, values: torch.Tensor, frame_active: torch.Tensor) -> torch.Tensor:
        triplets = frame_active[2:] & frame_active[1:-1] & frame_active[:-2]
        acceleration = values[2:] - 2.0 * values[1:-1] + values[:-2]
        return masked_smooth_l1(acceleration / self.sigma, triplets)


class ObjectRotationAccelerationLoss(nn.Module):
    """Penalize changes in inter-frame object angular velocity."""

    def __init__(self, sigma_radians: float) -> None:
        super().__init__()
        self.sigma_radians = float(sigma_radians)

    def forward(
        self, object_rotation: torch.Tensor, frame_active: torch.Tensor
    ) -> torch.Tensor:
        triplets = frame_active[2:] & frame_active[1:-1] & frame_active[:-2]
        relative = torch.bmm(object_rotation[1:], object_rotation[:-1].transpose(1, 2))
        angular_velocity = rotation_matrix_to_axis_angle(relative)
        angular_acceleration = angular_velocity[1:] - angular_velocity[:-1]
        return masked_smooth_l1(angular_acceleration / self.sigma_radians, triplets)


class ProjectedSilhouetteLoss(nn.Module):
    """Occlusion-aware symmetric sampled object-silhouette constraint."""

    def __init__(
        self,
        constraints: Any,
        *,
        sigma_pixels: float = 2.0,
        containment_weight: float = 0.5,
        coverage_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.sigma_pixels = float(sigma_pixels)
        self.containment_weight = float(containment_weight)
        self.coverage_weight = float(coverage_weight)
        self.register_buffer(
            "surface_points_object",
            torch.as_tensor(
                constraints.surface_points_object, dtype=torch.float32
            ).contiguous(),
        )
        self.register_buffer(
            "world_to_camera",
            torch.as_tensor(constraints.world_to_camera, dtype=torch.float32),
        )
        self.register_buffer(
            "intrinsics", torch.as_tensor(constraints.intrinsics, dtype=torch.float32)
        )
        self.register_buffer(
            "allowed_distance_fields",
            torch.as_tensor(
                constraints.allowed_distance_fields, dtype=torch.float32
            ).contiguous(),
        )
        self.register_buffer(
            "boundary_points_xy",
            torch.as_tensor(constraints.boundary_points_xy, dtype=torch.float32),
        )
        self.register_buffer(
            "boundary_valid",
            torch.as_tensor(constraints.boundary_valid, dtype=torch.bool),
        )
        self.register_buffer(
            "frame_active",
            torch.as_tensor(constraints.frame_active, dtype=torch.bool),
        )

    def forward(
        self, object_rotation: torch.Tensor, object_translation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = object_rotation.dtype
        device = object_rotation.device
        points = self.surface_points_object.to(device=device, dtype=dtype)
        world_to_camera = self.world_to_camera.to(device=device, dtype=dtype)
        intrinsics = self.intrinsics.to(device=device, dtype=dtype)
        distance_fields = self.allowed_distance_fields.to(device=device, dtype=dtype)
        boundary = self.boundary_points_xy.to(device=device, dtype=dtype)
        boundary_valid = self.boundary_valid.to(device=device)
        frame_active = self.frame_active.to(device=device)

        points_world = (
            torch.einsum("fij,sj->fsi", object_rotation, points)
            + object_translation[:, None]
        )
        points_camera = (
            torch.einsum("fij,fsj->fsi", world_to_camera[:, :3, :3], points_world)
            + world_to_camera[:, None, :3, 3]
        )
        depth = points_camera[..., 2]
        safe_depth = depth.clamp_min(1e-4)
        u = (
            intrinsics[:, None, 0, 0] * points_camera[..., 0] / safe_depth
            + intrinsics[:, None, 0, 2]
        )
        v = (
            intrinsics[:, None, 1, 1] * points_camera[..., 1] / safe_depth
            + intrinsics[:, None, 1, 2]
        )
        projected = torch.stack((u, v), dim=-1)

        height, width = distance_fields.shape[-2:]
        grid = projected.clone()
        grid[..., 0] = 2.0 * grid[..., 0] / max(width - 1, 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / max(height - 1, 1) - 1.0
        sampled_outside = F.grid_sample(
            distance_fields[:, None],
            grid[:, :, None],
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[:, 0, :, 0]
        image_overflow = (
            torch.relu(-u)
            + torch.relu(u - float(width - 1))
            + torch.relu(-v)
            + torch.relu(v - float(height - 1))
        )
        projected_valid = depth > 1e-4
        containment_values = (sampled_outside + image_overflow) / self.sigma_pixels
        containment_mask = projected_valid & frame_active[:, None]
        containment = masked_smooth_l1(containment_values, containment_mask)

        pairwise = torch.cdist(boundary, projected)
        pairwise = torch.where(
            projected_valid[:, None], pairwise, torch.full_like(pairwise, 1e3)
        )
        coverage_values = pairwise.min(dim=-1).values / self.sigma_pixels
        coverage_mask = boundary_valid & frame_active[:, None]
        coverage = masked_smooth_l1(coverage_values, coverage_mask)
        loss = self.containment_weight * containment + self.coverage_weight * coverage

        containment_frame = torch.where(
            containment_mask,
            sampled_outside + image_overflow,
            torch.full_like(sampled_outside, float("nan")),
        ).nanmean(dim=1)
        coverage_frame = torch.where(
            coverage_mask,
            pairwise.min(dim=-1).values,
            torch.full_like(coverage_values, float("nan")),
        ).nanmean(dim=1)
        return loss, 0.5 * (containment_frame + coverage_frame)


@dataclass
class FixedHandObjectLossOutput:
    """Differentiable total, named terms, poses, and diagnostic tensors."""

    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    object_rotation: torch.Tensor
    object_translation: torch.Tensor
    hand_object: torch.Tensor
    sdf_values: torch.Tensor
    nearest_contact_distances: torch.Tensor
    penetration_depths: torch.Tensor
    silhouette_error: torch.Tensor | None


class FixedHandObjectPoseLoss(nn.Module):
    """Compose all losses used by the fixed-hand rigid-object optimizer.

    The module owns no trainable parameters. Callers supply pose residuals from
    a network or create ``nn.Parameter`` tensors for direct optimization.
    """

    TERM_NAMES = (
        "contact",
        "penetration",
        "silhouette",
        "rotation_prior",
        "translation_prior",
        "rotation_velocity",
        "translation_velocity",
        "rotation_acceleration",
        "translation_acceleration",
        "object_rotation_acceleration",
        "object_translation_acceleration",
    )

    def __init__(
        self,
        *,
        sdf: Any,
        config: Any,
        silhouette_constraints: Any | None = None,
    ) -> None:
        super().__init__()
        self.sdf = VoxelSDFSampler.from_sdf(sdf)
        self.contact = ContactLoss(config.contact_sigma, config.contact_fraction)
        self.weighted_contact = WeightedVertexContactLoss(config.contact_sigma)
        self.object_region_contact = ObjectRegionWeightedContactLoss(
            config.contact_sigma
        )
        self.penetration = PenetrationLoss(
            config.penetration_sigma,
            config.penetration_tolerance,
        )
        self.rotation_prior = PosePriorLoss(config.rotation_prior_sigma_radians)
        self.translation_prior = PosePriorLoss(config.translation_prior_sigma)
        self.rotation_velocity = FirstOrderSmoothnessLoss(
            config.rotation_velocity_sigma_radians
        )
        self.translation_velocity = FirstOrderSmoothnessLoss(
            config.translation_velocity_sigma
        )
        self.rotation_acceleration = SecondOrderSmoothnessLoss(
            config.rotation_acceleration_sigma_radians
        )
        self.translation_acceleration = SecondOrderSmoothnessLoss(
            config.translation_acceleration_sigma
        )
        self.object_rotation_acceleration = ObjectRotationAccelerationLoss(
            config.object_rotation_acceleration_sigma_radians
        )
        self.object_translation_acceleration = SecondOrderSmoothnessLoss(
            config.object_translation_acceleration_sigma
        )
        self.silhouette = (
            ProjectedSilhouetteLoss(
                silhouette_constraints,
                sigma_pixels=config.silhouette_sigma_pixels,
                containment_weight=config.silhouette_containment_weight,
                coverage_weight=config.silhouette_coverage_weight,
            )
            if silhouette_constraints is not None and config.silhouette_weight > 0
            else None
        )
        self.weights = {
            name: float(getattr(config, f"{name}_weight")) for name in self.TERM_NAMES
        }

    def forward(
        self,
        *,
        hand_vertices_world: torch.Tensor,
        initial_object_rotation: torch.Tensor,
        initial_object_translation: torch.Tensor,
        rotation_delta: torch.Tensor,
        translation_delta: torch.Tensor,
        contact_vertex_indices: torch.Tensor,
        contact_group_active: torch.Tensor,
        frame_active: torch.Tensor,
        contact_vertex_weights: torch.Tensor | None = None,
        hand_vertex_active: torch.Tensor | None = None,
        object_contact_vertices: torch.Tensor | None = None,
    ) -> FixedHandObjectLossOutput:
        correction = axis_angle_to_matrix(rotation_delta)
        object_rotation = torch.bmm(correction, initial_object_rotation)
        object_translation = initial_object_translation + translation_delta
        hand_object = torch.bmm(
            hand_vertices_world - object_translation[:, None], object_rotation
        )
        sdf_values = self.sdf(hand_object)
        if object_contact_vertices is not None:
            if contact_vertex_weights is None:
                raise ValueError(
                    "Object-region contact requires HACO contact vertex weights"
                )
            contact, nearest = self.object_region_contact(
                hand_vertices_world,
                object_rotation,
                object_translation,
                object_contact_vertices,
                contact_vertex_weights,
                frame_active,
            )
        elif contact_vertex_weights is None:
            contact, nearest = self.contact(
                sdf_values, contact_vertex_indices, contact_group_active
            )
        else:
            contact, nearest = self.weighted_contact(
                sdf_values, contact_vertex_weights, frame_active
            )
        penetration, penetration_depths = self.penetration(
            sdf_values, frame_active, hand_vertex_active
        )

        if self.silhouette is None:
            silhouette = hand_vertices_world.new_zeros(())
            silhouette_error = None
        else:
            silhouette, silhouette_error = self.silhouette(
                object_rotation, object_translation
            )

        terms = {
            "contact": contact,
            "penetration": penetration,
            "silhouette": silhouette,
            "rotation_prior": self.rotation_prior(rotation_delta, frame_active),
            "translation_prior": self.translation_prior(
                translation_delta, frame_active
            ),
            "rotation_velocity": self.rotation_velocity(rotation_delta, frame_active),
            "translation_velocity": self.translation_velocity(
                translation_delta, frame_active
            ),
            "rotation_acceleration": self.rotation_acceleration(
                rotation_delta, frame_active
            ),
            "translation_acceleration": self.translation_acceleration(
                translation_delta, frame_active
            ),
            "object_rotation_acceleration": self.object_rotation_acceleration(
                object_rotation, frame_active
            ),
            "object_translation_acceleration": (
                self.object_translation_acceleration(object_translation, frame_active)
            ),
        }
        total = sum(self.weights[name] * terms[name] for name in self.TERM_NAMES)
        return FixedHandObjectLossOutput(
            total=total,
            terms=terms,
            object_rotation=object_rotation,
            object_translation=object_translation,
            hand_object=hand_object,
            sdf_values=sdf_values,
            nearest_contact_distances=nearest,
            penetration_depths=penetration_depths,
            silhouette_error=silhouette_error,
        )
