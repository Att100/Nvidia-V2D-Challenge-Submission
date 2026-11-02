"""Joint HACO-guided MANO articulation and rigid-object pose refinement."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial.transform import Rotation

from pipeline.utils import write_json

from .geometry import axis_angle_to_matrix
from .losses import FixedHandObjectPoseLoss, VoxelSDFSampler
from .mano import DifferentiableMANOSequence, project_world_points
from .optimizer import ContactPoseOptimizer, _finite_median
from .sdf import DifferentiableVoxelSDF
from .silhouette import build_constraints
from .types import (
    ContactOptimizationInput,
    JointHandObjectConfig,
    JointHandObjectOptimizationResult,
)


@dataclass(frozen=True)
class _JointRawResult:
    """In-memory outputs from one joint hand/object refinement."""

    object_to_world: np.ndarray
    rotation_vectors: np.ndarray
    translations: np.ndarray
    articulation_delta: np.ndarray
    hand_rotation_delta: np.ndarray
    hand_translation_delta: np.ndarray
    refined_vertices_world: np.ndarray
    refined_joints_world: np.ndarray
    refined_mano_hand_pose: np.ndarray
    frame_hand_active: np.ndarray
    loss_history: np.ndarray
    contact_distance_before: np.ndarray
    contact_distance_after: np.ndarray
    penetration_depth_before: np.ndarray
    penetration_depth_after: np.ndarray
    silhouette_error_before: np.ndarray
    silhouette_error_after: np.ndarray
    hand_reprojection_error_after: np.ndarray
    hand_mask_error_before: np.ndarray
    hand_mask_error_after: np.ndarray
    hand_vertex_displacement: np.ndarray
    object_object_penetration_before: np.ndarray
    object_object_penetration_after: np.ndarray
    object_pose_step_fraction: float


@dataclass(frozen=True)
class _PreparedCollisionObstacle:
    """GPU-ready fixed obstacle trajectory and differentiable SDF."""

    object_id: str
    sampler: VoxelSDFSampler
    rotation: torch.Tensor
    translation: torch.Tensor
    valid_frames: torch.Tensor


def _masked_smooth_l1(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 to zero over a leading mask with an empty-input guard."""

    if not bool(mask.any()):
        return values.new_zeros(())
    selected = values[mask]
    return F.smooth_l1_loss(selected, torch.zeros_like(selected))


@dataclass(frozen=True)
class _HandMaskConstraints:
    """Low-resolution independent left/right silhouette targets."""

    distance_fields: torch.Tensor
    boundary_xy: torch.Tensor
    boundary_valid: torch.Tensor
    active: torch.Tensor
    height: int
    width: int
    downsample: int


def _build_hand_mask_constraints(
    masks: np.ndarray | None,
    *,
    downsample: int,
    boundary_samples: int,
    device: torch.device,
) -> _HandMaskConstraints | None:
    """Precompute differentiable containment/coverage targets from SAM masks."""

    if masks is None:
        return None
    masks_np = np.asarray(masks, dtype=bool)
    if masks_np.ndim != 4:
        raise ValueError("hand_masks must have shape [T,H,image_height,image_width]")
    frame_count, hand_count, full_height, full_width = masks_np.shape
    factor = max(1, int(downsample))
    height = max(1, int(round(full_height / factor)))
    width = max(1, int(round(full_width / factor)))
    fields = np.zeros((frame_count, hand_count, height, width), dtype=np.float32)
    boundaries = np.zeros(
        (frame_count, hand_count, int(boundary_samples), 2), dtype=np.float32
    )
    boundary_valid = np.zeros(
        (frame_count, hand_count, int(boundary_samples)), dtype=bool
    )
    active = np.zeros((frame_count, hand_count), dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for frame_index in range(frame_count):
        for hand_index in range(hand_count):
            reduced = cv2.resize(
                masks_np[frame_index, hand_index].astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_AREA,
            ) > 0
            if int(reduced.sum()) < 16:
                continue
            active[frame_index, hand_index] = True
            fields[frame_index, hand_index] = cv2.distanceTransform(
                (~reduced).astype(np.uint8), cv2.DIST_L2, 3
            )
            edge = reduced & ~cv2.erode(reduced.astype(np.uint8), kernel).astype(bool)
            yy, xx = np.nonzero(edge)
            if not len(xx):
                continue
            choose = np.linspace(
                0, len(xx) - 1, min(int(boundary_samples), len(xx)), dtype=np.int64
            )
            count = len(choose)
            boundaries[frame_index, hand_index, :count, 0] = xx[choose]
            boundaries[frame_index, hand_index, :count, 1] = yy[choose]
            boundary_valid[frame_index, hand_index, :count] = True
    return _HandMaskConstraints(
        distance_fields=torch.as_tensor(fields, device=device),
        boundary_xy=torch.as_tensor(boundaries, device=device),
        boundary_valid=torch.as_tensor(boundary_valid, device=device),
        active=torch.as_tensor(active, device=device),
        height=height,
        width=width,
        downsample=factor,
    )


def _hand_mask_loss(
    vertices_world: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    constraints: _HandMaskConstraints | None,
    frame_hand_active: torch.Tensor,
    *,
    surface_samples: int,
    sigma_pixels: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric projected point-to-mask loss without a rasterizer dependency."""

    frame_count, hand_count, vertex_count = vertices_world.shape[:3]
    if constraints is None:
        return vertices_world.new_zeros(()), torch.full(
            (frame_count, hand_count), float("nan"), device=vertices_world.device
        )
    sample_count = min(max(1, int(surface_samples)), vertex_count)
    sample_ids = torch.linspace(
        0, vertex_count - 1, sample_count, device=vertices_world.device
    ).round().long()
    samples = vertices_world[:, :, sample_ids]
    pixels = project_world_points(samples, world_to_camera, intrinsics)
    pixels = pixels / float(constraints.downsample)
    flat_samples = samples.reshape(frame_count, -1, 3)
    camera = torch.einsum(
        "fij,fpj->fpi", world_to_camera[:, :3, :3], flat_samples
    ) + world_to_camera[:, None, :3, 3]
    positive_depth = camera[..., 2].reshape(frame_count, hand_count, sample_count) > 1e-5
    normalized = pixels.clone()
    normalized[..., 0] = 2.0 * normalized[..., 0] / max(constraints.width - 1, 1) - 1.0
    normalized[..., 1] = 2.0 * normalized[..., 1] / max(constraints.height - 1, 1) - 1.0
    grid = normalized.reshape(frame_count * hand_count, sample_count, 1, 2)
    sampled_distance = F.grid_sample(
        constraints.distance_fields.reshape(
            frame_count * hand_count, 1, constraints.height, constraints.width
        ),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(frame_count, hand_count, sample_count)
    outside_x = F.relu(-pixels[..., 0]) + F.relu(
        pixels[..., 0] - float(constraints.width - 1)
    )
    outside_y = F.relu(-pixels[..., 1]) + F.relu(
        pixels[..., 1] - float(constraints.height - 1)
    )
    containment_distance = sampled_distance + torch.sqrt(
        outside_x.square() + outside_y.square() + 1e-8
    )
    active = frame_hand_active & constraints.active
    containment_mask = active[..., None] & positive_depth
    containment = _masked_smooth_l1(
        containment_distance / float(sigma_pixels), containment_mask
    )

    projected_for_distance = torch.where(
        positive_depth[..., None], pixels, torch.full_like(pixels, 1.0e4)
    )
    boundary_distance = torch.cdist(
        constraints.boundary_xy.reshape(frame_count * hand_count, -1, 2),
        projected_for_distance.reshape(frame_count * hand_count, sample_count, 2),
    ).min(dim=-1).values.reshape(frame_count, hand_count, -1)
    coverage_mask = active[..., None] & constraints.boundary_valid
    coverage = _masked_smooth_l1(
        boundary_distance / float(sigma_pixels), coverage_mask
    )
    inside_sum = torch.where(
        positive_depth, containment_distance, torch.zeros_like(containment_distance)
    ).sum(dim=-1)
    inside_count = positive_depth.sum(dim=-1)
    boundary_sum = torch.where(
        constraints.boundary_valid, boundary_distance, torch.zeros_like(boundary_distance)
    ).sum(dim=-1)
    boundary_count = constraints.boundary_valid.sum(dim=-1)
    count = inside_count + boundary_count
    diagnostic = (inside_sum + boundary_sum) / count.clamp_min(1)
    diagnostic = torch.where(
        active & (count > 0), diagnostic, torch.full_like(diagnostic, float("nan"))
    )
    return 0.5 * (containment + coverage), diagnostic


class HACOJointHandObjectOptimizer(ContactPoseOptimizer):
    """Refine object SE(3), hand SE(3), and MANO articulation with HACO.

    Hand SE(3) is enabled by the Track-3 default. Its 2D evidence is the
    independent anatomical SAM mask, while the object retains the existing
    occlusion-aware silhouette constraint.
    """

    def __init__(
        self,
        config: JointHandObjectConfig | None = None,
        **kwargs: object,
    ) -> None:
        joint_config = config or JointHandObjectConfig()
        super().__init__(joint_config, **kwargs)
        self.joint_config = joint_config

    @staticmethod
    def _validate_mano_input(data: ContactOptimizationInput) -> None:
        missing = [
            name
            for name in (
                "mano_betas",
                "mano_global_orient",
                "mano_hand_pose",
                "hand_joints_2d",
            )
            if getattr(data, name) is None
        ]
        if missing:
            raise ValueError(f"Joint optimization requires MANO fields: {missing}")
        if data.contact_vertex_weights is None or data.hand_vertex_active is None:
            raise ValueError("Joint optimization requires HACO vertex associations")
        if not data.hand_labels:
            raise ValueError("Joint optimization requires anatomical hand labels")
        if data.hand_masks is None:
            raise ValueError("Joint optimization with 2D constraints requires hand_masks")

    @staticmethod
    def _filled_poses(data: ContactOptimizationInput) -> tuple[np.ndarray, np.ndarray]:
        original = data.initial_object_to_world.astype(np.float32)
        working = original.copy()
        finite = np.isfinite(working).all(axis=(1, 2))
        finite_indices = np.flatnonzero(finite)
        if not len(finite_indices):
            raise RuntimeError(f"{data.object_id} has no finite initial object pose")
        for frame_index in np.flatnonzero(~finite):
            nearest = finite_indices[np.argmin(np.abs(finite_indices - frame_index))]
            working[frame_index] = working[nearest]
        return original, working

    def _optimize_arrays(
        self,
        data: ContactOptimizationInput,
        mesh: trimesh.Trimesh,
        progress: Callable[[int, int, dict[str, float]], None] | None,
    ) -> tuple[_JointRawResult, dict[str, object]]:
        """Run Adam over object SE(3) residuals and MANO articulation."""

        self._validate_mano_input(data)
        cfg = self.joint_config
        if not data.valid_frames.any():
            raise RuntimeError(f"{data.object_id} has no valid HACO contact frames")
        device_name = cfg.device
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        device = torch.device(device_name)
        raw_vertices = np.asarray(data.hand_vertices_world, dtype=np.float32)
        raw_joints = np.asarray(data.hand_joints_world, dtype=np.float32)
        if raw_vertices.ndim != 4:
            raise ValueError("Joint optimization expects hand vertices shaped [T,H,V,3]")
        frame_count, hand_count = raw_vertices.shape[:2]
        weights_np = np.asarray(data.contact_vertex_weights, dtype=np.float32)
        vertex_active_np = np.asarray(data.hand_vertex_active, dtype=bool)
        frame_hand_active_np = (
            data.valid_frames[:, None]
            & (weights_np > 0.0).any(axis=2)
            & vertex_active_np.any(axis=2)
        )
        original_poses, poses_np = self._filled_poses(data)

        prepare_start = time.perf_counter()
        sdf = DifferentiableVoxelSDF.from_mesh(
            mesh,
            pitch=self.sdf_pitch,
            padding_voxels=self.sdf_padding_voxels,
        )
        surface_points_np, _ = trimesh.sample.sample_surface(
            mesh,
            max(32, int(cfg.object_object_surface_samples)),
            seed=0,
        )
        surface_points = torch.as_tensor(
            np.asarray(surface_points_np, dtype=np.float32), device=device
        )
        collision_obstacles: list[_PreparedCollisionObstacle] = []
        if cfg.object_object_penetration_sigma <= 0.0:
            raise ValueError("object_object_penetration_sigma must be positive")
        if cfg.object_object_penetration_tolerance < 0.0:
            raise ValueError("object_object_penetration_tolerance cannot be negative")
        configured_obstacles = (
            data.collision_obstacles
            if cfg.object_object_penetration_weight > 0.0
            else ()
        )
        for obstacle in configured_obstacles:
            obstacle_mesh = trimesh.load(obstacle.mesh_path, force="mesh", process=False)
            if not isinstance(obstacle_mesh, trimesh.Trimesh):
                raise TypeError(f"Expected a single mesh: {obstacle.mesh_path}")
            obstacle_sdf = DifferentiableVoxelSDF.from_mesh(
                obstacle_mesh,
                pitch=self.sdf_pitch,
                padding_voxels=self.sdf_padding_voxels,
            )
            obstacle_poses = np.asarray(obstacle.object_to_world, dtype=np.float32)
            if obstacle_poses.shape != (frame_count, 4, 4):
                raise ValueError(
                    f"Collision trajectory for {obstacle.object_id} has shape "
                    f"{obstacle_poses.shape}, expected {(frame_count, 4, 4)}"
                )
            obstacle_valid = np.asarray(obstacle.valid_frames, dtype=bool)
            if obstacle_valid.shape != (frame_count,):
                raise ValueError(
                    f"Collision validity for {obstacle.object_id} has shape "
                    f"{obstacle_valid.shape}, expected {(frame_count,)}"
                )
            collision_obstacles.append(
                _PreparedCollisionObstacle(
                    object_id=obstacle.object_id,
                    sampler=VoxelSDFSampler.from_sdf(obstacle_sdf).to(device),
                    rotation=torch.as_tensor(obstacle_poses[:, :3, :3], device=device),
                    translation=torch.as_tensor(obstacle_poses[:, :3, 3], device=device),
                    valid_frames=torch.as_tensor(
                        obstacle_valid, device=device
                    ),
                )
            )
        constraints, silhouette_report = build_constraints(
            mesh=mesh,
            masks=data.masks,
            intrinsics=data.intrinsics,
            world_to_camera=data.world_to_camera,
            valid_frames=data.valid_frames,
            downsample=self.silhouette_downsample,
            surface_sample_count=self.silhouette_surface_samples,
            boundary_sample_count=self.silhouette_boundary_samples,
            occluder_dilation=self.silhouette_occluder_dilation,
        )
        hand_mask_constraints = _build_hand_mask_constraints(
            data.hand_masks,
            downsample=cfg.hand_mask_downsample,
            boundary_samples=cfg.hand_mask_boundary_samples,
            device=device,
        )
        mano_root = Path(cfg.mano_model_root)
        if not mano_root.is_absolute():
            mano_root = Path(__file__).resolve().parents[3] / mano_root
        mano = DifferentiableMANOSequence(
            model_root=mano_root,
            hand_labels=data.hand_labels,
            vertices_world=raw_vertices,
            joints_world=raw_joints,
            mano_betas=np.asarray(data.mano_betas, dtype=np.float32),
            mano_global_orient=np.asarray(data.mano_global_orient, dtype=np.float32),
            mano_hand_pose=np.asarray(data.mano_hand_pose, dtype=np.float32),
            frame_hand_active=frame_hand_active_np,
            device=device,
        )
        preparation_seconds = time.perf_counter() - prepare_start

        initial_rotation = torch.as_tensor(poses_np[:, :3, :3], device=device)
        initial_translation = torch.as_tensor(poses_np[:, :3, 3], device=device)
        weights = torch.as_tensor(
            weights_np.reshape(frame_count, -1), dtype=torch.float32, device=device
        )
        vertex_active = torch.as_tensor(
            vertex_active_np.reshape(frame_count, -1), dtype=torch.bool, device=device
        )
        valid = torch.as_tensor(data.valid_frames, dtype=torch.bool, device=device)
        frame_hand_active = torch.as_tensor(
            frame_hand_active_np, dtype=torch.bool, device=device
        )
        world_to_camera = torch.as_tensor(data.world_to_camera, device=device)
        intrinsics = torch.as_tensor(data.intrinsics, device=device)
        objective = FixedHandObjectPoseLoss(
            sdf=sdf,
            config=cfg,
            silhouette_constraints=constraints,
        ).to(device)
        rotation_delta = torch.nn.Parameter(torch.zeros((frame_count, 3), device=device))
        translation_delta = torch.nn.Parameter(torch.zeros((frame_count, 3), device=device))
        articulation_delta = torch.nn.Parameter(
            torch.zeros((frame_count, hand_count, 45), device=device)
        )
        hand_rotation_delta = torch.nn.Parameter(
            torch.zeros((frame_count, hand_count, 3), device=device)
        )
        hand_translation_delta = torch.nn.Parameter(
            torch.zeros((frame_count, hand_count, 3), device=device)
        )
        hand_rigid_parameters = (
            [hand_rotation_delta, hand_translation_delta]
            if cfg.optimize_hand_se3
            else []
        )
        parameter_groups: list[dict[str, object]] = [
            {
                "params": [rotation_delta, translation_delta],
                "lr": float(cfg.learning_rate),
            },
            {"params": [articulation_delta], "lr": float(cfg.hand_learning_rate)},
        ]
        if hand_rigid_parameters:
            parameter_groups.append(
                {
                    "params": hand_rigid_parameters,
                    "lr": float(cfg.hand_se3_learning_rate),
                }
            )
        optimizer = torch.optim.Adam(parameter_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(cfg.iterations)), eta_min=2.0e-5
        )
        with torch.no_grad():
            target_joint_pixels = torch.as_tensor(
                np.asarray(data.hand_joints_2d, dtype=np.float32), device=device
            )
            target_joint_finite = torch.isfinite(target_joint_pixels).all(dim=(-1, -2))
            reprojection_active = frame_hand_active & target_joint_finite
            target_joint_pixels = torch.nan_to_num(target_joint_pixels)

        history: list[list[float]] = []
        optimization_start = time.perf_counter()

        def object_object_penetration(
            current_rotation: torch.Tensor,
            current_translation: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return symmetric-pass collision loss and per-frame max depth.

            Surface contact has zero cost. Only samples more than the configured
            tolerance inside another object's watertight voxel SDF are active.
            """

            if not collision_obstacles:
                return current_translation.new_zeros(()), torch.full(
                    (frame_count,),
                    float("nan"),
                    device=device,
                    dtype=current_translation.dtype,
                )
            current_world = (
                torch.einsum("fij,pj->fpi", current_rotation, surface_points)
                + current_translation[:, None]
            )
            losses: list[torch.Tensor] = []
            diagnostics: list[torch.Tensor] = []
            for obstacle in collision_obstacles:
                points_other = torch.einsum(
                    "fji,fpj->fpi",
                    obstacle.rotation,
                    current_world - obstacle.translation[:, None],
                )
                signed_distance = obstacle.sampler(points_other)
                penetration = torch.relu(
                    -signed_distance
                    - float(cfg.object_object_penetration_tolerance)
                )
                active = valid & obstacle.valid_frames
                # Normalize over penetrating samples only. Averaging over the
                # complete surface would let a small but deep collision vanish
                # among hundreds of exactly-zero samples.
                penetrating = active[:, None] & (penetration > 0.0)
                losses.append(
                    _masked_smooth_l1(
                        penetration / float(cfg.object_object_penetration_sigma),
                        penetrating,
                    )
                )
                per_frame = penetration.max(dim=1).values
                diagnostics.append(
                    torch.where(
                        active,
                        per_frame,
                        torch.full_like(per_frame, float("nan")),
                    )
                )
            stacked = torch.stack(diagnostics, dim=1)
            finite = torch.isfinite(stacked)
            per_frame = torch.where(
                finite.any(dim=1),
                torch.where(finite, stacked, torch.zeros_like(stacked)).max(dim=1).values,
                torch.full((frame_count,), float("nan"), device=device),
            )
            return torch.stack(losses).mean(), per_frame

        def refined_hand_geometry() -> tuple[torch.Tensor, torch.Tensor]:
            articulated_vertices, articulated_joints = mano(articulation_delta)
            if not cfg.optimize_hand_se3:
                return articulated_vertices, articulated_joints
            wrist_index = 5 if articulated_joints.shape[2] == 20 else 0
            pivot = articulated_joints[:, :, wrist_index]
            correction = axis_angle_to_matrix(hand_rotation_delta)
            refined_vertices = torch.einsum(
                "fhij,fhvj->fhvi",
                correction,
                articulated_vertices - pivot[:, :, None],
            ) + pivot[:, :, None] + hand_translation_delta[:, :, None]
            refined_joints = torch.einsum(
                "fhij,fhkj->fhki",
                correction,
                articulated_joints - pivot[:, :, None],
            ) + pivot[:, :, None] + hand_translation_delta[:, :, None]
            return refined_vertices, refined_joints

        for iteration in range(max(1, int(cfg.iterations))):
            optimizer.zero_grad(set_to_none=True)
            refined_vertices, refined_joints = refined_hand_geometry()
            base_output = objective(
                hand_vertices_world=refined_vertices.reshape(frame_count, -1, 3),
                initial_object_rotation=initial_rotation,
                initial_object_translation=initial_translation,
                rotation_delta=rotation_delta,
                translation_delta=translation_delta,
                contact_vertex_indices=torch.zeros(
                    (frame_count, 1, 1), dtype=torch.long, device=device
                ),
                contact_group_active=torch.zeros(
                    (frame_count, 1), dtype=torch.bool, device=device
                ),
                frame_active=valid,
                contact_vertex_weights=weights,
                hand_vertex_active=vertex_active,
            )
            current_rotation = torch.bmm(
                axis_angle_to_matrix(rotation_delta), initial_rotation
            )
            current_translation = initial_translation + translation_delta
            object_object_loss, _ = object_object_penetration(
                current_rotation, current_translation
            )
            articulation_prior = _masked_smooth_l1(
                articulation_delta / cfg.articulation_prior_sigma_radians,
                frame_hand_active,
            )
            adjacent = frame_hand_active[1:] & frame_hand_active[:-1]
            articulation_velocity = _masked_smooth_l1(
                (articulation_delta[1:] - articulation_delta[:-1])
                / cfg.articulation_velocity_sigma_radians,
                adjacent,
            )
            triplets = (
                frame_hand_active[2:]
                & frame_hand_active[1:-1]
                & frame_hand_active[:-2]
            )
            articulation_acceleration = _masked_smooth_l1(
                (
                    articulation_delta[2:]
                    - 2.0 * articulation_delta[1:-1]
                    + articulation_delta[:-2]
                )
                / cfg.articulation_acceleration_sigma_radians,
                triplets,
            )
            joint_pixels = project_world_points(
                refined_joints, world_to_camera, intrinsics
            )
            reprojection_residual = (
                joint_pixels - target_joint_pixels
            ) / cfg.hand_reprojection_sigma_pixels
            hand_reprojection = _masked_smooth_l1(
                reprojection_residual, reprojection_active
            )
            hand_mask, _ = _hand_mask_loss(
                refined_vertices,
                world_to_camera,
                intrinsics,
                hand_mask_constraints,
                frame_hand_active,
                surface_samples=cfg.hand_mask_surface_samples,
                sigma_pixels=cfg.hand_mask_sigma_pixels,
            )
            hand_rotation_prior = _masked_smooth_l1(
                hand_rotation_delta / cfg.hand_rotation_prior_sigma_radians,
                frame_hand_active,
            )
            hand_translation_prior = _masked_smooth_l1(
                hand_translation_delta / cfg.hand_translation_prior_sigma,
                frame_hand_active,
            )
            hand_rotation_velocity = _masked_smooth_l1(
                (hand_rotation_delta[1:] - hand_rotation_delta[:-1])
                / cfg.hand_rotation_velocity_sigma_radians,
                adjacent,
            )
            hand_translation_velocity = _masked_smooth_l1(
                (hand_translation_delta[1:] - hand_translation_delta[:-1])
                / cfg.hand_translation_velocity_sigma,
                adjacent,
            )
            hand_rotation_acceleration = _masked_smooth_l1(
                (
                    hand_rotation_delta[2:]
                    - 2.0 * hand_rotation_delta[1:-1]
                    + hand_rotation_delta[:-2]
                )
                / cfg.hand_rotation_acceleration_sigma_radians,
                triplets,
            )
            hand_translation_acceleration = _masked_smooth_l1(
                (
                    hand_translation_delta[2:]
                    - 2.0 * hand_translation_delta[1:-1]
                    + hand_translation_delta[:-2]
                )
                / cfg.hand_translation_acceleration_sigma,
                triplets,
            )
            total = (
                base_output.total
                + cfg.articulation_prior_weight * articulation_prior
                + cfg.articulation_velocity_weight * articulation_velocity
                + cfg.articulation_acceleration_weight * articulation_acceleration
                + cfg.hand_reprojection_weight * hand_reprojection
                + cfg.hand_mask_weight * hand_mask
                + cfg.hand_rotation_prior_weight * hand_rotation_prior
                + cfg.hand_translation_prior_weight * hand_translation_prior
                + cfg.hand_rotation_velocity_weight * hand_rotation_velocity
                + cfg.hand_translation_velocity_weight * hand_translation_velocity
                + cfg.hand_rotation_acceleration_weight * hand_rotation_acceleration
                + cfg.hand_translation_acceleration_weight * hand_translation_acceleration
                + cfg.object_object_penetration_weight * object_object_loss
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    rotation_delta,
                    translation_delta,
                    articulation_delta,
                    *hand_rigid_parameters,
                ],
                max_norm=5.0,
            )
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                rotation_norm = torch.linalg.norm(rotation_delta, dim=-1, keepdim=True)
                rotation_delta.mul_(
                    float(cfg.maximum_rotation_radians)
                    / rotation_norm.clamp_min(float(cfg.maximum_rotation_radians))
                )
                translation_norm = torch.linalg.norm(
                    translation_delta, dim=-1, keepdim=True
                )
                translation_delta.mul_(
                    float(cfg.maximum_translation)
                    / translation_norm.clamp_min(float(cfg.maximum_translation))
                )
                articulation_delta.clamp_(
                    -float(cfg.maximum_articulation_radians),
                    float(cfg.maximum_articulation_radians),
                )
                hand_rotation_norm = torch.linalg.norm(
                    hand_rotation_delta, dim=-1, keepdim=True
                )
                hand_rotation_delta.mul_(
                    float(cfg.maximum_hand_rotation_radians)
                    / hand_rotation_norm.clamp_min(
                        float(cfg.maximum_hand_rotation_radians)
                    )
                )
                hand_translation_norm = torch.linalg.norm(
                    hand_translation_delta, dim=-1, keepdim=True
                )
                hand_translation_delta.mul_(
                    float(cfg.maximum_hand_translation)
                    / hand_translation_norm.clamp_min(
                        float(cfg.maximum_hand_translation)
                    )
                )
                rotation_delta[~valid] = 0.0
                translation_delta[~valid] = 0.0
                articulation_delta[~frame_hand_active] = 0.0
                hand_rotation_delta[~frame_hand_active] = 0.0
                hand_translation_delta[~frame_hand_active] = 0.0
            values = {"total": float(total.detach())}
            values.update(
                {
                    name: float(base_output.terms[name].detach())
                    for name in objective.TERM_NAMES
                }
            )
            values.update(
                articulation_prior=float(articulation_prior.detach()),
                articulation_velocity=float(articulation_velocity.detach()),
                articulation_acceleration=float(articulation_acceleration.detach()),
                hand_reprojection=float(hand_reprojection.detach()),
                hand_mask=float(hand_mask.detach()),
                hand_rotation_prior=float(hand_rotation_prior.detach()),
                hand_translation_prior=float(hand_translation_prior.detach()),
                hand_rotation_velocity=float(hand_rotation_velocity.detach()),
                hand_translation_velocity=float(hand_translation_velocity.detach()),
                hand_rotation_acceleration=float(hand_rotation_acceleration.detach()),
                hand_translation_acceleration=float(hand_translation_acceleration.detach()),
                object_object_penetration=float(object_object_loss.detach()),
            )
            ordered_names = (
                *objective.TERM_NAMES,
                "articulation_prior",
                "articulation_velocity",
                "articulation_acceleration",
                "hand_reprojection",
                "hand_mask",
                "hand_rotation_prior",
                "hand_translation_prior",
                "hand_rotation_velocity",
                "hand_translation_velocity",
                "hand_rotation_acceleration",
                "hand_translation_acceleration",
                "object_object_penetration",
            )
            history.append([values["total"], *(values[name] for name in ordered_names)])
            if progress and (
                iteration == 0
                or (iteration + 1) % 50 == 0
                or iteration + 1 == cfg.iterations
            ):
                progress(iteration + 1, cfg.iterations, values)
        optimization_seconds = time.perf_counter() - optimization_start

        with torch.no_grad():
            refined_vertices_t, refined_joints_t = refined_hand_geometry()
            _, hand_mask_before_t = _hand_mask_loss(
                torch.as_tensor(np.nan_to_num(raw_vertices), device=device),
                world_to_camera,
                intrinsics,
                hand_mask_constraints,
                frame_hand_active,
                surface_samples=cfg.hand_mask_surface_samples,
                sigma_pixels=cfg.hand_mask_sigma_pixels,
            )
            _, hand_mask_after_t = _hand_mask_loss(
                refined_vertices_t,
                world_to_camera,
                intrinsics,
                hand_mask_constraints,
                frame_hand_active,
                surface_samples=cfg.hand_mask_surface_samples,
                sigma_pixels=cfg.hand_mask_sigma_pixels,
            )
            # The optimizer's aggregate objective may improve contact by moving
            # the object out of its observed silhouette. Select the largest
            # sequence-wide pose step that respects both median and tail 2D
            # degradation bounds. A single scalar preserves temporal motion and
            # avoids per-frame accept/reject pose jumps.
            _, silhouette_before_t = objective.silhouette(
                initial_rotation, initial_translation
            )
            before_np = silhouette_before_t.detach().cpu().numpy()
            guard_valid = np.asarray(data.valid_frames, dtype=bool) & np.isfinite(before_np)
            before_median = float(np.median(before_np[guard_valid]))
            before_p90 = float(np.quantile(before_np[guard_valid], 0.9))
            selected_fraction = 0.0
            selected_silhouette = silhouette_before_t
            guard_steps = max(1, int(cfg.silhouette_guard_steps))
            for fraction in np.linspace(1.0, 0.0, guard_steps + 1):
                fraction_value = float(fraction)
                candidate_rotation = torch.bmm(
                    axis_angle_to_matrix(rotation_delta * fraction_value),
                    initial_rotation,
                )
                candidate_translation = (
                    initial_translation + translation_delta * fraction_value
                )
                _, candidate_silhouette = objective.silhouette(
                    candidate_rotation, candidate_translation
                )
                candidate_np = candidate_silhouette.detach().cpu().numpy()
                candidate_valid = guard_valid & np.isfinite(candidate_np)
                if not candidate_valid.any():
                    continue
                if (
                    float(np.median(candidate_np[candidate_valid]))
                    <= before_median + float(cfg.silhouette_guard_median_tolerance)
                    and float(np.quantile(candidate_np[candidate_valid], 0.9))
                    <= before_p90 + float(cfg.silhouette_guard_p90_tolerance)
                ):
                    selected_fraction = fraction_value
                    selected_silhouette = candidate_silhouette
                    break
            accepted_rotation_delta = rotation_delta * selected_fraction
            accepted_translation_delta = translation_delta * selected_fraction
            correction = axis_angle_to_matrix(accepted_rotation_delta)
            refined_working = poses_np.copy()
            refined_working[:, :3, :3] = torch.bmm(
                correction, initial_rotation
            ).cpu().numpy()
            refined_working[:, :3, 3] = (
                initial_translation + accepted_translation_delta
            ).cpu().numpy()
            _, object_object_before_t = object_object_penetration(
                initial_rotation, initial_translation
            )
            _, object_object_after_t = object_object_penetration(
                torch.bmm(correction, initial_rotation),
                initial_translation + accepted_translation_delta,
            )
            joint_pixels = project_world_points(
                refined_joints_t, world_to_camera, intrinsics
            )
            reprojection_error = torch.linalg.norm(
                joint_pixels - target_joint_pixels, dim=-1
            ).cpu().numpy().astype(np.float32)
            reprojection_error[~reprojection_active.cpu().numpy()] = np.nan

        hand_before = np.nan_to_num(raw_vertices).reshape(frame_count, -1, 3)
        hand_after = refined_vertices_t.detach().cpu().numpy().astype(np.float32)
        hand_after_flat = hand_after.reshape(frame_count, -1, 3)
        dummy_indices = np.zeros((frame_count, 1, 1), dtype=np.int64)
        dummy_groups = np.zeros((frame_count, 1), dtype=bool)
        before = self._evaluate(
            hand_before,
            poses_np,
            sdf,
            dummy_indices,
            dummy_groups,
            data.valid_frames,
            device,
            weights_np.reshape(frame_count, -1),
            vertex_active_np.reshape(frame_count, -1),
        )
        after = self._evaluate(
            hand_after_flat,
            refined_working,
            sdf,
            dummy_indices,
            dummy_groups,
            data.valid_frames,
            device,
            weights_np.reshape(frame_count, -1),
            vertex_active_np.reshape(frame_count, -1),
        )
        silhouette_after_t = selected_silhouette
        refined = refined_working.copy()
        refined[~data.valid_frames] = original_poses[~data.valid_frames]
        articulation_np = articulation_delta.detach().cpu().numpy().astype(np.float32)
        refined_pose_matrices = np.asarray(data.mano_hand_pose, dtype=np.float32).copy()
        for hand_index in range(hand_count):
            slots = np.flatnonzero(frame_hand_active_np[:, hand_index])
            if not len(slots):
                continue
            if raw_joints.shape[2] == 20:
                base_axis_angle = Rotation.from_matrix(
                    refined_pose_matrices[slots, hand_index].reshape(-1, 3, 3)
                ).as_rotvec().reshape(len(slots), 15, 3)
                refined_pose_matrices[slots, hand_index] = Rotation.from_rotvec(
                    (base_axis_angle + articulation_np[slots, hand_index].reshape(
                        len(slots), 15, 3
                    )).reshape(-1, 3)
                ).as_matrix().reshape(len(slots), 15, 3, 3)
            else:
                correction = Rotation.from_rotvec(
                    articulation_np[slots, hand_index].reshape(-1, 3)
                ).as_matrix().reshape(len(slots), 15, 3, 3)
                refined_pose_matrices[slots, hand_index] = np.matmul(
                    correction,
                    refined_pose_matrices[slots, hand_index],
                )
        vertex_displacement = np.linalg.norm(
            hand_after - np.nan_to_num(raw_vertices), axis=-1
        ).astype(np.float32)
        raw = _JointRawResult(
            object_to_world=refined.astype(np.float32),
            rotation_vectors=accepted_rotation_delta.detach().cpu().numpy().astype(np.float32),
            translations=accepted_translation_delta.detach().cpu().numpy().astype(np.float32),
            articulation_delta=articulation_np,
            hand_rotation_delta=hand_rotation_delta.detach().cpu().numpy().astype(np.float32),
            hand_translation_delta=hand_translation_delta.detach().cpu().numpy().astype(np.float32),
            refined_vertices_world=hand_after,
            refined_joints_world=refined_joints_t.detach().cpu().numpy().astype(np.float32),
            refined_mano_hand_pose=refined_pose_matrices,
            frame_hand_active=frame_hand_active_np,
            loss_history=np.asarray(history, dtype=np.float32),
            contact_distance_before=before[0],
            contact_distance_after=after[0],
            penetration_depth_before=before[1],
            penetration_depth_after=after[1],
            silhouette_error_before=silhouette_before_t.cpu().numpy().astype(np.float32),
            silhouette_error_after=silhouette_after_t.cpu().numpy().astype(np.float32),
            hand_reprojection_error_after=reprojection_error,
            hand_mask_error_before=hand_mask_before_t.cpu().numpy().astype(np.float32),
            hand_mask_error_after=hand_mask_after_t.cpu().numpy().astype(np.float32),
            hand_vertex_displacement=vertex_displacement,
            object_object_penetration_before=(
                object_object_before_t.cpu().numpy().astype(np.float32)
            ),
            object_object_penetration_after=(
                object_object_after_t.cpu().numpy().astype(np.float32)
            ),
            object_pose_step_fraction=selected_fraction,
        )
        timing = {
            "preparation_seconds": preparation_seconds,
            "optimization_seconds": optimization_seconds,
            "device": str(device),
            "sdf_shape_xyz": list(sdf.shape_xyz),
            "silhouette": asdict(silhouette_report),
            "mano_model_root": str(mano_root),
            "object_pose_step_fraction": selected_fraction,
            "collision_obstacles": [item.object_id for item in collision_obstacles],
        }
        return raw, timing

    def run(
        self,
        data: ContactOptimizationInput,
        *,
        progress: Callable[[int, int, dict[str, float]], None] | None = None,
    ) -> JointHandObjectOptimizationResult:
        """Optimize one object/hand contact sequence and save typed artifacts."""

        data.output_dir.mkdir(parents=True, exist_ok=True)
        self._validate_mano_input(data)
        if not np.asarray(data.valid_frames, dtype=bool).any():
            frame_count = len(data.valid_frames)
            hand_vertices = np.asarray(data.hand_vertices_world, dtype=np.float32)
            hand_joints = np.asarray(data.hand_joints_world, dtype=np.float32)
            hand_count = hand_vertices.shape[1]
            poses_path = data.output_dir / "poses.npz"
            metrics_path = data.output_dir / "per_frame_metrics.npz"
            hand_path = data.output_dir / "hand_articulation.npz"
            report_path = data.output_dir / "report.json"
            unchanged = np.asarray(data.initial_object_to_world, dtype=np.float32)
            frame_hand_active = np.zeros((frame_count, hand_count), dtype=bool)
            articulation_delta = np.zeros(
                (frame_count, hand_count, 15, 3), dtype=np.float32
            )
            nan_metric = np.full(frame_count, np.nan, dtype=np.float32)
            np.savez_compressed(
                poses_path,
                object_to_world=unchanged,
                initial_object_to_world=unchanged,
                rotation_delta=np.zeros((frame_count, 3), dtype=np.float32),
                translation_delta=np.zeros((frame_count, 3), dtype=np.float32),
                valid=np.asarray(data.valid_frames, dtype=bool),
                contact_active=np.asarray(data.contact_active, dtype=bool),
                source_indices=data.source_indices,
            )
            np.savez_compressed(
                hand_path,
                vertices_world=hand_vertices,
                joints_world=hand_joints,
                mano_hand_pose=np.asarray(data.mano_hand_pose, dtype=np.float32),
                articulation_delta=articulation_delta,
                hand_rotation_delta=np.zeros((frame_count, hand_count, 3), dtype=np.float32),
                hand_translation_delta=np.zeros((frame_count, hand_count, 3), dtype=np.float32),
                frame_hand_active=frame_hand_active,
                hand_labels=np.asarray(data.hand_labels),
                source_indices=data.source_indices,
            )
            np.savez_compressed(
                metrics_path,
                source_indices=data.source_indices,
                valid=np.asarray(data.valid_frames, dtype=bool),
                frame_hand_active=frame_hand_active,
                contact_distance_before=nan_metric,
                contact_distance_after=nan_metric,
                penetration_depth_before=nan_metric,
                penetration_depth_after=nan_metric,
                silhouette_error_before=nan_metric,
                silhouette_error_after=nan_metric,
                hand_reprojection_error_after=np.full(
                    hand_joints.shape[:-1], np.nan, dtype=np.float32
                ),
                hand_mask_error_before=np.full(
                    (frame_count, hand_count), np.nan, dtype=np.float32
                ),
                hand_mask_error_after=np.full(
                    (frame_count, hand_count), np.nan, dtype=np.float32
                ),
                hand_vertex_displacement=np.zeros(
                    hand_vertices.shape[:-1], dtype=np.float32
                ),
                object_object_penetration_before=nan_metric,
                object_object_penetration_after=nan_metric,
                loss_history=np.empty((0, 0), dtype=np.float32),
                loss_columns=np.asarray(
                    (
                        "total",
                        *FixedHandObjectPoseLoss.TERM_NAMES,
                        "articulation_prior",
                        "articulation_velocity",
                        "articulation_acceleration",
                        "hand_reprojection",
                        "hand_mask",
                        "hand_rotation_prior",
                        "hand_translation_prior",
                        "hand_rotation_velocity",
                        "hand_translation_velocity",
                        "hand_rotation_acceleration",
                        "hand_translation_acceleration",
                        "object_object_penetration",
                    )
                ),
            )
            metrics = {
                "contact_distance_before_median": None,
                "contact_distance_after_median": None,
                "penetration_depth_before_median": None,
                "penetration_depth_after_median": None,
                "silhouette_error_before_median_lowres_px": None,
                "silhouette_error_after_median_lowres_px": None,
                "translation_delta_median": None,
                "rotation_delta_degrees_median": None,
                "articulation_delta_degrees_median": None,
                "articulation_delta_degrees_max": None,
                "hand_reprojection_error_median_pixels": None,
                "hand_reprojection_error_p90_pixels": None,
                "hand_mask_error_before_median_lowres_px": None,
                "hand_mask_error_after_median_lowres_px": None,
                "hand_vertex_displacement_median_m": None,
                "hand_vertex_displacement_p90_m": None,
                "object_pose_step_fraction": 0.0,
                "object_object_penetration_before_max_m": None,
                "object_object_penetration_after_max_m": None,
            }
            write_json(
                report_path,
                {
                    "schema_version": "ego-scene-hoi-haco-joint-hand-object-v1",
                    "object_id": data.object_id,
                    "object_name": data.object_name,
                    "hand_labels": list(data.hand_labels),
                    "frame_count": frame_count,
                    "optimized_frame_count": 0,
                    "optimized_hand_frame_count": 0,
                    "status": "skipped_no_valid_contact_frames",
                    "contact_selection": "haco_dense_vertex_weights",
                    "config": asdict(self.joint_config),
                    "timing": {
                        "preparation_seconds": 0.0,
                        "optimization_seconds": 0.0,
                        "device": None,
                        "sdf_shape_xyz": None,
                        "silhouette": None,
                        "mano_model_root": None,
                        "object_pose_step_fraction": 0.0,
                    },
                    "metrics": metrics,
                    "artifacts": {
                        "poses": str(poses_path),
                        "hand_articulation": str(hand_path),
                        "per_frame_metrics": str(metrics_path),
                    },
                },
            )
            return JointHandObjectOptimizationResult(
                object_id=data.object_id,
                poses_path=poses_path,
                metrics_path=metrics_path,
                report_path=report_path,
                active_frame_count=int(data.contact_active.sum()),
                optimized_frame_count=0,
                contact_distance_before_median=None,
                contact_distance_after_median=None,
                silhouette_error_before_median=None,
                silhouette_error_after_median=None,
                hand_articulation_path=hand_path,
                articulation_delta_degrees_median=None,
                hand_reprojection_error_median_pixels=None,
            )

        mesh = trimesh.load(data.mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected a single mesh: {data.mesh_path}")
        raw, timing = self._optimize_arrays(data, mesh, progress)
        poses_path = data.output_dir / "poses.npz"
        metrics_path = data.output_dir / "per_frame_metrics.npz"
        hand_path = data.output_dir / "hand_articulation.npz"
        report_path = data.output_dir / "report.json"
        np.savez_compressed(
            poses_path,
            object_to_world=raw.object_to_world,
            initial_object_to_world=data.initial_object_to_world.astype(np.float32),
            rotation_delta=raw.rotation_vectors,
            translation_delta=raw.translations,
            valid=data.valid_frames,
            contact_active=data.contact_active,
            source_indices=data.source_indices,
        )
        np.savez_compressed(
            hand_path,
            vertices_world=raw.refined_vertices_world,
            joints_world=raw.refined_joints_world,
            mano_hand_pose=raw.refined_mano_hand_pose,
            articulation_delta=raw.articulation_delta,
            hand_rotation_delta=raw.hand_rotation_delta,
            hand_translation_delta=raw.hand_translation_delta,
            frame_hand_active=raw.frame_hand_active,
            hand_labels=np.asarray(data.hand_labels),
            source_indices=data.source_indices,
        )
        np.savez_compressed(
            metrics_path,
            source_indices=data.source_indices,
            valid=data.valid_frames,
            frame_hand_active=raw.frame_hand_active,
            contact_distance_before=raw.contact_distance_before,
            contact_distance_after=raw.contact_distance_after,
            penetration_depth_before=raw.penetration_depth_before,
            penetration_depth_after=raw.penetration_depth_after,
            silhouette_error_before=raw.silhouette_error_before,
            silhouette_error_after=raw.silhouette_error_after,
            hand_reprojection_error_after=raw.hand_reprojection_error_after,
            hand_mask_error_before=raw.hand_mask_error_before,
            hand_mask_error_after=raw.hand_mask_error_after,
            hand_vertex_displacement=raw.hand_vertex_displacement,
            object_object_penetration_before=raw.object_object_penetration_before,
            object_object_penetration_after=raw.object_object_penetration_after,
            loss_history=raw.loss_history,
            loss_columns=np.asarray(
                (
                    "total",
                    *FixedHandObjectPoseLoss.TERM_NAMES,
                    "articulation_prior",
                    "articulation_velocity",
                    "articulation_acceleration",
                    "hand_reprojection",
                    "hand_mask",
                    "hand_rotation_prior",
                    "hand_translation_prior",
                    "hand_rotation_velocity",
                    "hand_translation_velocity",
                    "hand_rotation_acceleration",
                    "hand_translation_acceleration",
                    "object_object_penetration",
                )
            ),
        )
        valid = data.valid_frames
        articulation_degrees = np.rad2deg(np.abs(raw.articulation_delta))
        active_articulation = np.broadcast_to(
            raw.frame_hand_active[..., None], articulation_degrees.shape
        )
        active_reprojection = np.broadcast_to(
            raw.frame_hand_active[..., None], raw.hand_reprojection_error_after.shape
        )
        active_vertices = np.broadcast_to(
            raw.frame_hand_active[..., None], raw.hand_vertex_displacement.shape
        )
        metrics = {
            "contact_distance_before_median": _finite_median(raw.contact_distance_before, valid),
            "contact_distance_after_median": _finite_median(raw.contact_distance_after, valid),
            "penetration_depth_before_median": _finite_median(raw.penetration_depth_before, valid),
            "penetration_depth_after_median": _finite_median(raw.penetration_depth_after, valid),
            "silhouette_error_before_median_lowres_px": _finite_median(raw.silhouette_error_before, valid),
            "silhouette_error_after_median_lowres_px": _finite_median(raw.silhouette_error_after, valid),
            "translation_delta_median": _finite_median(np.linalg.norm(raw.translations, axis=1), valid),
            "rotation_delta_degrees_median": _finite_median(np.rad2deg(np.linalg.norm(raw.rotation_vectors, axis=1)), valid),
            "articulation_delta_degrees_median": _finite_median(articulation_degrees, active_articulation),
            "articulation_delta_degrees_max": float(np.max(articulation_degrees[active_articulation])) if active_articulation.any() else None,
            "hand_rotation_delta_degrees_median": _finite_median(
                np.rad2deg(np.linalg.norm(raw.hand_rotation_delta, axis=-1)),
                raw.frame_hand_active,
            ),
            "hand_translation_delta_median_m": _finite_median(
                np.linalg.norm(raw.hand_translation_delta, axis=-1),
                raw.frame_hand_active,
            ),
            "hand_reprojection_error_median_pixels": _finite_median(raw.hand_reprojection_error_after, active_reprojection),
            "hand_reprojection_error_p90_pixels": float(np.quantile(raw.hand_reprojection_error_after[active_reprojection], 0.9)) if active_reprojection.any() else None,
            "hand_mask_error_before_median_lowres_px": _finite_median(
                raw.hand_mask_error_before, raw.frame_hand_active
            ),
            "hand_mask_error_after_median_lowres_px": _finite_median(
                raw.hand_mask_error_after, raw.frame_hand_active
            ),
            "hand_vertex_displacement_median_m": _finite_median(raw.hand_vertex_displacement, active_vertices),
            "hand_vertex_displacement_p90_m": float(np.quantile(raw.hand_vertex_displacement[active_vertices], 0.9)) if active_vertices.any() else None,
            "object_pose_step_fraction": raw.object_pose_step_fraction,
            "object_object_penetration_before_max_m": (
                float(np.nanmax(raw.object_object_penetration_before))
                if np.isfinite(raw.object_object_penetration_before).any()
                else None
            ),
            "object_object_penetration_after_max_m": (
                float(np.nanmax(raw.object_object_penetration_after))
                if np.isfinite(raw.object_object_penetration_after).any()
                else None
            ),
            "object_object_penetration_before_p95_m": (
                float(np.nanquantile(raw.object_object_penetration_before, 0.95))
                if np.isfinite(raw.object_object_penetration_before).any()
                else None
            ),
            "object_object_penetration_after_p95_m": (
                float(np.nanquantile(raw.object_object_penetration_after, 0.95))
                if np.isfinite(raw.object_object_penetration_after).any()
                else None
            ),
        }
        report = {
            "schema_version": "ego-scene-hoi-haco-joint-hand-object-v1",
            "object_id": data.object_id,
            "object_name": data.object_name,
            "hand_labels": list(data.hand_labels),
            "frame_count": len(valid),
            "optimized_frame_count": int(valid.sum()),
            "optimized_hand_frame_count": int(raw.frame_hand_active.sum()),
            "contact_selection": "haco_dense_vertex_weights",
            "hand_parameterization": (
                "MANO 15-joint local axis-angle correction plus bounded per-frame "
                "wrist-centered SE(3); fixed shape and metric scale"
                if self.joint_config.optimize_hand_se3
                else "MANO 15-joint local axis-angle correction; fixed wrist SE(3), shape, and metric scale"
            ),
            "hand_reprojection_target": "source joints_2d_native in rectified-left pixels",
            "hand_mask_target": "independent anatomical SAM3 masks in rectified-left pixels",
            "config": asdict(self.joint_config),
            "timing": timing,
            "metrics": metrics,
            "artifacts": {
                "poses": str(poses_path),
                "hand_articulation": str(hand_path),
                "per_frame_metrics": str(metrics_path),
            },
        }
        write_json(report_path, report)
        return JointHandObjectOptimizationResult(
            object_id=data.object_id,
            poses_path=poses_path,
            metrics_path=metrics_path,
            report_path=report_path,
            active_frame_count=int(data.contact_active.sum()),
            optimized_frame_count=int(valid.sum()),
            contact_distance_before_median=metrics["contact_distance_before_median"],
            contact_distance_after_median=metrics["contact_distance_after_median"],
            silhouette_error_before_median=metrics["silhouette_error_before_median_lowres_px"],
            silhouette_error_after_median=metrics["silhouette_error_after_median_lowres_px"],
            hand_articulation_path=hand_path,
            articulation_delta_degrees_median=metrics["articulation_delta_degrees_median"],
            hand_reprojection_error_median_pixels=metrics["hand_reprojection_error_median_pixels"],
        )
