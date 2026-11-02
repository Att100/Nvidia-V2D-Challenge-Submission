"""Class-based fixed-hand object-pose optimizer for pipeline v2.

This module exposes typed Track-3 inputs and independent output artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import trimesh

from pipeline.utils import write_json

from .geometry import (
    FINGER_JOINTS,
    _select_finger_vertices,
    _select_palm_vertices,
    axis_angle_to_matrix,
)
from .losses import (
    ContactLoss,
    FixedHandObjectPoseLoss,
    PenetrationLoss,
    VoxelSDFSampler,
    WeightedVertexContactLoss,
)
from .sdf import DifferentiableVoxelSDF
from .silhouette import build_constraints
from .types import (
    ContactObjectConfig,
    ContactOptimizationInput,
    ContactOptimizationResult,
)


@dataclass(frozen=True)
class _RawResult:
    """In-memory optimizer output before writing project artifacts."""

    object_to_world: np.ndarray
    rotation_vectors: np.ndarray
    translations: np.ndarray
    loss_history: np.ndarray
    contact_distance_before: np.ndarray
    contact_distance_after: np.ndarray
    penetration_depth_before: np.ndarray
    penetration_depth_after: np.ndarray
    silhouette_error_before: np.ndarray
    silhouette_error_after: np.ndarray


def _finite_median(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if mask is not None:
        array = array[np.asarray(mask, dtype=bool)]
    array = array[np.isfinite(array)]
    return None if not len(array) else float(np.median(array))


class ContactPoseOptimizer:
    """Refine rigid object R/t while holding the surface-aligned MANO hand fixed.

    The optimizer uses object SDF contact and penetration losses, pose priors,
    temporal regularization, and an occlusion-aware 2D silhouette loss. The
    latter has a default weight of 2.0 as required by pipeline v2.
    """

    def __init__(
        self,
        config: ContactObjectConfig | None = None,
        *,
        contact_groups: tuple[str, ...] = ("thumb", "index", "middle", "palm"),
        sdf_pitch: float = 0.002,
        sdf_padding_voxels: int = 64,
        silhouette_downsample: int = 4,
        silhouette_surface_samples: int = 768,
        silhouette_boundary_samples: int = 128,
        silhouette_occluder_dilation: int = 2,
    ) -> None:
        self.config = config or ContactObjectConfig()
        if self.config.silhouette_weight != 2.0:
            raise ValueError("Pipeline v2 requires silhouette_weight=2.0")
        self.contact_groups = contact_groups
        self.sdf_pitch = float(sdf_pitch)
        self.sdf_padding_voxels = int(sdf_padding_voxels)
        self.silhouette_downsample = int(silhouette_downsample)
        self.silhouette_surface_samples = int(silhouette_surface_samples)
        self.silhouette_boundary_samples = int(silhouette_boundary_samples)
        self.silhouette_occluder_dilation = int(silhouette_occluder_dilation)

    def _build_contact_indices(self, data: ContactOptimizationInput) -> np.ndarray:
        """Select 96 MANO vertices around each requested anatomical patch."""

        frame_count = len(data.hand_vertices_world)
        indices = np.zeros((frame_count, len(self.contact_groups), 96), np.int64)
        for frame_index in np.flatnonzero(data.valid_frames):
            for group_index, name in enumerate(self.contact_groups):
                if name == "palm":
                    selected = _select_palm_vertices(
                        data.hand_vertices_world[frame_index],
                        data.hand_joints_world[frame_index],
                        96,
                    )
                else:
                    if name not in FINGER_JOINTS:
                        raise ValueError(f"Unknown contact group: {name}")
                    selected = _select_finger_vertices(
                        data.hand_vertices_world[frame_index],
                        data.hand_joints_world[frame_index],
                        FINGER_JOINTS[name],
                        96,
                    )
                indices[frame_index, group_index] = selected
        return indices

    def _evaluate(
        self,
        hand_vertices_world: np.ndarray,
        object_to_world: np.ndarray,
        sdf: DifferentiableVoxelSDF,
        contact_indices: np.ndarray,
        group_active: np.ndarray,
        frame_active: np.ndarray,
        device: torch.device,
        contact_vertex_weights: np.ndarray | None = None,
        hand_vertex_active: np.ndarray | None = None,
        object_contact_vertices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate contact distance and maximum penetration per frame."""

        hand = torch.as_tensor(hand_vertices_world, dtype=torch.float32, device=device)
        poses = torch.as_tensor(object_to_world, dtype=torch.float32, device=device)
        indices = torch.as_tensor(contact_indices, dtype=torch.long, device=device)
        active = torch.as_tensor(group_active, dtype=torch.bool, device=device)
        frame_active_t = torch.as_tensor(frame_active, dtype=torch.bool, device=device)
        sampler = VoxelSDFSampler.from_sdf(sdf).to(device)
        contact = ContactLoss(
            self.config.contact_sigma, self.config.contact_fraction
        ).to(device)
        penetration = PenetrationLoss(
            self.config.penetration_sigma,
            self.config.penetration_tolerance,
        ).to(device)
        with torch.inference_mode():
            hand_object = torch.bmm(
                hand - poses[:, None, :3, 3], poses[:, :3, :3]
            )
            values = sampler(hand_object)
            if object_contact_vertices is not None:
                if contact_vertex_weights is None:
                    raise ValueError(
                        "Object-region evaluation requires contact weights"
                    )
                region = torch.as_tensor(
                    object_contact_vertices, dtype=torch.float32, device=device
                )
                region_world = (
                    torch.einsum("fij,kj->fki", poses[:, :3, :3], region)
                    + poses[:, None, :3, 3]
                )
                weights = torch.as_tensor(
                    contact_vertex_weights, dtype=torch.float32, device=device
                )
                count = min(128, weights.shape[1])
                selected_weights, selected_indices = torch.topk(
                    weights, k=count, dim=1
                )
                selected_hand = torch.gather(
                    hand,
                    1,
                    selected_indices[..., None].expand(-1, -1, 3),
                )
                distances = torch.cdist(selected_hand, region_world).min(dim=-1).values
                active_weight = frame_active_t[:, None] & (selected_weights > 0)
                nearest = torch.where(
                    active_weight,
                    distances,
                    torch.full_like(distances, float("nan")),
                )
            elif contact_vertex_weights is None:
                _, nearest = contact(values, indices, active)
                nearest = torch.where(
                    active[..., None], nearest, torch.full_like(nearest, float("nan"))
                )
            else:
                weights = torch.as_tensor(
                    contact_vertex_weights, dtype=torch.float32, device=device
                )
                _, nearest = WeightedVertexContactLoss(
                    self.config.contact_sigma
                ).to(device)(values, weights, frame_active_t)
            vertex_active_t = (
                torch.as_tensor(hand_vertex_active, dtype=torch.bool, device=device)
                if hand_vertex_active is not None
                else None
            )
            _, depths = penetration(values, frame_active_t, vertex_active_t)
            distance = torch.nanmedian(nearest.reshape(len(hand), -1), dim=1).values
            if vertex_active_t is not None:
                # ``torch.nanmax`` is not available on every supported torch
                # build.  Mask with -inf and explicitly restore empty rows.
                masked_depths = torch.where(
                    vertex_active_t, depths, torch.full_like(depths, -torch.inf)
                )
                maximum_depth = masked_depths.max(dim=1).values
                maximum_depth = torch.where(
                    torch.isfinite(maximum_depth),
                    maximum_depth,
                    torch.zeros_like(maximum_depth),
                )
            else:
                maximum_depth = depths.max(dim=1).values
        return (
            distance.cpu().numpy().astype(np.float32),
            maximum_depth.cpu().numpy().astype(np.float32),
        )

    def _optimize_arrays(
        self,
        data: ContactOptimizationInput,
        mesh: trimesh.Trimesh,
        progress: Callable[[int, int, dict[str, float]], None] | None,
    ) -> tuple[_RawResult, dict[str, object]]:
        """Run Adam on per-frame object SE(3) residuals."""

        config = self.config
        if not data.valid_frames.any():
            raise RuntimeError(f"{data.object_id} has no valid contact frames")
        device_name = config.device
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            device_name = "cpu"
        device = torch.device(device_name)
        raw_hand = np.asarray(data.hand_vertices_world, dtype=np.float32)
        hand_np = np.nan_to_num(raw_hand).reshape(len(raw_hand), -1, 3).astype(np.float32)
        contact_weights_np = (
            np.asarray(data.contact_vertex_weights, dtype=np.float32).reshape(len(raw_hand), -1)
            if data.contact_vertex_weights is not None
            else None
        )
        hand_vertex_active_np = (
            np.asarray(data.hand_vertex_active, dtype=bool).reshape(len(raw_hand), -1)
            if data.hand_vertex_active is not None
            else None
        )
        object_contact_vertices_np = (
            np.asarray(data.object_contact_vertices, dtype=np.float32)
            if data.object_contact_vertices is not None
            else None
        )
        original_poses = data.initial_object_to_world.astype(np.float32)
        poses_np = original_poses.copy()
        pose_finite = np.isfinite(poses_np).all(axis=(1, 2))
        finite_indices = np.flatnonzero(pose_finite)
        if not len(finite_indices):
            raise RuntimeError(f"{data.object_id} has no finite initial object pose")
        # Held-pose tracks may intentionally contain NaNs before an object's
        # first observation. The loss evaluates tensors for the complete
        # sequence before applying masks, so fill those inactive slots with the
        # nearest finite pose to prevent NaN gradients. They are restored in the
        # saved output after optimization.
        for frame_index in np.flatnonzero(~pose_finite):
            nearest = finite_indices[np.argmin(np.abs(finite_indices - frame_index))]
            poses_np[frame_index] = poses_np[nearest]
        if contact_weights_np is None:
            contact_indices = self._build_contact_indices(data)
            group_active_np = np.broadcast_to(
                data.valid_frames[:, None], contact_indices.shape[:2]
            ).copy()
        else:
            # Dense HACO weights bypass anatomical groups; retain harmless
            # placeholders to keep the baseline loss API backward compatible.
            contact_indices = np.zeros((len(hand_np), 1, 1), dtype=np.int64)
            group_active_np = np.zeros((len(hand_np), 1), dtype=bool)

        sdf_start = time.perf_counter()
        sdf = DifferentiableVoxelSDF.from_mesh(
            mesh,
            pitch=self.sdf_pitch,
            padding_voxels=self.sdf_padding_voxels,
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
        preparation_seconds = time.perf_counter() - sdf_start

        hand = torch.as_tensor(hand_np, device=device)
        initial_rotation = torch.as_tensor(poses_np[:, :3, :3], device=device)
        initial_translation = torch.as_tensor(poses_np[:, :3, 3], device=device)
        contact_indices_t = torch.as_tensor(contact_indices, dtype=torch.long, device=device)
        group_active = torch.as_tensor(group_active_np, dtype=torch.bool, device=device)
        contact_weights = (
            torch.as_tensor(contact_weights_np, dtype=torch.float32, device=device)
            if contact_weights_np is not None
            else None
        )
        hand_vertex_active = (
            torch.as_tensor(hand_vertex_active_np, dtype=torch.bool, device=device)
            if hand_vertex_active_np is not None
            else None
        )
        object_contact_vertices = (
            torch.as_tensor(
                object_contact_vertices_np, dtype=torch.float32, device=device
            )
            if object_contact_vertices_np is not None
            else None
        )
        valid = torch.as_tensor(data.valid_frames, dtype=torch.bool, device=device)
        objective = FixedHandObjectPoseLoss(
            sdf=sdf,
            config=config,
            silhouette_constraints=constraints,
        ).to(device)
        rotation = torch.nn.Parameter(torch.zeros((len(hand), 3), device=device))
        translation = torch.nn.Parameter(torch.zeros((len(hand), 3), device=device))
        optimizer = torch.optim.Adam(
            [rotation, translation], lr=float(config.learning_rate)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(config.iterations)), eta_min=2e-5
        )

        history: list[list[float]] = []
        start = time.perf_counter()
        for iteration in range(max(1, int(config.iterations))):
            optimizer.zero_grad(set_to_none=True)
            output = objective(
                hand_vertices_world=hand,
                initial_object_rotation=initial_rotation,
                initial_object_translation=initial_translation,
                rotation_delta=rotation,
                translation_delta=translation,
                contact_vertex_indices=contact_indices_t,
                contact_group_active=group_active,
                frame_active=valid,
                contact_vertex_weights=contact_weights,
                hand_vertex_active=hand_vertex_active,
                object_contact_vertices=object_contact_vertices,
            )
            output.total.backward()
            torch.nn.utils.clip_grad_norm_([rotation, translation], max_norm=5.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                rotation_norm = torch.linalg.norm(rotation, dim=-1, keepdim=True)
                rotation.mul_(
                    float(config.maximum_rotation_radians)
                    / rotation_norm.clamp_min(float(config.maximum_rotation_radians))
                )
                translation_norm = torch.linalg.norm(translation, dim=-1, keepdim=True)
                translation.mul_(
                    float(config.maximum_translation)
                    / translation_norm.clamp_min(float(config.maximum_translation))
                )
                rotation[~valid] = 0.0
                translation[~valid] = 0.0
            values = {"total": float(output.total.detach())}
            values.update(
                {
                    name: float(output.terms[name].detach())
                    for name in objective.TERM_NAMES
                }
            )
            history.append([values["total"]] + [values[name] for name in objective.TERM_NAMES])
            if progress and (
                iteration == 0
                or (iteration + 1) % 50 == 0
                or iteration + 1 == config.iterations
            ):
                progress(iteration + 1, config.iterations, values)
        optimization_seconds = time.perf_counter() - start

        rotation_np = rotation.detach().cpu().numpy().astype(np.float32)
        translation_np = translation.detach().cpu().numpy().astype(np.float32)
        correction = axis_angle_to_matrix(rotation.detach()).cpu().numpy().astype(np.float32)
        refined_working = poses_np.copy()
        refined_working[:, :3, :3] = np.einsum(
            "fij,fjk->fik", correction, poses_np[:, :3, :3]
        )
        refined_working[:, :3, 3] += translation_np
        before = self._evaluate(
            hand_np,
            poses_np,
            sdf,
            contact_indices,
            group_active_np,
            data.valid_frames,
            device,
            contact_weights_np,
            hand_vertex_active_np,
            object_contact_vertices_np,
        )
        after = self._evaluate(
            hand_np,
            refined_working,
            sdf,
            contact_indices,
            group_active_np,
            data.valid_frames,
            device,
            contact_weights_np,
            hand_vertex_active_np,
            object_contact_vertices_np,
        )
        with torch.inference_mode():
            _, silhouette_before_t = objective.silhouette(
                initial_rotation, initial_translation
            )
            _, silhouette_after_t = objective.silhouette(
                torch.as_tensor(refined_working[:, :3, :3], device=device),
                torch.as_tensor(refined_working[:, :3, 3], device=device),
            )
        refined = refined_working.copy()
        refined[~data.valid_frames] = original_poses[~data.valid_frames]
        raw = _RawResult(
            object_to_world=refined.astype(np.float32),
            rotation_vectors=rotation_np,
            translations=translation_np,
            loss_history=np.asarray(history, dtype=np.float32),
            contact_distance_before=before[0],
            contact_distance_after=after[0],
            penetration_depth_before=before[1],
            penetration_depth_after=after[1],
            silhouette_error_before=silhouette_before_t.cpu().numpy().astype(np.float32),
            silhouette_error_after=silhouette_after_t.cpu().numpy().astype(np.float32),
        )
        timing = {
            "preparation_seconds": preparation_seconds,
            "optimization_seconds": optimization_seconds,
            "device": str(device),
            "sdf_shape_xyz": list(sdf.shape_xyz),
            "silhouette": asdict(silhouette_report),
        }
        return raw, timing

    def run(
        self,
        data: ContactOptimizationInput,
        *,
        progress: Callable[[int, int, dict[str, float]], None] | None = None,
    ) -> ContactOptimizationResult:
        """Optimize one object and persist poses, metrics, and a JSON report."""

        data.output_dir.mkdir(parents=True, exist_ok=True)
        # A semantic contact event does not guarantee that the selected hand,
        # object pose, mask, and (for HACO) predicted contact vertices overlap
        # in any usable frame.  This is expected on difficult egocentric
        # sequences and must not abort every later object.  Emit a schema-
        # compatible identity result so downstream consumers can distinguish
        # "skipped: no evidence" from a missing/failed stage.
        if not np.asarray(data.valid_frames, dtype=bool).any():
            frame_count = len(data.valid_frames)
            poses_path = data.output_dir / "poses.npz"
            metrics_path = data.output_dir / "per_frame_metrics.npz"
            report_path = data.output_dir / "report.json"
            unchanged = np.asarray(data.initial_object_to_world, dtype=np.float32)
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
                metrics_path,
                source_indices=data.source_indices,
                valid=np.asarray(data.valid_frames, dtype=bool),
                contact_distance_before=nan_metric,
                contact_distance_after=nan_metric,
                penetration_depth_before=nan_metric,
                penetration_depth_after=nan_metric,
                silhouette_error_before=nan_metric,
                silhouette_error_after=nan_metric,
                loss_history=np.empty((0, 0), dtype=np.float32),
            )
            report = {
                "schema_version": "ego-scene-hoi-contact-optimization-v2",
                "object_id": data.object_id,
                "object_name": data.object_name,
                "hand_side": data.hand_side,
                "mesh": str(data.mesh_path),
                "frame_count": frame_count,
                "contact_active_frame_count": int(data.contact_active.sum()),
                "optimized_frame_count": 0,
                "status": "skipped_no_valid_contact_frames",
                "config": asdict(self.config),
                "contact_groups": list(self.contact_groups),
                "contact_selection": (
                    "haco_dense_vertex_weights"
                    if data.contact_vertex_weights is not None
                    else "fixed_anatomical_patches"
                ),
                "positive_contact_vertex_observations": (
                    int((data.contact_vertex_weights > 0).sum())
                    if data.contact_vertex_weights is not None
                    else None
                ),
                "timing": {
                    "preparation_seconds": 0.0,
                    "optimization_seconds": 0.0,
                    "device": None,
                    "sdf_shape_xyz": None,
                    "silhouette": None,
                },
                "metrics": {
                    "contact_distance_before_median": None,
                    "contact_distance_after_median": None,
                    "penetration_depth_before_median": None,
                    "penetration_depth_after_median": None,
                    "silhouette_error_before_median_lowres_px": None,
                    "silhouette_error_after_median_lowres_px": None,
                    "translation_delta_median": None,
                    "rotation_delta_degrees_median": None,
                },
                "artifacts": {
                    "poses": str(poses_path),
                    "per_frame_metrics": str(metrics_path),
                },
            }
            write_json(report_path, report)
            return ContactOptimizationResult(
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
            )

        mesh = trimesh.load(data.mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected a single mesh: {data.mesh_path}")
        raw, timing = self._optimize_arrays(data, mesh, progress)
        poses_path = data.output_dir / "poses.npz"
        metrics_path = data.output_dir / "per_frame_metrics.npz"
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
            metrics_path,
            source_indices=data.source_indices,
            valid=data.valid_frames,
            contact_distance_before=raw.contact_distance_before,
            contact_distance_after=raw.contact_distance_after,
            penetration_depth_before=raw.penetration_depth_before,
            penetration_depth_after=raw.penetration_depth_after,
            silhouette_error_before=raw.silhouette_error_before,
            silhouette_error_after=raw.silhouette_error_after,
            loss_history=raw.loss_history,
        )
        valid = data.valid_frames
        report = {
            "schema_version": "ego-scene-hoi-contact-optimization-v2",
            "object_id": data.object_id,
            "object_name": data.object_name,
            "hand_side": data.hand_side,
            "mesh": str(data.mesh_path),
            "frame_count": len(valid),
            "contact_active_frame_count": int(data.contact_active.sum()),
            "optimized_frame_count": int(valid.sum()),
            "config": asdict(self.config),
            "contact_groups": list(self.contact_groups),
            "contact_selection": (
                "haco_hand_vertices_to_canonical_object_region"
                if data.object_contact_vertices is not None
                else "haco_dense_vertex_weights"
                if data.contact_vertex_weights is not None
                else "fixed_anatomical_patches"
            ),
            "object_contact_region_vertex_count": (
                int(len(data.object_contact_vertices))
                if data.object_contact_vertices is not None
                else None
            ),
            "positive_contact_vertex_observations": (
                int((data.contact_vertex_weights > 0).sum())
                if data.contact_vertex_weights is not None
                else None
            ),
            "timing": timing,
            "metrics": {
                "contact_distance_before_median": _finite_median(
                    raw.contact_distance_before, valid
                ),
                "contact_distance_after_median": _finite_median(
                    raw.contact_distance_after, valid
                ),
                "penetration_depth_before_median": _finite_median(
                    raw.penetration_depth_before, valid
                ),
                "penetration_depth_after_median": _finite_median(
                    raw.penetration_depth_after, valid
                ),
                "silhouette_error_before_median_lowres_px": _finite_median(
                    raw.silhouette_error_before, valid
                ),
                "silhouette_error_after_median_lowres_px": _finite_median(
                    raw.silhouette_error_after, valid
                ),
                "translation_delta_median": _finite_median(
                    np.linalg.norm(raw.translations, axis=1), valid
                ),
                "rotation_delta_degrees_median": _finite_median(
                    np.rad2deg(np.linalg.norm(raw.rotation_vectors, axis=1)), valid
                ),
            },
            "artifacts": {
                "poses": str(poses_path),
                "per_frame_metrics": str(metrics_path),
            },
        }
        write_json(report_path, report)
        return ContactOptimizationResult(
            object_id=data.object_id,
            poses_path=poses_path,
            metrics_path=metrics_path,
            report_path=report_path,
            active_frame_count=int(data.contact_active.sum()),
            optimized_frame_count=int(valid.sum()),
            contact_distance_before_median=report["metrics"]["contact_distance_before_median"],
            contact_distance_after_median=report["metrics"]["contact_distance_after_median"],
            silhouette_error_before_median=report["metrics"]["silhouette_error_before_median_lowres_px"],
            silhouette_error_after_median=report["metrics"]["silhouette_error_after_median_lowres_px"],
        )
