"""Differentiable MANO articulation in the aligned pipeline world frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import smplx
import torch
from torch import nn

from .geometry import axis_angle_to_matrix


_TIP_VERTEX_IDS = (745, 317, 445, 556, 673)
_MANO_TO_OPENPOSE = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)
# UmeTrack's public 20-landmark convention.  Keep this identical to the
# fitting module so the joint reprojection target remains anatomically valid.
_MANO_TO_UMETRACK = (16, 17, 18, 19, 20, 0, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9)


def _umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a proper metric similarity from row-vector source to target."""

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    covariance = centered_target.T @ centered_source / max(len(source), 1)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    variance = float(np.mean(np.sum(centered_source * centered_source, axis=1)))
    scale = float(np.sum(singular * sign) / max(variance, 1.0e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


@dataclass
class _MANOBatch:
    """One anatomical hand's valid frame subset and fixed world alignment."""

    slots: torch.Tensor
    model: nn.Module
    global_orient: torch.Tensor
    hand_pose: torch.Tensor
    betas: torch.Tensor
    scale: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor
    vertex_offset: torch.Tensor
    joint_offset: torch.Tensor
    mirror_x: bool
    axis_angle_mode: bool


class DifferentiableMANOSequence(nn.Module):
    """Evaluate MANO pose residuals while preserving upstream metric alignment.

    A fixed per-frame similarity maps the official MANO layer at zero residual
    onto the surface-aligned UmeTrack/MANO mesh. Only the 45-D articulated hand pose is
    variable; global hand rotation, translation, shape, and scale stay fixed.
    """

    def __init__(
        self,
        *,
        model_root: Path,
        hand_labels: tuple[str, ...],
        vertices_world: np.ndarray,
        joints_world: np.ndarray,
        mano_betas: np.ndarray,
        mano_global_orient: np.ndarray,
        mano_hand_pose: np.ndarray,
        frame_hand_active: np.ndarray,
        device: torch.device,
        mirror_left_from_right_model: bool = False,
    ) -> None:
        super().__init__()
        self.frame_count, self.hand_count = vertices_world.shape[:2]
        self.joint_count = int(joints_world.shape[2])
        if self.joint_count not in (20, 21):
            raise ValueError(
                f"Joint contact optimization supports 20/21 landmarks, got {self.joint_count}"
            )
        self.register_buffer(
            "base_vertices",
            torch.as_tensor(np.nan_to_num(vertices_world), dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "base_joints",
            torch.as_tensor(np.nan_to_num(joints_world), dtype=torch.float32, device=device),
        )
        self.batches: list[_MANOBatch | None] = []
        for hand_index, side in enumerate(hand_labels):
            parameter_finite = (
                np.isfinite(mano_betas[:, hand_index]).all(axis=1)
                & np.isfinite(mano_global_orient[:, hand_index]).all(axis=(1, 2, 3))
                & np.isfinite(mano_hand_pose[:, hand_index]).all(axis=(1, 2, 3))
                & np.isfinite(vertices_world[:, hand_index]).all(axis=(1, 2))
            )
            slots_np = np.flatnonzero(frame_hand_active[:, hand_index] & parameter_finite)
            if not len(slots_np):
                self.batches.append(None)
                continue
            mirror_x = bool(mirror_left_from_right_model and side == "left")
            model_side = "right" if mirror_x else side
            model_file = model_root / "mano" / f"MANO_{model_side.upper()}.pkl"
            axis_angle_mode = self.joint_count == 20
            model = (
                smplx.MANO(
                    model_path=str(model_file),
                    is_rhand=model_side == "right",
                    use_pca=False,
                    flat_hand_mean=False,
                    batch_size=len(slots_np),
                )
                if axis_angle_mode
                else smplx.MANOLayer(
                    model_path=str(model_file),
                    is_rhand=model_side == "right",
                    use_pca=False,
                    batch_size=len(slots_np),
                )
            ).to(device)
            if axis_angle_mode and model_side == "left":
                right_reference = smplx.MANO(
                    model_path=str(model_root / "mano" / "MANO_RIGHT.pkl"),
                    is_rhand=True,
                    use_pca=False,
                    flat_hand_mean=False,
                    batch_size=1,
                ).to(device)
                shapedirs_difference = torch.sum(
                    torch.abs(
                        model.shapedirs[:, 0, :]
                        - right_reference.shapedirs[:, 0, :]
                    )
                )
                if float(shapedirs_difference.detach().cpu()) < 1.0:
                    with torch.no_grad():
                        model.shapedirs[:, 0, :] *= -1.0
                del right_reference
            model.requires_grad_(False)
            betas = mano_betas[slots_np, hand_index].astype(np.float32)
            # The fitted sequence stores absolute rotations and evaluates MANO with
            # ``pose2rot=False``. Going through MANO's axis-angle path would
            # add its mean hand pose a second time.
            global_values = mano_global_orient[slots_np, hand_index]
            pose_values = mano_hand_pose[slots_np, hand_index]
            if axis_angle_mode:
                global_values = Rotation.from_matrix(
                    global_values.reshape(-1, 3, 3)
                ).as_rotvec().reshape(len(slots_np), 3).astype(np.float32)
                pose_values = Rotation.from_matrix(
                    pose_values.reshape(-1, 3, 3)
                ).as_rotvec().reshape(len(slots_np), 45).astype(np.float32)
            global_tensor = torch.as_tensor(global_values, device=device)
            pose_tensor = torch.as_tensor(pose_values, device=device)
            betas_tensor = torch.as_tensor(betas, device=device)
            with torch.no_grad():
                baseline_output = model(
                    global_orient=global_tensor,
                    hand_pose=pose_tensor,
                    betas=betas_tensor,
                )
                baseline = baseline_output.vertices.cpu().numpy()
                baseline_joints = self._mapped_joints(baseline_output).cpu().numpy()
                if mirror_x:
                    baseline[..., 0] *= -1.0
                    baseline_joints[..., 0] *= -1.0
            scales: list[float] = []
            rotations: list[np.ndarray] = []
            translations: list[np.ndarray] = []
            for row, frame_index in enumerate(slots_np):
                scale, rotation, translation = _umeyama(
                    baseline[row], vertices_world[frame_index, hand_index]
                )
                scales.append(scale)
                rotations.append(rotation)
                translations.append(translation)
            scales_np = np.asarray(scales, dtype=np.float32)
            rotations_np = np.stack(rotations).astype(np.float32)
            translations_np = np.stack(translations).astype(np.float32)
            aligned_vertices = scales_np[:, None, None] * np.einsum(
                "fij,fpj->fpi", rotations_np, baseline
            ) + translations_np[:, None]
            aligned_joints = scales_np[:, None, None] * np.einsum(
                "fij,fpj->fpi", rotations_np, baseline_joints
            ) + translations_np[:, None]
            vertex_offset = (
                vertices_world[slots_np, hand_index] - aligned_vertices
            ).astype(np.float32)
            joint_offset = (
                joints_world[slots_np, hand_index] - aligned_joints
            ).astype(np.float32)
            self.batches.append(
                _MANOBatch(
                    slots=torch.as_tensor(slots_np, dtype=torch.long, device=device),
                    model=model,
                    global_orient=global_tensor,
                    hand_pose=pose_tensor,
                    betas=betas_tensor,
                    scale=torch.as_tensor(scales_np, dtype=torch.float32, device=device),
                    rotation=torch.as_tensor(rotations_np, device=device),
                    translation=torch.as_tensor(translations_np, device=device),
                    vertex_offset=torch.as_tensor(vertex_offset, device=device),
                    joint_offset=torch.as_tensor(joint_offset, device=device),
                    mirror_x=mirror_x,
                    axis_angle_mode=axis_angle_mode,
                )
            )

    def _mapped_joints(self, output: object) -> torch.Tensor:
        """Map MANO joints to the source model's 20- or 21-point convention."""

        tips = output.vertices[:, _TIP_VERTEX_IDS]
        joints = torch.cat((output.joints[:, :16], tips), dim=1)
        mapping = _MANO_TO_UMETRACK if self.joint_count == 20 else _MANO_TO_OPENPOSE
        return joints[:, mapping]

    def forward(self, articulation_delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full-timeline world vertices and joints for pose residuals."""

        if articulation_delta.shape != (self.frame_count, self.hand_count, 45):
            raise ValueError(
                "articulation_delta must have shape "
                f"{(self.frame_count, self.hand_count, 45)}"
            )
        vertices = self.base_vertices.clone()
        joints = self.base_joints.clone()
        for hand_index, batch in enumerate(self.batches):
            if batch is None:
                continue
            if batch.axis_angle_mode:
                output = batch.model(
                    global_orient=batch.global_orient,
                    hand_pose=batch.hand_pose
                    + articulation_delta[batch.slots, hand_index],
                    betas=batch.betas,
                )
            else:
                correction = axis_angle_to_matrix(
                    articulation_delta[batch.slots, hand_index].reshape(-1, 3)
                ).reshape(-1, 15, 3, 3)
                output = batch.model(
                    global_orient=batch.global_orient,
                    hand_pose=torch.matmul(correction, batch.hand_pose),
                    betas=batch.betas,
                )
            model_vertices = output.vertices
            model_joints = self._mapped_joints(output)
            if batch.mirror_x:
                mirror = model_vertices.new_tensor((-1.0, 1.0, 1.0))
                model_vertices = model_vertices * mirror
                model_joints = model_joints * mirror
            batch_vertices = batch.scale[:, None, None] * torch.einsum(
                "fij,fpj->fpi", batch.rotation, model_vertices
            ) + batch.translation[:, None] + batch.vertex_offset
            batch_joints = batch.scale[:, None, None] * torch.einsum(
                "fij,fpj->fpi", batch.rotation, model_joints
            ) + batch.translation[:, None] + batch.joint_offset
            vertices[batch.slots, hand_index] = batch_vertices
            joints[batch.slots, hand_index] = batch_joints
        return vertices, joints


def project_world_points(
    points_world: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Project ``[T,...,3]`` world points with OpenCV camera matrices."""

    flat = points_world.reshape(len(points_world), -1, 3)
    camera = torch.einsum("fij,fpj->fpi", world_to_camera[:, :3, :3], flat)
    camera = camera + world_to_camera[:, None, :3, 3]
    homogeneous = torch.einsum("fij,fpj->fpi", intrinsics, camera)
    pixels = homogeneous[..., :2] / homogeneous[..., 2:3].clamp_min(1.0e-5)
    return pixels.reshape(*points_world.shape[:-1], 2)
