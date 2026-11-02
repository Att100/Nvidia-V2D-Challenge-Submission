"""Mesh voxelization and differentiable signed-distance sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F
import trimesh


@dataclass(frozen=True)
class DifferentiableVoxelSDF:
    """Object-local SDF volume and index transform."""

    values_zyx: np.ndarray
    object_to_index: np.ndarray
    pitch: float

    @classmethod
    def from_mesh(
        cls,
        mesh: trimesh.Trimesh,
        *,
        pitch: float = 0.002,
        padding_voxels: int = 8,
    ) -> "DifferentiableVoxelSDF":
        if pitch <= 0:
            raise ValueError("pitch must be positive")
        if padding_voxels < 2:
            raise ValueError("padding_voxels must be at least 2")
        voxel_grid = mesh.voxelized(float(pitch)).fill()
        occupied = np.asarray(voxel_grid.matrix, dtype=bool)
        padding = int(padding_voxels)
        occupied = np.pad(occupied, padding, mode="constant", constant_values=False)

        inside_distance = distance_transform_edt(occupied)
        outside_distance = distance_transform_edt(~occupied)
        # The half-voxel correction places the approximate zero crossing on the
        # mesh surface rather than at the centers of boundary voxels.
        sdf = np.where(
            occupied,
            -(inside_distance - 0.5) * float(pitch),
            (outside_distance - 0.5) * float(pitch),
        ).astype(np.float32)

        index_to_object = np.asarray(voxel_grid.transform, dtype=np.float64).copy()
        index_to_object[:3, 3] -= index_to_object[:3, :3] @ np.full(3, padding)
        object_to_index = np.linalg.inv(index_to_object)
        # grid_sample expects D,H,W = Z,Y,X while Trimesh stores X,Y,Z.
        return cls(
            values_zyx=np.ascontiguousarray(sdf.transpose(2, 1, 0)),
            object_to_index=object_to_index.astype(np.float32),
            pitch=float(pitch),
        )

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        depth, height, width = self.values_zyx.shape
        return width, height, depth

    def sample_torch(self, points_object: torch.Tensor) -> torch.Tensor:
        """Compatibility sampler; optimization uses ``VoxelSDFSampler``."""

        if points_object.shape[-1] != 3:
            raise ValueError("points_object must end in XYZ")
        device = points_object.device
        dtype = points_object.dtype
        transform = torch.as_tensor(self.object_to_index, dtype=dtype, device=device)
        indices = points_object @ transform[:3, :3].T + transform[:3, 3]
        nx, ny, nz = self.shape_xyz
        upper = torch.as_tensor([nx - 1, ny - 1, nz - 1], dtype=dtype, device=device)
        clamped_indices = torch.minimum(
            torch.maximum(indices, torch.zeros_like(indices)), upper
        )
        outside_distance = torch.linalg.norm(indices - clamped_indices, dim=-1) * float(
            self.pitch
        )
        normalized = torch.empty_like(clamped_indices)
        normalized[..., 0] = 2.0 * clamped_indices[..., 0] / max(nx - 1, 1) - 1.0
        normalized[..., 1] = 2.0 * clamped_indices[..., 1] / max(ny - 1, 1) - 1.0
        normalized[..., 2] = 2.0 * clamped_indices[..., 2] / max(nz - 1, 1) - 1.0
        original_shape = normalized.shape[:-1]
        grid = normalized.reshape(1, -1, 1, 1, 3)
        volume = torch.as_tensor(self.values_zyx, dtype=dtype, device=device)[
            None, None
        ]
        sampled = F.grid_sample(
            volume,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.reshape(original_shape) + outside_distance

    def sample_numpy(self, points_object: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            points = torch.as_tensor(np.asarray(points_object, dtype=np.float32))
            return self.sample_torch(points).numpy().astype(np.float32)
