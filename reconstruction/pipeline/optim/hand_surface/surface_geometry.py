from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceCorrespondences:
    predicted: np.ndarray
    observed: np.ndarray
    pixels: np.ndarray


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    include_edge_midpoints: bool = True,
) -> np.ndarray:
    """Densely and deterministically sample a triangular mesh surface."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected Vx3 vertices, got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected Fx3 faces, got {faces.shape}")
    triangles = vertices[faces]
    samples = [vertices, triangles.mean(axis=1)]
    if include_edge_midpoints:
        samples.extend(
            [
                0.5 * (triangles[:, 0] + triangles[:, 1]),
                0.5 * (triangles[:, 1] + triangles[:, 2]),
                0.5 * (triangles[:, 2] + triangles[:, 0]),
            ]
        )
    return np.concatenate(samples, axis=0).astype(np.float32)


def projective_surface_correspondences(
    surface_camera: np.ndarray,
    pointmap_camera: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    *,
    confidence_threshold: float,
    sample_mask: np.ndarray | None = None,
) -> SurfaceCorrespondences:
    """Match z-buffered mesh samples to an organized point map at the same pixels.

    Dense vertex/centroid/edge sampling approximates a surface rasterization while
    retaining the projective correspondence that a global point-cloud Chamfer loses.
    """
    surface = np.asarray(surface_camera, dtype=np.float64)
    pointmap = np.asarray(pointmap_camera, dtype=np.float64)
    scores = np.asarray(confidence, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if surface.ndim != 2 or surface.shape[1] != 3:
        raise ValueError(f"Expected Nx3 surface samples, got {surface.shape}")
    if pointmap.ndim != 3 or pointmap.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 point map, got {pointmap.shape}")
    height, width = pointmap.shape[:2]
    if scores.shape != (height, width):
        raise ValueError("Confidence must match the point-map resolution")
    allowed = None if sample_mask is None else np.asarray(sample_mask, dtype=bool)
    if allowed is not None and allowed.shape != (height, width):
        raise ValueError("sample_mask must match the point-map resolution")

    projected_h = surface @ k.T
    pixels_float = projected_h[:, :2] / projected_h[:, 2:3]
    pixels = np.rint(pixels_float).astype(np.int64)
    valid = (
        np.isfinite(surface).all(axis=1)
        & np.isfinite(pixels_float).all(axis=1)
        & (surface[:, 2] > 1e-5)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    candidate_indices = np.flatnonzero(valid)
    if not len(candidate_indices):
        empty_points = np.empty((0, 3), dtype=np.float32)
        return SurfaceCorrespondences(empty_points, empty_points.copy(), np.empty((0, 2), dtype=np.int32))

    candidate_pixels = pixels[candidate_indices]
    linear = candidate_pixels[:, 1] * width + candidate_pixels[:, 0]
    order = np.lexsort((surface[candidate_indices, 2], linear))
    ordered_indices = candidate_indices[order]
    ordered_linear = linear[order]
    _, first = np.unique(ordered_linear, return_index=True)
    surface_indices = ordered_indices[first]
    selected_pixels = pixels[surface_indices]
    observed = pointmap[selected_pixels[:, 1], selected_pixels[:, 0]]
    selected_scores = scores[selected_pixels[:, 1], selected_pixels[:, 0]]
    keep = (
        np.isfinite(observed).all(axis=1)
        & (observed[:, 2] > 1e-5)
        & np.isfinite(selected_scores)
        & (selected_scores >= float(confidence_threshold))
    )
    if allowed is not None:
        keep &= allowed[selected_pixels[:, 1], selected_pixels[:, 0]]
    return SurfaceCorrespondences(
        surface[surface_indices][keep].astype(np.float32),
        observed[keep].astype(np.float32),
        selected_pixels[keep].astype(np.int32),
    )


def rasterized_mesh_correspondences(
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    pointmap_camera: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    *,
    confidence_threshold: float,
    sample_mask: np.ndarray | None = None,
) -> SurfaceCorrespondences:
    """Rasterize the complete visible MANO surface and sample stereo depth at its pixels."""
    import torch
    from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
    from pytorch3d.structures import Meshes

    vertices = np.asarray(vertices_camera, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int64)
    pointmap = np.asarray(pointmap_camera, dtype=np.float32)
    scores = np.asarray(confidence, dtype=np.float32)
    k = np.asarray(intrinsics, dtype=np.float32)
    height, width = pointmap.shape[:2]
    allowed = None if sample_mask is None else np.asarray(sample_mask, dtype=bool)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected Vx3 vertices, got {vertices.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"Expected Fx3 faces, got {triangles.shape}")
    if pointmap.shape != (height, width, 3) or scores.shape != (height, width):
        raise ValueError("Point map and confidence dimensions do not match")
    if allowed is not None and allowed.shape != (height, width):
        raise ValueError("sample_mask must match the point-map resolution")

    # PyTorch3D uses +X left and +Y up. OpenCV/MANO camera coordinates use the
    # OpenCV convention (+X right, +Y down), so flip X/Y only for rasterization.
    raster_vertices = vertices.copy()
    raster_vertices[:, :2] *= -1.0
    vertices_tensor = torch.from_numpy(raster_vertices)[None]
    faces_tensor = torch.from_numpy(triangles)[None]
    cameras = PerspectiveCameras(
        focal_length=((float(k[0, 0]), float(k[1, 1])),),
        principal_point=((float(k[0, 2]), float(k[1, 2])),),
        image_size=((height, width),),
        in_ndc=False,
        device="cpu",
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
        ),
    )
    with torch.inference_mode():
        fragments = rasterizer(Meshes(vertices_tensor, faces_tensor))
    face_map = fragments.pix_to_face[0, :, :, 0].numpy()
    barycentric = fragments.bary_coords[0, :, :, 0].numpy()
    valid = face_map >= 0
    ys, xs = np.nonzero(valid)
    if not len(ys):
        empty_points = np.empty((0, 3), dtype=np.float32)
        return SurfaceCorrespondences(empty_points, empty_points.copy(), np.empty((0, 2), dtype=np.int32))
    face_ids = face_map[ys, xs]
    weights = barycentric[ys, xs]
    predicted = np.einsum(
        "ni,nij->nj", weights, vertices[triangles[face_ids]], optimize=True
    )
    observed = pointmap[ys, xs]
    keep = (
        np.isfinite(predicted).all(axis=1)
        & (predicted[:, 2] > 1e-5)
        & np.isfinite(observed).all(axis=1)
        & (observed[:, 2] > 1e-5)
        & np.isfinite(scores[ys, xs])
        & (scores[ys, xs] >= float(confidence_threshold))
    )
    if allowed is not None:
        keep &= allowed[ys, xs]
    return SurfaceCorrespondences(
        predicted[keep].astype(np.float32),
        observed[keep].astype(np.float32),
        np.column_stack([xs[keep], ys[keep]]).astype(np.int32),
    )


class BatchedSurfaceRasterizer:
    """Reusable PyTorch3D rasterizer for batched MANO/point-map matching."""

    def __init__(
        self,
        intrinsics: np.ndarray,
        image_size: tuple[int, int],
        *,
        device: str = "auto",
        bin_size: int | None = None,
        max_faces_per_bin: int | None = None,
    ) -> None:
        import torch
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA rasterization was requested but CUDA is unavailable")
        self.device = torch.device(device)
        self.intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if self.intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must have shape 3,3")
        self.height, self.width = map(int, image_size)
        if self.height <= 0 or self.width <= 0:
            raise ValueError("image_size values must be positive")
        self.rasterizer = MeshRasterizer(
            cameras=None,
            raster_settings=RasterizationSettings(
                image_size=(self.height, self.width),
                blur_radius=0.0,
                faces_per_pixel=1,
                cull_backfaces=False,
                bin_size=bin_size,
                max_faces_per_bin=max_faces_per_bin,
            ),
        )

    def __call__(
        self,
        vertices_camera: np.ndarray,
        faces: np.ndarray,
        pointmaps_camera: np.ndarray,
        confidence: np.ndarray,
        *,
        confidence_threshold: float,
        sample_masks: np.ndarray | None = None,
    ) -> list[SurfaceCorrespondences]:
        """Rasterize a batch and return one correspondence set per mesh."""

        import torch
        from pytorch3d.renderer import PerspectiveCameras
        from pytorch3d.structures import Meshes

        vertices = np.asarray(vertices_camera, dtype=np.float32)
        triangles = np.asarray(faces, dtype=np.int64)
        pointmaps = np.asarray(pointmaps_camera, dtype=np.float32)
        scores = np.asarray(confidence, dtype=np.float32)
        if vertices.ndim != 3 or vertices.shape[-1] != 3:
            raise ValueError("vertices_camera must have shape batch,vertices,3")
        batch_size = len(vertices)
        if triangles.ndim == 2:
            triangles = np.repeat(triangles[None], batch_size, axis=0)
        if triangles.ndim != 3 or triangles.shape[0] != batch_size or triangles.shape[-1] != 3:
            raise ValueError("faces must have shape faces,3 or batch,faces,3")
        expected_points = (batch_size, self.height, self.width, 3)
        if pointmaps.shape != expected_points:
            raise ValueError(f"pointmaps_camera must have shape {expected_points}")
        if scores.shape != expected_points[:-1]:
            raise ValueError("confidence must match the batched point maps")
        allowed = None
        if sample_masks is not None:
            allowed = np.asarray(sample_masks, dtype=bool)
            if allowed.shape != scores.shape:
                raise ValueError("sample_masks must match confidence")

        raster_vertices = vertices.copy()
        raster_vertices[..., :2] *= -1.0
        vertices_tensor = torch.as_tensor(raster_vertices, device=self.device)
        faces_tensor = torch.as_tensor(triangles, device=self.device)
        focal = torch.as_tensor(
            [[self.intrinsics[0, 0], self.intrinsics[1, 1]]],
            dtype=torch.float32,
            device=self.device,
        ).expand(batch_size, -1)
        principal = torch.as_tensor(
            [[self.intrinsics[0, 2], self.intrinsics[1, 2]]],
            dtype=torch.float32,
            device=self.device,
        ).expand(batch_size, -1)
        image_size = torch.as_tensor(
            [[self.height, self.width]], dtype=torch.float32, device=self.device
        ).expand(batch_size, -1)
        cameras = PerspectiveCameras(
            focal_length=focal,
            principal_point=principal,
            image_size=image_size,
            in_ndc=False,
            device=self.device,
        )
        meshes = Meshes(verts=vertices_tensor, faces=faces_tensor)
        with torch.inference_mode():
            fragments = self.rasterizer(meshes, cameras=cameras)
        face_maps = fragments.pix_to_face[..., 0].detach().cpu().numpy()
        barycentric = fragments.bary_coords[..., 0, :].detach().cpu().numpy()

        results = []
        for batch_id in range(batch_size):
            face_map = face_maps[batch_id]
            ys, xs = np.nonzero(face_map >= 0)
            if not len(ys):
                empty = np.empty((0, 3), dtype=np.float32)
                results.append(
                    SurfaceCorrespondences(
                        empty, empty.copy(), np.empty((0, 2), dtype=np.int32)
                    )
                )
                continue
            # ``pix_to_face`` indexes the packed face tensor across the batch.
            face_ids = face_map[ys, xs] - batch_id * triangles.shape[1]
            weights = barycentric[batch_id, ys, xs]
            predicted = np.einsum(
                "ni,nij->nj",
                weights,
                vertices[batch_id, triangles[batch_id, face_ids]],
                optimize=True,
            )
            observed = pointmaps[batch_id, ys, xs]
            keep = (
                np.isfinite(predicted).all(axis=1)
                & (predicted[:, 2] > 1e-5)
                & np.isfinite(observed).all(axis=1)
                & (observed[:, 2] > 1e-5)
                & np.isfinite(scores[batch_id, ys, xs])
                & (scores[batch_id, ys, xs] >= float(confidence_threshold))
            )
            if allowed is not None:
                keep &= allowed[batch_id, ys, xs]
            results.append(
                SurfaceCorrespondences(
                    predicted[keep].astype(np.float32),
                    observed[keep].astype(np.float32),
                    np.column_stack([xs[keep], ys[keep]]).astype(np.int32),
                )
            )
        return results


def robust_median(values: np.ndarray, *, minimum_count: int = 16) -> tuple[float, int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) < minimum_count:
        return float("nan"), len(values)
    center = float(np.median(values))
    deviations = np.abs(values - center)
    mad = float(np.median(deviations))
    threshold = max(1e-5, 3.5 * 1.4826 * mad)
    inliers = deviations <= threshold
    if inliers.sum() < minimum_count:
        return float("nan"), int(inliers.sum())
    return float(np.median(values[inliers])), int(inliers.sum())


def robust_frame_scale(
    correspondences: SurfaceCorrespondences,
    *,
    minimum_count: int = 32,
    ratio_bounds: tuple[float, float] = (0.5, 2.0),
) -> tuple[float, int]:
    predicted_z = correspondences.predicted[:, 2]
    observed_z = correspondences.observed[:, 2]
    ratio = observed_z / predicted_z
    valid = (
        np.isfinite(ratio)
        & (ratio >= ratio_bounds[0])
        & (ratio <= ratio_bounds[1])
    )
    log_scale, count = robust_median(np.log(ratio[valid]), minimum_count=minimum_count)
    return (float(np.exp(log_scale)) if np.isfinite(log_scale) else float("nan"), count)


def robust_weighted_scale(
    frame_scales: np.ndarray,
    support: np.ndarray,
    *,
    fallback: float = 1.0,
) -> tuple[float, np.ndarray]:
    scales = np.asarray(frame_scales, dtype=np.float64)
    weights = np.asarray(support, dtype=np.float64)
    valid = np.isfinite(scales) & (scales > 0.0) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return float(fallback), valid
    logs = np.log(scales[valid])
    center = float(np.median(logs))
    mad = float(np.median(np.abs(logs - center)))
    threshold = max(0.025, 3.5 * 1.4826 * mad)
    selected = np.zeros_like(valid)
    selected[np.flatnonzero(valid)] = np.abs(logs - center) <= threshold
    selected_values = scales[selected]
    selected_weights = weights[selected]
    order = np.argsort(selected_values)
    cumulative = np.cumsum(selected_weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    return float(selected_values[order[min(index, len(order) - 1)]]), selected


def camera_ray(pixel: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    pixel = np.asarray(pixel, dtype=np.float64).reshape(2)
    k = np.asarray(intrinsics, dtype=np.float64)
    ray = np.linalg.inv(k) @ np.asarray([pixel[0], pixel[1], 1.0], dtype=np.float64)
    norm = float(np.linalg.norm(ray))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("Invalid camera ray")
    return (ray / norm).astype(np.float32)


def robust_ray_offset(
    correspondences: SurfaceCorrespondences,
    scale: float,
    ray: np.ndarray,
    *,
    minimum_count: int = 32,
) -> tuple[float, int]:
    residual = correspondences.observed - correspondences.predicted * float(scale)
    signed = residual @ np.asarray(ray, dtype=np.float64).reshape(3)
    return robust_median(signed, minimum_count=minimum_count)


def smooth_scalar_offsets(
    raw_offsets: np.ndarray,
    support: np.ndarray,
    *,
    smoothness_weight: float = 8.0,
    prior_weight: float = 0.02,
) -> np.ndarray:
    """Weighted second-order smoothing with interpolation through missing frames."""
    values = np.asarray(raw_offsets, dtype=np.float64).reshape(-1)
    counts = np.asarray(support, dtype=np.float64).reshape(-1)
    if len(values) != len(counts):
        raise ValueError("raw_offsets and support must have equal length")
    frame_count = len(values)
    valid = np.isfinite(values) & np.isfinite(counts) & (counts > 0.0)
    if not valid.any():
        return np.zeros(frame_count, dtype=np.float32)
    weights = np.zeros(frame_count, dtype=np.float64)
    weights[valid] = np.sqrt(counts[valid] / np.median(counts[valid]))
    weights = np.clip(weights, 0.25, 4.0)
    targets = np.where(valid, values, 0.0)
    system = np.diag(weights + float(prior_weight))
    right_hand_side = weights * targets
    if frame_count >= 3 and smoothness_weight > 0.0:
        second = np.zeros((frame_count - 2, frame_count), dtype=np.float64)
        indices = np.arange(frame_count - 2)
        second[indices, indices] = 1.0
        second[indices, indices + 1] = -2.0
        second[indices, indices + 2] = 1.0
        system += float(smoothness_weight) * (second.T @ second)
    solution = np.linalg.solve(system, right_hand_side)
    return solution.astype(np.float32)


def apply_radial_surface_refinement(
    points_camera: np.ndarray,
    scale: float,
    ray: np.ndarray,
    offset: float,
) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float64)
    return (points * float(scale) + float(offset) * np.asarray(ray, dtype=np.float64)).astype(
        np.float32
    )
