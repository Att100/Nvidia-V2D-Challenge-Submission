"""FoundationPose runtime loading, mesh adaptation, rendering, and scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Callable

import cv2
import numpy as np
import torch
import trimesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FOUNDATIONPOSE_ROOT = REPOSITORY_ROOT / "thirdparty/FoundationPose"


@dataclass(frozen=True)
class FoundationPoseRuntime:
    """Lazily imported FoundationPose implementation objects."""

    rasterizer: Any
    estimator_class: type
    score_predictor_class: type
    refine_predictor_class: type
    draw_box: Callable[..., np.ndarray]
    draw_axis: Callable[..., np.ndarray]
    make_mesh_tensors: Callable[..., dict]
    render_mesh: Callable[..., tuple]
    set_logging_format: Callable[[], None]
    set_seed: Callable[[int], None]


@dataclass(frozen=True)
class PoseConfidence:
    """2D/depth agreement for one object-to-camera pose candidate."""

    silhouette_iou: float
    visible_mask_coverage: float
    rendered_visibility_fraction: float
    observed_mask_pixels: int
    rendered_mask_pixels: int
    normalized_center_error: float
    depth_median_meters: float
    depth_p80_meters: float
    valid_depth_pixels: int
    quality_cost: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable confidence record."""

        return asdict(self)


def load_foundationpose_runtime() -> FoundationPoseRuntime:
    """Import the repository-local FoundationPose runtime lazily."""

    if not FOUNDATIONPOSE_ROOT.is_dir():
        raise FileNotFoundError(FOUNDATIONPOSE_ROOT)
    # FoundationStereo and FoundationPose both expose an un-namespaced
    # top-level ``Utils.py``.  A full in-process stereo run loads the former
    # first, so a normal import would silently reuse the wrong cached module.
    # Put FoundationPose first and evict only a foreign Utils module before
    # importing any FoundationPose files that execute ``from Utils import *``.
    foundationpose_path = str(FOUNDATIONPOSE_ROOT)
    sys.path[:] = [entry for entry in sys.path if entry != foundationpose_path]
    sys.path.insert(0, foundationpose_path)
    cached_utils = sys.modules.get("Utils")
    if cached_utils is not None:
        cached_file = getattr(cached_utils, "__file__", None)
        try:
            is_foundationpose_utils = (
                cached_file is not None
                and Path(cached_file).resolve().is_relative_to(FOUNDATIONPOSE_ROOT.resolve())
            )
        except (OSError, RuntimeError):
            is_foundationpose_utils = False
        if not is_foundationpose_utils:
            del sys.modules["Utils"]
    rasterizer = import_module("nvdiffrast.torch")
    estimator_module = import_module("estimater")
    refine_module = import_module("learning.training.predict_pose_refine")
    score_module = import_module("learning.training.predict_score")
    utility_module = import_module("Utils")
    return FoundationPoseRuntime(
        rasterizer=rasterizer,
        estimator_class=getattr(estimator_module, "FoundationPose"),
        score_predictor_class=getattr(score_module, "ScorePredictor"),
        refine_predictor_class=getattr(refine_module, "PoseRefinePredictor"),
        draw_box=getattr(utility_module, "draw_posed_3d_box"),
        draw_axis=getattr(utility_module, "draw_xyz_axis"),
        make_mesh_tensors=getattr(utility_module, "make_mesh_tensors"),
        render_mesh=getattr(utility_module, "nvdiffrast_render"),
        set_logging_format=getattr(utility_module, "set_logging_format"),
        set_seed=getattr(utility_module, "set_seed"),
    )


def load_metric_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a centered metric GLB/OBJ mesh and preserve its texture."""

    mesh_path = Path(path)
    loaded = trimesh.load(mesh_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.dump())
        if not geometries:
            raise ValueError(f"Mesh scene contains no geometry: {mesh_path}")
        for geometry in geometries:
            visual = geometry.visual
            if not isinstance(visual, trimesh.visual.texture.TextureVisuals):
                continue
            material = visual.material
            texture = getattr(material, "baseColorTexture", None)
            if getattr(material, "image", None) is None and texture is not None:
                visual.material = trimesh.visual.material.SimpleMaterial(image=texture)
        loaded = geometries[0] if len(geometries) == 1 else trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise TypeError(f"Expected triangle geometry: {mesh_path}")
    return loaded


def render_mesh_overlay(
    rgb: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    mesh_tensors: dict,
    glctx: Any,
    render_mesh: Callable[..., tuple],
) -> np.ndarray:
    """Blend a posed textured mesh into an RGB image."""

    rendered, depth, _ = render_mesh(
        K=intrinsics,
        H=rgb.shape[0],
        W=rgb.shape[1],
        ob_in_cams=torch.as_tensor(pose[None], device="cuda", dtype=torch.float32),
        glctx=glctx,
        mesh_tensors=mesh_tensors,
        use_light=True,
    )
    color = np.rint(rendered[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    visible = depth[0].detach().cpu().numpy() > 0
    output = rgb.copy()
    output[visible] = np.rint(0.45 * output[visible] + 0.55 * color[visible]).clip(0, 255)
    return output.astype(np.uint8)


def render_candidate_depth(
    rgb: np.ndarray,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    mesh_tensors: dict,
    glctx: Any,
    render_mesh: Callable[..., tuple],
) -> np.ndarray:
    """Render camera-space depth for confidence scoring."""

    _, depth, _ = render_mesh(
        K=intrinsics,
        H=rgb.shape[0],
        W=rgb.shape[1],
        ob_in_cams=torch.as_tensor(pose[None], device="cuda", dtype=torch.float32),
        glctx=glctx,
        mesh_tensors=mesh_tensors,
        use_light=False,
    )
    return depth[0].detach().cpu().numpy().astype(np.float32)


def evaluate_rendered_candidate(
    observed_mask: np.ndarray,
    observed_depth_meters: np.ndarray,
    rendered_depth_meters: np.ndarray,
    *,
    depth_scale_meters: float,
    mask_erode_pixels: int = 1,
) -> PoseConfidence:
    """Score rendered silhouette containment, center, and depth agreement."""

    observed = np.asarray(observed_mask, dtype=bool)
    depth = np.asarray(observed_depth_meters, dtype=np.float32)
    rendered = np.asarray(rendered_depth_meters, dtype=np.float32)
    if mask_erode_pixels > 0:
        size = 2 * mask_erode_pixels + 1
        observed = cv2.erode(observed.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool)
    predicted = np.isfinite(rendered) & (rendered > 0)
    intersection = observed & predicted
    observed_count = int(observed.sum())
    predicted_count = int(predicted.sum())
    intersection_count = int(intersection.sum())
    union_count = int((observed | predicted).sum())
    iou = intersection_count / max(union_count, 1)
    coverage = intersection_count / max(observed_count, 1)
    visibility = intersection_count / max(predicted_count, 1)

    def center(mask: np.ndarray) -> tuple[np.ndarray, float]:
        ys, xs = np.where(mask)
        if not len(xs):
            return np.array([np.nan, np.nan]), 1.0
        return np.array([xs.mean(), ys.mean()]), max(float(np.hypot(np.ptp(xs) + 1, np.ptp(ys) + 1)), 1.0)

    predicted_center, _ = center(predicted)
    observed_center, diagonal = center(observed)
    center_error = (
        float(np.linalg.norm(predicted_center - observed_center) / diagonal)
        if np.isfinite(predicted_center).all() and np.isfinite(observed_center).all()
        else float("inf")
    )
    valid_depth = intersection & np.isfinite(depth) & (depth > 0)
    residual = np.abs(rendered[valid_depth] - depth[valid_depth])
    median = float(np.median(residual)) if len(residual) else float("inf")
    p80 = float(np.quantile(residual, 0.8)) if len(residual) else float("inf")
    scale = max(float(depth_scale_meters), 1e-4)
    cost = (
        0.10 * min((1 - iou) / 0.70, 2.0)
        + 0.20 * min((1 - coverage) / 0.50, 2.0)
        + 0.20 * min(center_error / 0.15, 2.0)
        + 0.50 * min(median / scale, 3.0)
    )
    return PoseConfidence(
        iou,
        coverage,
        visibility,
        observed_count,
        predicted_count,
        center_error,
        median,
        p80,
        int(len(residual)),
        float(cost),
    )
