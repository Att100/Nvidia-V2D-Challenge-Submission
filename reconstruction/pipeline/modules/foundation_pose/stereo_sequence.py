"""FoundationPose experiment driven by rectified left-eye stereo geometry.

It segments each canonical object in the rectified left view, fits
a configured metric mesh scale from FoundationStereo depth, prepares the
standard FoundationPose RGB-D dataset, and runs camera-motion-compensated
bidirectional tracking with VIPE camera poses.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
import trimesh

from pipeline.modules.foundation_pose.tracker import run_bidirectional_tracking
from pipeline.modules.object_scale.fitting import (
    _axis_scales,
    _mesh_frame,
    _robust_frame,
    _scale_transform,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class StereoObject:
    """One canonical object, its prompt, seed frame, and reconstructed mesh."""

    object_id: str
    name: str
    prompt: str
    seed_frame: int
    mesh: Path
    metric_mesh: bool = False


@dataclass(frozen=True)
class StereoFoundationPoseConfig:
    """Inputs and conservative settings for the focused stereo experiment."""

    stereo_root: Path
    source_run: Path
    output_dir: Path
    reasoning_path: Path | None = None
    object_tracks_dir: Path | None = None
    # The rectified left stream is monochrome and the thin wooden spoon has a
    # low text-grounding score despite a clean mask (about 0.08 at frame 312).
    # We keep the highest-scoring instance and still enforce a minimum area.
    confidence_threshold: float = 0.05
    minimum_mask_area: int = 64
    extent_quantile: float = 0.01
    thickness_weight: float = 0.15
    scale_mode: str = "isotropic"
    max_anisotropy: float = 2.0
    estimation_iterations: int = 5
    tracking_iterations: int = 2
    adaptive_reregister: bool = False
    appearance_mode: str = "grayscale"
    maximum_reregister_rotation_degrees: float = 25.0
    maximum_reregister_translation_meters: float = 0.03
    maximum_reregister_translation_fraction: float = 0.15
    registration_only: bool = False
    write_overlay: bool = True
    # Preserve released geometry, UVs and materials. Registration memory is
    # bounded by rendering pose hypotheses in chunks instead of simplifying.
    render_batch_size: int = 8


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_-") or "object"


def _seed_prompt_candidates(item: StereoObject) -> tuple[str, ...]:
    """Return conservative text-grounding fallbacks for a canonical seed.

    Rectified stereo images are monochrome and lower resolution than the RGB
    stream used by Codex.  Detailed colour/shape phrases can consequently score
    below SAM3's grounding threshold even when the object is plainly visible.
    Keep the original prompt first, then progressively remove attributes while
    preserving the object category.
    """

    values = [item.prompt, item.name]
    lowered = f"{item.name} {item.prompt}".lower()
    if "spoon" in lowered:
        values.extend(("wooden spoon", "spoon"))
    elif "saucepan" in lowered or "pan" in lowered:
        values.extend(("white saucepan", "saucepan"))
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


def _video_metadata(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        capture.release()


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = tuple(loaded.dump())
        if not geometries:
            raise ValueError(f"Mesh scene contains no geometry: {path}")
        loaded = geometries[0] if len(geometries) == 1 else trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError(f"Mesh contains no triangles: {path}")
    return loaded


class StereoFoundationPoseExperiment:
    """Prepare and run FoundationPose using one rectified stereo left eye."""

    def __init__(self, config: StereoFoundationPoseConfig) -> None:
        if config.scale_mode not in {"isotropic", "anisotropic"}:
            raise ValueError(
                "scale_mode must be 'isotropic' or 'anisotropic', got "
                f"{config.scale_mode!r}"
            )
        if config.max_anisotropy < 1.0:
            raise ValueError("max_anisotropy must be at least 1")
        self.config = config
        self.stereo_root = config.stereo_root.expanduser().resolve()
        self.source_run = config.source_run.expanduser().resolve()
        self.output = config.output_dir.expanduser().resolve()
        self.reasoning_path = (
            config.reasoning_path.expanduser().resolve()
            if config.reasoning_path is not None
            else self.source_run / "semantics/reasoning/contact_events.json"
        )
        self.object_tracks_dir = (
            config.object_tracks_dir.expanduser().resolve()
            if config.object_tracks_dir is not None
            else None
        )
        self.video = self.stereo_root / "calibration/left_rectified_working.mp4"
        self.calibration_path = self.stereo_root / "calibration/calibration.npz"
        self.depth_dir = self.stereo_root / "depth/frames"
        pose_dir = self.stereo_root / "vipe/pose"
        canonical_pose = pose_dir / "rectified_left.npz"
        legacy_candidates = sorted(pose_dir.glob("*_rectified_left.npz"))
        self.pose_path = (
            canonical_pose
            if canonical_pose.is_file()
            else legacy_candidates[0]
            if legacy_candidates
            else Path("__missing__")
        )
        for path in (self.video, self.calibration_path, self.depth_dir, self.pose_path):
            if not path.exists():
                raise FileNotFoundError(path)
        if not self.reasoning_path.is_file():
            raise FileNotFoundError(self.reasoning_path)
        if self.object_tracks_dir is not None and not self.object_tracks_dir.is_dir():
            raise FileNotFoundError(self.object_tracks_dir)
        self.frame_count, self.width, self.height, self.fps = _video_metadata(self.video)
        with np.load(self.calibration_path) as archive:
            self.K = np.asarray(
                archive["K_working"] if "K_working" in archive else archive["K"],
                dtype=np.float64,
            )
            self.left_valid = np.asarray(
                archive.get("left_valid", np.ones((self.height, self.width), dtype=bool)),
                dtype=bool,
            )
        with np.load(self.pose_path) as archive:
            poses = np.asarray(archive["data"], dtype=np.float64)
            indices = np.asarray(
                archive["inds"] if "inds" in archive else np.arange(len(poses)),
                dtype=np.int64,
            )
        pose_by_frame = {int(index): pose for index, pose in zip(indices, poses)}
        self.camera_to_world = np.stack(
            [pose_by_frame.get(index, np.eye(4, dtype=np.float64)) for index in range(self.frame_count)]
        )

    def discover_objects(self) -> list[StereoObject]:
        """Join Codex canonical records to passing primary SAM3D meshes."""
        reconstruction_path = self.source_run / "objects/sam3d_objects/reconstruction_manifest.json"
        reasoning = json.loads(self.reasoning_path.read_text(encoding="utf-8"))
        reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8"))
        jobs = {
            str(record["global_object_id"]): record
            for record in reconstruction["jobs"]
            if record.get("role") == "primary" and record.get("status") == "pass"
        }
        objects: list[StereoObject] = []
        for record in reasoning.get("objects", []):
            object_id = str(record["global_object_id"])
            if object_id not in jobs:
                continue
            canonical = record.get("canonical", {})
            seed = int(canonical.get("frame", self.frame_count - 1))
            objects.append(
                StereoObject(
                    object_id=object_id,
                    name=str(record["name"]),
                    prompt=str(record.get("sam3_prompt") or record["name"]),
                    seed_frame=int(np.clip(seed, 0, self.frame_count - 1)),
                    mesh=Path(jobs[object_id]["output_glb_path"]).resolve(),
                    metric_mesh=bool(jobs[object_id].get("metric_mesh", False)),
                )
            )
        if not objects:
            raise RuntimeError("No passing primary object meshes match the reasoning output")
        return objects

    def _tracked_masks(self, item: StereoObject) -> np.ndarray:
        """Load one Codex-bbox-seeded full-video SAM3 mask sequence."""

        if self.object_tracks_dir is None:
            raise RuntimeError("No SAM3 object-track directory was configured")
        direct_path = self.object_tracks_dir / f"{item.object_id}.npz"
        path = (
            direct_path
            if direct_path.is_file()
            else self.object_tracks_dir / "tracklets" / f"{item.object_id}.npz"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as archive:
            width = int(np.asarray(archive["mask_width"]).item())
            masks = np.unpackbits(
                np.asarray(archive["packed_masks"], dtype=np.uint8), axis=2
            )[..., :width].astype(bool)
            source_key = (
                "source_indices"
                if "source_indices" in archive.files
                else "source_frames"
            )
            source_frames = np.asarray(archive[source_key], dtype=np.int64)
        expected_frames = np.arange(self.frame_count, dtype=np.int64)
        if masks.shape != (self.frame_count, self.height, self.width):
            raise ValueError(f"Unexpected tracked-mask shape in {path}: {masks.shape}")
        if not np.array_equal(source_frames, expected_frames):
            raise ValueError(f"Tracked masks are not dense frame-aligned data: {path}")
        return masks

    def _read_rgb(self, frame_id: int) -> np.ndarray:
        capture = cv2.VideoCapture(str(self.video))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not decode left frame {frame_id}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()

    def segment_seed_masks(self, objects: list[StereoObject]) -> dict[str, np.ndarray]:
        """Run SAM3 once per distinct seed image and retain the best prompt instance."""
        if self.object_tracks_dir is not None:
            mask_root = self.output / "seed_masks"
            mask_root.mkdir(parents=True, exist_ok=True)
            masks: dict[str, np.ndarray] = {}
            for item in objects:
                full_masks = self._tracked_masks(item)
                mask = full_masks[item.seed_frame]
                if int(mask.sum()) < self.config.minimum_mask_area:
                    raise RuntimeError(
                        f"Tracked seed mask for {item.object_id} at frame "
                        f"{item.seed_frame} has only {int(mask.sum())} pixels"
                    )
                masks[item.object_id] = mask
                rgb = self._read_rgb(item.seed_frame)
                object_dir = mask_root / item.object_id
                object_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(object_dir / "mask.png"), mask.astype(np.uint8) * 255)
                overlay = rgb.copy()
                overlay[mask] = np.rint(
                    0.35 * overlay[mask] + 0.65 * np.asarray([255, 70, 20])
                ).astype(np.uint8)
                cv2.putText(
                    overlay,
                    f"{item.object_id}: Codex-bbox SAM3 track seed",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                imageio.imwrite(object_dir / "overlay.png", overlay)
                (object_dir / "report.json").write_text(
                    json.dumps(
                        {
                            "object_id": item.object_id,
                            "object_name": item.name,
                            "seed_frame": item.seed_frame,
                            "mask_pixels": int(mask.sum()),
                            "source": str(
                                self.object_tracks_dir
                                / f"{item.object_id}.npz"
                            ),
                            "prompt_type": "codex_canonical_bbox",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            return masks

        sam3_parent = REPOSITORY_ROOT / "thirdparty"
        if str(sam3_parent) not in sys.path:
            sys.path.insert(0, str(sam3_parent))
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(device="cuda")
        processor = Sam3Processor(
            model,
            device="cuda",
            confidence_threshold=self.config.confidence_threshold,
        )
        mask_root = self.output / "seed_masks"
        mask_root.mkdir(parents=True, exist_ok=True)
        masks: dict[str, np.ndarray] = {}
        frames: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
        try:
            for item in objects:
                if item.seed_frame not in frames:
                    rgb = self._read_rgb(item.seed_frame)
                    frames[item.seed_frame] = (
                        rgb,
                        processor.set_image(Image.fromarray(rgb)),
                    )
                rgb, state = frames[item.seed_frame]
                best_result: tuple[float, str, np.ndarray, int] | None = None
                attempted_prompts = _seed_prompt_candidates(item)
                for prompt in attempted_prompts:
                    state = processor.set_text_prompt(prompt, state)
                    candidates = state.get("masks")
                    scores = state.get("scores")
                    candidate_array = (
                        candidates.detach().cpu().numpy()
                        if hasattr(candidates, "detach")
                        else np.asarray(candidates)
                    )
                    score_array = (
                        scores.detach().cpu().numpy()
                        if hasattr(scores, "detach")
                        else np.asarray(scores)
                    ).reshape(-1)
                    if candidate_array.ndim == 4 and candidate_array.shape[1] == 1:
                        candidate_array = candidate_array[:, 0]
                    accepted = [
                        index
                        for index, candidate in enumerate(candidate_array)
                        if int(np.asarray(candidate, dtype=bool).sum())
                        >= self.config.minimum_mask_area
                    ]
                    if accepted:
                        candidate_index = max(
                            accepted, key=lambda index: float(score_array[index])
                        )
                        result = (
                            float(score_array[candidate_index]),
                            prompt,
                            np.asarray(candidate_array[candidate_index], dtype=bool),
                            int(len(candidate_array)),
                        )
                        if best_result is None or result[0] > best_result[0]:
                            best_result = result
                    processor.reset_all_prompts(state)
                if best_result is None:
                    raise RuntimeError(
                        f"SAM3 found no usable mask for {item.object_id} at frame "
                        f"{item.seed_frame}; prompts={attempted_prompts!r}"
                    )
                best_score, used_prompt, mask, candidate_count = best_result
                if mask.shape != (self.height, self.width):
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (self.width, self.height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                masks[item.object_id] = mask
                object_dir = mask_root / item.object_id
                object_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(object_dir / "mask.png"), mask.astype(np.uint8) * 255)
                overlay = rgb.copy()
                overlay[mask] = np.rint(
                    0.35 * overlay[mask] + 0.65 * np.asarray([255, 70, 20])
                ).astype(np.uint8)
                cv2.putText(
                    overlay,
                    f"{item.object_id}: {item.name} score={best_score:.3f}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                imageio.imwrite(object_dir / "overlay.png", overlay)
                (object_dir / "report.json").write_text(
                    json.dumps(
                        {
                            "object_id": item.object_id,
                            "object_name": item.name,
                            "prompt": item.prompt,
                            "attempted_prompts": list(attempted_prompts),
                            "used_prompt": used_prompt,
                            "seed_frame": item.seed_frame,
                            "score": best_score,
                            "candidate_count": candidate_count,
                            "mask_pixels": int(mask.sum()),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        finally:
            del processor, model
            gc.collect()
            torch.cuda.empty_cache()
        return masks

    def _metric_mesh(
        self, item: StereoObject, mask: np.ndarray
    ) -> tuple[Path, dict[str, Any]]:
        if item.metric_mesh:
            mesh = _load_trimesh(item.mesh)
            original_vertices = int(len(mesh.vertices))
            original_faces = int(len(mesh.faces))
            output_dir = self.output / "objects" / item.object_id / "scale"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_mesh = output_dir / f"{_safe_name(item.name)}_public_metric.glb"
            trimesh.Scene(mesh).export(output_mesh)
            report = {
                "schema_version": "ego-scene-hoi-public-metric-object-mesh-v1",
                "object_id": item.object_id,
                "object_name": item.name,
                "source_frame": item.seed_frame,
                "source_mesh": str(item.mesh),
                "output_mesh": str(output_mesh),
                "depth_source": None,
                "scale_mode": "released_metric_mesh",
                "uniform_scale_m_per_unit": 1.0,
                "scaled_mesh_extents_m": mesh.extents.tolist(),
                "input_to_metric_centered": np.eye(4).tolist(),
                "source_vertices": original_vertices,
                "source_faces": original_faces,
                "tracking_vertices": int(len(mesh.vertices)),
                "tracking_faces": int(len(mesh.faces)),
                "geometry_simplified": False,
                "appearance_preserved": mesh.visual.kind == "texture",
            }
            (output_dir / "report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            return output_mesh, report
        depth = np.asarray(np.load(self.depth_dir / f"{item.seed_frame:06d}.npy"), dtype=np.float32)
        yy, xx = np.indices(depth.shape, dtype=np.float64)
        valid = mask & self.left_valid & np.isfinite(depth) & (depth > 0.10) & (depth < 10.0)
        z = depth[valid].astype(np.float64)
        points = np.column_stack(
            (
                (xx[valid] - self.K[0, 2]) * z / self.K[0, 0],
                (yy[valid] - self.K[1, 2]) * z / self.K[1, 1],
                z,
            )
        )
        cloud_frame = _robust_frame(
            points,
            np.ones(len(points), dtype=np.float64),
            confidence_quantile=0.0,
            extent_quantile=self.config.extent_quantile,
        )
        mesh = _load_trimesh(item.mesh)
        mesh_center, mesh_axes, mesh_extents = _mesh_frame(mesh)
        scales = _axis_scales(
            mesh_extents,
            cloud_frame.extents,
            mode=self.config.scale_mode,
            thickness_weight=self.config.thickness_weight,
            max_anisotropy=self.config.max_anisotropy,
        )
        transform = _scale_transform(mesh_center, mesh_axes, scales)
        metric_mesh = mesh.copy()
        metric_mesh.apply_transform(transform)
        output_dir = self.output / "objects" / item.object_id / "scale"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_mesh = output_dir / f"{_safe_name(item.name)}_stereo_metric_centered.glb"
        trimesh.Scene(metric_mesh).export(output_mesh)
        report = {
            "schema_version": "ego-scene-hoi-stereo-object-scale-v1",
            "object_id": item.object_id,
            "object_name": item.name,
            "source_frame": item.seed_frame,
            "source_mesh": str(item.mesh),
            "output_mesh": str(output_mesh),
            "depth_source": "FoundationStereo metric left depth",
            "scale_mode": self.config.scale_mode,
            "mask_pixels": int(mask.sum()),
            "valid_depth_pixels": int(valid.sum()),
            "robust_inlier_points": int(cloud_frame.inliers.sum()),
            "pointcloud_extents_m": cloud_frame.extents.tolist(),
            "mesh_extents_input": mesh_extents.tolist(),
            "uniform_scale_m_per_unit": (
                float(scales[0]) if self.config.scale_mode == "isotropic" else None
            ),
            "axis_scales_m_per_unit": scales.tolist(),
            "max_anisotropy": float(self.config.max_anisotropy),
            "scaled_mesh_extents_m": (mesh_extents * scales).tolist(),
            "input_to_metric_centered": transform.tolist(),
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return output_mesh, report

    def prepare_shared_rgbd(self) -> Path:
        """Export rectified RGB/depth once for all object datasets."""
        shared = self.output / "dataset"
        rgb_dir, depth_out = shared / "rgb", shared / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_out.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(self.video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {self.video}")
        try:
            for frame_id in range(self.frame_count):
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode frame {frame_id}")
                depth = np.asarray(
                    np.load(self.depth_dir / f"{frame_id:06d}.npy"), dtype=np.float32
                )
                valid = self.left_valid & np.isfinite(depth) & (depth > 0.0) & (depth < 65.535)
                millimeters = np.zeros(depth.shape, dtype=np.uint16)
                millimeters[valid] = np.clip(
                    np.rint(depth[valid] * 1000.0), 1, np.iinfo(np.uint16).max
                ).astype(np.uint16)
                stem = f"{frame_id:06d}"
                cv2.imwrite(str(rgb_dir / f"{stem}.png"), bgr)
                cv2.imwrite(str(depth_out / f"{stem}.png"), millimeters)
        finally:
            capture.release()
        return shared

    def prepare_object_dataset(
        self,
        item: StereoObject,
        seed_mask: np.ndarray,
        shared: Path,
    ) -> Path:
        """Create a standard all-frame dataset with the measured seed mask."""
        dataset = self.output / "objects" / item.object_id / "dataset"
        dataset.mkdir(parents=True, exist_ok=True)
        for name in ("rgb", "depth"):
            link = dataset / name
            if link.is_symlink() or link.exists():
                if link.resolve() != (shared / name).resolve():
                    raise FileExistsError(link)
            else:
                link.symlink_to((shared / name).resolve(), target_is_directory=True)
        masks = dataset / "masks"
        intrinsics_dir = dataset / "cam_K_per_frame"
        masks.mkdir(exist_ok=True)
        intrinsics_dir.mkdir(exist_ok=True)
        full_masks: np.ndarray | None = None
        if self.config.adaptive_reregister:
            if self.object_tracks_dir is not None:
                full_masks = self._tracked_masks(item)
            else:
                archive_path = self.stereo_root / "tracks/sam3/refined_masks/object_masks.npz"
                if not archive_path.is_file():
                    raise FileNotFoundError(
                        f"Adaptive re-registration requires {archive_path}; rerun the mask stage"
                    )
                with np.load(archive_path) as archive:
                    object_ids = [str(value) for value in archive["object_ids"].tolist()]
                    if item.object_id not in object_ids:
                        raise KeyError(f"{item.object_id} not present in {archive_path}")
                    packed = np.asarray(
                        archive["packed_masks"][object_ids.index(item.object_id)], dtype=np.uint8
                    )
                    width = int(np.asarray(archive["mask_width"]).item())
                full_masks = np.unpackbits(packed, axis=2)[..., :width].astype(bool)
            if full_masks.shape != (self.frame_count, self.height, self.width):
                raise ValueError(
                    f"Unexpected object-mask shape {full_masks.shape} for {item.object_id}"
                )
        blank = np.zeros((self.height, self.width), dtype=np.uint8)
        for frame_id in range(self.frame_count):
            stem = f"{frame_id:06d}"
            frame_mask = (
                full_masks[frame_id]
                if full_masks is not None
                else seed_mask if frame_id == item.seed_frame else blank
            )
            cv2.imwrite(
                str(masks / f"{stem}.png"),
                np.asarray(frame_mask, dtype=np.uint8) * 255,
            )
            np.savetxt(intrinsics_dir / f"{stem}.txt", self.K, fmt="%.10f")
        intrinsics = np.repeat(self.K[None], self.frame_count, axis=0)
        source_indices = np.arange(self.frame_count, dtype=np.int32)
        np.savez_compressed(
            dataset / "camera_K.npz",
            intrinsics=intrinsics,
            median_intrinsics=self.K,
            source_indices=source_indices,
        )
        np.savez_compressed(
            dataset / "camera_poses_world.npz",
            rigid_camera_to_world=self.camera_to_world,
            camera_to_world=self.camera_to_world,
            source_indices=source_indices,
            timestamps_ns=np.rint(source_indices / self.fps * 1e9).astype(np.int64),
        )
        np.savez_compressed(
            dataset / "frame_mapping.npz",
            local_frame_indices=source_indices,
            pipeline_frame_indices=source_indices,
            source_indices=source_indices,
        )
        np.savetxt(dataset / "cam_K.txt", self.K, fmt="%.10f")
        report = {
            "format": "ego-scene-hoi-foundationpose-stereo-left-v1",
            "object_id": item.object_id,
            "object_name": item.name,
            "fps": self.fps,
            "image_size": [self.height, self.width],
            "frame_count": self.frame_count,
            "seed_frame": item.seed_frame,
            "intrinsics_source": str(self.calibration_path),
            "depth_source": "FoundationStereo left metric depth",
            "camera_trajectory_source": str(self.pose_path),
            "uses_gt_camera_extrinsics": False,
            "mask_sequence": "sam3_per_object" if full_masks is not None else "canonical_only",
        }
        (dataset / "dataset_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return dataset

    def run(self) -> Path:
        """Execute segmentation, metric scaling, tracking, and video visualization."""
        self.output.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        objects = self.discover_objects()
        masks = self.segment_seed_masks(objects)
        shared = self.prepare_shared_rgbd()
        records: list[dict[str, Any]] = []
        for item in objects:
            metric_mesh, scale_report = self._metric_mesh(item, masks[item.object_id])
            dataset = self.prepare_object_dataset(item, masks[item.object_id], shared)
            tracking_dir = (
                self.output / "objects" / item.object_id /
                ("tracking_adaptive" if self.config.adaptive_reregister else "tracking")
            )
            arguments = SimpleNamespace(
                dataset_dir=dataset,
                mesh=metric_mesh,
                output_dir=tracking_dir,
                object_label=_safe_name(item.name),
                seed_frame=item.seed_frame,
                est_refine_iter=self.config.estimation_iterations,
                track_refine_iter=self.config.tracking_iterations,
                adaptive_reregister=self.config.adaptive_reregister,
                appearance_mode=self.config.appearance_mode,
                reregister_interval=20,
                minimum_silhouette_iou=0.30,
                minimum_mask_coverage=0.50,
                reregister_accept_margin=0.02,
                maximum_reregister_rotation_degrees=(
                    self.config.maximum_reregister_rotation_degrees
                ),
                maximum_reregister_translation_meters=(
                    self.config.maximum_reregister_translation_meters
                ),
                maximum_reregister_translation_fraction=(
                    self.config.maximum_reregister_translation_fraction
                ),
                max_frames=None,
                write_overlay=self.config.write_overlay,
                registration_only=self.config.registration_only,
                render_batch_size=self.config.render_batch_size,
            )
            run_bidirectional_tracking(arguments)
            records.append(
                {
                    "object_id": item.object_id,
                    "object_name": item.name,
                    "prompt": item.prompt,
                    "seed_frame": item.seed_frame,
                    "metric_mesh": str(metric_mesh),
                    "scale": scale_report,
                    "poses": str(tracking_dir / "poses.npz"),
                    "overlay": (
                        str(tracking_dir / "foundationpose_tracking_vis.mp4")
                        if self.config.write_overlay
                        else None
                    ),
                }
            )
            gc.collect()
            torch.cuda.empty_cache()
        comparison = (
            self._write_comparison_video(records)
            if self.config.write_overlay
            else None
        )
        manifest = self.output / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "ego-scene-hoi-stereo-foundationpose-v1",
                    "status": "pass",
                    "left_video": str(self.video),
                    "depth_source": str(self.depth_dir),
                    "intrinsics_source": str(self.calibration_path),
                    "camera_poses_source": str(self.pose_path),
                    "uses_gt_camera_extrinsics": False,
                    "adaptive_reregistration": self.config.adaptive_reregister,
                    "registration_only": self.config.registration_only,
                    "reasoning_source": str(self.reasoning_path),
                    "object_tracks_source": (
                        str(self.object_tracks_dir)
                        if self.object_tracks_dir is not None
                        else None
                    ),
                    "objects": records,
                    "comparison_video": str(comparison) if comparison is not None else None,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest

    def _write_comparison_video(self, records: list[dict[str, Any]]) -> Path:
        """Place source and per-object pose overlays in one diagnostic video."""
        output = self.output / (
            "foundationpose_overlay_comparison_adaptive.mp4"
            if self.config.adaptive_reregister
            else "foundationpose_overlay_comparison.mp4"
        )
        paths = [self.video] + [Path(record["overlay"]) for record in records]
        labels = ["rectified left"] + [str(record["object_name"]) for record in records]
        captures = [cv2.VideoCapture(str(path)) for path in paths]
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.width * len(paths), self.height),
        )
        if not all(capture.isOpened() for capture in captures) or not writer.isOpened():
            raise RuntimeError("Could not create FoundationPose comparison video")
        try:
            for frame_id in range(self.frame_count):
                panels = []
                for capture, label in zip(captures, labels):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Comparison video ended at frame {frame_id}")
                    cv2.rectangle(frame, (0, 0), (min(self.width, 320), 36), (0, 0, 0), -1)
                    cv2.putText(
                        frame,
                        label,
                        (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    panels.append(frame)
                writer.write(np.concatenate(panels, axis=1))
        finally:
            for capture in captures:
                capture.release()
            writer.release()
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reasoning", type=Path)
    parser.add_argument("--object-tracks-dir", type=Path)
    parser.add_argument(
        "--adaptive-reregister",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--registration-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Register the canonical frame without running track_one.",
    )
    parser.add_argument(
        "--write-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write FoundationPose's per-object diagnostic overlays.",
    )
    parser.add_argument(
        "--appearance-mode",
        choices=("rgb", "grayscale", "depth"),
        default="grayscale",
        help="Appearance presented to the FoundationPose refiner and scorer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = StereoFoundationPoseExperiment(
        StereoFoundationPoseConfig(
            stereo_root=args.stereo_root,
            source_run=args.source_run,
            output_dir=args.output_dir,
            reasoning_path=args.reasoning,
            object_tracks_dir=args.object_tracks_dir,
            adaptive_reregister=args.adaptive_reregister,
            registration_only=args.registration_only,
            write_overlay=args.write_overlay,
            appearance_mode=args.appearance_mode,
        )
    )
    print(experiment.run())


if __name__ == "__main__":
    main()
