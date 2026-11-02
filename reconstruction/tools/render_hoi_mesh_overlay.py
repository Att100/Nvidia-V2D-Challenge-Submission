#!/usr/bin/env python3
"""Render hand and untextured object meshes over a pipeline video.

The renderer consumes the shared world-coordinate outputs produced by pipeline
v2. It uses nvdiffrast for a single depth-tested pass, so hands and objects
occlude one another correctly. Object visibility comes from the global pose
track rather than the held-pose policy used outside observed intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh


OBJECT_COLORS = (
    (0.95, 0.38, 0.16),  # orange-red
    (0.16, 0.68, 0.95),  # cyan-blue
    (0.68, 0.34, 0.92),  # violet
    (0.95, 0.76, 0.16),  # amber
    (0.12, 0.72, 0.72),  # teal
    (0.88, 0.52, 0.18),  # ochre
    (0.35, 0.48, 0.96),  # royal blue
    (0.72, 0.64, 0.18),  # olive-gold
    (0.72, 0.30, 0.30),  # brick red
    (0.42, 0.28, 0.82),  # indigo
)
HAND_COLORS = (
    (0.28, 0.90, 0.36),  # left: green
    (0.98, 0.36, 0.78),  # right: pink
)


@dataclass(frozen=True)
class StaticMesh:
    """A canonical mesh and its complete object-to-world pose sequence."""

    object_id: str
    label: str
    vertices: torch.Tensor
    normals: torch.Tensor
    faces: torch.Tensor
    poses: torch.Tensor
    observed: np.ndarray
    color: tuple[float, float, float]


@dataclass(frozen=True)
class HandSequence:
    """World-space MANO vertices, topology, and visibility over time."""

    labels: tuple[str, ...]
    vertices_world: torch.Tensor
    faces: tuple[torch.Tensor, ...]
    valid: np.ndarray
    colors: tuple[tuple[float, float, float], ...]


def _first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("None of the candidate paths exists")


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    """Load a GLB/mesh and bake scene-node transforms."""

    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_geometry"):
            loaded = loaded.to_geometry()
        else:
            loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected triangle mesh: {path}")
    return loaded


def _object_labels(run_root: Path) -> dict[str, str]:
    path = run_root / "objects/object_pose_tracks.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(record.get("object_id")): str(
            record.get("object_name", record.get("object_id"))
        )
        for record in value.get("objects", [])
    }


class HOIMeshVideoRenderer:
    """Depth-test world-space object/MANO meshes and alpha-blend them on RGB."""

    def __init__(
        self,
        *,
        run_root: Path,
        hand_motion: Path | None = None,
        foundationpose_manifest: Path | None = None,
        object_alpha: float = 0.94,
        hand_alpha: float = 0.90,
        object_pose_stage: str = "auto",
        include_supports: bool = True,
        device: str = "cuda",
    ) -> None:
        self.run_root = run_root.resolve()
        self.hand_motion = hand_motion.resolve() if hand_motion is not None else None
        self.foundationpose_manifest = (
            foundationpose_manifest.resolve()
            if foundationpose_manifest is not None
            else None
        )
        self.device = torch.device(
            device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu"
        )
        if self.device.type != "cuda":
            raise RuntimeError("nvdiffrast mesh-video rendering currently requires CUDA")
        if not 0.0 <= object_alpha <= 1.0 or not 0.0 <= hand_alpha <= 1.0:
            raise ValueError("Mesh alpha values must be in [0, 1]")
        self.object_alpha = float(object_alpha)
        self.hand_alpha = float(hand_alpha)
        self.object_pose_stage = object_pose_stage
        self.include_supports = bool(include_supports)
        self.context = dr.RasterizeCudaContext(device=self.device)

        camera_archive = np.load(self.run_root / "4d/cameras.npz")
        camera_to_world = camera_archive["camera_poses"].astype(np.float32)
        self.world_to_camera = torch.as_tensor(
            np.linalg.inv(camera_to_world).astype(np.float32), device=self.device
        )
        # The render camera must be the same camera that defines c2w and all
        # world-space meshes.  Stereo runs use rectified-left geometry, while
        # ``input/cameras.npz`` describes the native RGB stream and therefore
        # cannot be used for this overlay.
        if "intrinsics" in camera_archive.files:
            self.intrinsics = camera_archive["intrinsics"].astype(np.float32)
        else:
            input_camera = np.load(self.run_root / "input/cameras.npz")
            self.intrinsics = input_camera["intrinsics"].astype(np.float32)
        self.frame_count = len(camera_to_world)
        self.objects = self._load_objects()
        self.hands = self._load_hands()

    def _load_manifest_objects(self) -> list[StaticMesh]:
        """Load meshes and poses directly from a FoundationPose experiment."""

        if self.foundationpose_manifest is None:
            return []
        manifest = json.loads(
            self.foundationpose_manifest.read_text(encoding="utf-8")
        )
        result: list[StaticMesh] = []
        for index, record in enumerate(manifest.get("objects", [])):
            object_id = str(record["object_id"])
            mesh_path = Path(record["metric_mesh"])
            pose_path = Path(record["poses"])
            with np.load(pose_path) as archive:
                pose_key = (
                    "object_to_world"
                    if "object_to_world" in archive.files
                    else "ob_in_world"
                )
                poses = np.asarray(archive[pose_key], dtype=np.float32)
            observed = np.isfinite(poses).all(axis=(1, 2))
            track_path = (
                self.run_root
                / "tracks/sam3/refined_masks"
                / f"{object_id}.npz"
            )
            if track_path.is_file():
                with np.load(track_path) as archive:
                    width = int(np.asarray(archive["mask_width"]).item())
                    masks = np.unpackbits(
                        np.asarray(archive["packed_masks"], dtype=np.uint8),
                        axis=2,
                    )[..., :width]
                observed &= masks.any(axis=(1, 2))
            if len(poses) != self.frame_count:
                raise ValueError(f"Timeline mismatch for {object_id}")
            mesh = _load_trimesh(mesh_path)
            result.append(
                StaticMesh(
                    object_id=object_id,
                    label=str(record.get("object_name", object_id)),
                    vertices=torch.as_tensor(
                        np.asarray(mesh.vertices, dtype=np.float32),
                        device=self.device,
                    ),
                    normals=torch.as_tensor(
                        np.asarray(mesh.vertex_normals, dtype=np.float32),
                        device=self.device,
                    ),
                    faces=torch.as_tensor(
                        np.asarray(mesh.faces, dtype=np.int32), device=self.device
                    ),
                    poses=torch.as_tensor(poses, device=self.device),
                    observed=observed,
                    color=OBJECT_COLORS[index % len(OBJECT_COLORS)],
                )
            )
        if not result:
            raise RuntimeError(
                f"No objects found in {self.foundationpose_manifest}"
            )
        return result

    def _load_objects(self) -> tuple[StaticMesh, ...]:
        """Load final interaction-object and static-support geometry."""

        labels = _object_labels(self.run_root)
        result = self._load_manifest_objects()
        object_dirs = (
            []
            if self.foundationpose_manifest is not None
            else sorted(
                path
                for path in (self.run_root / "objects").glob("object_*")
                if path.is_dir()
            )
        )
        for index, object_dir in enumerate(object_dirs):
            object_id = object_dir.name
            mesh_path = _first_existing(
                [
                    *sorted((object_dir / "scale_primary").glob("*_metric_centered.glb")),
                    *sorted((object_dir / "scale_primary").glob("*.glb")),
                    *sorted((object_dir / "canonical").glob("*.glb")),
                ]
            )
            pose_candidates = {
                "foundationpose": [object_dir / "pose_track/poses.npz"],
                "surface": [
                    self.run_root / "optim/object_surface_v1" / object_id / "poses.npz",
                    object_dir / "pose_track/poses.npz",
                ],
                "haco": [
                    self.run_root
                    / "optim/contact_haco_joint_v2"
                    / object_id
                    / "poses.npz",
                    self.run_root / "optim/contact_haco_v1" / object_id / "poses.npz",
                ],
                "auto": [
                    self.run_root
                    / "optim/contact_haco_joint_v2"
                    / object_id
                    / "poses.npz",
                    self.run_root / "optim/contact_haco_v1" / object_id / "poses.npz",
                    self.run_root
                    / "optim/object_surface_v1"
                    / object_id
                    / "poses.npz",
                    object_dir / "pose_track/poses.npz",
                ],
            }
            pose_path = _first_existing(pose_candidates[self.object_pose_stage])
            pose_archive = np.load(pose_path)
            pose_key = (
                "object_to_world"
                if "object_to_world" in pose_archive.files
                else "ob_in_world"
            )
            poses = pose_archive[pose_key].astype(np.float32)
            track = np.load(object_dir / "pose_track/poses.npz")
            observed = track["observed"].astype(bool)
            if len(poses) != self.frame_count or len(observed) != self.frame_count:
                raise ValueError(f"Timeline mismatch for {object_id}")
            mesh = _load_trimesh(mesh_path)
            result.append(
                StaticMesh(
                    object_id=object_id,
                    label=labels.get(object_id, object_id),
                    vertices=torch.as_tensor(
                        np.asarray(mesh.vertices, dtype=np.float32), device=self.device
                    ),
                    normals=torch.as_tensor(
                        np.asarray(mesh.vertex_normals, dtype=np.float32),
                        device=self.device,
                    ),
                    faces=torch.as_tensor(
                        np.asarray(mesh.faces, dtype=np.int32), device=self.device
                    ),
                    poses=torch.as_tensor(poses, device=self.device),
                    observed=observed,
                    color=OBJECT_COLORS[index % len(OBJECT_COLORS)],
                )
            )

        support_manifest = self.run_root / "support_objects/pose/manifest.json"
        if self.include_supports and support_manifest.is_file():
            manifest = json.loads(support_manifest.read_text(encoding="utf-8"))
            for record in manifest.get("objects", []):
                support_id = str(record["global_support_id"])
                mesh_path = Path(record["metric_mesh"])
                pose_path = Path(record["poses"])
                if not mesh_path.is_file():
                    mesh_path = support_manifest.parent / support_id / mesh_path.name
                if not pose_path.is_file():
                    pose_path = support_manifest.parent / support_id / pose_path.name
                pose_archive = np.load(pose_path)
                if "ob_in_world" in pose_archive.files:
                    poses = pose_archive["ob_in_world"].astype(np.float32)
                elif "object_to_world" in pose_archive.files:
                    poses = pose_archive["object_to_world"].astype(np.float32)
                else:
                    poses = (
                        pose_archive["camera_to_world"]
                        @ pose_archive["ob_in_cam"]
                    ).astype(np.float32)
                if len(poses) != self.frame_count:
                    raise ValueError(f"Timeline mismatch for {support_id}")
                mesh = _load_trimesh(mesh_path)
                color_index = len(result)
                result.append(
                    StaticMesh(
                        object_id=support_id,
                        label=f"support: {record.get('name', support_id)}",
                        vertices=torch.as_tensor(
                            np.asarray(mesh.vertices, dtype=np.float32),
                            device=self.device,
                        ),
                        normals=torch.as_tensor(
                            np.asarray(mesh.vertex_normals, dtype=np.float32),
                            device=self.device,
                        ),
                        faces=torch.as_tensor(
                            np.asarray(mesh.faces, dtype=np.int32),
                            device=self.device,
                        ),
                        poses=torch.as_tensor(poses, device=self.device),
                        observed=np.ones(self.frame_count, dtype=bool),
                        color=OBJECT_COLORS[color_index % len(OBJECT_COLORS)],
                    )
                )
        return tuple(result)

    def _load_hands(self) -> HandSequence:
        hand_path = (
            self.hand_motion
            if self.hand_motion is not None
            else _first_existing([
                self.run_root / "hands/umetrack_mano_surface/hand_motion.npz",
                self.run_root / "hands/haco_joint_refined/hand_motion.npz",
            ])
        )
        archive = np.load(hand_path, allow_pickle=True)
        labels = tuple(str(item) for item in archive["hand_labels"].tolist())
        vertices = archive["vertices_world"].astype(np.float32)
        valid = archive["valid"].astype(bool)
        if len(vertices) != self.frame_count:
            raise ValueError("Hand and camera timelines differ")
        faces: list[torch.Tensor] = []
        for label in labels:
            key = f"faces_{label}"
            if key not in archive.files:
                raise KeyError(f"Missing {key} in {hand_path}")
            faces.append(
                torch.as_tensor(archive[key].astype(np.int32), device=self.device)
            )
        return HandSequence(
            labels=labels,
            vertices_world=torch.as_tensor(vertices, device=self.device),
            faces=tuple(faces),
            valid=valid,
            colors=tuple(
                HAND_COLORS[0]
                if label.lower().startswith("left")
                else HAND_COLORS[1]
                for label in labels
            ),
        )

    @staticmethod
    def _camera_to_clip(
        vertices_camera: torch.Tensor,
        intrinsic: torch.Tensor,
        width: int,
        height: int,
        near: float = 0.01,
        far: float = 100.0,
    ) -> torch.Tensor:
        """Map OpenCV camera coordinates to OpenGL homogeneous clip space."""

        x, y, z = vertices_camera.unbind(dim=1)
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        x_clip = (2.0 * fx / width) * x + (2.0 * cx / width - 1.0) * z
        y_clip = (-2.0 * fy / height) * y + (1.0 - 2.0 * cy / height) * z
        z_clip = ((far + near) / (far - near)) * z - (
            2.0 * far * near / (far - near)
        )
        return torch.stack((x_clip, y_clip, z_clip, z), dim=1)

    def _frame_geometry(
        self, frame_index: int
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[tuple[str, tuple[float, float, float]]],
    ]:
        """Assemble visible meshes into one indexed GPU triangle buffer."""

        vertices_world: list[torch.Tensor] = []
        normals_world: list[torch.Tensor] = []
        faces: list[torch.Tensor] = []
        colors: list[torch.Tensor] = []
        alpha: list[torch.Tensor] = []
        legend: list[tuple[str, tuple[float, float, float]]] = []
        offset = 0
        for item in self.objects:
            pose = item.poses[frame_index]
            if not item.observed[frame_index] or not bool(torch.isfinite(pose).all()):
                continue
            world = item.vertices @ pose[:3, :3].T + pose[:3, 3]
            normal = torch.nn.functional.normalize(
                item.normals @ pose[:3, :3].T, dim=1, eps=1.0e-8
            )
            vertices_world.append(world)
            normals_world.append(normal)
            faces.append(item.faces + offset)
            colors.append(
                torch.tensor(item.color, dtype=torch.float32, device=self.device)
                .expand(len(world), -1)
            )
            alpha.append(
                torch.full(
                    (len(world), 1),
                    self.object_alpha,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            offset += len(world)
            legend.append((item.label, item.color))
        for hand_index, label in enumerate(self.hands.labels):
            if not self.hands.valid[frame_index, hand_index]:
                continue
            world = self.hands.vertices_world[frame_index, hand_index]
            if not bool(torch.isfinite(world).all()):
                continue
            hand_faces = self.hands.faces[hand_index]
            normal = self._vertex_normals(world, hand_faces)
            vertices_world.append(world)
            normals_world.append(normal)
            faces.append(hand_faces + offset)
            hand_color = self.hands.colors[hand_index]
            colors.append(
                torch.tensor(hand_color, dtype=torch.float32, device=self.device)
                .expand(len(world), -1)
            )
            alpha.append(
                torch.full(
                    (len(world), 1),
                    self.hand_alpha,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            offset += len(world)
            legend.append((f"{label} hand", hand_color))
        if not vertices_world:
            empty_float = torch.empty((0, 3), device=self.device)
            empty_faces = torch.empty((0, 3), dtype=torch.int32, device=self.device)
            return (
                empty_float,
                empty_float,
                empty_faces,
                empty_float,
                torch.empty((0, 1), device=self.device),
                legend,
            )
        return (
            torch.cat(vertices_world, dim=0),
            torch.cat(normals_world, dim=0),
            torch.cat(faces, dim=0),
            torch.cat(colors, dim=0),
            torch.cat(alpha, dim=0),
            legend,
        )

    @staticmethod
    def _vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Compute area-weighted smooth vertex normals for a dynamic mesh."""

        triangles = vertices[faces.long()]
        face_normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1,
        )
        normals = torch.zeros_like(vertices)
        for corner in range(3):
            normals.index_add_(0, faces[:, corner].long(), face_normals)
        return torch.nn.functional.normalize(normals, dim=1, eps=1.0e-8)

    @staticmethod
    def _shade_material(
        vertices_camera: torch.Tensor,
        normals_camera: torch.Tensor,
        base_color: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ambient, two-light Lambert, and restrained specular shading."""

        normals = torch.nn.functional.normalize(normals_camera, dim=1, eps=1.0e-8)
        view = torch.nn.functional.normalize(-vertices_camera, dim=1, eps=1.0e-8)
        key = torch.tensor(
            (-0.35, -0.45, -1.0), device=vertices_camera.device
        )
        fill = torch.tensor(
            (0.65, 0.20, -1.0), device=vertices_camera.device
        )
        key = torch.nn.functional.normalize(key, dim=0)
        fill = torch.nn.functional.normalize(fill, dim=0)
        # Two-sided diffuse is robust to left/right MANO winding while still
        # producing a conventional shaded clay appearance on visible faces.
        key_diffuse = torch.abs(torch.sum(normals * key[None], dim=1))
        fill_diffuse = torch.abs(torch.sum(normals * fill[None], dim=1))
        half_vector = torch.nn.functional.normalize(
            key[None] + view, dim=1, eps=1.0e-8
        )
        specular = torch.abs(torch.sum(normals * half_vector, dim=1)).pow(40.0)
        intensity = (
            0.24 + 0.52 * key_diffuse + 0.18 * fill_diffuse + 0.18 * specular
        ).clamp(0.20, 1.0)
        rgb = (base_color * intensity[:, None]).clamp(0.0, 1.0)
        return torch.cat((rgb, alpha), dim=1)

    def render_frame(
        self, frame_bgr: np.ndarray, frame_index: int
    ) -> tuple[np.ndarray, list[tuple[str, tuple[float, float, float]]]]:
        """Render one depth-tested RGBA layer and composite it over BGR."""

        height, width = frame_bgr.shape[:2]
        world, normals_world, faces, colors, alpha, legend = self._frame_geometry(
            frame_index
        )
        if not len(world):
            return frame_bgr, legend
        world_to_camera = self.world_to_camera[frame_index]
        camera = world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        normals_camera = torch.nn.functional.normalize(
            normals_world @ world_to_camera[:3, :3].T,
            dim=1,
            eps=1.0e-8,
        )
        attributes = self._shade_material(camera, normals_camera, colors, alpha)
        intrinsic = torch.as_tensor(
            self.intrinsics[min(frame_index, len(self.intrinsics) - 1)],
            device=self.device,
        )
        clip = self._camera_to_clip(camera, intrinsic, width, height)
        rast, _ = dr.rasterize(
            self.context, clip[None], faces, resolution=(height, width)
        )
        layer, _ = dr.interpolate(attributes[None], rast, faces)
        layer = dr.antialias(layer, rast, clip[None], faces)[0]
        # nvdiffrast returns OpenGL bottom-up raster rows, whereas OpenCV video
        # frames use a top-left origin.
        layer = torch.flip(layer, dims=(0,))
        layer_np = layer.detach().clamp(0.0, 1.0).cpu().numpy()
        alpha = layer_np[..., 3:4]
        rgb = frame_bgr[..., ::-1].astype(np.float32) / 255.0
        composed = layer_np[..., :3] * alpha + rgb * (1.0 - alpha)
        return np.clip(composed[..., ::-1] * 255.0, 0, 255).astype(np.uint8), legend

    @staticmethod
    def _draw_legend(
        frame: np.ndarray,
        frame_index: int,
        legend: list[tuple[str, tuple[float, float, float]]],
    ) -> None:
        cv2.putText(
            frame,
            f"frame {frame_index:06d}",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y = 58
        for label, color in legend:
            bgr = tuple(int(round(255 * value)) for value in color[::-1])
            cv2.rectangle(frame, (18, y - 13), (34, y + 3), bgr, thickness=-1)
            cv2.putText(
                frame,
                label,
                (42, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y += 23

    def render_video(
        self,
        *,
        input_video: Path,
        output_video: Path,
        start_frame: int = 0,
        end_frame: int | None = None,
        two_column_grayscale: bool = False,
    ) -> None:
        """Render an inclusive-exclusive frame range to an MP4 video."""

        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {input_video}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        stop = min(
            self.frame_count,
            video_frames,
            self.frame_count if end_frame is None else int(end_frame),
        )
        start = max(0, int(start_frame))
        if start >= stop:
            raise ValueError(f"Empty frame range [{start}, {stop})")
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width * (2 if two_column_grayscale else 1), height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video: {output_video}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        try:
            with torch.inference_mode():
                for frame_index in range(start, stop):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Failed to decode frame {frame_index}")
                    if two_column_grayscale:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    else:
                        background = frame
                    rendered, legend = self.render_frame(background, frame_index)
                    self._draw_legend(rendered, frame_index, legend)
                    if two_column_grayscale:
                        left = background.copy()
                        cv2.putText(
                            left,
                            f"rectified left | frame {frame_index:06d}",
                            (18, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.72,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        writer.write(np.concatenate((left, rendered), axis=1))
                    else:
                        writer.write(rendered)
                    if frame_index == start or (frame_index + 1) % 50 == 0:
                        print(f"[mesh-overlay] {frame_index + 1}/{stop}", flush=True)
        finally:
            writer.release()
            capture.release()


def parse_args() -> argparse.Namespace:
    """Parse rendering options for a pipeline output directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--input-video", type=Path)
    parser.add_argument(
        "--hand-motion",
        type=Path,
        help="Optional hand_motion.npz override for ablations or diagnostics.",
    )
    parser.add_argument(
        "--foundationpose-manifest",
        type=Path,
        help=(
            "Optional FoundationPose experiment manifest whose metric meshes "
            "and poses replace the run-root object outputs."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--object-alpha", type=float, default=0.94)
    parser.add_argument("--hand-alpha", type=float, default=0.90)
    parser.add_argument(
        "--object-pose-stage",
        choices=("auto", "foundationpose", "surface", "haco"),
        default="auto",
        help="Select the object-pose result rendered in the right column.",
    )
    parser.add_argument(
        "--exclude-supports",
        action="store_true",
        help="Do not render static support-object reconstructions.",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument(
        "--two-column-grayscale",
        action="store_true",
        help="Place the raw grayscale frame left of the colored mesh overlay.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    """Load the latest joint-refined meshes and render the requested range."""

    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    stereo_video = run_root / "calibration/left_rectified_working.mp4"
    input_video = (
        args.input_video.expanduser().resolve()
        if args.input_video
        else (
            stereo_video
            if stereo_video.is_file()
            else run_root / "input/rgb_upright.mp4"
        )
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else run_root / "visualization/hoi_mesh_white_lit_overlay.mp4"
    )
    renderer = HOIMeshVideoRenderer(
        run_root=run_root,
        hand_motion=(args.hand_motion.expanduser() if args.hand_motion else None),
        foundationpose_manifest=(
            args.foundationpose_manifest.expanduser()
            if args.foundationpose_manifest
            else None
        ),
        object_alpha=args.object_alpha,
        hand_alpha=args.hand_alpha,
        object_pose_stage=args.object_pose_stage,
        include_supports=not args.exclude_supports,
        device=args.device,
    )
    renderer.render_video(
        input_video=input_video,
        output_video=output,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        two_column_grayscale=args.two_column_grayscale,
    )
    print(f"[mesh-overlay] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
