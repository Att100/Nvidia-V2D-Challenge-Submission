#!/usr/bin/env python3
"""Render hands and objects using only an exported sequence deliverable.

The script intentionally does not import the ego-scene-hoi pipeline.  It reads
the public delivery contract (``manifest.json``, NumPy-array pickle files, GLB
meshes, and the rectified-left video), reconstructs official MANO meshes, and
renders all valid hands and objects in the rectified-left camera.

MANO model files are not redistributed by the deliverable because of their
license.  Supply a directory containing ``MANO_LEFT.pkl`` and
``MANO_RIGHT.pkl`` with ``--mano-model-dir``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import nvdiffrast.torch as dr
import numpy as np
import smplx
import torch
import trimesh


OBJECT_COLORS = (
    (0.95, 0.38, 0.16),
    (0.16, 0.68, 0.95),
    (0.68, 0.34, 0.92),
    (0.95, 0.76, 0.16),
    (0.12, 0.72, 0.72),
    (0.88, 0.52, 0.18),
)
HAND_COLORS = ((0.28, 0.90, 0.36), (0.98, 0.36, 0.78))


@dataclass(frozen=True)
class ObjectSequence:
    """One canonical object mesh and its rectified-left pose sequence."""

    label: str
    vertices: torch.Tensor
    normals: torch.Tensor
    faces: torch.Tensor
    object_to_camera: torch.Tensor
    valid: np.ndarray
    color: tuple[float, float, float]


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a dictionary in {path}")
    return value


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = (
            loaded.to_geometry()
            if hasattr(loaded, "to_geometry")
            else loaded.dump(concatenate=True)
        )
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh: {path}")
    return loaded


def _vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
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


def _camera_to_clip(
    vertices: torch.Tensor,
    intrinsic: torch.Tensor,
    width: int,
    height: int,
    near: float = 0.01,
    far: float = 100.0,
) -> torch.Tensor:
    """Map OpenCV camera coordinates to OpenGL homogeneous clip space."""

    x, y, z = vertices.unbind(dim=1)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x_clip = (2.0 * fx / width) * x + (2.0 * cx / width - 1.0) * z
    y_clip = (-2.0 * fy / height) * y + (1.0 - 2.0 * cy / height) * z
    z_clip = ((far + near) / (far - near)) * z - 2.0 * far * near / (far - near)
    return torch.stack((x_clip, y_clip, z_clip, z), dim=1)


def _shade(
    vertices_camera: torch.Tensor,
    normals_camera: torch.Tensor,
    colors: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Apply two-sided Lambert lighting and a small specular component."""

    normals = torch.nn.functional.normalize(normals_camera, dim=1, eps=1.0e-8)
    view = torch.nn.functional.normalize(-vertices_camera, dim=1, eps=1.0e-8)
    key = torch.nn.functional.normalize(
        torch.tensor((-0.35, -0.45, -1.0), device=vertices_camera.device), dim=0
    )
    fill = torch.nn.functional.normalize(
        torch.tensor((0.65, 0.20, -1.0), device=vertices_camera.device), dim=0
    )
    key_diffuse = torch.abs(torch.sum(normals * key[None], dim=1))
    fill_diffuse = torch.abs(torch.sum(normals * fill[None], dim=1))
    half_vector = torch.nn.functional.normalize(
        key[None] + view, dim=1, eps=1.0e-8
    )
    specular = torch.abs(torch.sum(normals * half_vector, dim=1)).pow(40.0)
    intensity = (
        0.24 + 0.52 * key_diffuse + 0.18 * fill_diffuse + 0.18 * specular
    ).clamp(0.20, 1.0)
    return torch.cat(((colors * intensity[:, None]).clamp(0.0, 1.0), alpha), dim=1)


def _create_mano_layers(
    model_dir: Path, device: torch.device
) -> tuple[torch.nn.Module, torch.nn.Module]:
    layers = []
    for label, is_right in (("LEFT", False), ("RIGHT", True)):
        model_path = model_dir / f"MANO_{label}.pkl"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        layers.append(
            smplx.MANO(
                model_path=str(model_path),
                is_rhand=is_right,
                use_pca=False,
                flat_hand_mean=False,
            ).to(device).eval()
        )
    left, right = layers
    if float(torch.sum(torch.abs(left.shapedirs[:, 0] - right.shapedirs[:, 0]))) < 1.0:
        with torch.no_grad():
            left.shapedirs[:, 0] *= -1.0
    return left, right


class DeliverableRenderer:
    """Reconstruct and render a complete exported HOI sequence."""

    def __init__(
        self,
        root: Path,
        mano_model_dir: Path,
        device: str = "cuda",
        object_alpha: float = 0.94,
        hand_alpha: float = 0.90,
    ) -> None:
        self.root = root.resolve()
        self.device = torch.device(
            device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu"
        )
        if self.device.type != "cuda":
            raise RuntimeError("The nvdiffrast validation renderer requires CUDA")
        self.object_alpha = float(object_alpha)
        self.hand_alpha = float(hand_alpha)
        self.context = dr.RasterizeCudaContext(device=self.device)
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_file():
            self.manifest = json.loads(manifest_path.read_text())
        else:
            # The exporter invokes this renderer once before writing its final
            # manifest.  Discovering the stable delivery layout here also
            # makes partially copied deliveries diagnosable.
            object_records = []
            for object_dir in sorted((self.root / "objects").glob("*")):
                mesh = object_dir / "textured_mesh_metric.glb"
                poses = object_dir / "poses.pkl"
                if mesh.is_file() and poses.is_file():
                    object_records.append(
                        {
                            "id": object_dir.name,
                            "mesh": mesh.relative_to(self.root).as_posix(),
                            "poses": poses.relative_to(self.root).as_posix(),
                        }
                    )
            self.manifest = {
                "files": {
                    "rectified_left_video": "videos/rectified_left.mp4",
                    "objects": object_records,
                }
            }
        self.intrinsics = _load_pickle(self.root / "cameras/intrinsics.pkl")
        self.trajectory = _load_pickle(self.root / "cameras/trajectory.pkl")
        self.K = torch.as_tensor(
            np.asarray(self.intrinsics["K_rectified_left"], dtype=np.float32),
            device=self.device,
        )
        self.world_to_camera = torch.as_tensor(
            np.asarray(
                self.trajectory["rectified_left_world_to_camera"], dtype=np.float32
            ),
            device=self.device,
        )
        self.frame_count = len(self.world_to_camera)
        self.objects = self._load_objects()
        self.hand_vertices, self.hand_faces, self.hand_valid = self._load_hands(
            mano_model_dir.resolve()
        )

    def _load_objects(self) -> tuple[ObjectSequence, ...]:
        result = []
        for index, record in enumerate(self.manifest["files"]["objects"]):
            mesh = _load_mesh(self.root / record["mesh"])
            pose = _load_pickle(self.root / record["poses"])
            object_to_camera = np.asarray(
                pose["object_to_rectified_left_camera"], dtype=np.float32
            )
            valid = np.asarray(pose["valid"], dtype=bool)[..., 0]
            if len(object_to_camera) != self.frame_count:
                raise ValueError(f"Object timeline mismatch: {record['id']}")
            result.append(
                ObjectSequence(
                    label=str(record.get("name", record["id"])),
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
                    object_to_camera=torch.as_tensor(
                        object_to_camera, device=self.device
                    ),
                    valid=valid,
                    color=OBJECT_COLORS[index % len(OBJECT_COLORS)],
                )
            )
        return tuple(result)

    def _load_hands(
        self, mano_model_dir: Path
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], np.ndarray]:
        parameter_data = _load_pickle(self.root / "hands/mano_parameters.pkl")
        scale_data = _load_pickle(self.root / "hands/mano_scale.pkl")
        parameters = parameter_data["parameters"]
        valid = np.asarray(parameter_data["valid"], dtype=bool)[..., 0]
        scale = np.asarray(scale_data["scale"], dtype=np.float32)
        frame_count, hand_count = valid.shape
        if frame_count != self.frame_count or hand_count != 2:
            raise ValueError("MANO and camera timelines differ")
        layers = _create_mano_layers(mano_model_dir, self.device)
        vertices_world = torch.full(
            (frame_count, hand_count, 778, 3),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        faces: list[torch.Tensor] = []
        with torch.inference_mode():
            for hand_id, layer in enumerate(layers):
                ids = np.flatnonzero(valid[:, hand_id])
                if len(ids):
                    output = layer(
                        betas=torch.as_tensor(
                            parameters["betas"][ids, hand_id], device=self.device
                        ),
                        global_orient=torch.as_tensor(
                            parameters["global_orient"][ids, hand_id],
                            device=self.device,
                        ),
                        hand_pose=torch.as_tensor(
                            parameters["hand_pose"][ids, hand_id], device=self.device
                        ),
                        transl=torch.as_tensor(
                            parameters["transl"][ids, hand_id], device=self.device
                        ),
                    )
                    vertices_world[ids, hand_id] = float(scale[hand_id]) * output.vertices
                faces.append(
                    torch.as_tensor(
                        np.asarray(layer.faces, dtype=np.int32), device=self.device
                    )
                )
        return vertices_world, tuple(faces), valid

    def _frame_geometry(
        self, frame_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vertices_camera: list[torch.Tensor] = []
        normals_camera: list[torch.Tensor] = []
        faces: list[torch.Tensor] = []
        colors: list[torch.Tensor] = []
        alphas: list[torch.Tensor] = []
        offset = 0
        for item in self.objects:
            pose = item.object_to_camera[frame_id]
            if not item.valid[frame_id] or not bool(torch.isfinite(pose).all()):
                continue
            camera = item.vertices @ pose[:3, :3].T + pose[:3, 3]
            normal = torch.nn.functional.normalize(
                item.normals @ pose[:3, :3].T, dim=1, eps=1.0e-8
            )
            vertices_camera.append(camera)
            normals_camera.append(normal)
            faces.append(item.faces + offset)
            colors.append(
                torch.tensor(item.color, device=self.device).expand(len(camera), -1)
            )
            alphas.append(
                torch.full(
                    (len(camera), 1), self.object_alpha, device=self.device
                )
            )
            offset += len(camera)
        w2c = self.world_to_camera[frame_id]
        for hand_id in range(2):
            if not self.hand_valid[frame_id, hand_id]:
                continue
            world = self.hand_vertices[frame_id, hand_id]
            if not bool(torch.isfinite(world).all()):
                continue
            camera = world @ w2c[:3, :3].T + w2c[:3, 3]
            hand_faces = self.hand_faces[hand_id]
            world_normal = _vertex_normals(world, hand_faces)
            normal = torch.nn.functional.normalize(
                world_normal @ w2c[:3, :3].T, dim=1, eps=1.0e-8
            )
            vertices_camera.append(camera)
            normals_camera.append(normal)
            faces.append(hand_faces + offset)
            colors.append(
                torch.tensor(HAND_COLORS[hand_id], device=self.device).expand(
                    len(camera), -1
                )
            )
            alphas.append(
                torch.full((len(camera), 1), self.hand_alpha, device=self.device)
            )
            offset += len(camera)
        if not vertices_camera:
            raise RuntimeError(f"No valid hand/object geometry at frame {frame_id}")
        return (
            torch.cat(vertices_camera),
            torch.cat(normals_camera),
            torch.cat(faces),
            torch.cat(colors),
            torch.cat(alphas),
        )

    def render_frame(self, frame_bgr: np.ndarray, frame_id: int) -> np.ndarray:
        height, width = frame_bgr.shape[:2]
        vertices, normals, faces, colors, alphas = self._frame_geometry(frame_id)
        clip = _camera_to_clip(vertices, self.K, width, height)
        attributes = _shade(vertices, normals, colors, alphas)
        raster, _ = dr.rasterize(
            self.context, clip[None], faces, resolution=(height, width)
        )
        layer, _ = dr.interpolate(attributes[None], raster, faces)
        layer = dr.antialias(layer, raster, clip[None], faces)[0]
        layer = torch.flip(layer, dims=(0,))
        rgba = layer.detach().clamp(0.0, 1.0).cpu().numpy()
        alpha = rgba[..., 3:4]
        background = frame_bgr[..., ::-1].astype(np.float32) / 255.0
        composed = rgba[..., :3] * alpha + background * (1.0 - alpha)
        return np.clip(composed[..., ::-1] * 255.0, 0, 255).astype(np.uint8)

    def render_video(self, output: Path, start: int = 0, end: int | None = None) -> None:
        video = self.root / self.manifest["files"]["rectified_left_video"]
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 20.0
        stop = min(
            self.frame_count,
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            self.frame_count if end is None else int(end),
        )
        start = max(0, int(start))
        if start >= stop:
            raise ValueError(f"Empty frame range [{start}, {stop})")
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create {output}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        try:
            with torch.inference_mode():
                for frame_id in range(start, stop):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Failed to decode frame {frame_id}")
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    rendered = self.render_frame(background, frame_id)
                    cv2.putText(
                        rendered,
                        f"export reconstruction | frame {frame_id:06d}",
                        (14, 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    writer.write(np.concatenate((background, rendered), axis=1))
                    if frame_id == start or (frame_id + 1) % 50 == 0:
                        print(f"[deliverable-render] {frame_id + 1}/{stop}", flush=True)
        finally:
            writer.release()
            capture.release()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mano-model-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--object-alpha", type=float, default=0.94)
    parser.add_argument("--hand-alpha", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output
        else root / "validation/rendered_from_export.mp4"
    )
    renderer = DeliverableRenderer(
        root=root,
        mano_model_dir=args.mano_model_dir.expanduser().resolve(),
        device=args.device,
        object_alpha=args.object_alpha,
        hand_alpha=args.hand_alpha,
    )
    renderer.render_video(output, args.start, args.end)
    print(f"[deliverable-render] wrote {output}")


if __name__ == "__main__":
    main()
