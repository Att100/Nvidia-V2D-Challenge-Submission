"""Inference-only HACO integration and object-specific contact association.

The upstream release predicts a contact probability for each of the 778
right-MANO vertices. Existing UmeTrack/MANO detections are reused so the HACO detector
dependencies are not needed.  Left hands are mirrored into the right-hand
image convention; MANO left/right releases share vertex indices (only face
winding differs), which is verified against the motion archive before use.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterator

import cv2
import numpy as np
from scipy.spatial import cKDTree
import torch
import trimesh

from pipeline.modules.haco.mano_regions import MANOPalmarRegion
from pipeline.utils import parse_worker_args, write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict dense MANO contacts with HACO")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--haco-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hand-motion", type=Path, required=True)
    parser.add_argument("--backbone", default="vit-b-16")
    parser.add_argument(
        "--image-source",
        choices=("rgb", "rectified_left"),
        default="rgb",
        help="Image stream used by the HACO appearance network.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--temporal-alpha", type=float, default=0.8)
    parser.add_argument("--minimum-component-vertices", type=int, default=20)
    parser.add_argument("--bbox-expand-ratio", type=float, default=2.0)
    parser.add_argument("--object-mask-dilation", type=int, default=24)
    parser.add_argument("--geometry-soft-radius", type=float, default=0.03)
    parser.add_argument("--geometry-soft-sigma", type=float, default=0.04)
    parser.add_argument("--geometry-maximum-distance", type=float, default=0.15)
    parser.add_argument(
        "--contact-region",
        choices=("palmar", "all"),
        default="palmar",
        help="MANO surface region allowed to attract the object.",
    )
    parser.add_argument(
        "--palmar-maximum-normal-y",
        type=float,
        default=-0.10,
        help="Maximum neutral-template local-Y normal retained as palm/pad side.",
    )
    return parser


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _crop_hand(
    image_rgb: np.ndarray,
    bbox_xyxy: np.ndarray,
    *,
    expand_ratio: float,
    mirror: bool,
) -> np.ndarray:
    """Apply the release's expanded affine hand crop without its detector."""

    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float32)
    width = max(float(x2 - x1), 2.0) * float(expand_ratio)
    height = max(float(y2 - y1), 2.0) * float(expand_ratio)
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], np.float32)
    source = np.array(
        [
            center,
            center + np.array([0.0, height * 0.5], np.float32),
            center + np.array([width * 0.5, 0.0], np.float32),
        ],
        np.float32,
    )
    target_center = np.array([128.0, 128.0], np.float32)
    target = np.array(
        [target_center, target_center + [0.0, 128.0], target_center + [128.0, 0.0]],
        np.float32,
    )
    transform = cv2.getAffineTransform(source, target)
    crop = cv2.warpAffine(
        image_rgb,
        transform,
        (256, 256),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return np.ascontiguousarray(crop[:, ::-1] if mirror else crop)


def _component_filter(mask: np.ndarray, faces: np.ndarray, minimum: int) -> np.ndarray:
    adjacency: list[set[int]] = [set() for _ in range(len(mask))]
    for a, b, c in np.asarray(faces, dtype=np.int32):
        adjacency[a].update((int(b), int(c)))
        adjacency[b].update((int(a), int(c)))
        adjacency[c].update((int(a), int(b)))
    visited = np.zeros(len(mask), dtype=bool)
    output = np.zeros(len(mask), dtype=bool)
    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        queue: deque[int] = deque([int(start)])
        component: list[int] = []
        while queue:
            vertex = queue.popleft()
            if visited[vertex] or not mask[vertex]:
                continue
            visited[vertex] = True
            component.append(vertex)
            queue.extend(adjacency[vertex])
        if len(component) >= minimum:
            output[component] = True
    return output


def _unpack_masks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(path)
    width = int(archive["mask_width"])
    masks = np.unpackbits(archive["packed_masks"], axis=2)[:, :, :width].astype(bool)
    visible = (
        archive["visible"].astype(bool)
        if "visible" in archive.files
        else masks.reshape(len(masks), -1).any(axis=1)
    )
    return masks, visible


def _load_mesh_vertices(path: Path) -> np.ndarray:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry in {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected mesh at {path}")
    return np.asarray(loaded.vertices, dtype=np.float32)


class HACOContactPredictor:
    """Predict per-hand contact probabilities and associate them to objects."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments

    @classmethod
    def from_worker_job(cls) -> "HACOContactPredictor":
        return cls(parse_worker_args(_parser()))

    def _load_model(self) -> tuple[torch.nn.Module, object]:
        args = self.arguments
        root = args.haco_root.expanduser().resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        with _working_directory(root):
            from lib.core.config import cfg, update_config
            from lib.models.model import HACO

            update_config(backbone_type=args.backbone, exp_dir=str(root / ".runtime"))
            model = HACO().to(args.device).eval()
            checkpoint = torch.load(
                args.checkpoint.expanduser().resolve(),
                map_location=args.device,
                weights_only=False,
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        return model, cfg

    def _prediction_video(self) -> Path:
        """Resolve the explicitly configured HACO appearance stream."""

        args = self.arguments
        if args.image_source == "rgb":
            candidates = (
                args.run_root / "registration/rgb_left/rgb_undistorted.mp4",
                args.run_root / "input/rgb_upright.mp4",
            )
        else:
            candidates = (args.run_root / "calibration/left_rectified_working.mp4",)
        video = next((path for path in candidates if path.is_file()), None)
        if video is None:
            raise FileNotFoundError(
                f"No {args.image_source} HACO input video exists; checked: "
                + ", ".join(str(path) for path in candidates)
            )
        return video

    def _predict(self, hand: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        args = self.arguments
        model, cfg = self._load_model()
        video_path = self._prediction_video()
        labels = [str(item) for item in hand["hand_labels"].tolist()]
        valid = np.asarray(hand["valid"], dtype=bool)
        box_key = (
            "bbox_xyxy_rgb"
            if args.image_source == "rgb" and "bbox_xyxy_rgb" in hand
            else "bbox_xyxy"
        )
        boxes = np.asarray(hand[box_key], dtype=np.float32)
        frame_count, hand_count = valid.shape
        probabilities = np.zeros((frame_count, hand_count, 778), dtype=np.float32)

        mean = torch.tensor(cfg.MODEL.img_mean, dtype=torch.float32)[:, None, None]
        std = torch.tensor(cfg.MODEL.img_std, dtype=torch.float32)[:, None, None]
        batch_crops: list[np.ndarray] = []
        batch_slots: list[tuple[int, int]] = []

        def flush() -> None:
            if not batch_crops:
                return
            array = np.stack(batch_crops).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            tensor = (torch.from_numpy(array) - mean[None]) / std[None]
            with torch.inference_mode():
                values = model({"input": {"image": tensor.to(args.device)}})[
                    "contact_out"
                ].sigmoid()
            for slot, value in zip(batch_slots, values.float().cpu().numpy(), strict=True):
                probabilities[slot] = value
            batch_crops.clear()
            batch_slots.clear()

        capture = cv2.VideoCapture(str(video_path))
        frame_index = 0
        while frame_index < frame_count:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            for hand_index, side in enumerate(labels):
                if not valid[frame_index, hand_index] or not np.isfinite(
                    boxes[frame_index, hand_index]
                ).all():
                    continue
                batch_crops.append(
                    _crop_hand(
                        frame_rgb,
                        boxes[frame_index, hand_index],
                        expand_ratio=args.bbox_expand_ratio,
                        mirror=side == "left",
                    )
                )
                batch_slots.append((frame_index, hand_index))
                if len(batch_crops) >= args.batch_size:
                    flush()
            frame_index += 1
        capture.release()
        flush()
        if frame_index != frame_count:
            raise RuntimeError(f"Video ended at frame {frame_index}; expected {frame_count}")

        smoothed = np.zeros_like(probabilities)
        selected = np.zeros_like(probabilities, dtype=bool)
        faces_right = np.asarray(hand["faces_right"], dtype=np.int32)
        faces_left = np.asarray(hand["faces_left"], dtype=np.int32)
        if not np.array_equal(np.sort(faces_left, axis=1), np.sort(faces_right, axis=1)):
            raise RuntimeError("Left/right MANO vertex indices are not topology-compatible")
        for hand_index in range(hand_count):
            previous: np.ndarray | None = None
            previous_frame = -2
            for frame_index in range(frame_count):
                if not valid[frame_index, hand_index]:
                    previous = None
                    continue
                current = probabilities[frame_index, hand_index]
                if previous is None or frame_index != previous_frame + 1:
                    filtered = current
                else:
                    filtered = args.temporal_alpha * previous + (1.0 - args.temporal_alpha) * current
                smoothed[frame_index, hand_index] = filtered
                selected[frame_index, hand_index] = _component_filter(
                    filtered >= args.threshold,
                    faces_right,
                    args.minimum_component_vertices,
                )
                previous = filtered
                previous_frame = frame_index
        return probabilities, smoothed, selected

    def _write_overlay(
        self,
        hand: np.lib.npyio.NpzFile,
        smoothed: np.ndarray,
        selected: np.ndarray,
        output_path: Path,
    ) -> None:
        stereo_video = (
            self.arguments.run_root / "calibration/left_rectified_working.mp4"
        )
        video_path = (
            stereo_video
            if stereo_video.is_file()
            else self.arguments.run_root / "input/rgb_upright.mp4"
        )
        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        vertices = np.asarray(hand["vertices_camera"], dtype=np.float32)
        intrinsics = np.asarray(hand["intrinsics_video"], dtype=np.float32)
        valid = np.asarray(hand["valid"], dtype=bool)
        labels = [str(item) for item in hand["hand_labels"].tolist()]
        for frame_index in range(len(vertices)):
            ok, frame = capture.read()
            if not ok:
                break
            for hand_index, side in enumerate(labels):
                if not valid[frame_index, hand_index]:
                    continue
                points = vertices[frame_index, hand_index]
                finite = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-5)
                pixels_h = (intrinsics @ points[finite].T).T
                pixels = np.rint(pixels_h[:, :2] / pixels_h[:, 2:3]).astype(np.int32)
                vertex_ids = np.flatnonzero(finite)
                for vertex_id, (x, y) in zip(vertex_ids, pixels, strict=True):
                    if 0 <= x < width and 0 <= y < height:
                        probability = float(smoothed[frame_index, hand_index, vertex_id])
                        if selected[frame_index, hand_index, vertex_id]:
                            color = (0, int(255 * (1.0 - probability)), 255)
                            cv2.circle(frame, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"HACO {side}: {int(selected[frame_index, hand_index].sum())}",
                    (12, 28 + 26 * hand_index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
        capture.release()
        writer.release()

    def _associate_objects(
        self,
        hand: np.lib.npyio.NpzFile,
        smoothed: np.ndarray,
        selected: np.ndarray,
        output_dir: Path,
    ) -> list[dict[str, object]]:
        args = self.arguments
        run_root = args.run_root
        labels = [str(item) for item in hand["hand_labels"].tolist()]
        vertices_world = np.asarray(hand["vertices_world"], dtype=np.float32)
        hand_valid = np.asarray(hand["valid"], dtype=bool)
        camera_to_world = np.asarray(hand["camera_to_world"], dtype=np.float32)
        world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
        intrinsics = np.asarray(hand["intrinsics_video"], dtype=np.float32)
        source_indices = np.asarray(hand["source_indices"], dtype=np.int32)
        frame_count = len(source_indices)
        events = json.loads(
            (run_root / "semantics/reasoning/contact_events.json").read_text(encoding="utf-8")
        ).get("events", [])
        records = json.loads(
            (run_root / "tracks/sam3/refined_masks/manifest.json").read_text(encoding="utf-8")
        ).get("objects", [])
        results: list[dict[str, object]] = []
        association_dir = output_dir / "object_contacts"
        association_dir.mkdir(parents=True, exist_ok=True)

        for object_index, record in enumerate(records):
            object_id = str(record["global_object_id"])
            object_name = str(record["object_name"])
            matching = [
                event
                for event in events
                if _normalize_name(str(event.get("object", ""))) == _normalize_name(object_name)
            ]
            if not matching and object_index < len(events):
                matching = [events[object_index]]
            event_active = np.zeros((frame_count, len(labels)), dtype=bool)
            support_active = np.zeros_like(event_active)
            event_ids: list[str] = []
            for event in matching:
                event_ids.append(str(event.get("event_id", "")))
                for atom in event.get("atoms", []) or [event]:
                    side = str(atom.get("hand", "right"))
                    if side not in labels:
                        continue
                    hand_index = labels.index(side)
                    start = max(0, int(atom.get("start_frame", event.get("start_frame", 0))))
                    end = min(frame_count - 1, int(atom.get("end_frame", event.get("end_frame", start))))
                    event_active[start : end + 1, hand_index] = True
                    support = str(atom.get("physical_support", atom.get("support", "false"))).lower()
                    if support in {"true", "1", "yes"}:
                        support_active[start : end + 1, hand_index] = True

            masks, object_visible = _unpack_masks(
                run_root / "tracks/sam3/refined_masks" / f"{object_id}.npz"
            )
            object_dir = run_root / "objects" / object_id
            mesh_candidates = sorted((object_dir / "scale_primary").glob("*_metric_centered.glb"))
            if not mesh_candidates:
                # Released challenge meshes are already metric and therefore
                # deliberately bypass the scale fitter/legacy filename.
                mesh_candidates = sorted((object_dir / "scale_primary").glob("*.glb"))
            if not mesh_candidates:
                mesh_candidates = sorted((object_dir / "canonical").glob("*.glb"))
            if not mesh_candidates:
                raise FileNotFoundError(f"No object mesh for {object_id}")
            mesh_path = mesh_candidates[0]
            mesh_vertices = _load_mesh_vertices(mesh_path)
            tree = cKDTree(mesh_vertices)
            pose_archive = np.load(object_dir / "pose_track/poses.npz")
            object_to_world = np.asarray(pose_archive["object_to_world"], dtype=np.float32)
            observed = np.asarray(pose_archive["observed"], dtype=bool)
            weights = np.zeros_like(smoothed, dtype=np.float32)
            mask_gate = np.zeros_like(selected, dtype=bool)
            geometry_distance = np.full_like(smoothed, np.nan, dtype=np.float32)
            kernel_size = 2 * int(args.object_mask_dilation) + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            for frame_index in range(frame_count):
                if not object_visible[frame_index] or not observed[frame_index]:
                    continue
                pose = object_to_world[frame_index]
                if not np.isfinite(pose).all():
                    continue
                dilated = cv2.dilate(masks[frame_index].astype(np.uint8), kernel) > 0
                camera = world_to_camera[frame_index]
                for hand_index in range(len(labels)):
                    candidates = (
                        event_active[frame_index, hand_index]
                        & hand_valid[frame_index, hand_index]
                        & selected[frame_index, hand_index]
                    )
                    if not np.any(candidates):
                        continue
                    points_world = vertices_world[frame_index, hand_index, candidates]
                    points_camera = (camera[:3, :3] @ points_world.T).T + camera[:3, 3]
                    projected = (intrinsics @ points_camera.T).T
                    pixels = np.rint(projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)).astype(np.int32)
                    inside = (
                        (points_camera[:, 2] > 1e-5)
                        & (pixels[:, 0] >= 0)
                        & (pixels[:, 0] < masks.shape[2])
                        & (pixels[:, 1] >= 0)
                        & (pixels[:, 1] < masks.shape[1])
                    )
                    gate = np.zeros(len(points_world), dtype=bool)
                    valid_pixels = np.flatnonzero(inside)
                    gate[valid_pixels] = dilated[pixels[valid_pixels, 1], pixels[valid_pixels, 0]]
                    points_object = (points_world - pose[:3, 3]) @ pose[:3, :3]
                    distances = tree.query(points_object, workers=-1)[0].astype(np.float32)
                    geometry = np.exp(
                        -0.5
                        * (
                            np.maximum(distances - args.geometry_soft_radius, 0.0)
                            / args.geometry_soft_sigma
                        )
                        ** 2
                    ).astype(np.float32)
                    geometry[distances > args.geometry_maximum_distance] = 0.0
                    vertex_ids = np.flatnonzero(candidates)
                    mask_gate[frame_index, hand_index, vertex_ids] = gate
                    geometry_distance[frame_index, hand_index, vertex_ids] = distances
                    weights[frame_index, hand_index, vertex_ids] = (
                        smoothed[frame_index, hand_index, vertex_ids]
                        * gate.astype(np.float32)
                        * geometry
                    )

            vertex_active = np.repeat(
                (hand_valid & event_active)[:, :, None], 778, axis=2
            )
            valid_frames = (
                object_visible
                & observed
                & np.isfinite(object_to_world).all(axis=(1, 2))
                & (weights > 0).any(axis=(1, 2))
            )
            path = association_dir / f"{object_id}.npz"
            np.savez_compressed(
                path,
                weights=weights,
                selected=selected,
                mask_gate=mask_gate,
                geometry_distance=geometry_distance,
                event_active=event_active,
                support_active=support_active,
                hand_vertex_active=vertex_active,
                valid_frames=valid_frames,
                source_indices=source_indices,
                hand_labels=np.asarray(labels),
            )
            active_weights = weights[weights > 0]
            results.append(
                {
                    "object_id": object_id,
                    "object_name": object_name,
                    "event_ids": event_ids,
                    "association": str(path),
                    "valid_frame_count": int(valid_frames.sum()),
                    "selected_vertex_observations": int((weights > 0).sum()),
                    "mean_positive_weight": (
                        float(active_weights.mean()) if len(active_weights) else None
                    ),
                    "mesh": str(mesh_path),
                }
            )
        return results

    def _contact_region(self) -> tuple[np.ndarray, dict[str, object]]:
        """Return the topology mask allowed to produce contact attraction."""

        args = self.arguments
        if args.contact_region == "all":
            mask = np.ones(778, dtype=bool)
            return mask, {
                "name": "all",
                "vertex_count": 778,
                "definition": "complete MANO surface",
            }
        region = MANOPalmarRegion.from_model_root(
            args.haco_root / "data/base_data/human_models/mano",
            maximum_normal_y=args.palmar_maximum_normal_y,
        )
        return region.mask, {
            "name": "palmar",
            "vertex_count": int(region.mask.sum()),
            "maximum_neutral_normal_y": region.maximum_normal_y,
            "definition": (
                "neutral MANO palm and finger-pad side; local surface "
                "normal_y <= threshold"
            ),
        }

    def run(self) -> None:
        args = self.arguments
        args.run_root = args.run_root.expanduser().resolve()
        args.haco_root = args.haco_root.expanduser().resolve()
        args.checkpoint = args.checkpoint.expanduser().resolve()
        args.hand_motion = args.hand_motion.expanduser().resolve()
        output_dir = args.run_root / "hands/haco"
        output_dir.mkdir(parents=True, exist_ok=True)
        hand = np.load(args.hand_motion, allow_pickle=True)
        probabilities, smoothed, selected_raw = self._predict(hand)
        contact_region_mask, contact_region_report = self._contact_region()
        selected = selected_raw & contact_region_mask[None, None, :]
        predictions_path = output_dir / "contact_predictions.npz"
        np.savez_compressed(
            predictions_path,
            probabilities=probabilities.astype(np.float16),
            smoothed_probabilities=smoothed.astype(np.float16),
            selected=selected,
            selected_raw=selected_raw,
            contact_region_mask=contact_region_mask,
            valid=np.asarray(hand["valid"], dtype=bool),
            source_indices=np.asarray(hand["source_indices"], dtype=np.int32),
            timestamps_seconds=np.asarray(hand["timestamps_seconds"], dtype=np.float64),
            hand_labels=np.asarray(hand["hand_labels"]),
        )
        overlay_path = output_dir / "contact_overlay.mp4"
        self._write_overlay(hand, smoothed, selected, overlay_path)
        objects = self._associate_objects(hand, smoothed, selected, output_dir)
        labels = [str(item) for item in hand["hand_labels"].tolist()]
        selected_counts = {
            side: {
                "valid_frames": int(np.asarray(hand["valid"])[:, index].sum()),
                "frames_with_contact": int(selected[:, index].any(axis=1).sum()),
                "median_vertices": float(
                    np.median(selected[:, index].sum(axis=1)[np.asarray(hand["valid"])[:, index]])
                )
                if np.asarray(hand["valid"])[:, index].any()
                else None,
                "median_vertices_before_region_constraint": float(
                    np.median(
                        selected_raw[:, index].sum(axis=1)[
                            np.asarray(hand["valid"])[:, index]
                        ]
                    )
                )
                if np.asarray(hand["valid"])[:, index].any()
                else None,
            }
            for index, side in enumerate(labels)
        }
        report = {
            "schema_version": "ego-scene-hoi-haco-contact-v1",
            "upstream_commit": "c774211b968bdf836f9511dbd5ff23bb19fe1c44",
            "upstream_license": "CC-BY-NC-4.0",
            "backbone": args.backbone,
            "image_source": args.image_source,
            "prediction_video": str(self._prediction_video()),
            "prediction_bbox_coordinates": (
                "undistorted RGB" if args.image_source == "rgb" else "rectified left"
            ),
            "overlay_coordinates": "rectified left",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "threshold": args.threshold,
            "temporal_alpha": args.temporal_alpha,
            "minimum_component_vertices": args.minimum_component_vertices,
            "contact_region": contact_region_report,
            "left_hand_conversion": "horizontal image mirror; identity MANO vertex IDs after topology verification",
            "hand_summary": selected_counts,
            "objects": objects,
            "artifacts": {
                "predictions": str(predictions_path),
                "overlay": str(overlay_path),
            },
        }
        write_json(output_dir / "report.json", report)
        write_json(
            output_dir / "object_contacts/manifest.json",
            {
                "schema_version": "ego-scene-hoi-haco-object-contact-v1",
                "source_predictions": str(predictions_path),
                "objects": objects,
            },
        )


def main() -> None:
    HACOContactPredictor.from_worker_job().run()


if __name__ == "__main__":
    main()
