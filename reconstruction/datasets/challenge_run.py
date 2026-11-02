"""NVIDIA Track-3 completed-run optimizer adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from pipeline.optim.contact.types import ContactOptimizationInput


def _normalize_name(value: str) -> str:
    """Normalize object names across reasoning, SAM, and FoundationPose."""

    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _unpack_mask_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode the packed-mask schema used by the local SAM 3 scripts."""

    archive = np.load(path)
    width = int(archive["mask_width"])
    masks = np.unpackbits(archive["packed_masks"], axis=2)[:, :, :width].astype(bool)
    visible = (
        archive["visible"].astype(bool)
        if "visible" in archive.files
        else masks.reshape(len(masks), -1).any(axis=1)
    )
    return masks, visible


class ChallengeRunAdapter:
    """Convert an existing pipeline run into optimizer-neutral arrays.

    This adapter is intentionally the only code that knows the Track-3 run
    layout. Contact optimization consumes ``ContactOptimizationInput`` only.
    """

    HAND_CANDIDATES = (
        "hands/umetrack_mano_surface/hand_motion.npz",
    )

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.resolve()

    def _first_existing(self, relative_paths: tuple[str, ...]) -> Path:
        for relative in relative_paths:
            path = self.run_root / relative
            if path.is_file():
                return path
        raise FileNotFoundError(f"None of these artifacts exists: {relative_paths}")

    def object_records(self) -> list[dict[str, Any]]:
        """Return global object IDs and names in the SAM 3 manifest order."""

        path = self.run_root / "tracks/sam3/global_masks/manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("objects", []))

    def _video_intrinsics(self, frame_count: int) -> np.ndarray:
        """Load intrinsics expressed in the full-resolution mask/RGB pixels."""

        stereo_path = self.run_root / "calibration/calibration.npz"
        if stereo_path.is_file():
            with np.load(stereo_path) as archive:
                intrinsic = np.asarray(archive["K_working"], dtype=np.float32)
            return np.repeat(intrinsic[None], frame_count, axis=0)
        path = self.run_root / "input/cameras.npz"
        archive = np.load(path)
        intrinsics = np.asarray(archive["intrinsics"], dtype=np.float32)
        if intrinsics.shape == (3, 3):
            intrinsics = np.repeat(intrinsics[None], frame_count, axis=0)
        if intrinsics.shape != (frame_count, 3, 3):
            raise ValueError(
                f"Video intrinsics {intrinsics.shape} do not match {frame_count} frames"
            )
        return intrinsics

    @staticmethod
    def _project_hand_mask(
        vertices_camera: np.ndarray,
        intrinsic: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Rasterize a conservative convex MANO occluder when no SAM hand mask exists."""

        mask = np.zeros(shape, dtype=np.uint8)
        vertices = np.asarray(vertices_camera, dtype=np.float64)
        valid = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 1.0e-5)
        if valid.sum() < 3:
            return mask.astype(bool)
        projected = vertices[valid] @ np.asarray(intrinsic, dtype=np.float64).T
        pixels = projected[:, :2] / projected[:, 2:3]
        cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(pixels).astype(np.int32)), 1)
        return mask.astype(bool)

    def _load_stereo_hand_masks(
        self,
        *,
        labels: tuple[str, ...],
        frame_count: int,
        output_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Decode rectified-left SAM masks in the requested hand order.

        Stereo hand masks pack the width along axis 3 because they contain an
        explicit hand dimension.  The generic SAM archive helper handles only
        ``[T,H,W]`` object masks and therefore must not be used here.
        """

        path = self.run_root / "tracks/sam3_hands_left/hand_masks.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=True) as archive:
            width = int(archive["mask_width"])
            decoded = np.unpackbits(archive["packed_masks"], axis=3)[..., :width]
            decoded = decoded.astype(bool)
            archive_labels = tuple(str(value) for value in archive["hand_labels"].tolist())
            archive_valid = (
                np.asarray(archive["valid"], dtype=bool)
                if "valid" in archive.files
                else decoded.reshape(decoded.shape[:2] + (-1,)).any(axis=2)
            )
        if len(decoded) != frame_count:
            raise ValueError(
                f"Stereo hand masks have {len(decoded)} frames; expected {frame_count}"
            )
        ordered = np.zeros((frame_count, len(labels), *output_shape), dtype=bool)
        for output_index, side in enumerate(labels):
            if side not in archive_labels:
                continue
            source_index = archive_labels.index(side)
            for frame_index in np.flatnonzero(archive_valid[:, source_index]):
                ordered[frame_index, output_index] = cv2.resize(
                    decoded[frame_index, source_index].astype(np.uint8),
                    (output_shape[1], output_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        return ordered

    def load_contact_input(self, object_id: str) -> ContactOptimizationInput:
        """Load one object, its contacting hand, masks, cameras, and poses."""

        from pipeline.optim.contact.types import ContactOptimizationInput

        records = {str(item["global_object_id"]): item for item in self.object_records()}
        if object_id not in records:
            raise KeyError(f"Unknown object ID {object_id}")
        record = records[object_id]
        object_name = str(record["object_name"])

        events_path = self.run_root / "semantics/reasoning/contact_events.json"
        events = json.loads(events_path.read_text(encoding="utf-8")).get("events", [])
        matching = [
            event
            for event in events
            if _normalize_name(str(event.get("object", ""))) == _normalize_name(object_name)
        ]
        if not matching:
            # Stable ordering is a fallback for lexical variants such as
            # "blue controller" versus "blue game controller".
            object_index = int(object_id.rsplit("_", 1)[-1])
            if object_index < len(events):
                matching = [events[object_index]]
        if not matching:
            raise RuntimeError(
                f"No Codex interaction matches {object_id}: {object_name}"
            )

        hand_path = self._first_existing(self.HAND_CANDIDATES)
        hand = np.load(hand_path, allow_pickle=True)
        labels = [str(value) for value in hand["hand_labels"].tolist()]
        atom_hands = [
            str(atom.get("hand", "right"))
            for event in matching
            for atom in event.get("atoms", [])
        ]
        candidates = atom_hands or ["right"]
        hand_side = max(set(candidates), key=candidates.count)
        if hand_side not in labels:
            raise RuntimeError(f"Hand {hand_side!r} is not present in {labels}")
        hand_index = labels.index(hand_side)
        vertices = hand["vertices_world"][:, hand_index].astype(np.float32)
        joints = hand["joints_world"][:, hand_index].astype(np.float32)
        hand_valid = hand["valid"][:, hand_index].astype(bool)
        frame_count = len(vertices)

        contact_active = np.zeros(frame_count, dtype=bool)
        for event in matching:
            atoms = event.get("atoms", []) or [event]
            for atom in atoms:
                if str(atom.get("hand", hand_side)) != hand_side:
                    continue
                start = max(0, int(atom.get("start_frame", event.get("start_frame", 0))))
                end = min(
                    frame_count - 1,
                    int(atom.get("end_frame", event.get("end_frame", start))),
                )
                contact_active[start : end + 1] = True

        object_dir = self.run_root / "objects" / object_id
        pose_archive = np.load(object_dir / "pose_track/poses.npz")
        poses = pose_archive["object_to_world"].astype(np.float32)
        observed = pose_archive["observed"].astype(bool)
        source_indices = pose_archive["source_indices"].astype(np.int32)
        if len(poses) != frame_count:
            raise ValueError("Hand and object timelines have different lengths")

        cameras = np.load(self.run_root / "4d/cameras.npz")
        camera_to_world = cameras["camera_poses"].astype(np.float32)
        world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
        # Masks and 2D observations are in full-resolution RGB pixels. The
        # Stereo optimization uses the rectified-left working-resolution K.
        intrinsics = self._video_intrinsics(frame_count)

        object_masks, object_visible = _unpack_mask_archive(
            self.run_root / "tracks/sam3/global_masks" / f"{object_id}.npz"
        )
        hand_mask_path = (
            self.run_root
            / "tracks/sam3_hands_full/global_masks"
            / f"{hand_side}_hand.npz"
        )
        if not hand_mask_path.is_file():
            hand_mask_path = (
                self.run_root
                / "tracks/sam3_human_exclusion/global_masks"
                / f"{hand_side}_hand.npz"
            )
        hand_masks = np.zeros_like(object_masks)
        if hand_mask_path.is_file():
            raw_hand_masks, _ = _unpack_mask_archive(hand_mask_path)
            for frame_index, raw in enumerate(raw_hand_masks):
                hand_masks[frame_index] = cv2.resize(
                    raw.astype(np.uint8),
                    (object_masks.shape[2], object_masks.shape[1]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        elif (self.run_root / "calibration/calibration.npz").is_file():
            camera_vertices = np.asarray(hand["vertices_camera"], dtype=np.float32)
            for frame_index in np.flatnonzero(hand_valid):
                hand_masks[frame_index] = self._project_hand_mask(
                    camera_vertices[frame_index, hand_index],
                    intrinsics[frame_index],
                    object_masks.shape[1:],
                )
        masks = np.zeros(object_masks.shape, dtype=np.uint8)
        masks[hand_masks] = 255
        masks[object_masks] = 1

        mesh_candidates = sorted(
            (object_dir / "scale_primary").glob("*_metric_centered.glb")
        )
        if not mesh_candidates:
            mesh_candidates = sorted((object_dir / "scale_primary").glob("*.glb"))
        if not mesh_candidates:
            mesh_candidates = sorted((object_dir / "canonical").glob("*.glb"))
        if not mesh_candidates:
            raise FileNotFoundError(f"No centered object mesh for {object_id}")

        finite = np.isfinite(vertices).all(axis=(1, 2)) & np.isfinite(poses).all(
            axis=(1, 2)
        )
        valid = contact_active & hand_valid & observed & object_visible & finite
        return ContactOptimizationInput(
            object_id=object_id,
            object_name=object_name,
            mesh_path=mesh_candidates[0],
            output_dir=self.run_root / "optim/contact_v2" / object_id,
            hand_side=hand_side,
            hand_vertices_world=vertices,
            hand_joints_world=joints,
            hand_joints_2d=hand["joints_2d_native"][:, hand_index].astype(np.float32),
            initial_object_to_world=poses,
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            masks=masks,
            contact_active=contact_active,
            valid_frames=valid,
            source_indices=source_indices,
        )

    def load_haco_contact_input(self, object_id: str) -> ContactOptimizationInput:
        """Load both hands and HACO vertex weights for one rigid object.

        Unlike :meth:`load_contact_input`, this path does not collapse a
        bimanual event to a dominant hand.  Its ``[T,H,V]`` hand/contact arrays
        preserve the left/right association produced by HACO; the optimizer
        flattens ``H*V`` only after loading.
        """

        from pipeline.optim.contact.types import ContactOptimizationInput

        records = {str(item["global_object_id"]): item for item in self.object_records()}
        if object_id not in records:
            raise KeyError(f"Unknown object ID {object_id}")
        record = records[object_id]
        object_name = str(record["object_name"])

        hand_path = self._first_existing(self.HAND_CANDIDATES)
        hand = np.load(hand_path, allow_pickle=True)
        labels = tuple(str(value) for value in hand["hand_labels"].tolist())
        vertices = hand["vertices_world"].astype(np.float32)
        joints = hand["joints_world"].astype(np.float32)
        hand_valid = hand["valid"].astype(bool)
        frame_count = len(vertices)

        association_path = (
            self.run_root / "hands/haco/object_contacts" / f"{object_id}.npz"
        )
        if not association_path.is_file():
            raise FileNotFoundError(f"Missing HACO object association: {association_path}")
        association = np.load(association_path, allow_pickle=True)
        association_labels = tuple(
            str(value) for value in association["hand_labels"].tolist()
        )
        if association_labels != labels:
            raise ValueError(
                f"HACO hand order {association_labels} differs from motion {labels}"
            )
        weights = association["weights"].astype(np.float32)
        hand_vertex_active = association["hand_vertex_active"].astype(bool)
        contact_active_per_hand = association["event_active"].astype(bool)
        contact_active = contact_active_per_hand.any(axis=1)
        association_valid = association["valid_frames"].astype(bool)
        if weights.shape != vertices.shape[:-1]:
            raise ValueError(
                f"HACO weights {weights.shape} do not match hands {vertices.shape[:-1]}"
            )

        object_dir = self.run_root / "objects" / object_id
        pose_archive = np.load(object_dir / "pose_track/poses.npz")
        poses = pose_archive["object_to_world"].astype(np.float32)
        observed = pose_archive["observed"].astype(bool)
        source_indices = pose_archive["source_indices"].astype(np.int32)
        if len(poses) != frame_count:
            raise ValueError("Hand and object timelines have different lengths")

        cameras = np.load(self.run_root / "4d/cameras.npz")
        camera_to_world = cameras["camera_poses"].astype(np.float32)
        world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
        # Keep 4d camera extrinsics, but use K in the 1280x720 SAM3/RGB pixel
        # coordinate system for silhouette and native-joint reprojection.
        intrinsics = self._video_intrinsics(frame_count)

        object_masks, object_visible = _unpack_mask_archive(
            self.run_root / "tracks/sam3/global_masks" / f"{object_id}.npz"
        )
        independent_hand_masks = self._load_stereo_hand_masks(
            labels=labels,
            frame_count=frame_count,
            output_shape=object_masks.shape[1:],
        )
        hand_masks = np.zeros_like(object_masks)
        if independent_hand_masks is not None:
            hand_masks = independent_hand_masks.any(axis=1)
        for side in labels:
            if independent_hand_masks is not None:
                break
            hand_mask_path = (
                self.run_root / "tracks/sam3_hands_full/global_masks" / f"{side}_hand.npz"
            )
            if not hand_mask_path.is_file():
                hand_mask_path = (
                    self.run_root
                    / "tracks/sam3_human_exclusion/global_masks"
                    / f"{side}_hand.npz"
                )
            if not hand_mask_path.is_file():
                continue
            raw_masks, _ = _unpack_mask_archive(hand_mask_path)
            for frame_index, raw in enumerate(raw_masks):
                hand_masks[frame_index] |= cv2.resize(
                    raw.astype(np.uint8),
                    (object_masks.shape[2], object_masks.shape[1]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        if (
            not hand_masks.any()
            and (self.run_root / "calibration/calibration.npz").is_file()
        ):
            camera_vertices = np.asarray(hand["vertices_camera"], dtype=np.float32)
            for frame_index in range(frame_count):
                for hand_index in np.flatnonzero(hand_valid[frame_index]):
                    hand_masks[frame_index] |= self._project_hand_mask(
                        camera_vertices[frame_index, hand_index],
                        intrinsics[frame_index],
                        object_masks.shape[1:],
                    )
        masks = np.zeros(object_masks.shape, dtype=np.uint8)
        masks[hand_masks] = 255
        masks[object_masks] = 1

        mesh_candidates = sorted(
            (object_dir / "scale_primary").glob("*_metric_centered.glb")
        )
        if not mesh_candidates:
            mesh_candidates = sorted((object_dir / "scale_primary").glob("*.glb"))
        if not mesh_candidates:
            mesh_candidates = sorted((object_dir / "canonical").glob("*.glb"))
        if not mesh_candidates:
            raise FileNotFoundError(f"No centered object mesh for {object_id}")

        finite_hand = np.isfinite(vertices).all(axis=(2, 3))
        hand_vertex_active &= (hand_valid & finite_hand)[:, :, None]
        finite_pose = np.isfinite(poses).all(axis=(1, 2))
        valid = (
            association_valid
            & observed
            & object_visible
            & finite_pose
            & (weights > 0).any(axis=(1, 2))
        )
        return ContactOptimizationInput(
            object_id=object_id,
            object_name=object_name,
            mesh_path=mesh_candidates[0],
            output_dir=self.run_root / "optim/contact_haco_v1" / object_id,
            hand_side="both",
            hand_vertices_world=vertices,
            hand_joints_world=joints,
            hand_joints_2d=hand["joints_2d_native"].astype(np.float32),
            initial_object_to_world=poses,
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            masks=masks,
            contact_active=contact_active,
            valid_frames=valid,
            source_indices=source_indices,
            hand_masks=independent_hand_masks,
            contact_vertex_weights=weights,
            hand_vertex_active=hand_vertex_active,
            hand_labels=labels,
            mano_betas=hand["mano_betas"].astype(np.float32),
            mano_global_orient=hand["mano_global_orient"].astype(np.float32),
            mano_hand_pose=hand["mano_hand_pose"].astype(np.float32),
            hand_motion_path=hand_path,
        )
