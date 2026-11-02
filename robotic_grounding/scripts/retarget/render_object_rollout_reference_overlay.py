#!/usr/bin/env python3
"""Overlay an object-only physics rollout with the original timed reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _offline_video import OfflineVideoRenderer  # noqa: E402
from robotic_grounding.retarget.retarget_utils import setup_sharpa_kinematics  # noqa: E402


SIM_COLOR = np.array([35, 190, 255, 205], dtype=np.uint8)
REFERENCE_COLOR = np.array([255, 105, 35, 145], dtype=np.uint8)
RIGHT_HAND_COLOR = (45, 225, 80, 220)
LEFT_HAND_COLOR = (235, 65, 225, 220)


def _tinted(mesh: trimesh.Trimesh, color: np.ndarray) -> trimesh.Trimesh:
    result = mesh.copy()
    # Reconstructed OBJs usually carry TextureVisuals; assigning face_colors on
    # that object does not reliably replace the texture in pyrender.  Install an
    # explicit ColorVisuals instance so the two trajectories remain legible.
    result.visual = trimesh.visual.ColorVisuals(
        mesh=result,
        face_colors=np.tile(color[None, :], (len(result.faces), 1)),
    )
    return result


def _transform(position: np.ndarray, rotation: Rotation) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = rotation.as_matrix()
    value[:3, 3] = position
    return value


def _outlined_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def _reference_hand_qpos(reference: dict, side: str, frame: int, kinematics: object) -> np.ndarray:
    """Build the exact retarget-reference hand configuration stored in parquet."""
    qpos = kinematics.robot.q0.copy()
    qpos[:3] = np.asarray(reference[f"robot_{side}_wrist_position"][frame])
    # Parquet stores wxyz while Pinocchio free joints use xyzw.
    qpos[3:7] = np.asarray(reference[f"robot_{side}_wrist_wxyz"][frame])[[1, 2, 3, 0]]
    qpos[7:] = np.asarray(reference[f"robot_{side}_finger_joints"][frame])
    return qpos


def _minimum_tip_distance_mm(reference: dict, side: str, frame: int) -> float | None:
    """Return the reconstruction MANO fingertip-to-object distance, if present."""
    values = reference.get(f"mano_{side}_tips_distance")
    if values is None or frame >= len(values):
        return None
    distances = np.asarray(values[frame], dtype=np.float64).reshape(-1)
    distances = distances[np.isfinite(distances)]
    return None if distances.size == 0 else 1000.0 * float(np.min(distances))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-parquet", type=Path, required=True)
    parser.add_argument("--simulation-npz", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = pq.ParquetFile(args.reference_parquet.resolve()).read().to_pylist()[0]
    reference_position = np.asarray(reference["object_body_position"], dtype=np.float64)
    reference_wxyz = np.asarray(reference["object_body_wxyz"], dtype=np.float64)
    simulation = np.load(args.simulation_npz.resolve())
    simulation_position = np.asarray(simulation["positions"], dtype=np.float64)
    simulation_xyzw = np.asarray(simulation["quaternions_xyzw"], dtype=np.float64)
    names = [str(value) for value in simulation["names"].tolist()]
    fps = float(simulation["fps"])
    if names != list(reference["object_body_names"]):
        raise ValueError(f"object mismatch: simulation={names}, reference={reference['object_body_names']}")
    frame_count = min(len(simulation_position), len(reference_position))
    if frame_count < 1:
        raise ValueError("overlay has no frames")

    renderer = OfflineVideoRenderer(fps=int(round(fps)))
    right_kinematics = setup_sharpa_kinematics(side="right")
    left_kinematics = setup_sharpa_kinematics(side="left")
    renderer.add_robot("reference_right", right_kinematics, color=RIGHT_HAND_COLOR)
    renderer.add_robot("reference_left", left_kinematics, color=LEFT_HAND_COLOR)
    for name, mesh_path in zip(names, reference["object_mesh_paths"], strict=True):
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        renderer.add_object(f"simulation:{name}", _tinted(mesh, SIM_COLOR))
        renderer.add_object(f"reference:{name}", _tinted(mesh, REFERENCE_COLOR))

    support = json.loads(args.support_report.read_text())
    table = trimesh.creation.cylinder(radius=float(support["radius_m"]), height=float(support["thickness_m"]))
    table.apply_translation(
        [*support["center_xy_m"], float(support["top_height_m"]) - 0.5 * float(support["thickness_m"])]
    )
    table.visual.face_colors = np.array([135, 135, 140, 255], dtype=np.uint8)
    renderer.add_object("simulation_tabletop", table)
    renderer.update_object("simulation_tabletop", np.eye(4))
    points = np.concatenate(
        [
            simulation_position[:frame_count].reshape(-1, 3),
            reference_position[:frame_count].reshape(-1, 3),
            np.asarray(reference["robot_right_wrist_position"][:frame_count]).reshape(-1, 3),
            np.asarray(reference["robot_left_wrist_position"][:frame_count]).reshape(-1, 3),
            np.asarray([[*support["center_xy_m"], support["top_height_m"]]]),
        ],
        axis=0,
    )
    # Missing hands are conventionally hidden at z=-100; exclude those outliers.
    center = np.median(reference_position[:frame_count].reshape(-1, 3), axis=0)
    inliers = points[np.linalg.norm(points - center, axis=1) < 5.0]
    points = inliers if len(inliers) else points
    renderer.fit_camera(points, padding=1.8, min_extent=0.5)

    reference_rotation = Rotation.from_quat(reference_wxyz[:frame_count].reshape(-1, 4), scalar_first=True)
    reference_rotation = reference_rotation.as_quat().reshape(frame_count, len(names), 4)
    for frame in range(frame_count):
        renderer.update_robot(
            "reference_right", _reference_hand_qpos(reference, "right", frame, right_kinematics)
        )
        renderer.update_robot(
            "reference_left", _reference_hand_qpos(reference, "left", frame, left_kinematics)
        )
        translation_errors = []
        rotation_errors = []
        for body, name in enumerate(names):
            sim_rotation = Rotation.from_quat(simulation_xyzw[frame, body])
            ref_rotation = Rotation.from_quat(reference_rotation[frame, body])
            renderer.update_object(
                f"simulation:{name}",
                _transform(simulation_position[frame, body], sim_rotation),
            )
            renderer.update_object(
                f"reference:{name}",
                _transform(reference_position[frame, body], ref_rotation),
            )
            translation_errors.append(
                1000.0 * float(np.linalg.norm(simulation_position[frame, body] - reference_position[frame, body]))
            )
            rotation_errors.append(float(np.degrees((ref_rotation.inv() * sim_rotation).magnitude())))
        renderer.capture()
        image = renderer._frames[-1]  # noqa: SLF001 - annotate the just-captured offline frame.
        _outlined_text(image, "CYAN: optimized frame-0 free physics", (24, 34), (255, 210, 50))
        _outlined_text(image, "ORANGE: original timed reference replay", (24, 64), (35, 105, 255))
        _outlined_text(image, "GREEN: right reference hand   MAGENTA: left reference hand", (24, 94), (245, 245, 245))
        _outlined_text(
            image,
            f"frame {frame:04d}  t={frame / fps:.2f}s  error={max(translation_errors):.1f} mm / {max(rotation_errors):.1f} deg",
            (24, 124),
            (245, 245, 245),
        )
        right_distance = _minimum_tip_distance_mm(reference, "right", frame)
        left_distance = _minimum_tip_distance_mm(reference, "left", frame)
        if right_distance is not None or left_distance is not None:
            right_text = "n/a" if right_distance is None else f"{right_distance:.1f} mm"
            left_text = "n/a" if left_distance is None else f"{left_distance:.1f} mm"
            _outlined_text(
                image,
                f"reconstruction min fingertip-object distance: R={right_text}  L={left_text}",
                (24, 154),
                (245, 245, 245),
            )

    renderer.save(args.output.resolve())
    renderer.close()
    print(args.output.resolve())


if __name__ == "__main__":
    main()
