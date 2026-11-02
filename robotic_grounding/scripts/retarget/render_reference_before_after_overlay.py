#!/usr/bin/env python3
"""Render before/after object and retarget-hand references in one scene."""

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


BEFORE_OBJECT = np.array([255, 125, 25, 125], dtype=np.uint8)
AFTER_OBJECT = np.array([30, 195, 255, 205], dtype=np.uint8)
BEFORE_RIGHT = (255, 190, 35, 115)
BEFORE_LEFT = (255, 75, 55, 115)
AFTER_RIGHT = (45, 225, 80, 225)
AFTER_LEFT = (235, 65, 225, 225)


def _tinted(mesh: trimesh.Trimesh, color: np.ndarray) -> trimesh.Trimesh:
    result = mesh.copy()
    result.visual = trimesh.visual.ColorVisuals(
        mesh=result,
        face_colors=np.tile(color[None, :], (len(result.faces), 1)),
    )
    return result


def _transform(position: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_quat(wxyz, scalar_first=True).as_matrix()
    value[:3, 3] = position
    return value


def _hand_qpos(row: dict, side: str, frame: int, kinematics: object) -> np.ndarray:
    qpos = kinematics.robot.q0.copy()
    qpos[:3] = np.asarray(row[f"robot_{side}_wrist_position"][frame])
    qpos[3:7] = np.asarray(row[f"robot_{side}_wrist_wxyz"][frame])[[1, 2, 3, 0]]
    qpos[7:] = np.asarray(row[f"robot_{side}_finger_joints"][frame])
    return qpos


def _outlined_text(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.60, (15, 15, 15), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)


def _rotation_error_deg(before: np.ndarray, after: np.ndarray) -> float:
    a = Rotation.from_quat(before, scalar_first=True)
    b = Rotation.from_quat(after, scalar_first=True)
    return float(np.degrees((a.inv() * b).magnitude()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-parquet", type=Path, required=True)
    parser.add_argument("--after-parquet", type=Path, required=True)
    parser.add_argument("--support-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = pq.ParquetFile(args.before_parquet.resolve()).read().to_pylist()[0]
    after = pq.ParquetFile(args.after_parquet.resolve()).read().to_pylist()[0]
    if before["object_body_names"] != after["object_body_names"]:
        raise ValueError("before/after object names or ordering differ")
    names = list(before["object_body_names"])
    frame_count = min(len(before["object_body_position"]), len(after["object_body_position"]))
    fps = float(after["fps"])

    renderer = OfflineVideoRenderer(fps=int(round(fps)))
    kinematics: dict[tuple[str, str], object] = {}
    for version in ("before", "after"):
        for side in ("right", "left"):
            kin = setup_sharpa_kinematics(side=side)
            kinematics[(version, side)] = kin
            color = {
                ("before", "right"): BEFORE_RIGHT,
                ("before", "left"): BEFORE_LEFT,
                ("after", "right"): AFTER_RIGHT,
                ("after", "left"): AFTER_LEFT,
            }[(version, side)]
            renderer.add_robot(f"{version}_{side}", kin, color=color)

    for name, mesh_path in zip(names, before["object_mesh_paths"], strict=True):
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        renderer.add_object(f"before:{name}", _tinted(mesh, BEFORE_OBJECT))
        renderer.add_object(f"after:{name}", _tinted(mesh, AFTER_OBJECT))

    support = json.loads(args.support_report.read_text())
    table = trimesh.creation.cylinder(radius=float(support["radius_m"]), height=float(support["thickness_m"]))
    table.apply_translation(
        [*support["center_xy_m"], float(support["top_height_m"]) - 0.5 * float(support["thickness_m"])]
    )
    table.visual.face_colors = np.array([135, 135, 140, 255], dtype=np.uint8)
    renderer.add_object("tabletop", table)
    renderer.update_object("tabletop", np.eye(4))

    point_sets = []
    for row in (before, after):
        point_sets.extend(
            [
                np.asarray(row["object_body_position"][:frame_count]).reshape(-1, 3),
                np.asarray(row["robot_right_wrist_position"][:frame_count]).reshape(-1, 3),
                np.asarray(row["robot_left_wrist_position"][:frame_count]).reshape(-1, 3),
            ]
        )
    points = np.concatenate(point_sets, axis=0)
    object_center = np.median(np.asarray(after["object_body_position"][:frame_count]).reshape(-1, 3), axis=0)
    inliers = points[np.linalg.norm(points - object_center, axis=1) < 5.0]
    renderer.fit_camera(inliers if len(inliers) else points, padding=1.55, min_extent=0.5)

    before_position = np.asarray(before["object_body_position"], dtype=np.float64)
    after_position = np.asarray(after["object_body_position"], dtype=np.float64)
    before_wxyz = np.asarray(before["object_body_wxyz"], dtype=np.float64)
    after_wxyz = np.asarray(after["object_body_wxyz"], dtype=np.float64)
    before_wrists = {
        side: np.asarray(before[f"robot_{side}_wrist_position"], dtype=np.float64)
        for side in ("right", "left")
    }
    after_wrists = {
        side: np.asarray(after[f"robot_{side}_wrist_position"], dtype=np.float64)
        for side in ("right", "left")
    }
    active = {
        side: np.any(np.asarray(after[f"mano_{side}_object_contact_part_ids"]) > 0, axis=1)
        for side in ("right", "left")
    }

    for frame in range(frame_count):
        for version, row in (("before", before), ("after", after)):
            for side in ("right", "left"):
                renderer.update_robot(
                    f"{version}_{side}",
                    _hand_qpos(row, side, frame, kinematics[(version, side)]),
                )
        object_translation_errors = []
        object_rotation_errors = []
        for body, name in enumerate(names):
            renderer.update_object(
                f"before:{name}", _transform(before_position[frame, body], before_wxyz[frame, body])
            )
            renderer.update_object(
                f"after:{name}", _transform(after_position[frame, body], after_wxyz[frame, body])
            )
            object_translation_errors.append(
                1000.0 * float(np.linalg.norm(after_position[frame, body] - before_position[frame, body]))
            )
            object_rotation_errors.append(
                _rotation_error_deg(before_wxyz[frame, body], after_wxyz[frame, body])
            )

        renderer.capture()
        image = renderer._frames[-1]  # noqa: SLF001
        right_delta = 1000.0 * np.linalg.norm(after_wrists["right"][frame] - before_wrists["right"][frame])
        left_delta = 1000.0 * np.linalg.norm(after_wrists["left"][frame] - before_wrists["left"][frame])
        contact_text = f"contact R={'ON' if active['right'][frame] else 'off'}  L={'ON' if active['left'][frame] else 'off'}"
        _outlined_text(image, "BEFORE: orange object / yellow right / red left", (24, 32), (35, 150, 255))
        _outlined_text(image, "AFTER: cyan object / green right / magenta left", (24, 62), (255, 225, 60))
        _outlined_text(
            image,
            f"frame {frame:04d}  t={frame / fps:.2f}s  {contact_text}",
            (24, 94),
            (245, 245, 245),
        )
        _outlined_text(
            image,
            f"object delta={max(object_translation_errors):.1f} mm / {max(object_rotation_errors):.1f} deg   wrist delta R={right_delta:.1f} mm L={left_delta:.1f} mm",
            (24, 126),
            (245, 245, 245),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    renderer.save(args.output.resolve())
    renderer.close()
    print(args.output.resolve())


if __name__ == "__main__":
    main()
