"""Preparation of an already-upright, undistorted pinhole video."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil

import cv2
import numpy as np


class GenericVideoPreparer:
    """Copy a video into the pipeline layout and synthesize pinhole metadata."""

    def __init__(self, *, horizontal_fov_degrees: float = 60.0) -> None:
        self.horizontal_fov_degrees = float(horizontal_fov_degrees)

    def prepare(
        self,
        video_path: Path,
        run_root: Path,
        frame_count: int,
    ) -> tuple[Path, ...]:
        """Prepare an MP4 without using dataset camera extrinsics or calibration."""

        source = video_path.expanduser().resolve()
        output_root = run_root.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise OSError(source)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if frame_count != total:
            raise ValueError(
                "GenericVideoPreparer currently preserves the source MP4 exactly; "
                f"requested {frame_count} frames but the video contains {total}"
            )
        if min(width, height, fps) <= 0:
            raise ValueError(f"Invalid video metadata: {source}")

        input_dir = output_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        prepared_video = input_dir / "rgb_upright.mp4"
        if source != prepared_video:
            shutil.copy2(source, prepared_video)

        focal = 0.5 * width / math.tan(
            math.radians(self.horizontal_fov_degrees) * 0.5
        )
        matrix = np.asarray(
            [
                [focal, 0.0, (width - 1) * 0.5],
                [0.0, focal, (height - 1) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        source_indices = np.arange(frame_count, dtype=np.int32)
        timestamps_ns = np.rint(
            source_indices.astype(np.float64) * 1.0e9 / fps
        ).astype(np.int64)
        camera_path = input_dir / "cameras.npz"
        np.savez_compressed(
            camera_path,
            source_indices=source_indices,
            timestamps_ns=timestamps_ns,
            intrinsics=np.repeat(matrix[None], frame_count, axis=0),
            measured_fps=np.asarray(fps, dtype=np.float64),
            encoded_fps=np.asarray(fps, dtype=np.float64),
        )
        valid_path = input_dir / "valid_masks.npz"
        np.savez_compressed(
            valid_path,
            packed_masks=np.full(
                (frame_count, height, (width + 7) // 8), 255, dtype=np.uint8
            ),
            mask_width=np.asarray(width, dtype=np.int32),
            source_indices=source_indices,
        )
        input_manifest_path = input_dir / "manifest.json"
        input_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "ego-scene-hoi-generic-video-v1",
                    "source_video": str(source),
                    "video": str(prepared_video),
                    "camera_policy": "synthetic_pinhole_fov",
                    "uses_gt_intrinsics": False,
                    "uses_gt_camera_extrinsics": False,
                    "undistorted": False,
                    "already_upright": True,
                    "frame_count": frame_count,
                    "image_size": [height, width],
                    "measured_fps": fps,
                    "horizontal_fov_degrees": self.horizontal_fov_degrees,
                    "intrinsics": {
                        "storage": "cameras.npz:intrinsics",
                        "matrix": matrix.tolist(),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        sequence_manifest_path = output_root / "sequence_manifest.json"
        sequence_manifest_path.write_text(
            json.dumps(
                {
                    "format": "ego-scene-hoi-generic-video-v1",
                    "sequence_id": source.stem,
                    "source_sequence": str(source),
                    "video": str(prepared_video),
                    "frame_count": frame_count,
                    "coordinate_system": "input RGB image coordinates",
                    "camera_policy": {
                        "distortion": "input treated as already-undistorted pinhole",
                        "intrinsics": (
                            "synthetic pinhole from image size and horizontal FoV "
                            f"{self.horizontal_fov_degrees:g} degrees"
                        ),
                        "extrinsics": "not supplied during input preparation",
                        "uses_gt_camera_intrinsics": False,
                        "uses_gt_camera_extrinsics": False,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return (
            prepared_video,
            camera_path,
            valid_path,
            input_manifest_path,
            sequence_manifest_path,
        )
