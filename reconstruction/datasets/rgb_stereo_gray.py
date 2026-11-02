"""Preparation for synchronized native RGB plus grayscale stereo videos."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import cv2

from .generic_video import GenericVideoPreparer


def _metadata(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


class RGBStereoGrayPreparer:
    """Normalize three synchronized videos without assuming cross-camera calibration."""

    def __init__(
        self,
        *,
        stereo_left: Path,
        stereo_right: Path,
        horizontal_fov_degrees: float = 60.0,
    ) -> None:
        self.stereo_left = stereo_left.expanduser().resolve()
        self.stereo_right = stereo_right.expanduser().resolve()
        self.horizontal_fov_degrees = float(horizontal_fov_degrees)

    def prepare(self, rgb_video: Path, run_root: Path, frame_count: int) -> tuple[Path, ...]:
        rgb_video = rgb_video.expanduser().resolve()
        run_root = run_root.expanduser().resolve()
        base_artifacts = GenericVideoPreparer(
            horizontal_fov_degrees=self.horizontal_fov_degrees
        ).prepare(rgb_video, run_root, frame_count)
        metadata = {
            "rgb": _metadata(rgb_video),
            "stereo_left": _metadata(self.stereo_left),
            "stereo_right": _metadata(self.stereo_right),
        }
        if len({values[0] for values in metadata.values()}) != 1:
            raise ValueError(f"Synchronized streams have different frame counts: {metadata}")
        if max(values[1] for values in metadata.values()) - min(
            values[1] for values in metadata.values()
        ) > 1e-3:
            raise ValueError(f"Synchronized streams have different FPS: {metadata}")
        input_dir = run_root / "input"
        prepared = []
        for source, name in (
            (self.stereo_left, "stereo_left_raw.mp4"),
            (self.stereo_right, "stereo_right_raw.mp4"),
        ):
            target = input_dir / name
            if source != target:
                shutil.copy2(source, target)
            prepared.append(target)
        manifest_path = input_dir / "multiview_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "ego-scene-hoi-rgb-stereo-gray-input-v1",
                    "input_mode": "rgb_plus_stereo_grayscale",
                    "synchronization": "strict_frame_index",
                    "frame_count": frame_count,
                    "fps": metadata["rgb"][1],
                    "streams": {
                        "rgb": {
                            "source": str(rgb_video),
                            "prepared": str(base_artifacts[0]),
                            "size": [metadata["rgb"][3], metadata["rgb"][2]],
                        },
                        "stereo_left": {
                            "source": str(self.stereo_left),
                            "prepared": str(prepared[0]),
                            "size": [metadata["stereo_left"][3], metadata["stereo_left"][2]],
                        },
                        "stereo_right": {
                            "source": str(self.stereo_right),
                            "prepared": str(prepared[1]),
                            "size": [metadata["stereo_right"][3], metadata["stereo_right"][2]],
                        },
                    },
                    "uses_gt_camera_extrinsics": False,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return (*base_artifacts, *prepared, manifest_path)
