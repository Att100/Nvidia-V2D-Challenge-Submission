"""VIPE pose-only integration with external stereo depth and dynamic masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Iterator

import cv2
import numpy as np
from omegaconf import OmegaConf
import torch


@dataclass(frozen=True)
class ViPEPoseConfig:
    """Configuration for fixed-intrinsic, metric-depth VIPE inference."""

    video: Path
    calibration_npz: Path
    depth_dir: Path
    dynamic_mask_npz: Path
    vipe_repository: Path
    output_dir: Path
    stream_name: str = "episode_000003_rectified_left"
    robust_kernel: str | None = "huber"
    robust_kernel_threshold: float = 3.0


class ViPEMetricPoseEstimator:
    """Estimate c2w poses while preserving externally supplied geometry.

    FoundationStereo depth supplies metric inverse-depth anchors, AnyCalib and
    stereo rectification supply fixed intrinsics, and the external SAM3 union
    supplies the only dynamic-pixel mask used by VIPE.
    """

    def __init__(self, config: ViPEPoseConfig) -> None:
        self.config = config

    def run(self) -> Path:
        cfg = self.config
        repository = cfg.vipe_repository.resolve()
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))

        from vipe.pipeline.pose_only import PoseOnlyAnnotationPipeline
        from vipe.streams.base import FrameAttribute, VideoFrame, VideoStream
        from vipe.utils.cameras import CameraType

        calibration = np.load(cfg.calibration_npz)
        intrinsic_matrix = np.asarray(calibration["K_working"], dtype=np.float32)
        intrinsic = torch.tensor(
            [
                intrinsic_matrix[0, 0],
                intrinsic_matrix[1, 1],
                intrinsic_matrix[0, 2],
                intrinsic_matrix[1, 2],
            ],
            dtype=torch.float32,
        )
        rectified_valid = np.asarray(calibration["left_valid"], dtype=bool)
        mask_archive = np.load(cfg.dynamic_mask_npz)
        packed_masks = np.asarray(mask_archive["packed_masks"], dtype=np.uint8)
        mask_width = int(mask_archive["mask_width"])
        dynamic_masks = np.unpackbits(packed_masks, axis=2, count=mask_width).astype(bool)

        class ExternalMetricRGBDStream(VideoStream):
            """Lazy rectified RGB-D stream implementing VIPE's stream API."""

            def __init__(self) -> None:
                capture = cv2.VideoCapture(str(cfg.video))
                if not capture.isOpened():
                    raise FileNotFoundError(cfg.video)
                self._length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self._width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._fps = float(capture.get(cv2.CAP_PROP_FPS))
                capture.release()
                if dynamic_masks.shape != (self._length, self._height, self._width):
                    raise ValueError(
                        f"Dynamic masks {dynamic_masks.shape} != video "
                        f"{(self._length, self._height, self._width)}"
                    )

            def frame_size(self) -> tuple[int, int]:
                return self._height, self._width

            def name(self) -> str:
                return cfg.stream_name

            def fps(self) -> float:
                return self._fps

            def __len__(self) -> int:
                return self._length

            def attributes(self) -> set[FrameAttribute]:
                return {
                    FrameAttribute.INTRINSICS,
                    FrameAttribute.CAMERA_TYPE,
                    FrameAttribute.MASK,
                    FrameAttribute.METRIC_DEPTH,
                }

            def __iter__(self) -> Iterator[VideoFrame]:
                capture = cv2.VideoCapture(str(cfg.video))
                if not capture.isOpened():
                    raise FileNotFoundError(cfg.video)
                try:
                    for frame_id in range(self._length):
                        ok, bgr = capture.read()
                        if not ok:
                            raise RuntimeError(f"Failed to decode frame {frame_id}")
                        depth = np.load(cfg.depth_dir / f"{frame_id:06d}.npy")
                        if depth.shape != (self._height, self._width):
                            raise ValueError(f"Depth shape mismatch at frame {frame_id}: {depth.shape}")
                        valid_static = (
                            rectified_valid
                            & ~dynamic_masks[frame_id]
                            & np.isfinite(depth)
                            & (depth > 0)
                        )
                        masked_depth = np.where(valid_static, depth, 0.0).astype(np.float32)
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        yield VideoFrame(
                            raw_frame_idx=frame_id,
                            rgb=torch.from_numpy(rgb.copy()).float().div_(255.0).cuda(),
                            intrinsics=intrinsic.clone().cuda(),
                            camera_type=CameraType.PINHOLE,
                            mask=torch.from_numpy(valid_static.copy()).cuda(),
                            metric_depth=torch.from_numpy(masked_depth).cuda(),
                        )
                finally:
                    capture.release()

        slam = OmegaConf.load(repository / "configs" / "slam" / "default.yaml")
        slam.keyframe_depth = None
        slam.optimize_intrinsics = False
        slam.visualize = False
        slam.ba.robust_kernel = cfg.robust_kernel
        slam.ba.robust_kernel_threshold = cfg.robust_kernel_threshold
        # The fused kernel does not implement robust residual kernels.
        slam.ba.fused = False if cfg.robust_kernel is not None else "auto"
        init = OmegaConf.create(
            {
                "camera_type": "pinhole",
                "intrinsics": "gt",
                "async_prefetch": True,
                "prefetch_queue_size": 8,
                "instance": None,
            }
        )
        output_root = cfg.output_dir.resolve()
        output = OmegaConf.create(
            {
                "path": str(output_root),
                "skip_exists": False,
                "save_artifacts": True,
                "save_slam_map": True,
                "save_viz": False,
                "viz_downsample": 2,
                "viz_attributes": [["rgb"]],
            }
        )
        started = time.perf_counter()
        pipeline = PoseOnlyAnnotationPipeline(init=init, slam=slam, output=output)
        pipeline.run(ExternalMetricRGBDStream())

        pose_path = output_root / "pose" / f"{cfg.stream_name}.npz"
        intrinsics_path = output_root / "intrinsics" / f"{cfg.stream_name}.npz"
        slam_map_path = output_root / "vipe" / f"{cfg.stream_name}_slam_map.pt"
        for required in (pose_path, intrinsics_path, slam_map_path):
            if not required.exists():
                raise FileNotFoundError(f"VIPE did not create {required}")
        poses = np.load(pose_path)["data"]
        translation_steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
        report = {
            "schema_version": "ego-scene-hoi-vipe-external-stereo-v1",
            "result": "PASS",
            "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(cfg).items()},
            "external_inputs": {
                "fixed_intrinsics": True,
                "metric_depth": "FoundationStereo per-frame depth",
                "dynamic_mask": "SAM3 Codex-object + second-pass hand + body union",
                "vipe_default_depth_disabled": True,
                "vipe_default_dynamic_segmentation_disabled": True,
            },
            "pose_count": int(len(poses)),
            "translation_step_m": {
                "median": float(np.median(translation_steps)),
                "p95": float(np.quantile(translation_steps, 0.95)),
                "max": float(translation_steps.max()),
                "max_frame": int(np.argmax(translation_steps) + 1),
            },
            "pose_path": str(pose_path),
            "intrinsics_path": str(intrinsics_path),
            "slam_map_path": str(slam_map_path),
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_path = output_root / "report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report_path

