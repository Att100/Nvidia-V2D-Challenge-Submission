"""Command-line entry point for the focused stereo + VIPE experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from omegaconf import OmegaConf

from .calibration import FixedStereoCalibrator, StereoCalibrationConfig
from .fusion import MetricFusionConfig, MetricStaticSceneFuser
from .inference import (
    DynamicMaskConfig,
    FoundationStereoSequenceConfig,
    FoundationStereoSequenceEstimator,
    SAM3DynamicMaskBuilder,
)
from .vipe_pose import ViPEMetricPoseEstimator, ViPEPoseConfig


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def run(config_path: Path, stages: list[str]) -> Path:
    """Run selected stages in dependency order and write a stage manifest."""

    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    assert isinstance(raw, dict)
    root = _path(str(raw["repository_root"]))
    output = _path(str(raw["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    calibration_dir = output / "calibration"
    mask_dir = output / "dynamic_mask"
    depth_dir = output / "depth"
    vipe_dir = output / "vipe"
    fusion_dir = output / "fusion"
    stage_set = set(stages)
    known_reports = {
        "calibration": calibration_dir / "report.json",
        "mask": mask_dir / "report.json",
        "depth": depth_dir / "report.json",
        "vipe": vipe_dir / "report.json",
        "fusion": fusion_dir / "report.json",
    }
    reports: dict[str, str] = {
        stage: str(path.resolve()) for stage, path in known_reports.items() if path.exists()
    }
    started = time.perf_counter()

    if "calibration" in stage_set:
        reports["calibration"] = str(
            FixedStereoCalibrator(
                StereoCalibrationConfig(
                    camera_b_video=_path(str(raw["camera_b_video"])),
                    camera_c_video=_path(str(raw["camera_c_video"])),
                    output_dir=calibration_dir,
                    baseline_m=float(raw.get("baseline_m", 0.075)),
                    work_scale=float(raw.get("work_scale", 0.5)),
                    sample_count=int(raw.get("sample_count", 6)),
                    selection_mode=str(
                        raw.get("calibration_selection_mode", "multi_frame_candidates")
                    ),
                    max_rectified_center_offset_fraction=float(
                        raw.get("calibration_max_center_offset_fraction", 0.15)
                    ),
                    min_rectified_valid_fraction=float(
                        raw.get("calibration_min_valid_fraction", 0.70)
                    ),
                    candidate_consensus_weight=float(
                        raw.get("calibration_candidate_consensus_weight", 0.10)
                    ),
                )
            ).run()
        )
    if "mask" in stage_set:
        reports["mask"] = str(
            SAM3DynamicMaskBuilder(
                DynamicMaskConfig(
                    video=calibration_dir / "left_rectified_working.mp4",
                    reasoning_json=_path(str(raw["reasoning_json"])),
                    output_dir=mask_dir,
                    repository_root=root,
                    object_tracks_dir=(
                        _path(str(raw["object_tracks_dir"]))
                        if raw.get("object_tracks_dir")
                        else None
                    ),
                    checkpoint=_path(str(raw["sam3_checkpoint"])) if raw.get("sam3_checkpoint") else None,
                    confidence_threshold=float(raw.get("sam3_confidence_threshold", 0.3)),
                    dilation_pixels=int(raw.get("mask_dilation_pixels", 4)),
                )
            ).run()
        )
    if "depth" in stage_set:
        reports["depth"] = str(
            FoundationStereoSequenceEstimator(
                FoundationStereoSequenceConfig(
                    left_video=calibration_dir / "left_rectified_working.mp4",
                    right_video=calibration_dir / "right_rectified_working.mp4",
                    calibration_npz=calibration_dir / "calibration.npz",
                    repository=root / "thirdparty" / "FoundationStereo",
                    checkpoint=_path(str(raw["foundation_stereo_checkpoint"])),
                    output_dir=depth_dir,
                    valid_iters=int(raw.get("foundation_stereo_iters", 32)),
                    sample_count=int(raw.get("sample_count", 6)),
                )
            ).run()
        )
    if "vipe" in stage_set:
        reports["vipe"] = str(
            ViPEMetricPoseEstimator(
                ViPEPoseConfig(
                    video=calibration_dir / "left_rectified_working.mp4",
                    calibration_npz=calibration_dir / "calibration.npz",
                    depth_dir=depth_dir / "frames",
                    dynamic_mask_npz=mask_dir / "dynamic_mask.npz",
                    vipe_repository=root / "thirdparty" / "vipe",
                    output_dir=vipe_dir,
                    stream_name=str(raw.get("stream_name", "rectified_left")),
                )
            ).run()
        )
    if "fusion" in stage_set:
        name = str(raw.get("stream_name", "rectified_left"))
        reports["fusion"] = str(
            MetricStaticSceneFuser(
                MetricFusionConfig(
                    video=calibration_dir / "left_rectified_working.mp4",
                    calibration_npz=calibration_dir / "calibration.npz",
                    depth_dir=depth_dir / "frames",
                    dynamic_mask_npz=mask_dir / "dynamic_mask.npz",
                    vipe_pose_npz=vipe_dir / "pose" / f"{name}.npz",
                    output_dir=fusion_dir,
                    pixel_stride=int(raw.get("fusion_pixel_stride", 4)),
                    voxel_size_m=float(raw.get("fusion_voxel_size_m", 0.01)),
                    sample_count=int(raw.get("sample_count", 6)),
                )
            ).run()
        )

    manifest = {
        "schema_version": "ego-scene-hoi-stereo-vipe-run-v1",
        "result": "PASS",
        "config": str(config_path.resolve()),
        "executed_stages": stages,
        "completed_stages": sorted(reports),
        "reports": reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["calibration", "mask", "depth", "vipe", "fusion"],
        choices=["calibration", "mask", "depth", "vipe", "fusion"],
    )
    args = parser.parse_args()
    print(run(args.config, args.stages))


if __name__ == "__main__":
    main()

