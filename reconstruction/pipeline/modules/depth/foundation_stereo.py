#!/usr/bin/env python3
"""Run FoundationStereo without opening an Open3D window.

This wrapper preserves the official inference path while making checkpoint
loading explicit and safe under Torch 2.6+.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import typing
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from pipeline.utils import parse_worker_args


def load_trusted_checkpoint(path: Path) -> dict:
    """Load the audited official checkpoint with the minimal known allowlist."""
    safe_types = [
        dict,
        list,
        int,
        typing.Any,
        collections.defaultdict,
        DictConfig,
        ListConfig,
        AnyNode,
        ContainerMetadata,
        Metadata,
        np.core.multiarray.scalar,
        np.dtype,
        np.dtypes.Float64DType,
    ]
    with torch.serialization.safe_globals(safe_types):
        return torch.load(path, weights_only=True, map_location="cpu")


def read_intrinsics(path: Path) -> tuple[np.ndarray, float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Expected intrinsics and baseline in {path}")
    intrinsic = np.asarray([float(value) for value in lines[0].split()]).reshape(3, 3)
    return intrinsic, float(lines[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foundationstereo-dir", type=Path, required=True)
    parser.add_argument("--left-file", type=Path, required=True)
    parser.add_argument("--right-file", type=Path, required=True)
    parser.add_argument("--intrinsic-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--hiera", action="store_true")
    parser.add_argument("--z-far", type=float, default=10.0)
    parser.add_argument("--denoise-nb-points", type=int, default=30)
    parser.add_argument("--denoise-radius", type=float, default=0.03)
    args = parse_worker_args(parser)
    if not 0 < args.scale <= 1:
        raise ValueError("--scale must be in (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("FoundationStereo validation requires CUDA")

    repository = args.foundationstereo_dir.resolve()
    sys.path.insert(0, str(repository))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder
    from Utils import depth2xyzmap, vis_disparity

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.autograd.set_grad_enabled(False)
    torch.cuda.reset_peak_memory_stats()

    config = OmegaConf.load(args.checkpoint.parent / "cfg.yaml")
    if "vit_size" not in config:
        config["vit_size"] = "vitl"
    config.valid_iters = args.valid_iters
    config.hiera = int(args.hiera)
    model = FoundationStereo(OmegaConf.create(config))
    checkpoint = load_trusted_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()

    left_original = imageio.imread(args.left_file)
    right_original = imageio.imread(args.right_file)
    if left_original.shape != right_original.shape:
        raise RuntimeError(
            f"Stereo image shapes differ: {left_original.shape} vs {right_original.shape}"
        )
    left = cv2.resize(left_original, dsize=None, fx=args.scale, fy=args.scale)
    right = cv2.resize(right_original, dsize=None, fx=args.scale, fy=args.scale)
    height, width = left.shape[:2]
    left_tensor = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

    inference_start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        if args.hiera:
            disparity_tensor = model.run_hierachical(
                left_tensor,
                right_tensor,
                iters=args.valid_iters,
                test_mode=True,
                small_ratio=0.5,
            )
        else:
            disparity_tensor = model.forward(
                left_tensor, right_tensor, iters=args.valid_iters, test_mode=True
            )
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - inference_start
    disparity = padder.unpad(disparity_tensor.float()).cpu().numpy().reshape(height, width)
    np.save(args.output_dir / "disparity_px.npy", disparity)
    disparity_visualization = vis_disparity(disparity)
    imageio.imwrite(
        args.output_dir / "vis.png", np.concatenate([left, disparity_visualization], axis=1)
    )

    intrinsic, baseline = read_intrinsics(args.intrinsic_file)
    intrinsic = intrinsic.astype(np.float32)
    intrinsic[:2] *= args.scale
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    visible = (xx - disparity) >= 0
    valid_disparity = np.isfinite(disparity) & (disparity > 0) & visible
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    depth[valid_disparity] = intrinsic[0, 0] * baseline / disparity[valid_disparity]
    depth[(depth <= 0) | (depth > args.z_far)] = np.nan
    np.save(args.output_dir / "depth_meter.npy", depth)

    xyz = depth2xyzmap(depth, intrinsic)
    valid_points = np.isfinite(xyz).all(axis=2) & np.isfinite(depth)
    points = xyz[valid_points]
    colors = left[valid_points].astype(np.float64) / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(args.output_dir / "cloud.ply"), cloud)

    _, inlier_indices = cloud.remove_radius_outlier(
        nb_points=args.denoise_nb_points, radius=args.denoise_radius
    )
    denoised = cloud.select_by_index(inlier_indices)
    o3d.io.write_point_cloud(str(args.output_dir / "cloud_denoise.ply"), denoised)

    finite_depth = depth[np.isfinite(depth)]
    positive_disparity = disparity[np.isfinite(disparity) & (disparity > 0)]
    report = {
        "result": "PASS" if len(points) > 0 else "FAIL",
        "inputs": {
            "left_file": str(args.left_file.resolve()),
            "right_file": str(args.right_file.resolve()),
            "intrinsic_file": str(args.intrinsic_file.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
        },
        "checkpoint": {
            "global_step": int(checkpoint["global_step"]),
            "epoch": int(checkpoint["epoch"]),
            "model_tensors": len(checkpoint["model"]),
            "weights_only": True,
        },
        "inference": {
            "input_shape": [height, width],
            "scale": args.scale,
            "valid_iters": args.valid_iters,
            "hierarchical": args.hiera,
            "seconds": inference_seconds,
            "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_vram_gib": torch.cuda.max_memory_reserved() / 2**30,
        },
        "geometry": {
            "rectified_intrinsic": intrinsic.tolist(),
            "baseline_m": baseline,
            "positive_disparity_pixels": int(positive_disparity.size),
            "positive_disparity_min_px": float(positive_disparity.min()),
            "positive_disparity_median_px": float(np.median(positive_disparity)),
            "positive_disparity_max_px": float(positive_disparity.max()),
            "valid_depth_pixels": int(finite_depth.size),
            "depth_min_m": float(finite_depth.min()),
            "depth_median_m": float(np.median(finite_depth)),
            "depth_max_m": float(finite_depth.max()),
            "raw_point_count": len(cloud.points),
            "denoised_point_count": len(denoised.points),
            "all_output_points_finite": bool(np.isfinite(points).all()),
        },
        "total_seconds": time.perf_counter() - start_time,
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
