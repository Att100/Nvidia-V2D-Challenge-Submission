# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Reconstruct one masked object and export a textured GLB mesh."""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "checkpoints" / "hf" / "pipeline.yaml"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export a textured mesh GLB from an image and object mask."
    )
    parser.add_argument(
        "--image", required=True, type=Path, help="Input RGB/RGBA image."
    )
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument("--mask", type=Path, help="Path to a binary mask image.")
    mask_group.add_argument(
        "--mask-index",
        type=int,
        help="Mask index, loaded as <mask-dir>/<index>.png.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help="Directory containing indexed masks; defaults to the image directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path; defaults to <image-stem>_textured.glb.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sequentially offload inactive models to CPU (enabled by default).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mask_dir is not None and args.mask_index is None:
        parser.error("--mask-dir requires --mask-index")
    if not args.image.is_file():
        parser.error(f"image does not exist: {args.image}")
    if not args.config.is_file():
        parser.error(f"config does not exist: {args.config}")
    if "CONDA_PREFIX" not in os.environ:
        parser.error("activate the sam3d-objects Conda environment before running")

    sys.path.insert(0, str(REPO_ROOT / "notebook"))
    import torch
    import trimesh
    from inference import Inference, load_image, load_mask, load_single_mask

    image = load_image(args.image)
    if args.mask is not None:
        if not args.mask.is_file():
            parser.error(f"mask does not exist: {args.mask}")
        mask = load_mask(args.mask)
    elif args.mask_index is not None:
        mask_dir = args.mask_dir or args.image.parent
        mask_path = mask_dir / f"{args.mask_index}.png"
        if not mask_path.is_file():
            parser.error(f"indexed mask does not exist: {mask_path}")
        mask = load_single_mask(mask_dir, index=args.mask_index)
    elif image.shape[-1] == 4:
        mask = image[..., 3] > 0
    else:
        parser.error("provide --mask or --mask-index for an RGB input image")

    output_path = args.output or Path(f"{args.image.stem}_textured.glb")
    if output_path.suffix.lower() != ".glb":
        parser.error("--output must use the .glb extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    inference = Inference(
        str(args.config),
        compile=False,
        cpu_offload=args.cpu_offload,
    )
    rgba = inference.merge_mask_to_rgba(image, mask)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = inference._pipeline.run(
        rgba,
        None,
        seed=args.seed,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        with_layout_postprocess=False,
        use_vertex_color=False,
        pointmap=None,
        decode_formats=["mesh", "gaussian"],
    )
    elapsed = time.monotonic() - started

    glb = result.get("glb")
    if glb is None:
        raise RuntimeError("the pipeline did not return a GLB mesh")
    glb.export(str(output_path))

    scene = trimesh.load(str(output_path), force="scene")
    geometries = list(scene.geometry.values())
    texture_sizes = []
    for geometry in geometries:
        material = getattr(geometry.visual, "material", None)
        texture = getattr(material, "baseColorTexture", None)
        if getattr(geometry.visual, "kind", None) != "texture" or texture is None:
            raise RuntimeError(
                "exported GLB does not contain an embedded base-color texture"
            )
        texture_sizes.append(texture.size)

    vertices = sum(len(geometry.vertices) for geometry in geometries)
    faces = sum(len(geometry.faces) for geometry in geometries)
    peak_vram = torch.cuda.max_memory_allocated() / 2**30
    print(f"Exported textured GLB: {output_path.resolve()}")
    print(f"Geometry: {vertices} vertices, {faces} faces")
    print(f"Embedded textures: {texture_sizes}")
    print(f"Elapsed: {elapsed:.1f}s; peak VRAM: {peak_vram:.2f} GiB")


if __name__ == "__main__":
    main()
