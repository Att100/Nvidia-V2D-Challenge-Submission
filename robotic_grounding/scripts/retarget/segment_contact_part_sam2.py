#!/usr/bin/env python3
"""Run SAM 2 point/box prompting on calibrated object renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# The shared Isaac environment can contain a mixed cuDNN plugin set. SAM 2 image
# inference remains correct with native CUDA convolution kernels and only runs five images.
torch.backends.cudnn.enabled = False

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def overlay(image: Image.Image, mask: np.ndarray, prompt: dict[str, object]) -> Image.Image:
    base = image.convert("RGBA")
    layer = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    layer[mask] = np.array([20, 220, 80, 120], dtype=np.uint8)
    base = Image.alpha_composite(base, Image.fromarray(layer, mode="RGBA"))
    draw = ImageDraw.Draw(base)
    box = prompt.get("box")
    if box:
        draw.rectangle(tuple(box), outline=(255, 210, 30, 255), width=4)
    for x, y in prompt.get("positive_points", []):
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(20, 255, 70, 255), outline="black", width=2)
    for x, y in prompt.get("negative_points", []):
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(255, 45, 45, 255), outline="white", width=2)
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--views-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(args.prompts.read_text())

    model = build_sam2(args.model_cfg, str(args.checkpoint), device="cuda")
    predictor = SAM2ImagePredictor(model)
    report: dict[str, object] = {"checkpoint": str(args.checkpoint.resolve()), "views": {}}
    for view, prompt in prompts["views"].items():
        image = Image.open(args.views_dir / f"{view}.png").convert("RGB")
        array = np.asarray(image)
        positive = np.asarray(prompt.get("positive_points", []), dtype=np.float32).reshape(-1, 2)
        negative = np.asarray(prompt.get("negative_points", []), dtype=np.float32).reshape(-1, 2)
        points = np.concatenate((positive, negative), axis=0)
        labels = np.concatenate(
            (np.ones(len(positive), dtype=np.int32), np.zeros(len(negative), dtype=np.int32)), axis=0
        )
        box = np.asarray(prompt["box"], dtype=np.float32) if prompt.get("box") else None
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(array)
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=True,
            )
        selected = int(np.argmax(scores))
        mask = masks[selected].astype(bool)
        np.save(args.output_dir / f"{view}_mask.npy", mask)
        overlay(image, mask, prompt).save(args.output_dir / f"{view}_overlay.png")
        ys, xs = np.nonzero(mask)
        report["views"][view] = {
            "selected": selected,
            "scores": [float(x) for x in scores],
            "mask_pixels": int(mask.sum()),
            "mask_fraction": float(mask.mean()),
            "mask_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
            "prompt": prompt,
        }
    (args.output_dir / "sam2_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
