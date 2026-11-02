#!/usr/bin/env python3
"""Re-segment FLUX-completed object references with the SAM 3 image model."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from pipeline.utils import parse_worker_args

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--flux-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--schema-version", required=True)
    parser.add_argument("--allow-empty", action="store_true")
    return parse_worker_args(parser)


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "thirdparty"))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    selection_dir = args.selection_dir.resolve()
    flux_dir = args.flux_dir.resolve()
    selection_records = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))
        for path in selection_dir.glob("*/selection.json")
    }
    completion_manifest = json.loads(
        (flux_dir / "completion_manifest.json").read_text(encoding="utf-8")
    )
    if not completion_manifest.get("jobs") and args.allow_empty:
        result = flux_dir / "segmentation_manifest.json"
        result.write_text(
            json.dumps(
                {
                    "schema_version": args.schema_version,
                    "jobs": [],
                    "status": "skipped_no_support_objects",
                    "elapsed_seconds": 0.0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(result)
        return
    kwargs: dict[str, Any] = {"device": "cuda"}
    if args.checkpoint is not None:
        kwargs["checkpoint_path"] = str(args.checkpoint.resolve())
        kwargs["load_from_HF"] = False
    model = build_sam3_image_model(**kwargs)
    processor = Sam3Processor(
        model,
        device="cuda",
        confidence_threshold=args.confidence_threshold,
    )
    outputs = []
    started = time.perf_counter()
    try:
        for job in completion_manifest["jobs"]:
            object_id = job["global_object_id"]
            role = job["role"]
            record = selection_records[object_id]
            selected = next(item for item in record["selected"] if item["role"] == role)
            image_path = Path(job["output_image_path"])
            original_mask_path = Path(selected["mask_path"])
            image = Image.open(image_path).convert("RGB")
            original_mask = (
                np.asarray(Image.open(original_mask_path).convert("L")) > 127
            )
            prompt_names = list(
                dict.fromkeys([record["object_name"], *record["observed_object_names"]])
            )
            candidates = []
            for prompt in prompt_names:
                state = processor.set_image(image)
                state = processor.set_text_prompt(prompt=prompt, state=state)
                masks = as_numpy(state["masks"]).astype(bool)
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
                scores = as_numpy(state["scores"]).astype(float).reshape(-1)
                for candidate_index, (mask, score) in enumerate(zip(masks, scores)):
                    if mask.shape != original_mask.shape:
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (original_mask.shape[1], original_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    intersection = int(np.logical_and(mask, original_mask).sum())
                    union = int(np.logical_or(mask, original_mask).sum())
                    original_area = int(original_mask.sum())
                    containment = intersection / original_area if original_area else 0.0
                    iou = intersection / union if union else 0.0
                    association_score = (
                        0.45 * float(score) + 0.40 * containment + 0.15 * iou
                    )
                    candidates.append(
                        {
                            "prompt": prompt,
                            "candidate_index": candidate_index,
                            "sam_score": float(score),
                            "mask_pixels": int(mask.sum()),
                            "original_visible_containment": containment,
                            "original_mask_iou": iou,
                            "association_score": association_score,
                            "mask": mask,
                        }
                    )
            no_completed_candidate = not candidates
            winner = (
                max(candidates, key=lambda item: item["association_score"])
                if candidates
                else {
                    "prompt": None,
                    "candidate_index": -1,
                    "sam_score": 0.0,
                    "mask_pixels": 0,
                    "original_visible_containment": 0.0,
                    "original_mask_iou": 0.0,
                    "association_score": 0.0,
                    "mask": np.zeros_like(original_mask, dtype=bool),
                }
            )
            output_mask_path = image_path.parent / "completed_mask.png"
            Image.fromarray(winner["mask"].astype(np.uint8) * 255).save(
                output_mask_path
            )
            rgb = np.asarray(image).copy()
            mask = winner["mask"]
            rgb[mask] = (0.55 * rgb[mask] + 0.45 * np.array([40, 230, 90])).astype(
                np.uint8
            )
            overlay_path = image_path.parent / "completed_mask_overlay.png"
            Image.fromarray(rgb).save(overlay_path)
            # FLUX can occasionally move an otherwise unobstructed object to a
            # different image location. A high-confidence SAM mask is not
            # sufficient in that case: reconstructing it would silently change
            # the canonical observation. Preserve the failed generated-image
            # segmentation above for diagnosis, but route SAM3D to the original
            # canonical RGB and mask. This is a conservative no-completion
            # fallback, not an acceptance of the relocated generation.
            use_original_reference = (
                no_completed_candidate
                or winner["original_visible_containment"] < 0.5
            )
            reconstruction_image_path = image_path
            reconstruction_mask_path = output_mask_path
            reconstruction_overlay_path = overlay_path
            if use_original_reference:
                reconstruction_image_path = Path(selected["image_path"])
                reconstruction_mask_path = image_path.parent / "original_reference_fallback_mask.png"
                Image.fromarray(original_mask.astype(np.uint8) * 255).save(
                    reconstruction_mask_path
                )
                original_rgb = np.asarray(
                    Image.open(reconstruction_image_path).convert("RGB")
                ).copy()
                original_rgb[original_mask] = (
                    0.55 * original_rgb[original_mask]
                    + 0.45 * np.array([40, 230, 90])
                ).astype(np.uint8)
                reconstruction_overlay_path = (
                    image_path.parent / "original_reference_fallback_overlay.png"
                )
                Image.fromarray(original_rgb).save(reconstruction_overlay_path)
            candidate_records = []
            for candidate in candidates:
                candidate_records.append(
                    {key: value for key, value in candidate.items() if key != "mask"}
                )
            result = {
                "global_object_id": object_id,
                "object_name": record["object_name"],
                "role": role,
                "source_frame": job["source_frame"],
                "completed_image_path": str(reconstruction_image_path.resolve()),
                "original_visible_mask_path": str(original_mask_path.resolve()),
                "completed_mask_path": str(reconstruction_mask_path.resolve()),
                "overlay_path": str(reconstruction_overlay_path.resolve()),
                "selected_prompt": (
                    "original_reference_mask"
                    if use_original_reference
                    else winner["prompt"]
                ),
                "selected_candidate_index": winner["candidate_index"],
                "sam_score": winner["sam_score"],
                "mask_pixels": (
                    int(original_mask.sum())
                    if use_original_reference
                    else winner["mask_pixels"]
                ),
                "original_visible_containment": (
                    1.0
                    if use_original_reference
                    else winner["original_visible_containment"]
                ),
                "original_mask_iou": (
                    1.0 if use_original_reference else winner["original_mask_iou"]
                ),
                "association_score": winner["association_score"],
                "candidates": candidate_records,
                "input_variant": (
                    "fallback_original_reference"
                    if use_original_reference
                    else "flux_completed_reference"
                ),
                "fallback_reason": (
                    "no_sam_candidate_on_flux_output"
                    if no_completed_candidate
                    else "flux_sam_mask_below_original_visible_containment_0.5"
                    if use_original_reference
                    else None
                ),
                "failed_completed_mask_path": (
                    str(output_mask_path.resolve())
                    if use_original_reference
                    else None
                ),
                "failed_completed_overlay_path": (
                    str(overlay_path.resolve())
                    if use_original_reference
                    else None
                ),
                "status": "pass",
            }
            (image_path.parent / "segmentation.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            outputs.append(result)
            print(
                f"{object_id}/{role}: prompt={winner['prompt']!r} "
                f"score={winner['sam_score']:.3f} containment="
                f"{winner['original_visible_containment']:.3f}",
                flush=True,
            )
    finally:
        del processor, model
        gc.collect()
        torch.cuda.empty_cache()
    manifest = {
        "schema_version": args.schema_version,
        "completion_manifest": str((flux_dir / "completion_manifest.json").resolve()),
        "confidence_threshold": args.confidence_threshold,
        "elapsed_seconds": time.perf_counter() - started,
        "jobs": outputs,
        "status": "pass"
        if outputs and all(x["status"] == "pass" for x in outputs)
        else "fail",
    }
    manifest_path = flux_dir / "segmentation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
