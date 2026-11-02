#!/usr/bin/env python3
"""Run FLUX.2 Klein 9B completion on selected sparse object references."""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.hooks.layerwise_casting import apply_layerwise_casting
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import Qwen3Config, Qwen3ForCausalLM
from pipeline.utils import parse_worker_args

LAYERWISE_CASTABLE = (
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
    torch.nn.Linear,
    torch.nn.Embedding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--roles", nargs="+", default=["primary"])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--schema-version", required=True)
    parser.add_argument("--allow-empty", action="store_true")
    return parse_worker_args(parser)


def promote_unhooked_fp8_parameters(module: torch.nn.Module) -> int:
    promoted_numel = 0
    with torch.no_grad():
        for submodule in module.modules():
            if isinstance(submodule, LAYERWISE_CASTABLE):
                continue
            for parameter in submodule.parameters(recurse=False):
                if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
                    promoted_numel += parameter.numel()
                    parameter.data = parameter.data.to(dtype=torch.bfloat16)
            for name, buffer in submodule.named_buffers(recurse=False):
                if buffer.is_floating_point() and buffer.dtype != torch.bfloat16:
                    promoted_numel += buffer.numel()
                    submodule._buffers[name] = buffer.to(dtype=torch.bfloat16)
    return promoted_numel


def gpu_memory() -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    gib = 1024**3
    return {
        "free_gib": free / gib,
        "used_gib": (total - free) / gib,
        "torch_allocated_gib": torch.cuda.memory_allocated() / gib,
        "torch_reserved_gib": torch.cuda.memory_reserved() / gib,
        "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "torch_peak_reserved_gib": torch.cuda.max_memory_reserved() / gib,
    }


def load_pipeline(model_id: str) -> tuple[Flux2KleinPipeline, dict[str, int]]:
    text_encoder_config = Qwen3Config.from_pretrained(
        model_id, subfolder="text_encoder"
    )
    with init_empty_weights():
        text_encoder = Qwen3ForCausalLM(text_encoder_config)
    snapshot_dir = Path(
        snapshot_download(repo_id=model_id, allow_patterns=["text_encoder/*"])
    )
    text_encoder = load_checkpoint_and_dispatch(
        text_encoder,
        checkpoint=snapshot_dir / "text_encoder",
        device_map={"": "cpu"},
        dtype=torch.float8_e4m3fn,
        strict=True,
    )
    apply_layerwise_casting(
        text_encoder,
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
        skip_modules_pattern=("embed_tokens",),
    )
    text_encoder.model.embed_tokens.to(dtype=torch.bfloat16)
    text_promoted = promote_unhooked_fp8_parameters(text_encoder)
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        dtype=torch.float8_e4m3fn,
        low_cpu_mem_usage=True,
    )
    apply_layerwise_casting(
        transformer,
        storage_dtype=torch.float8_e4m3fn,
        compute_dtype=torch.bfloat16,
        skip_modules_pattern=(),
    )
    transformer_promoted = promote_unhooked_fp8_parameters(transformer)
    pipeline = Flux2KleinPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=transformer,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipeline.enable_model_cpu_offload()
    return pipeline, {
        "text_encoder_promoted_numel": text_promoted,
        "transformer_promoted_numel": transformer_promoted,
    }


def completion_prompt(object_name: str) -> str:
    return (
        f"Prepare this exact {object_name} for single-image 3D reconstruction. "
        f"Reveal a complete unobstructed view of the same {object_name} at its current "
        "location and viewpoint. Remove only hands or fingers that occlude or touch the "
        "object and reconstruct the hidden object surface naturally. Preserve the exact "
        "visible shape, proportions, color, material, texture, markings, lighting, camera "
        "viewpoint, scale, and background. Do not move, replace, redesign, duplicate, or "
        "add parts to the object. If the object is already unobstructed, leave it unchanged."
    )


def run_completion(args: argparse.Namespace) -> None:
    """Complete selected object references with FLUX.2 Klein."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    selection_dir = args.selection_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = []
    for path in sorted(selection_dir.glob("*/selection.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        selected_by_role = {item["role"]: item for item in record["selected"]}
        for role in args.roles:
            if role in selected_by_role:
                selections.append((record, selected_by_role[role]))
    if not selections and not args.allow_empty:
        raise RuntimeError("No selected references matched --roles")

    if not selections:
        (output_dir / "completion_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": args.schema_version,
                    "model_id": args.model_id,
                    "roles": args.roles,
                    "jobs": [],
                    "status": "skipped_no_support_objects",
                    "elapsed_seconds": 0.0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(output_dir / "completion_manifest.json")
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run = {
        "schema_version": args.schema_version,
        "model_id": args.model_id,
        "dtype_policy": "fp8_storage_bf16_layerwise_compute_component_cpu_offload",
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "base_seed": args.seed,
        "roles": args.roles,
        "memory_before_load": gpu_memory(),
        "jobs": [],
        "status": "started",
    }
    started = time.perf_counter()
    try:
        pipeline, promoted = load_pipeline(args.model_id)
        run.update(promoted)
        run["load_seconds"] = time.perf_counter() - started
        run["memory_after_load"] = gpu_memory()
        for index, (record, selected) in enumerate(selections):
            object_id = record["global_object_id"]
            object_name = record["object_name"]
            role = selected["role"]
            image_path = Path(selected["image_path"])
            mask_path = Path(selected["mask_path"])
            image = Image.open(image_path).convert("RGB")
            if image.width % 16 or image.height % 16:
                raise ValueError(
                    f"FLUX input size must be divisible by 16: {image.size}"
                )
            prompt = completion_prompt(object_name)
            seed = args.seed + index
            generator = torch.Generator(device="cuda").manual_seed(seed)
            job_dir = output_dir / object_id / role
            job_dir.mkdir(parents=True, exist_ok=True)
            inference_start = time.perf_counter()
            result = pipeline(
                image=image,
                prompt=prompt,
                height=image.height,
                width=image.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - inference_start
            output_path = job_dir / "completed_rgb.png"
            result.save(output_path)
            before = np.asarray(image, dtype=np.float32)
            after = np.asarray(result.convert("RGB"), dtype=np.float32)
            mask = np.asarray(Image.open(mask_path).convert("L")) > 127
            absolute_difference = np.abs(after - before).mean(axis=2)
            job = {
                "global_object_id": object_id,
                "object_name": object_name,
                "role": role,
                "source_frame": selected["source_frame"],
                "input_image_path": str(image_path.resolve()),
                "input_mask_path": str(mask_path.resolve()),
                "output_image_path": str(output_path.resolve()),
                "prompt": prompt,
                "seed": seed,
                "inference_seconds": inference_seconds,
                "mean_absolute_pixel_difference": float(absolute_difference.mean()),
                "mean_difference_inside_visible_mask": float(
                    absolute_difference[mask].mean()
                ),
                "mean_difference_outside_visible_mask": float(
                    absolute_difference[~mask].mean()
                ),
                "memory_after_inference": gpu_memory(),
                "status": "generated_unvalidated",
            }
            (job_dir / "completion.json").write_text(
                json.dumps(job, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            run["jobs"].append(job)
            print(
                f"{object_id}/{role} frame={selected['source_frame']} "
                f"seconds={inference_seconds:.2f} diff={absolute_difference.mean():.2f}",
                flush=True,
            )
        run["status"] = "generated_unvalidated"
    except Exception as error:
        run["status"] = "fail"
        run["exception"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run["elapsed_seconds"] = time.perf_counter() - started
        run["process_peak_rss_gib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        )
        if torch.cuda.is_available():
            run["final_memory"] = gpu_memory()
        (output_dir / "completion_manifest.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if "pipeline" in locals():
            del pipeline
        gc.collect()
        torch.cuda.empty_cache()
    print(output_dir / "completion_manifest.json")


class Flux2ObjectCompleter:
    """Worker-facing class for occluded-object appearance completion."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments

    @classmethod
    def from_worker_job(cls) -> "Flux2ObjectCompleter":
        return cls(parse_args())

    def run(self) -> None:
        run_completion(self.arguments)


def main() -> None:
    Flux2ObjectCompleter.from_worker_job().run()


if __name__ == "__main__":
    main()
