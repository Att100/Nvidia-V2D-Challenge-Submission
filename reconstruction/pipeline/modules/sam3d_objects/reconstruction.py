#!/usr/bin/env python3
"""Batch FLUX-completed references through CPU-offloaded SAM 3D Objects."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from pathlib import Path
from pipeline.utils import parse_worker_args

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", default=["primary"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpu-offload", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--schema-version", required=True)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--object-id")
    parser.add_argument("--manifest-name", default="reconstruction_manifest.json")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--isolate-jobs", action=argparse.BooleanOptionalAction, default=True
    )
    return parse_worker_args(parser)


def _completed_candidate(output_dir: Path, job: dict) -> dict | None:
    """Return a validated completed reconstruction for resumable isolation."""

    job_dir = output_dir / job["global_object_id"] / job["role"]
    report_path = job_dir / "reconstruction.json"
    mesh_path = job_dir / "textured_mesh.glb"
    if not report_path.is_file() or not mesh_path.is_file():
        return None
    try:
        candidate = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if candidate.get("status") != "pass":
        return None
    if Path(str(candidate.get("input_image_path", ""))).resolve() != Path(
        job["completed_image_path"]
    ).resolve():
        return None
    if Path(str(candidate.get("input_mask_path", ""))).resolve() != Path(
        job["completed_mask_path"]
    ).resolve():
        return None
    candidate["reused"] = True
    return candidate


def _run_isolated_reconstruction(arguments: dict) -> None:
    """Child-process target that loads SAM3D for exactly one object."""

    run_reconstruction(argparse.Namespace(**arguments))


def run_reconstruction(args: argparse.Namespace) -> None:
    """Reconstruct canonical textured meshes from completed references."""
    sam3d_root = args.config.expanduser().resolve().parents[2]
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_BACKEND", "spconv")
    sys.path.insert(0, str(sam3d_root))
    sys.path.insert(0, str(sam3d_root / "notebook"))

    manifest_path = args.segmentation_manifest.resolve()
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = [
        item
        for item in input_manifest["jobs"]
        if item["role"] in args.roles and item["status"] == "pass"
    ]
    if args.object_id is not None:
        jobs = [job for job in jobs if job["global_object_id"] == args.object_id]
    if not jobs and not args.allow_empty:
        raise RuntimeError(
            "No passing completed-reference segmentations to reconstruct"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_manifest = output_dir / args.manifest_name
    if not jobs:
        result_manifest.write_text(
            json.dumps(
                {
                    "schema_version": args.schema_version,
                    "segmentation_manifest": str(manifest_path),
                    "jobs": [],
                    "status": "skipped_no_support_objects",
                    "elapsed_seconds": 0.0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(result_manifest)
        return

    # SAM3D CPU offload can retain more than 20 GiB after one textured object.
    # Run each object in a fresh spawned process so process exit, rather than
    # allocator heuristics, provides the memory-lifetime boundary.
    if len(jobs) > 1 and args.isolate_jobs:
        started = time.perf_counter()
        records: list[dict] = []
        child_manifests: list[str] = []
        context = mp.get_context("spawn")
        for index, job in enumerate(jobs):
            cached = _completed_candidate(output_dir, job)
            if cached is not None:
                records.append(cached)
                continue
            child_name = (
                f"reconstruction_{job['global_object_id']}_{job['role']}.json"
            )
            child_args = vars(args).copy()
            child_args.update(
                object_id=job["global_object_id"],
                manifest_name=child_name,
                seed_offset=args.seed_offset + index,
                isolate_jobs=False,
            )
            process = context.Process(
                target=_run_isolated_reconstruction,
                args=(child_args,),
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Isolated SAM3D process failed for "
                    f"{job['global_object_id']}/{job['role']} with exit "
                    f"code {process.exitcode}"
                )
            child_manifest = output_dir / child_name
            child_run = json.loads(child_manifest.read_text(encoding="utf-8"))
            if child_run.get("status") != "pass" or len(child_run.get("jobs", [])) != 1:
                raise RuntimeError(f"Invalid isolated SAM3D manifest: {child_manifest}")
            records.extend(child_run["jobs"])
            child_manifests.append(str(child_manifest))
        result_manifest.write_text(
            json.dumps(
                {
                    "schema_version": args.schema_version,
                    "segmentation_manifest": str(manifest_path),
                    "config": str(args.config.resolve()),
                    "cpu_offload": args.cpu_offload,
                    "base_seed": args.seed,
                    "roles": args.roles,
                    "jobs": records,
                    "status": "pass",
                    "execution": "one_spawned_process_per_object",
                    "child_manifests": child_manifests,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(result_manifest)
        return

    import torch
    import trimesh
    from inference import Inference, load_image, load_mask
    run = {
        "schema_version": args.schema_version,
        "segmentation_manifest": str(manifest_path),
        "config": str(args.config.resolve()),
        "cpu_offload": args.cpu_offload,
        "base_seed": args.seed,
        "roles": args.roles,
        "jobs": [],
        "status": "started",
    }
    started = time.perf_counter()
    try:
        inference = Inference(
            str(args.config.resolve()), compile=False, cpu_offload=args.cpu_offload
        )
        run["load_seconds"] = time.perf_counter() - started
        for index, job in enumerate(jobs):
            object_id, role = job["global_object_id"], job["role"]
            image_path = Path(job["completed_image_path"])
            mask_path = Path(job["completed_mask_path"])
            image = load_image(image_path)
            mask = load_mask(mask_path)
            rgba = inference.merge_mask_to_rgba(image, mask)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            inference_start = time.perf_counter()
            result = inference._pipeline.run(
                rgba,
                None,
                seed=args.seed + args.seed_offset + index,
                stage1_only=False,
                with_mesh_postprocess=True,
                with_texture_baking=True,
                with_layout_postprocess=False,
                use_vertex_color=False,
                pointmap=None,
                decode_formats=["mesh", "gaussian"],
            )
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - inference_start
            glb = result.get("glb")
            if glb is None:
                raise RuntimeError(f"SAM 3D returned no GLB for {object_id}/{role}")
            job_dir = output_dir / object_id / role
            job_dir.mkdir(parents=True, exist_ok=True)
            output_path = job_dir / "textured_mesh.glb"
            glb.export(str(output_path))
            scene = trimesh.load(str(output_path), force="scene")
            geometries = list(scene.geometry.values())
            texture_sizes = []
            for geometry in geometries:
                material = getattr(geometry.visual, "material", None)
                texture = getattr(material, "baseColorTexture", None)
                if (
                    getattr(geometry.visual, "kind", None) != "texture"
                    or texture is None
                ):
                    raise RuntimeError(f"Missing embedded texture in {output_path}")
                texture_sizes.append(list(texture.size))
            candidate = {
                "global_object_id": object_id,
                "object_name": job["object_name"],
                "role": role,
                "source_frame": job["source_frame"],
                "seed": args.seed + args.seed_offset + index,
                "input_image_path": str(image_path.resolve()),
                "input_mask_path": str(mask_path.resolve()),
                "output_glb_path": str(output_path.resolve()),
                "geometry_count": len(geometries),
                "vertices": sum(len(geometry.vertices) for geometry in geometries),
                "faces": sum(len(geometry.faces) for geometry in geometries),
                "embedded_texture_sizes": texture_sizes,
                "inference_seconds": inference_seconds,
                "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_vram_gib": torch.cuda.max_memory_reserved() / 2**30,
                "status": "pass",
            }
            (job_dir / "reconstruction.json").write_text(
                json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            run["jobs"].append(candidate)
            print(
                f"{object_id}/{role}: {candidate['vertices']} vertices, "
                f"{candidate['faces']} faces, {inference_seconds:.1f}s, "
                f"{candidate['peak_allocated_vram_gib']:.2f} GiB",
                flush=True,
            )
            del scene, geometries, glb, result, rgba, image, mask
            gc.collect()
            torch.cuda.empty_cache()
        run["status"] = "pass"
    except Exception as error:
        run["status"] = "fail"
        run["exception"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run["elapsed_seconds"] = time.perf_counter() - started
        run["process_peak_rss_gib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        )
        result_manifest.write_text(
            json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if "inference" in locals():
            del inference
        gc.collect()
        torch.cuda.empty_cache()
    print(result_manifest)


class SAM3DObjectReconstructor:
    """Worker-facing class for CPU-offloaded SAM 3D Objects inference."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments

    @classmethod
    def from_worker_job(cls) -> "SAM3DObjectReconstructor":
        return cls(parse_args())

    def run(self) -> None:
        run_reconstruction(self.arguments)


def main() -> None:
    SAM3DObjectReconstructor.from_worker_job().run()


if __name__ == "__main__":
    main()
