"""Project-level optimization stages using dataset adapters."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from datasets import ChallengeRunAdapter
from pipeline.config import get_config
from pipeline.utils import PipelineRequest, StageResult, StageStatus, write_json

from .contact import (
    ContactObjectConfig,
    ContactPoseOptimizer,
    HACOJointHandObjectOptimizer,
    JointHandObjectConfig,
    ObjectCollisionObstacle,
)


def _surface_hand_path(run_root: Path) -> Path:
    """Return the authoritative stereo UmeTrack/MANO hand sequence."""

    return run_root / "hands/umetrack_mano_surface/hand_motion.npz"


def _static_support_collision_obstacles(
    run_root: Path,
    frame_count: int,
) -> tuple[ObjectCollisionObstacle, ...]:
    """Load fixed support meshes as world-space collision obstacles.

    Static supports are reconstructed and registered independently from the
    interaction-object registry. They nevertheless occupy the same VIPE world
    coordinate system and must participate in the rigid-object penetration
    term. Their poses remain fixed while the interaction object and hand
    variables are optimized.
    """

    manifest_path = run_root / "support_objects/pose/manifest.json"
    if not manifest_path.is_file():
        return ()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    obstacles: list[ObjectCollisionObstacle] = []
    for record in manifest.get("objects", []):
        support_id = str(record["global_support_id"])
        mesh_path = Path(record["metric_mesh"])
        pose_path = Path(record["poses"])
        if not mesh_path.is_absolute():
            mesh_path = manifest_path.parent / mesh_path
        if not pose_path.is_absolute():
            pose_path = manifest_path.parent / pose_path
        if not mesh_path.is_file() or not pose_path.is_file():
            raise FileNotFoundError(
                f"Incomplete static support collision asset for {support_id}: "
                f"mesh={mesh_path}, poses={pose_path}"
            )
        with np.load(pose_path) as archive:
            pose_key = (
                "object_to_world"
                if "object_to_world" in archive.files
                else "ob_in_world"
            )
            object_to_world = np.asarray(archive[pose_key], dtype=np.float32)
        if object_to_world.shape != (frame_count, 4, 4):
            raise ValueError(
                f"Static support trajectory for {support_id} has shape "
                f"{object_to_world.shape}, expected {(frame_count, 4, 4)}"
            )
        obstacles.append(
            ObjectCollisionObstacle(
                object_id=support_id,
                mesh_path=mesh_path.resolve(),
                object_to_world=object_to_world,
                valid_frames=np.isfinite(object_to_world).all(axis=(1, 2)),
            )
        )
    return tuple(obstacles)


class HACOContactOptimizationStage:
    """Refine object poses using only HACO-associated MANO vertices."""

    name = "haco_contact_pose_optimization"

    def __init__(
        self,
        *,
        iterations: int = 800,
        device: str = "cuda",
        contact_weight: float = 0.1,
    ) -> None:
        config = replace(
            ContactObjectConfig(),
            iterations=int(iterations),
            device=device,
            contact_weight=float(contact_weight),
            silhouette_weight=2.0,
        )
        self.optimizer = ContactPoseOptimizer(config)

    def run(self, request: PipelineRequest) -> StageResult:
        """Optimize every object with dense per-vertex HACO confidence."""

        manifest_path = request.run_root / "optim/contact_haco_v1/manifest.json"
        if request.dry_run:
            return StageResult(
                stage=self.name,
                status=StageStatus.SKIPPED,
                artifacts=(manifest_path,),
                metrics={"silhouette_weight": 2.0},
            )
        stereo_hand: Path | None = None
        if (
            str(get_config(request.config, "dataset.input_mode", ""))
            == "rgb_plus_stereo_grayscale"
        ):
            stereo_hand = (
                request.run_root
                / "hands/umetrack_mano_surface/hand_motion.npz"
            )
            if not stereo_hand.is_file():
                raise FileNotFoundError(
                    "Stereo HACO optimization requires the UmeTrack/MANO surface "
                    f"output and has no fallback: {stereo_hand}"
                )
        adapter = ChallengeRunAdapter(request.run_root)
        records = adapter.object_records()
        expected = tuple(
            request.run_root
            / "optim/contact_haco_v1"
            / str(item["global_object_id"])
            / "poses.npz"
            for item in records
        ) + (manifest_path,)
        cache_valid = all(path.is_file() for path in expected)
        if cache_valid and stereo_hand is not None:
            try:
                cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                cache_valid = cached_manifest.get("source_hand") == str(stereo_hand)
                cache_valid = cache_valid and np.isclose(
                    float(cached_manifest.get("contact_weight", float("nan"))),
                    float(self.optimizer.config.contact_weight),
                )
                source_paths = (
                    stereo_hand,
                    request.run_root / "hands/haco/object_contacts/manifest.json",
                )
                cache_valid = cache_valid and min(
                    path.stat().st_mtime for path in expected
                ) >= max(path.stat().st_mtime for path in source_paths)
            except (FileNotFoundError, OSError, ValueError, TypeError):
                cache_valid = False
        if request.resume and not request.force and cache_valid:
            return StageResult(
                stage=self.name,
                status=StageStatus.REUSED,
                artifacts=expected,
                message="Validated existing HACO-contact pose tracks.",
            )
        start = time.perf_counter()
        results = []
        for record in records:
            object_id = str(record["global_object_id"])
            data = adapter.load_haco_contact_input(object_id)
            print(
                f"[haco-contact] {object_id} {data.object_name}: "
                f"{int(data.valid_frames.sum())} valid frames, "
                f"{int((data.contact_vertex_weights > 0).sum())} weighted vertices",
                flush=True,
            )
            result = self.optimizer.run(
                data,
                progress=lambda current, total, losses, oid=object_id: print(
                    f"[haco-contact] {oid} {current}/{total} "
                    f"loss={losses['total']:.5f} 2d={losses['silhouette']:.5f}",
                    flush=True,
                ),
            )
            results.append(
                {
                    "object_id": result.object_id,
                    "poses": str(result.poses_path),
                    "report": str(result.report_path),
                    "optimized_frame_count": result.optimized_frame_count,
                    "contact_distance_before_median": result.contact_distance_before_median,
                    "contact_distance_after_median": result.contact_distance_after_median,
                    "silhouette_error_before_median": result.silhouette_error_before_median,
                    "silhouette_error_after_median": result.silhouette_error_after_median,
                }
            )
        write_json(
            manifest_path,
            {
                "schema_version": "ego-scene-hoi-haco-contact-manifest-v1",
                "source_contact": str(
                    request.run_root / "hands/haco/object_contacts/manifest.json"
                ),
                "source_hand": str(_surface_hand_path(request.run_root)),
                "silhouette_weight": 2.0,
                "contact_weight": float(self.optimizer.config.contact_weight),
                "objects": results,
            },
        )
        return StageResult(
            stage=self.name,
            status=StageStatus.COMPLETED,
            artifacts=expected,
            elapsed_seconds=time.perf_counter() - start,
            metrics={
                "object_count": len(results),
                "silhouette_weight": 2.0,
                "contact_weight": float(self.optimizer.config.contact_weight),
            },
        )


def _relative_change(new: float | None, old: float | None) -> float | None:
    """Return signed fractional change while handling absent/zero baselines."""

    if new is None or old is None or not np.isfinite(old) or abs(old) < 1.0e-12:
        return None
    return float((new - old) / old)


class HACOJointContactOptimizationStage:
    """Jointly refine object R/t and MANO articulation with HACO contacts."""

    name = "haco_joint_hand_object_optimization"

    def __init__(
        self,
        config: JointHandObjectConfig | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.config = config or JointHandObjectConfig()
        self.enabled = bool(enabled)
        self.optimizer = HACOJointHandObjectOptimizer(self.config)

    def _run_passthrough(
        self,
        *,
        request: PipelineRequest,
        records: list[dict[str, object]],
        output_root: Path,
        manifest_path: Path,
        comparison_path: Path,
        merged_hand_path: Path,
    ) -> StageResult:
        """Expose fixed-hand HACO and surface-aligned MANO as final outputs.

        Keeping the usual compatibility paths prevents visualizers and export
        tools from accidentally consuming stale joint-optimization artifacts
        when articulation refinement is disabled.
        """

        started = time.perf_counter()
        source_hand_path = _surface_hand_path(request.run_root)
        if not source_hand_path.is_file():
            raise FileNotFoundError(source_hand_path)
        merged_hand_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_hand_path, merged_hand_path)

        objects: list[dict[str, object]] = []
        comparisons: list[dict[str, object]] = []
        artifacts: list[Path] = []
        for record in records:
            object_id = str(record["global_object_id"])
            object_name = str(record.get("object_name", object_id))
            source_dir = request.run_root / "optim/contact_haco_v1" / object_id
            source_pose = source_dir / "poses.npz"
            source_report = source_dir / "report.json"
            if not source_pose.is_file():
                raise FileNotFoundError(source_pose)
            target_dir = output_root / object_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target_pose = target_dir / "poses.npz"
            target_report = target_dir / "report.json"
            shutil.copy2(source_pose, target_pose)
            (target_dir / "hand_articulation.npz").unlink(missing_ok=True)
            write_json(
                target_report,
                {
                    "schema_version": "ego-scene-hoi-haco-joint-passthrough-v1",
                    "status": "disabled_passthrough",
                    "object_id": object_id,
                    "object_name": object_name,
                    "source_pose": str(source_pose),
                    "source_report": str(source_report),
                    "hand_articulation_adjustment": False,
                },
            )
            artifacts.append(target_pose)
            objects.append(
                {
                    "object_id": object_id,
                    "object_name": object_name,
                    "poses": str(target_pose),
                    "report": str(target_report),
                    "hand_articulation": None,
                    "optimized_frame_count": 0,
                    "source": str(source_pose),
                }
            )
            comparisons.append(
                {
                    "object_id": object_id,
                    "object_name": object_name,
                    "fixed_hand_report": str(source_report),
                    "candidate": "disabled; identical to fixed-hand result",
                }
            )

        write_json(
            comparison_path,
            {
                "schema_version": "ego-scene-hoi-haco-joint-comparison-v1",
                "enabled": False,
                "baseline": "fixed-hand HACO object-pose optimization",
                "candidate": "disabled passthrough",
                "objects": comparisons,
            },
        )
        write_json(
            manifest_path,
            {
                "schema_version": "ego-scene-hoi-haco-joint-manifest-v1",
                "enabled": False,
                "mode": "fixed_hand_haco_and_surface_hand_passthrough",
                "source_hand": str(source_hand_path),
                "hand_articulation_adjustment": False,
                "objects": objects,
                "merged_hand": {
                    "path": str(merged_hand_path),
                    "source": str(source_hand_path),
                    "refined_hand_frames": 0,
                    "merge_policy": "passthrough",
                },
                "comparison": str(comparison_path),
            },
        )
        expected = tuple(artifacts) + (
            manifest_path,
            comparison_path,
            merged_hand_path,
        )
        return StageResult(
            stage=self.name,
            status=StageStatus.COMPLETED,
            artifacts=expected,
            elapsed_seconds=time.perf_counter() - started,
            metrics={
                "object_count": len(objects),
                "refined_hand_frames": 0,
                "hand_articulation_adjustment": False,
            },
            message="Joint HACO articulation disabled; previous-stage results passed through.",
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, object] | None:
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None

    def _merge_hand_results(
        self,
        *,
        source_path: Path,
        candidates: list[tuple[str, Path, np.ndarray]],
        output_path: Path,
    ) -> dict[str, object]:
        """Merge per-object hands using dominant per-hand HACO evidence.

        Object optimizations are independent. If one hand is associated with
        multiple objects in the same frame, the candidate with the largest sum
        of HACO vertex confidence is selected. This keeps MANO rotations valid
        and avoids averaging independently optimized joint rotations.
        """

        source = np.load(source_path, allow_pickle=True)
        merged = {name: source[name].copy() for name in source.files}
        frame_count, hand_count = merged["vertices_world"].shape[:2]
        owner = np.full((frame_count, hand_count), "", dtype="<U32")
        best_evidence = np.zeros((frame_count, hand_count), dtype=np.float32)
        selected_count: dict[str, int] = {}
        for object_id, articulation_path, evidence in candidates:
            refined = np.load(articulation_path, allow_pickle=True)
            active = refined["frame_hand_active"].astype(bool)
            if evidence.shape != best_evidence.shape:
                raise ValueError(
                    f"HACO evidence {evidence.shape} does not match hand timeline "
                    f"{best_evidence.shape} for {object_id}"
                )
            select = active & (evidence > best_evidence)
            for key in ("vertices_world", "joints_world", "mano_hand_pose"):
                values = refined[key]
                merged[key][select] = values[select]
            best_evidence[select] = evidence[select]
            owner[select] = object_id
            selected_count[object_id] = int(select.sum())

        # Per-object joint optimization produces refined geometry in the
        # shared world frame. Keep the active camera-space arrays consistent
        # with those selected world-space results; leaving the source values
        # here made downstream overlays depend on which coordinate-space key
        # a consumer happened to read.
        if "camera_to_world" not in merged:
            raise ValueError("Merged hand motion is missing camera_to_world")
        camera_to_world = np.asarray(merged["camera_to_world"], dtype=np.float64)
        if camera_to_world.shape != (frame_count, 4, 4):
            raise ValueError(
                "camera_to_world must have shape "
                f"({frame_count}, 4, 4), got {camera_to_world.shape}"
            )
        for frame_id, hand_id in np.argwhere(owner != ""):
            rotation = camera_to_world[frame_id, :3, :3]
            translation = camera_to_world[frame_id, :3, 3]
            merged["vertices_camera"][frame_id, hand_id] = (
                merged["vertices_world"][frame_id, hand_id] - translation
            ) @ rotation
            merged["joints_camera"][frame_id, hand_id] = (
                merged["joints_world"][frame_id, hand_id] - translation
            ) @ rotation
        merged["haco_joint_refinement_owner"] = owner
        merged["haco_joint_refinement_evidence"] = best_evidence
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **merged)
        return {
            "path": str(output_path),
            "refined_hand_frames": int((owner != "").sum()),
            "selected_hand_frames_by_object": selected_count,
            "merge_policy": "maximum summed HACO vertex confidence per frame and hand",
        }

    def run(self, request: PipelineRequest) -> StageResult:
        """Run the joint ablation and compare it with fixed-hand HACO v1."""

        output_root = request.run_root / "optim/contact_haco_joint_v2"
        manifest_path = output_root / "manifest.json"
        comparison_path = output_root / "comparison_with_fixed_hand.json"
        merged_hand_path = request.run_root / "hands/haco_joint_refined/hand_motion.npz"
        if request.dry_run:
            return StageResult(
                stage=self.name,
                status=StageStatus.SKIPPED,
                artifacts=(manifest_path, comparison_path, merged_hand_path),
            )
        adapter = ChallengeRunAdapter(request.run_root)
        records = adapter.object_records()
        if not self.enabled:
            return self._run_passthrough(
                request=request,
                records=records,
                output_root=output_root,
                manifest_path=manifest_path,
                comparison_path=comparison_path,
                merged_hand_path=merged_hand_path,
            )
        expected = tuple(
            output_root / str(item["global_object_id"]) / "poses.npz"
            for item in records
        ) + (manifest_path, comparison_path, merged_hand_path)
        if request.resume and not request.force and all(path.is_file() for path in expected):
            return StageResult(
                stage=self.name,
                status=StageStatus.REUSED,
                artifacts=expected,
                message="Validated existing joint HACO hand/object refinement.",
            )

        start = time.perf_counter()
        object_results: list[dict[str, object]] = []
        comparisons: list[dict[str, object]] = []
        hand_candidates: list[tuple[str, Path, np.ndarray]] = []
        source_hand_path: Path | None = None
        inputs = {
            str(record["global_object_id"]): adapter.load_haco_contact_input(
                str(record["global_object_id"])
            )
            for record in records
        }
        # Gauss--Seidel collision state: later objects see the already-refined
        # trajectory of earlier objects instead of every object independently
        # colliding against the same stale initialization.
        collision_poses = {
            object_id: data.initial_object_to_world.copy()
            for object_id, data in inputs.items()
        }
        frame_count = next(
            (len(data.initial_object_to_world) for data in inputs.values()),
            0,
        )
        support_obstacles = (
            _static_support_collision_obstacles(request.run_root, frame_count)
            if frame_count
            else ()
        )
        for record in records:
            object_id = str(record["global_object_id"])
            base_data = inputs[object_id]
            interaction_obstacles = tuple(
                ObjectCollisionObstacle(
                    object_id=other_id,
                    mesh_path=other.mesh_path,
                    object_to_world=collision_poses[other_id],
                    valid_frames=(
                        np.isfinite(collision_poses[other_id]).all(axis=(1, 2))
                        & (np.asarray(other.masks) == 1).any(axis=(1, 2))
                    ),
                )
                for other_id, other in inputs.items()
                if other_id != object_id
            )
            obstacles = interaction_obstacles + support_obstacles
            data = replace(
                base_data,
                output_dir=output_root / object_id,
                collision_obstacles=obstacles,
            )
            if data.hand_motion_path is None:
                raise ValueError("HACO joint optimization requires a source hand archive")
            if source_hand_path is None:
                source_hand_path = data.hand_motion_path
            elif source_hand_path != data.hand_motion_path:
                raise ValueError("All objects must share the same source hand archive")
            print(
                f"[haco-joint] {object_id} {data.object_name}: "
                f"{int(data.valid_frames.sum())} valid frames",
                flush=True,
            )
            result = self.optimizer.run(
                data,
                progress=lambda current, total, losses, oid=object_id: print(
                    f"[haco-joint] {oid} {current}/{total} "
                    f"loss={losses['total']:.5f} "
                    f"contact={losses['contact']:.5f} "
                    f"hand2d={losses['hand_reprojection']:.5f}",
                    flush=True,
                ),
            )
            refined_archive = np.load(result.poses_path)
            collision_poses[object_id] = np.asarray(
                refined_archive["object_to_world"], dtype=np.float32
            )
            joint_report = self._load_json(result.report_path) or {}
            joint_metrics = joint_report.get("metrics", {})
            if not isinstance(joint_metrics, dict):
                joint_metrics = {}
            fixed_report_path = (
                request.run_root / "optim/contact_haco_v1" / object_id / "report.json"
            )
            fixed_report = self._load_json(fixed_report_path)
            fixed_metrics = fixed_report.get("metrics", {}) if fixed_report else {}
            if not isinstance(fixed_metrics, dict):
                fixed_metrics = {}
            comparison_metrics: dict[str, object] = {}
            for name in (
                "contact_distance_after_median",
                "penetration_depth_after_median",
                "silhouette_error_after_median_lowres_px",
                "translation_delta_median",
                "rotation_delta_degrees_median",
            ):
                old_value = fixed_metrics.get(name)
                new_value = joint_metrics.get(name)
                comparison_metrics[name] = {
                    "fixed_hand": old_value,
                    "joint_hand_object": new_value,
                    "relative_change": _relative_change(new_value, old_value),
                }
            comparisons.append(
                {
                    "object_id": object_id,
                    "object_name": data.object_name,
                    "fixed_hand_report": str(fixed_report_path),
                    "fixed_hand_available": fixed_report is not None,
                    "joint_report": str(result.report_path),
                    "metrics": comparison_metrics,
                    "hand_metrics": {
                        key: value
                        for key, value in joint_metrics.items()
                        if key.startswith(("articulation_", "hand_"))
                    },
                }
            )
            evidence = np.asarray(data.contact_vertex_weights).sum(axis=2)
            hand_candidates.append((object_id, result.hand_articulation_path, evidence))
            object_results.append(
                {
                    "object_id": object_id,
                    "object_name": data.object_name,
                    "poses": str(result.poses_path),
                    "hand_articulation": str(result.hand_articulation_path),
                    "report": str(result.report_path),
                    "optimized_frame_count": result.optimized_frame_count,
                }
            )
        if source_hand_path is None:
            raise RuntimeError("No objects were available for joint HACO optimization")
        merged_hand = self._merge_hand_results(
            source_path=source_hand_path,
            candidates=hand_candidates,
            output_path=merged_hand_path,
        )
        write_json(
            comparison_path,
            {
                "schema_version": "ego-scene-hoi-haco-joint-comparison-v1",
                "baseline": "fixed-hand HACO object-pose optimization",
                "candidate": "joint MANO-articulation and object-pose HACO optimization",
                "interpretation": "Negative relative change is better for all listed errors.",
                "objects": comparisons,
            },
        )
        write_json(
            manifest_path,
            {
                "schema_version": "ego-scene-hoi-haco-joint-manifest-v1",
                "source_contact": str(
                    request.run_root / "hands/haco/object_contacts/manifest.json"
                ),
                "source_hand": str(source_hand_path),
                "config": self.config.__dict__,
                "objects": object_results,
                "merged_hand": merged_hand,
                "comparison": str(comparison_path),
            },
        )
        return StageResult(
            stage=self.name,
            status=StageStatus.COMPLETED,
            artifacts=expected,
            elapsed_seconds=time.perf_counter() - start,
            metrics={
                "object_count": len(object_results),
                "refined_hand_frames": merged_hand["refined_hand_frames"],
                "silhouette_weight": 2.0,
            },
        )
