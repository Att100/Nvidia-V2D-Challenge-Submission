"""Compare reconstruction, IK-target, and retargeted fingertip/object distances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from robotic_grounding.retarget.params import MANO_JOINTS_ORDER, SHARPA_TO_MANO_MAPPING


def load_row(partition: Path) -> dict:
    files = sorted(partition.glob("*.parquet"))
    if len(files) != 1:
        raise ValueError(f"expected one parquet in {partition}, got {len(files)}")
    return pq.ParquetFile(files[0]).read().to_pylist()[0]


def surface_distances(
    points_world: np.ndarray,
    meshes: list[trimesh.Trimesh],
    sampled_surface_trees: list[cKDTree | None],
    body_positions: np.ndarray,
    body_wxyz: np.ndarray,
) -> np.ndarray:
    output = np.full(points_world.shape[:2], np.inf, dtype=np.float64)
    # Transform the small set of tracked sites for all frames at once, then do
    # one proximity query per body.  The previous frame-by-frame form rebuilt
    # query batches hundreds of times and became prohibitively slow for dense
    # reconstructed meshes (for example the 98k-face pot in episode_000028).
    for body, (mesh, sampled_tree) in enumerate(
        zip(meshes, sampled_surface_trees, strict=True)
    ):
        rotation = Rotation.from_quat(
            body_wxyz[:, body], scalar_first=True
        ).as_matrix()
        delta = points_world - body_positions[:, body, None, :]
        points_local = np.einsum("tni,tij->tnj", delta, rotation)
        flat = points_local.reshape(-1, 3)
        if sampled_tree is None:
            _, distances, _ = trimesh.proximity.closest_point(mesh, flat)
        else:
            distances, _ = sampled_tree.query(flat, workers=-1)
        output = np.minimum(output, distances.reshape(output.shape))
    return output


def summarize(reference: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    delta = target - reference
    contact = reference <= threshold
    result = {
        "all_point_count": int(reference.size),
        "mean_abs_error_mm": float(np.mean(np.abs(delta)) * 1000.0),
        "p95_abs_error_mm": float(np.percentile(np.abs(delta), 95) * 1000.0),
        "mean_bias_mm": float(np.mean(delta) * 1000.0),
        "reference_contact_point_count": int(contact.sum()),
    }
    if contact.any():
        result.update(
            contact_mean_abs_error_mm=float(np.mean(np.abs(delta[contact])) * 1000.0),
            contact_p95_abs_error_mm=float(np.percentile(np.abs(delta[contact]), 95) * 1000.0),
            contact_mean_bias_mm=float(np.mean(delta[contact]) * 1000.0),
            contact_recall=float(np.mean(target[contact] <= threshold)),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.1)
    parser.add_argument("--contact-threshold", type=float, default=0.01)
    args = parser.parse_args()

    row = load_row(args.partition)
    meshes = [trimesh.load(path, force="mesh", process=False) for path in row["object_mesh_paths"]]
    sampled_surface_trees: list[cKDTree | None] = []
    distance_methods = []
    rng_state = np.random.get_state()
    for body, mesh in enumerate(meshes):
        if len(mesh.faces) > 50_000:
            np.random.seed(body)
            samples, _ = trimesh.sample.sample_surface(mesh, 200_000)
            sampled_surface_trees.append(cKDTree(samples))
            distance_methods.append(
                {"body": body, "method": "sampled_surface_kdtree", "samples": 200_000}
            )
        else:
            sampled_surface_trees.append(None)
            distance_methods.append({"body": body, "method": "exact_triangle_surface"})
    np.random.set_state(rng_state)
    body_positions = np.asarray(row["object_body_position"], dtype=np.float64)
    body_wxyz = np.asarray(row["object_body_wxyz"], dtype=np.float64)
    sequence_parts = [part for part in args.partition.parts if part.startswith("sequence_id=")]
    sequence_id = sequence_parts[-1].split("=", 1)[1] if sequence_parts else "unknown"
    report = {
        "sequence_id": sequence_id,
        "scale": args.scale,
        "contact_threshold_mm": args.contact_threshold * 1000.0,
        "distance_methods": distance_methods,
        "sides": {},
    }

    mapping = {pattern.replace(".*", ""): source for pattern, (source, _, _) in SHARPA_TO_MANO_MAPPING.items()}
    for side in ("right", "left"):
        task_names = row[f"{side}_robot_frame_task_names"]
        frame_names = row[f"{side}_robot_frame_names"]
        frame_indices = [frame_names.index(name) for name in task_names]
        source_names = []
        for name in task_names:
            suffix = name.removeprefix(f"{side}_")
            source_names.append(mapping[f"_{suffix}"])
        source_indices = [MANO_JOINTS_ORDER.index(name) for name in source_names]

        mano = np.asarray(row[f"mano_{side}_joints"], dtype=np.float64)
        reconstruction_points = mano[:, source_indices]
        wrist = mano[:, :1]
        ik_target_points = wrist + args.scale * (reconstruction_points - wrist)
        robot_frames = np.asarray(row[f"robot_{side}_frames"], dtype=np.float64)
        robot_points = robot_frames[:, frame_indices, :3]

        reconstruction_distance = surface_distances(
            reconstruction_points, meshes, sampled_surface_trees, body_positions, body_wxyz
        )
        ik_target_distance = surface_distances(
            ik_target_points, meshes, sampled_surface_trees, body_positions, body_wxyz
        )
        robot_distance = surface_distances(
            robot_points, meshes, sampled_surface_trees, body_positions, body_wxyz
        )
        tip_mask = np.asarray(["tip_site" in name for name in task_names])

        source_tips_stored = np.asarray(row[f"mano_{side}_tips_distance"], dtype=np.float64)
        exact_tip_distance = reconstruction_distance[:, tip_mask]
        side_report = {
            "task_names": task_names,
            "reconstruction_vs_stored_sampled_tip_distance": summarize(
                source_tips_stored, exact_tip_distance, args.contact_threshold
            ),
            "all_mapped_sites": {
                "ik_target_vs_reconstruction": summarize(
                    reconstruction_distance, ik_target_distance, args.contact_threshold
                ),
                "robot_vs_reconstruction": summarize(
                    reconstruction_distance, robot_distance, args.contact_threshold
                ),
                "robot_vs_ik_target": summarize(
                    ik_target_distance, robot_distance, args.contact_threshold
                ),
            },
            "fingertips": {
                "ik_target_vs_reconstruction": summarize(
                    exact_tip_distance, ik_target_distance[:, tip_mask], args.contact_threshold
                ),
                "robot_vs_reconstruction": summarize(
                    exact_tip_distance, robot_distance[:, tip_mask], args.contact_threshold
                ),
                "robot_vs_ik_target": summarize(
                    ik_target_distance[:, tip_mask], robot_distance[:, tip_mask], args.contact_threshold
                ),
            },
        }
        report["sides"][side] = side_report

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
