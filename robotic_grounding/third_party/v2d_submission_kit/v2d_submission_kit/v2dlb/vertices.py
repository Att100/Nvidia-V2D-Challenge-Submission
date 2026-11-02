"""Build the canonical per-object vertex sample that the ADD-AUC metric transforms.

WHY THIS FILE IS THE CANONICAL SOURCE
-------------------------------------
ADD is the mean distance between a fixed set of object-frame points transformed by the achieved
pose and by the reference pose. The value therefore depends on WHICH points, so the host and the
participant must use the same ones or their two computations of AUC will disagree in the fourth
decimal and produce "my local score does not match the leaderboard" support tickets.

The evaluator's own sampler (`flash_chord/evaluation/rollout.py::sample_object_vertices`) draws
from the *simulation model's* visible mesh shapes, which are built by the asset pipeline and are
not reproducible from the released `.glb` files alone. So this module defines the canonical set
from the released meshes instead, using the evaluator's sampling convention, and the intent is
that the evaluator loads this set rather than re-sampling. Until that lands, treat any small
AUC discrepancy as this mismatch and not as a scoring bug.

Sampling convention, matched to `sample_object_vertices`:
    generator = np.random.default_rng(seed)                      # seed 0, not 42
    chosen = generator.choice(n_vertices, size=count, replace=n_vertices < count)
    sampled = vertices[chosen]
`ADD_VERTEX_COUNT = 500`, `ADD_VERTEX_SEED = 0` in `flash_chord/evaluation/metrics.py`.

Where an object ships both a visual and a collision mesh, the VISUAL one is used: the evaluator
filters sim shapes on the "visible" flag.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

#: Matches flash_chord.evaluation.metrics.ADD_VERTEX_COUNT / ADD_VERTEX_SEED.
ADD_VERTEX_COUNT = 500
ADD_VERTEX_SEED = 0


def mesh_path_for(mesh_root: Path, name: str) -> Path:
    """The .glb to sample for an object, preferring the visual variant."""
    directory = Path(mesh_root) / name
    for candidate in (f"{name}_visual.glb", f"{name}.glb"):
        path = directory / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"no mesh for object {name!r} under {directory}")


def sample_vertices(mesh_file: Path, count: int = ADD_VERTEX_COUNT, seed: int = ADD_VERTEX_SEED) -> np.ndarray:
    """(count, 3) float64 object-frame points, sampled the way the evaluator samples."""
    import trimesh  # offline only; the generated Kaggle metric never imports this

    mesh = trimesh.load(str(mesh_file), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        raise ValueError(f"{mesh_file}: mesh has no vertices")
    generator = np.random.default_rng(seed)
    chosen = generator.choice(vertices.shape[0], size=count, replace=vertices.shape[0] < count)
    return vertices[chosen]


def build_blob(mesh_root: Path, names: list[str]) -> tuple[str, dict[str, np.ndarray]]:
    """base64(npz) of {object_name: (count, 3) float32} for embedding in a metric notebook."""
    sampled = {name: sample_vertices(mesh_path_for(mesh_root, name)).astype(np.float32) for name in sorted(names)}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **sampled)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), sampled


def wrap_b64(blob: str, width: int = 116) -> str:
    """Fixed-width lines so the generated file stays diffable.

    base64.b64decode ignores embedded whitespace, so splitting anywhere is safe.
    """
    return "\n".join(blob[i : i + width] for i in range(0, len(blob), width))
