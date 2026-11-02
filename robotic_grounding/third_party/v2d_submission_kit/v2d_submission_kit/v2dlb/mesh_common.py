"""Triangle-mesh helpers shared by every metric that scores a submitted object mesh.

Track 1 (the object in a reconstruction) and Track 3's CD-O (the reconstructed object alone)
both take an object as a fixed-budget mesh in long rows. Sampling its surface the same way in
both keeps a mesh scored identically wherever it is submitted.
"""

from __future__ import annotations

import numpy as np

# --- CORE BEGIN (inlined into the generated Kaggle metric; edit here, never there) ---


def _mesh_faces(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    """Face rows as integer vertex indices, validated."""
    rounded = np.rint(faces)
    if np.abs(faces - rounded).max() > 1e-6:
        raise ValueError("object_mesh_faces must hold integer vertex indices")
    rounded = rounded.astype(np.int64)
    if rounded.min() < 0 or rounded.max() >= vertex_count:
        raise ValueError(f"object_mesh_faces index outside [0, {vertex_count})")
    return rounded


def _sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, count: int,
                         seed: int) -> np.ndarray:
    """``count`` points uniform over the mesh surface, deterministic for a given seed.

    Area-weighted face choice and the square-root barycentric draw, so zero-area padding faces
    are never chosen. numpy's PCG64 stream is stable across versions, which makes the sample, and
    so the score, reproducible on any host.
    """
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("object mesh has no surface area")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(faces), size=count, p=area / total)
    r1, r2 = rng.random(count), rng.random(count)
    s = np.sqrt(r1)[:, None]
    return (1 - s) * a[chosen] + s * (1 - r2[:, None]) * b[chosen] + s * r2[:, None] * c[chosen]


# --- CORE END ---------------------------------------------------------------------


#: Public aliases for local tooling.
mesh_faces = _mesh_faces
sample_mesh_surface = _sample_mesh_surface
