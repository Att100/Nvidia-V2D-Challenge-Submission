"""Reduce triangle meshes to fixed vertex and face budgets, then add zero-area padding."""

from __future__ import annotations

import numpy as np


def budget_mesh(path, faces: int, vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """Weld vertices, simplify to both limits, and pad to the exact row counts."""
    import fast_simplification
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    welded = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
    v = np.asarray(welded.vertices, dtype=np.float64)
    f = np.asarray(welded.faces, dtype=np.int64)
    while len(f) > faces or len(v) > vertices:
        target = faces if len(v) <= vertices else min(faces, int(len(f) * vertices / len(v) * 0.95))
        if target < 1:
            raise ValueError(f"{path}: cannot fit a surface within the mesh budget")
        old_size = (len(v), len(f))
        v, f = fast_simplification.simplify(v, f, target_count=target)
        reduced = trimesh.Trimesh(v, f, process=False)
        reduced.remove_unreferenced_vertices()
        v = np.asarray(reduced.vertices, dtype=np.float64)
        f = np.asarray(reduced.faces, dtype=np.int64)
        if not len(f) or (len(v), len(f)) == old_size:
            raise ValueError(f"{path}: mesh simplification cannot satisfy the budget")
    # Padding: a repeated vertex, and degenerate faces on it, which have zero area.
    v = np.concatenate([v, np.repeat(v[:1], vertices - len(v), axis=0)])
    f = np.concatenate([f, np.zeros((faces - len(f), 3), dtype=np.int64)])
    return v, f
