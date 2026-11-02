"""Semantic MANO surface regions used by contact prediction.

MANO left and right meshes share vertex identities.  Their neutral templates
are mirrored only along the lateral axis, so the palmar surface is represented
by the same vertex IDs for both hands.  In the released MANO coordinate frame,
the palm and finger-pad surface points toward negative local Y.
"""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted outward vertex normals for a triangular mesh."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangle_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], triangle_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(lengths, 1.0e-12)).astype(np.float32)


def _canonical_faces(faces: np.ndarray) -> np.ndarray:
    """Return unoriented triangles in a row-order-independent representation."""

    canonical = np.sort(np.asarray(faces, dtype=np.int32), axis=1)
    order = np.lexsort((canonical[:, 2], canonical[:, 1], canonical[:, 0]))
    return canonical[order]


@dataclass(frozen=True)
class MANOPalmarRegion:
    """A topology-stable mask for the palm and finger-pad side of MANO."""

    mask: np.ndarray
    normal_y: np.ndarray
    maximum_normal_y: float

    @classmethod
    def from_model_root(
        cls,
        model_root: Path,
        *,
        maximum_normal_y: float = -0.10,
    ) -> "MANOPalmarRegion":
        """Build the shared left/right palmar mask from neutral MANO meshes.

        A small negative threshold excludes the lateral transition band as
        well as the dorsal surface.  This keeps the planar palm and volar
        (finger-pad) faces while avoiding unstable edge vertices.
        """

        model_root = Path(model_root).expanduser().resolve()
        models: dict[str, dict[str, object]] = {}
        for side in ("RIGHT", "LEFT"):
            path = model_root / f"MANO_{side}.pkl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing MANO model: {path}")
            with path.open("rb") as handle:
                models[side] = pickle.load(handle, encoding="latin1")

        right = models["RIGHT"]
        left = models["LEFT"]
        right_faces = np.asarray(right["f"], dtype=np.int32)
        left_faces = np.asarray(left["f"], dtype=np.int32)
        if not np.array_equal(_canonical_faces(right_faces), _canonical_faces(left_faces)):
            raise RuntimeError("MANO left/right vertex topology is incompatible")

        right_normals = _vertex_normals(
            np.asarray(right["v_template"], dtype=np.float32), right_faces
        )
        left_normals = _vertex_normals(
            np.asarray(left["v_template"], dtype=np.float32), left_faces
        )
        # Mirroring reverses face winding, but the released models preserve
        # the local Y direction and therefore the palm-side vertex IDs.
        right_mask = right_normals[:, 1] <= float(maximum_normal_y)
        left_mask = left_normals[:, 1] <= float(maximum_normal_y)
        if not np.array_equal(right_mask, left_mask):
            mismatch = int(np.count_nonzero(right_mask != left_mask))
            raise RuntimeError(
                f"MANO left/right palmar masks differ at {mismatch} vertices"
            )
        if right_mask.shape != (778,) or int(right_mask.sum()) < 100:
            raise RuntimeError("Unexpected MANO palmar-region topology")
        return cls(
            mask=right_mask,
            normal_y=right_normals[:, 1],
            maximum_normal_y=float(maximum_normal_y),
        )

