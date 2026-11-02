"""Track 3 CD-O: is each object reconstructed with the right shape?

Track 3's evaluation videos ship without object meshes, so a participant reconstructs every object
they manipulate. This scores that mesh against the scanned reference with the V2D team's own
object-reconstruction measure, ``reconstruction/modules/v2d_hoi_object_reconstruction/tools/
evaluate_mesh_quality.py``, in its ``shape_scale_normalized`` form, transcribed without trimesh so
it runs in a Kaggle metric. Registration is Sim(3): pose AND size are removed, so CD-O scores shape.

  1. scale: rescale the candidate about its surface centroid by the ratio of the two meshes'
     exact area-weighted RMS surface radii (an analytic integral over each triangle, so it does not
     depend on sampling or on how finely the surface is triangulated),
  2. sample both surfaces uniformly (10,000 points to score, 3,000 to align),
  3. rotation and translation: 25 starts from the proper PCA axis mappings, each refined by 40
     iterations of ICP that keeps the closest 90% of pairs,
  4. report the symmetric Chamfer distance, the mean of the two directed mean nearest-neighbour
     distances, in centimetres at the REFERENCE's scale.

Both sides are sampled with the same seed, so a submission equal to the reference, or equal to it
under any similarity transform, scores 0.

SUBMISSION. One fixed-budget triangle mesh per (episode, object), in long rows:

    row_id, episode_index, object_slot, object_name, array_name, point_index, x, y, z

``row_id`` names the object; ``object_slot`` is its position in the episode's reference data.
``array_name`` is ``object_mesh_vertices`` or ``object_mesh_faces`` (vertex indices as floats); a
smaller mesh is padded, vertices with any finite value and faces with ``0, 0, 0``, which have zero
area and are never sampled.
"""

from __future__ import annotations

import numpy as np

from v2dlb.mesh_common import _mesh_faces, _sample_mesh_surface  # inlined ahead of this CORE

#: The submitted arrays, both carried once per (episode, object) and numbered 0..P-1.
_ARRAYS = ("object_mesh_vertices", "object_mesh_faces")
KEY_COLUMNS = ["episode_index", "object_slot", "object_name", "array_name", "point_index"]
POINT_COLUMNS = ["x", "y", "z"]
REF_COLUMNS = ["ref_x", "ref_y", "ref_z"]


def row_id(track_number: int, episode: int, object_name: str, array_index: int, point: int) -> str:
    """Stable row key for a mesh row. It names the object rather than its slot.

    Track 3's episodes do not all list their objects in one order: in four of them the per-frame
    ``observation.objects`` is not sorted by name although the schema metadata is. Naming the
    object in the key makes a row say which object it describes, so a participant's mesh cannot
    land in the wrong slot whatever order they believe in.
    """
    return f"t{track_number}m_{episode:06d}_{object_name}_{array_index:d}_{point:06d}"


#: Parses a row id back into (episode, object_name, array_index, point).
ROW_ID_PATTERN = r"^t\d+m_(\d{6})_(.+)_(\d)_(\d{6})$"


# --- CORE BEGIN (inlined into the generated Kaggle metric; edit here, never there) ---

#: evaluate_mesh_quality.py's defaults.
_METRIC_SAMPLES = 10_000
_ALIGNMENT_SAMPLES = 3_000
_ICP_ITERATIONS = 40
_ALIGNMENT_STARTS = 25
_TRIM_FRACTION = 0.9


def _to_meshes(block, value_columns: list[str]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Long rows for ONE episode into {object_slot: (vertices [V, 3], faces [F, 3] int)}."""
    meshes = {}
    for slot, rows in block.groupby("object_slot", sort=True):
        parts = {}
        for name in ("object_mesh_vertices", "object_mesh_faces"):
            part = rows[rows["array_name"] == name].sort_values("point_index")
            if part.empty:
                raise ValueError(f"object slot {slot}: no rows for {name!r}")
            if not np.array_equal(part["point_index"].to_numpy(), np.arange(len(part))):
                raise ValueError(f"object slot {slot}: {name} point_index must run 0..{len(part) - 1}")
            parts[name] = part[value_columns].to_numpy(dtype=np.float64)
        vertices = parts["object_mesh_vertices"]
        meshes[int(slot)] = (vertices, _mesh_faces(parts["object_mesh_faces"], len(vertices)))
    return meshes


def _pca_basis(points):
    centered = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered / max(len(centered), 1))
    basis = vectors[:, ::-1]
    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0
    return basis


def _initial_rotations(reference, candidate, maximum):
    """Identity, then every proper mapping between the two PCA frames, up to ``maximum``."""
    import itertools

    ref_basis, cand_basis = _pca_basis(reference), _pca_basis(candidate)
    rotations = [np.eye(3)]
    seen = {tuple(np.round(rotations[0], 10).ravel())}
    for permutation in itertools.permutations(range(3)):
        permuted = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mapping = permuted @ np.diag(signs)
            if np.linalg.det(mapping) <= 0.0:
                continue
            rotation = ref_basis @ mapping @ cand_basis.T
            if np.linalg.det(rotation) < 0.0:
                continue
            key = tuple(np.round(rotation, 10).ravel())
            if key in seen:
                continue
            seen.add(key)
            rotations.append(rotation)
            if len(rotations) >= maximum:
                return rotations
    return rotations


def _kabsch(source, target):
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    left, _, right_t = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    return rotation, target_center - rotation @ source_center


def _symmetric_mean_distance(reference, candidate):
    from scipy.spatial import cKDTree

    to_reference = cKDTree(reference).query(candidate, k=1)[0]
    to_candidate = cKDTree(candidate).query(reference, k=1)[0]
    return 0.5 * (float(np.mean(to_reference)) + float(np.mean(to_candidate)))


def _align_rigid(reference, candidate):
    """Best rotation and translation taking ``candidate`` onto ``reference``; no scale."""
    import math

    from scipy.spatial import cKDTree

    tree = cKDTree(reference)
    best = None
    for start in _initial_rotations(reference, candidate, _ALIGNMENT_STARTS):
        rotation = start.copy()
        translation = reference.mean(axis=0) - rotation @ candidate.mean(axis=0)
        previous = math.inf
        for _ in range(_ICP_ITERATIONS):
            moved = candidate @ rotation.T + translation
            distances, indices = tree.query(moved, k=1)
            keep = max(3, int(math.ceil(len(distances) * _TRIM_FRACTION)))
            chosen = (np.argpartition(distances, keep - 1)[:keep] if keep < len(distances)
                      else np.arange(len(distances)))
            step_rotation, step_translation = _kabsch(moved[chosen], reference[indices[chosen]])
            rotation = step_rotation @ rotation
            translation = step_rotation @ translation + step_translation
            error = float(np.mean(distances[chosen]))
            if math.isfinite(previous) and abs(previous - error) <= 1e-8 * max(previous, 1.0):
                break
            previous = error
        score = _symmetric_mean_distance(reference, candidate @ rotation.T + translation)
        if best is None or score < best[2]:
            best = (rotation, translation, score)
    return best[0], best[1]


def _surface_moments(vertices, faces):
    """Exact area-weighted surface centroid and RMS radius of a triangle mesh.

    evaluate_mesh_quality.py's ``area_weighted_surface_moments``: integrated analytically per
    triangle, so it depends on the surface, not on sampling or triangle density. Zero-area
    (padding) triangles carry no weight.
    """
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1)
    keep = np.isfinite(areas) & (areas > 1e-12)
    if not keep.any():
        raise ValueError("object mesh has no surface area")
    triangles, areas = triangles[keep], areas[keep]
    total = float(areas.sum())
    centroid = (triangles.mean(axis=1) * areas[:, None]).sum(axis=0) / total
    centered = triangles - centroid
    squared = np.einsum("tij,tij->ti", centered, centered).sum(axis=1)
    cross = (np.einsum("ti,ti->t", centered[:, 0], centered[:, 1])
             + np.einsum("ti,ti->t", centered[:, 0], centered[:, 2])
             + np.einsum("ti,ti->t", centered[:, 1], centered[:, 2]))
    radius = float(np.sqrt(max(float((areas * (squared + cross) / 6.0).sum() / total), 0.0)))
    if not np.isfinite(radius) or radius <= 1e-12:
        raise ValueError("object mesh has zero or invalid extent")
    return centroid, radius


def _object_chamfer_cm(reference_mesh, candidate_mesh, seed: int) -> float:
    """Symmetric Chamfer distance in cm after Sim(3) registration (shape only)."""
    reference_v, reference_f = reference_mesh
    candidate_v, candidate_f = candidate_mesh
    _, reference_radius = _surface_moments(reference_v, reference_f)
    candidate_center, candidate_radius = _surface_moments(candidate_v, candidate_f)
    ratio = candidate_radius / reference_radius
    candidate_v = (candidate_v - candidate_center) / ratio + candidate_center
    reference_metric = _sample_mesh_surface(reference_v, reference_f, _METRIC_SAMPLES, seed)
    candidate_metric = _sample_mesh_surface(candidate_v, candidate_f, _METRIC_SAMPLES, seed)
    reference_align = _sample_mesh_surface(reference_v, reference_f, _ALIGNMENT_SAMPLES, seed + 1)
    candidate_align = _sample_mesh_surface(candidate_v, candidate_f, _ALIGNMENT_SAMPLES, seed + 1)
    rotation, translation = _align_rigid(reference_align, candidate_align)
    return 100.0 * _symmetric_mean_distance(reference_metric,
                                            candidate_metric @ rotation.T + translation)


def _object_seed(episode: int, slot: int) -> int:
    """The sampling seed for one (episode, object); identical on both sides."""
    return 1_000_003 * int(episode) + 101 * int(slot)


# --- CORE END ---------------------------------------------------------------------


#: Public aliases for local tooling.
ARRAYS = _ARRAYS
to_meshes = _to_meshes
object_chamfer_cm = _object_chamfer_cm
object_seed = _object_seed
align_rigid = _align_rigid
surface_moments = _surface_moments
