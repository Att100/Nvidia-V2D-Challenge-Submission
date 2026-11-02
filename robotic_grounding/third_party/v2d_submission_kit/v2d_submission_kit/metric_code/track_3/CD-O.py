"""Kaggle metric scoring code. Reference arrays and embedded model or vertex assets are omitted from this participant copy."""

import numpy as np
import pandas as pd


class ParticipantVisibleError(Exception):
    pass


                                                                               
                                                                    
def _mesh_faces(faces: np.ndarray, vertex_count: int) -> np.ndarray:

    rounded = np.rint(faces)
    if np.abs(faces - rounded).max() > 1e-6:
        raise ValueError("object_mesh_faces must hold integer vertex indices")
    rounded = rounded.astype(np.int64)
    if rounded.min() < 0 or rounded.max() >= vertex_count:
        raise ValueError(f"object_mesh_faces index outside [0, {vertex_count})")
    return rounded


def _sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray, count: int,
                         seed: int) -> np.ndarray:

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


                                                                               
                                                                    
                                       
_METRIC_SAMPLES = 10_000
_ALIGNMENT_SAMPLES = 3_000
_ICP_ITERATIONS = 40
_ALIGNMENT_STARTS = 25
_TRIM_FRACTION = 0.9


def _to_meshes(block, value_columns: list[str]) -> dict[int, tuple[np.ndarray, np.ndarray]]:

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

    return 1_000_003 * int(episode) + 101 * int(slot)



_METRIC = "cd_o_cm"
_METRIC_DISPLAY = "CD-O (cm)"
_KEY_COLUMNS = ['episode_index', 'object_slot', 'object_name', 'array_name', 'point_index']
_POINT_COLUMNS = ['x', 'y', 'z']
_REF_COLUMNS = ['ref_x', 'ref_y', 'ref_z']
_PROVENANCE_COLUMNS = ['code_commit_url']
_UNIFORM_PROVENANCE_COLUMNS = ['code_commit_url']
_PER_EPISODE_PROVENANCE_COLUMNS = []
_COMMIT_PATTERN = '^https://[\\w.-]*github[\\w.-]*/[\\w.-]+/[\\w.-]+/commit/[0-9a-fA-F]{7,40}/?$'

_REQUIRED_SUBMISSION_COLUMNS = ["row_id"] + _POINT_COLUMNS + _PROVENANCE_COLUMNS


def _check_provenance(merged):

    import re

    for column in _PROVENANCE_COLUMNS:
                                                                                         
                                                                                       
        if merged[column].isna().any():
            raise ParticipantVisibleError(f"every row must carry a non-empty '{column}'")
        values = merged[column].astype(str).str.strip()
        if (values == "").any() or values.isin({"nan", "None", "NaN"}).any():
            raise ParticipantVisibleError(f"every row must carry a non-empty '{column}'")

    for column in _UNIFORM_PROVENANCE_COLUMNS:
        unique = set(merged[column].astype(str).str.strip())
        if len(unique) != 1:
            raise ParticipantVisibleError(
                f"'{column}' must be identical on every row; found {len(unique)} distinct values. "
                "One submission is one evaluator build, run from one commit."
            )

                                                                                            
    for column in _PER_EPISODE_PROVENANCE_COLUMNS:
        counts = merged.groupby("episode_index")[column].apply(
            lambda block: block.astype(str).str.strip().nunique()
        )
        offenders = sorted(counts[counts > 1].index.tolist())
        if offenders:
            raise ParticipantVisibleError(
                f"'{column}' must be identical within each episode; episode(s) {offenders[:5]} "
                "carry more than one value."
            )

    commit = str(merged["code_commit_url"].iloc[0]).strip()
    if not re.match(_COMMIT_PATTERN, commit):
        raise ParticipantVisibleError(
            "'code_commit_url' must be a GitHub commit URL of the form "
            "https://github.com/<owner>/<repo>/commit/<sha>; a branch or tag moves and so pins "
            f"nothing. Got: {commit[:120]!r}"
        )


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:

    if row_id_column_name in solution.columns and "Usage" in solution.columns:
        solution = solution.drop(columns=["Usage"])

                                                                                                      
    for column in _KEY_COLUMNS + _REF_COLUMNS:
        if column not in solution.columns:
            raise ValueError(f"solution is missing '{column}'")

    missing = [c for c in _REQUIRED_SUBMISSION_COLUMNS if c not in submission.columns]
    if missing:
        raise ParticipantVisibleError(f"submission is missing column(s): {missing}")

    for frame, who in ((solution, "solution"), (submission, "submission")):
        frame[row_id_column_name] = frame[row_id_column_name].astype(str)

    if submission[row_id_column_name].duplicated().any():
        raise ParticipantVisibleError("submission contains duplicate row_id values")

    merged = solution.merge(
        submission, on=row_id_column_name, how="left", validate="one_to_one"
    )
    if merged[_POINT_COLUMNS].isna().any().any():
        n = int(merged[_POINT_COLUMNS].isna().any(axis=1).sum())
        raise ParticipantVisibleError(
            f"{n:,} required row(s) are missing from the submission, or carry a blank coordinate"
        )

    for column in _POINT_COLUMNS:
        values = pd.to_numeric(merged[column], errors="coerce")
        if values.isna().any():
            raise ParticipantVisibleError(f"'{column}' contains a non-numeric value")
        if not np.isfinite(values.to_numpy()).all():
            raise ParticipantVisibleError(f"'{column}' contains a non-finite value")
        merged[column] = values

    _check_provenance(merged)

    values = []
    for episode, block in merged.groupby("episode_index", sort=True):
        try:
            predicted = _to_meshes(block, _POINT_COLUMNS)
            reference = _to_meshes(block, _REF_COLUMNS)
        except ValueError as error:
            raise ParticipantVisibleError(f"episode {episode}: {error}") from None
        for slot in sorted(reference):
            name = block.loc[block["object_slot"] == slot, "object_name"].iloc[0]
            try:
                values.append(_object_chamfer_cm(reference[slot], predicted[slot],
                                                 _object_seed(episode, slot)))
            except ValueError as error:
                raise ParticipantVisibleError(f"episode {episode}, {name}: {error}") from None

    if not values:
        raise ValueError(f"no object produced a {_METRIC_DISPLAY} value")
    result = float(np.mean(values))
    if not np.isfinite(result):
        raise ValueError(f"computed {_METRIC_DISPLAY} is not finite: {result}")
    return result
