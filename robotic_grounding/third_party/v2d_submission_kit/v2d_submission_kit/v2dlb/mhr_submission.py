"""Track 1's submission schema: the long table that carries an MHR reconstruction.

Tracks 2 and 3 submit one pose per (episode, frame, object). Track 1 submits a reconstruction,
which is several arrays of very different shapes, so the row key gains an ``array_name`` axis:

    row_id, episode_index, frame_index, array_name, point_index, x, y, z

and each row carries three numbers. Submissions may be CSV or Parquet; Parquet is the documented
format, because a full Track 1 submission is ~0.7M rows.

THE HUMAN IS MHR PARAMETERS. Per frame, the 136 pose parameters (``mhr_model_params[:136]``: root
translation, root rotation, body pose); per episode, the 68 body scales (``mhr_model_params[136:]``)
and the 45 identity shape parameters. The pseudo-GT holds scales and shape constant within an
episode, so a submission does too. Numbers are packed three to a row in order, and the unused tail
of the last row must be 0. The host runs MHR itself (:mod:`v2dlb.mhr_body`) to get the joints,
the body surface, the alignment vertices and the hand points every metric reads.

THE OBJECT IS A MESH AND A SIM(3) POSE. Its shape is sent once per episode as a fixed-budget
triangle mesh (``object_mesh_vertices``, ``object_mesh_faces``), its size as one scale per episode
(``object_scale``, in ``x``; ``y`` and ``z`` must be 0), and its motion as a rotation and a
translation per frame:

    point_scene = s * R @ point_mesh + t

The host samples the surface points Bowen's metric expects from that mesh, with a seed fixed per
episode, and poses them. Both sides of every comparison go through the same code, so a submission
identical to the reference scores exactly zero, and Chamfer, acceleration and contact penetration
all score the SAME body and object.

The mesh budget is fixed because Kaggle compares row sets: every submission carries exactly the
reference's vertex and face rows. A smaller mesh is padded, vertices with any finite value and
faces with ``0, 0, 0``, which have zero area and so are never sampled.

WHICH FRAMES. Every valid frame inside the hand-object contact span is scored, grouped into
contiguous stretches separated by invalid frames. ``frame_index`` is the video frame number, so a
row says exactly which frame it describes. Acceleration is a second difference and is taken within
each stretch, never across a gap (:func:`_episode_metrics`).
"""

from __future__ import annotations

import numpy as np

from v2dlb.mesh_common import _mesh_faces, _sample_mesh_surface  # inlined ahead, like mhr_metrics
from v2dlb.mhr_metrics import (  # inlined ahead of this module's CORE in the Kaggle metric
    MHR_TABLE3_BODY_JOINT_INDICES,
    _select_body_joints,
    acceleration_error_per_frame,
    apply_similarity_transform,
    fit_similarity_transform,
    mhr_table3_metric_series,
    sdf_interpenetration_distance_per_frame,
)

#: The submitted arrays: (carried per "frame" or once per "episode", what point_index means).
#: All are numbered 0..P-1.
_ARRAYS = {
    "mhr_pose": ("frame", "ordinal"),              # 136 pose parameters, 3 per row: 46 rows
    "mhr_scales": ("episode", "ordinal"),          # 68 body scales: 23 rows
    "mhr_shape": ("episode", "ordinal"),           # 45 identity parameters: 15 rows
    "object_rotation": ("frame", "ordinal"),       # 3 rows: the rows of R
    "object_translation": ("frame", "ordinal"),    # 1 row
    "object_scale": ("episode", "ordinal"),        # 1 row: (s, 0, 0)
    "object_mesh_vertices": ("episode", "ordinal"),
    "object_mesh_faces": ("episode", "ordinal"),   # vertex indices, written as floats
}

#: Long-format columns. Kept identical in spirit to Tracks 2/3 so one reader serves both.
KEY_COLUMNS = ["episode_index", "frame_index", "array_name", "point_index"]
POINT_COLUMNS = ["x", "y", "z"]
REF_COLUMNS = ["ref_x", "ref_y", "ref_z"]


def row_id(track_number: int, episode: int, frame: int, array_index: int, point: int) -> str:
    """Stable, sortable row key. Mirrors the Tracks 2/3 scheme so tooling can stay shared."""
    return f"t{track_number}_{episode:06d}_{frame:06d}_{array_index:d}_{point:06d}"


def array_order() -> list[str]:
    """The canonical array order, which fixes ``array_index`` in a row_id."""
    return list(ARRAYS)


# --- CORE BEGIN (inlined into the generated Kaggle metric; edit here, never there) ---

#: frame_index carried by the arrays that exist once per episode rather than once per frame.
_EPISODE_FRAME = 999999

#: How many numbers each packed parameter array carries.
_PARAMETER_COUNTS = {"mhr_pose": 136, "mhr_scales": 68, "mhr_shape": 45}

#: How far from orthonormal a submitted rotation may be before it is refused rather than
#: projected. Loose enough for rounded CSV values, tight enough to catch a transposed or scaled R.
_ROTATION_TOLERANCE = 1e-3


def _to_arrays(block, value_columns: list[str]) -> dict[str, np.ndarray]:
    """Long rows for ONE episode into the submitted arrays.

    ``block`` must already be restricted to one episode. Per-frame arrays come back [T, P, 3],
    per-episode arrays [P, 3]. A missing or duplicated row is an error rather than a hole: the
    reshape is positional, so a gap would silently misalign every later point.
    """
    out: dict[str, np.ndarray] = {}
    per_frame = block[block["array_name"].map(lambda n: _ARRAYS.get(n, ("",))[0]) == "frame"]
    frames = np.unique(per_frame["frame_index"].to_numpy())
    for name, (scope, kind) in _ARRAYS.items():
        rows = block[block["array_name"] == name]
        if rows.empty:
            raise ValueError(f"no rows for array {name!r}")
        points = np.unique(rows["point_index"].to_numpy())
        # Ordinal arrays run 0..P-1: a per-episode array is one frame, so a dropped point would
        # still reshape cleanly and silently shift every later vertex or face.
        if kind == "ordinal" and not np.array_equal(points, np.arange(len(points))):
            raise ValueError(f"{name}: point_index must run 0..{len(points) - 1} without gaps")
        count = len(frames) if scope == "frame" else 1
        if len(rows) != count * len(points):
            raise ValueError(
                f"{name}: expected {count * len(points):,} rows ({count:,} frame(s) x "
                f"{len(points):,} points), got {len(rows):,}"
            )
        ordered = rows.sort_values(["frame_index", "point_index"])
        grid = ordered[value_columns].to_numpy(dtype=np.float64).reshape(count, len(points), 3)
        out[name] = grid if scope == "frame" else grid[0]
    return out


def _rotations(rows: np.ndarray) -> np.ndarray:
    """[T, 3, 3] submitted rotation rows, checked and projected onto SO(3)."""
    if rows.shape[1:] != (3, 3):
        raise ValueError(f"object_rotation must carry 3 rows per frame, got {rows.shape[1]}")
    error = np.abs(rows @ np.swapaxes(rows, 1, 2) - np.eye(3)).max()
    if not error <= _ROTATION_TOLERANCE:
        raise ValueError(
            f"object_rotation is not a rotation matrix (max |R R^T - I| = {error:.3g}); "
            "rows must be orthonormal"
        )
    if (np.linalg.det(rows) <= 0).any():
        raise ValueError("object_rotation contains a reflection (det <= 0)")
    u, _, vt = np.linalg.svd(rows)
    return u @ vt


def _unpack(rows: np.ndarray, count: int, name: str) -> np.ndarray:
    """Numbers packed three to a row, back to [..., count]; the unused tail must be exactly 0."""
    flat = rows.reshape(*rows.shape[:-2], -1)
    if flat.shape[-1] != 3 * -(-count // 3):
        raise ValueError(f"{name} must carry {count} numbers in {-(-count // 3)} rows")
    if np.any(flat[..., count:] != 0):
        raise ValueError(f"{name}: the unused tail of the last row must be 0 (it carries "
                         f"{count} numbers); a nonzero tail means the numbers are misaligned")
    return flat[..., :count]


def _object_scale(rows: np.ndarray) -> float:
    """The episode's object scale, from its single (s, 0, 0) row."""
    s, y, z = rows[0]
    if not (np.isfinite(s) and s > 0):
        raise ValueError(f"object_scale must be a positive number, got {s}")
    if y != 0 or z != 0:
        raise ValueError("object_scale carries the scale in x; y and z must be 0")
    return float(s)


def _to_frames(block, value_columns: list[str], seed: int, body):
    """Long rows for ONE episode into the five arrays ``mhr_table3_metrics`` expects.

    Returns ``(arrays, frames, scene)``. ``frames`` are the scored video frame numbers, in order.
    ``body`` is the MHR model (:class:`_MHRBody`) that turns the parameters into the 22 scored
    joints and the alignment, body-surface and hand vertices. ``scene`` is what contact penetration
    needs and Bowen's five do not: the 512 hand points (left then right) and the object mesh with
    its pose. The object's surface points are sampled from its mesh with as many points as the
    body carries, so the combined Chamfer weights both evenly.
    """
    raw = _to_arrays(block, value_columns)
    frames = np.unique(block.loc[block["array_name"].map(lambda n: _ARRAYS.get(n, ("",))[0])
                                 == "frame", "frame_index"])
    pose = _unpack(raw["mhr_pose"], _PARAMETER_COUNTS["mhr_pose"], "mhr_pose")
    scales = _unpack(raw["mhr_scales"], _PARAMETER_COUNTS["mhr_scales"], "mhr_scales")
    shape = _unpack(raw["mhr_shape"], _PARAMETER_COUNTS["mhr_shape"], "mhr_shape")
    params = np.concatenate([pose, np.broadcast_to(scales, (len(pose), len(scales)))], axis=1)
    vertices, joints = body(params, shape)
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise ValueError("the MHR parameters produce a non-finite body")

    mesh_vertices = raw["object_mesh_vertices"]
    faces = _mesh_faces(raw["object_mesh_faces"], len(mesh_vertices))
    rotation = _rotations(raw["object_rotation"])
    translation = raw["object_translation"][:, 0, :]
    scale = _object_scale(raw["object_scale"])
    local = _sample_mesh_surface(mesh_vertices, faces, len(body.roles["body"]), seed)
    arrays = {
        "mhr_vertices": vertices[:, body.roles["alignment"]],
        "mhr_joints": joints[:, list(MHR_TABLE3_BODY_JOINT_INDICES)],
        "human_surface_points": vertices[:, body.roles["body"]],
        "object_surface_points": scale * np.einsum("tij,pj->tpi", rotation, local)
        + translation[:, None],
        "object_translation": translation,
    }
    scene = {
        "hands": vertices[:, np.concatenate([body.roles["left_hand"], body.roles["right_hand"]])],
        "mesh_vertices": mesh_vertices,
        "mesh_faces": faces,
        "rotation": rotation,
        "translation": translation,
        "scale": scale,
    }
    return arrays, frames, scene


def _stretches(frames: np.ndarray) -> list[slice]:
    """Maximal runs of consecutive frame numbers, as slices into the frame axis."""
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    edges = [0, *breaks.tolist(), len(frames)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def _episode_metrics(pred, target, frames, body_joint_indices) -> dict[str, float]:
    """Bowen's five Table 3 metrics for one episode whose scored frames may have gaps.

    Chamfer is per frame, so it is his series unchanged. Acceleration is a second difference, and
    across a gap it would difference two frames that are not neighbours, so it is his own
    ``acceleration_error_per_frame`` applied to each contiguous stretch separately. The alignment
    stays ONE similarity transform per episode, fitted at the first scored frame exactly as he
    fits it. With a single stretch this is identical to ``mhr_table3_metrics``.
    """
    series = mhr_table3_metric_series(pred, target, body_joint_indices=body_joint_indices)
    transform = fit_similarity_transform(pred["mhr_vertices"][0], target["mhr_vertices"][0])
    pred_joints = _select_body_joints(apply_similarity_transform(pred["mhr_joints"], transform),
                                      body_joint_indices)
    target_joints = _select_body_joints(target["mhr_joints"], body_joint_indices)
    pred_translation = apply_similarity_transform(pred["object_translation"], transform)
    acc_h, acc_o = [], []
    for part in _stretches(frames):
        if part.stop - part.start < 3:
            raise ValueError(f"a scored stretch of {part.stop - part.start} frames is too short "
                             "for acceleration")
        acc_h.append(acceleration_error_per_frame(pred_joints[part], target_joints[part]))
        acc_o.append(acceleration_error_per_frame(pred_translation[part],
                                                  target["object_translation"][part]))
    series["acc_h_cm"] = np.concatenate(acc_h) * 100.0
    series["acc_o_cm"] = np.concatenate(acc_o) * 100.0
    return {key: float(np.mean(value)) for key, value in series.items()}


def _winding_numbers(points, v0, v1, v2, chunk: int = 500_000) -> np.ndarray:
    """Generalized winding number of each point w.r.t. a triangle soup (Jacobson et al. 2013).

    About 1 inside a closed surface and 0 outside, and it degrades gracefully across holes, which
    a participant's mesh may have. Zero-area padding faces contribute exactly 0.
    """
    out = np.zeros(len(points))
    per = max(1, chunk // max(1, len(v0)))
    for s in range(0, len(points), per):
        q = points[s:s + per, None, :]
        a, b, c = v0[None] - q, v1[None] - q, v2[None] - q
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        num = np.einsum("qfi,qfi->qf", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("qfi,qfi->qf", a, b) * lc
               + np.einsum("qfi,qfi->qf", b, c) * la + np.einsum("qfi,qfi->qf", c, a) * lb)
        out[s:s + per] = np.arctan2(num, den).sum(axis=1) / (2.0 * np.pi)
    return out


def _point_triangle_distance(points, v0, v1, v2, chunk: int = 500_000) -> np.ndarray:
    """Exact distance from each point to the nearest triangle (Ericson, Real-Time Collision
    Detection, 5.1.5), vectorised over points and faces."""
    out = np.empty(len(points))
    per = max(1, chunk // max(1, len(v0)))

    def dot(x, y):
        return np.einsum("...i,...i->...", x, y)

    for s in range(0, len(points), per):
        p = points[s:s + per, None, :]
        a, b, c = v0[None], v1[None], v2[None]
        ab, ac = b - a, c - a
        ap, bp, cp = p - a, p - b, p - c
        d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
        va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = va + vb + vc
            v = np.where(denom != 0, vb / denom, 0.0)
            w = np.where(denom != 0, vc / denom, 0.0)
            closest = a + ab * v[..., None] + ac * w[..., None]
            t = np.nan_to_num((d4 - d3) / ((d4 - d3) + (d5 - d6)))
            m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
            closest = np.where(m[..., None], b + (c - b) * t[..., None], closest)
            t = np.nan_to_num(d2 / (d2 - d6))
            m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
            closest = np.where(m[..., None], a + ac * t[..., None], closest)
            m = (d6 >= 0) & (d5 <= d6)
            closest = np.where(m[..., None], np.broadcast_to(c, closest.shape), closest)
            t = np.nan_to_num(d1 / (d1 - d3))
            m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
            closest = np.where(m[..., None], a + ab * t[..., None], closest)
            m = (d3 >= 0) & (d4 <= d3)
            closest = np.where(m[..., None], np.broadcast_to(b, closest.shape), closest)
            m = (d1 <= 0) & (d2 <= 0)
            closest = np.where(m[..., None], np.broadcast_to(a, closest.shape), closest)
        out[s:s + per] = np.linalg.norm(p - closest, axis=-1).min(axis=1)
    return out


def _compile_penetration_kernels():
    """Compiled inside test and distance, when numba is importable; otherwise None.

    Same maths as :func:`_winding_numbers` and :func:`_point_triangle_distance`, one point per
    parallel iteration with no temporaries, which is what makes the worst case affordable: the
    cost is (hand points inside the object's box) x faces, and a submission can put far more hand
    points there than the reference does. numba is in Kaggle's kernel image; if it is missing the
    numpy path gives the same answer, slower.
    """
    try:
        import numba
    except Exception:  # noqa: BLE001 - any import failure means "use numpy"
        return None

    @numba.njit(parallel=True, cache=False)
    def winding(points, v0, v1, v2):
        out = np.zeros(points.shape[0])
        for i in numba.prange(points.shape[0]):
            px, py, pz = points[i, 0], points[i, 1], points[i, 2]
            total = 0.0
            for f in range(v0.shape[0]):
                ax, ay, az = v0[f, 0] - px, v0[f, 1] - py, v0[f, 2] - pz
                bx, by, bz = v1[f, 0] - px, v1[f, 1] - py, v1[f, 2] - pz
                cx, cy, cz = v2[f, 0] - px, v2[f, 1] - py, v2[f, 2] - pz
                la = np.sqrt(ax * ax + ay * ay + az * az)
                lb = np.sqrt(bx * bx + by * by + bz * bz)
                lc = np.sqrt(cx * cx + cy * cy + cz * cz)
                num = ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)
                den = (la * lb * lc + (ax * bx + ay * by + az * bz) * lc
                       + (bx * cx + by * cy + bz * cz) * la + (cx * ax + cy * ay + cz * az) * lb)
                total += np.arctan2(num, den)
            out[i] = total / (2.0 * np.pi)
        return out

    @numba.njit(cache=False)
    def closest_sq(px, py, pz, a0, a1, a2, b0, b1, b2, c0, c1, c2):
        ab0, ab1, ab2 = b0 - a0, b1 - a1, b2 - a2
        ac0, ac1, ac2 = c0 - a0, c1 - a1, c2 - a2
        ap0, ap1, ap2 = px - a0, py - a1, pz - a2
        d1 = ab0 * ap0 + ab1 * ap1 + ab2 * ap2
        d2 = ac0 * ap0 + ac1 * ap1 + ac2 * ap2
        if d1 <= 0.0 and d2 <= 0.0:
            q0, q1, q2 = a0, a1, a2
        else:
            bp0, bp1, bp2 = px - b0, py - b1, pz - b2
            d3 = ab0 * bp0 + ab1 * bp1 + ab2 * bp2
            d4 = ac0 * bp0 + ac1 * bp1 + ac2 * bp2
            cp0, cp1, cp2 = px - c0, py - c1, pz - c2
            d5 = ab0 * cp0 + ab1 * cp1 + ab2 * cp2
            d6 = ac0 * cp0 + ac1 * cp1 + ac2 * cp2
            vc = d1 * d4 - d3 * d2
            vb = d5 * d2 - d1 * d6
            va = d3 * d6 - d5 * d4
            if d3 >= 0.0 and d4 <= d3:
                q0, q1, q2 = b0, b1, b2
            elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                denom = d1 - d3
                t = d1 / denom if denom != 0.0 else 0.0
                q0, q1, q2 = a0 + t * ab0, a1 + t * ab1, a2 + t * ab2
            elif d6 >= 0.0 and d5 <= d6:
                q0, q1, q2 = c0, c1, c2
            elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                denom = d2 - d6
                t = d2 / denom if denom != 0.0 else 0.0
                q0, q1, q2 = a0 + t * ac0, a1 + t * ac1, a2 + t * ac2
            elif va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                denom = (d4 - d3) + (d5 - d6)
                t = (d4 - d3) / denom if denom != 0.0 else 0.0
                q0, q1, q2 = b0 + t * (c0 - b0), b1 + t * (c1 - b1), b2 + t * (c2 - b2)
            else:
                denom = va + vb + vc
                v = vb / denom if denom != 0.0 else 0.0
                w = vc / denom if denom != 0.0 else 0.0
                q0, q1, q2 = a0 + ab0 * v + ac0 * w, a1 + ab1 * v + ac1 * w, a2 + ab2 * v + ac2 * w
        return (px - q0) ** 2 + (py - q1) ** 2 + (pz - q2) ** 2

    @numba.njit(parallel=True, cache=False)
    def distance(points, v0, v1, v2):
        out = np.zeros(points.shape[0])
        for i in numba.prange(points.shape[0]):
            best = np.inf
            for f in range(v0.shape[0]):
                d = closest_sq(points[i, 0], points[i, 1], points[i, 2],
                               v0[f, 0], v0[f, 1], v0[f, 2], v1[f, 0], v1[f, 1], v1[f, 2],
                               v2[f, 0], v2[f, 1], v2[f, 2])
                if d < best:
                    best = d
            out[i] = np.sqrt(best)
        return out

    return winding, distance


_PENETRATION_KERNELS = _compile_penetration_kernels()


def _penetration_depth(points, vertices, faces) -> np.ndarray:
    """Depth (>= 0) of each mesh-frame point inside the mesh; 0 outside.

    Inside means |winding number| > 0.5, which does not depend on face orientation. A point
    outside the mesh's bounding box is outside its convex hull, where |winding number| <= 0.5,
    so it is skipped without being tested; that is what keeps this affordable.
    """
    depth = np.zeros(len(points))
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    near = np.flatnonzero(((points >= lo) & (points <= hi)).all(axis=1))
    if near.size == 0:
        return depth
    v0, v1, v2 = (np.ascontiguousarray(vertices[faces[:, k]], dtype=np.float64) for k in range(3))
    # Zero-area faces do not contribute to the mesh surface.
    area = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    surface = area > 0.0
    if not surface.any():
        return depth
    v0, v1, v2 = (np.ascontiguousarray(v[surface]) for v in (v0, v1, v2))
    if _PENETRATION_KERNELS is not None:
        winding, distance = _PENETRATION_KERNELS
    else:
        winding = lambda q, *f: _in_parallel(_winding_numbers, q, *f)          # noqa: E731
        distance = lambda q, *f: _in_parallel(_point_triangle_distance, q, *f)  # noqa: E731
    query = np.ascontiguousarray(points[near], dtype=np.float64)
    inside_local = np.flatnonzero(np.abs(winding(query, v0, v1, v2)) > 0.5)
    if inside_local.size:
        depth[near[inside_local]] = distance(query[inside_local], v0, v1, v2)
    return depth


def _in_parallel(function, points, *faces):
    """``function(points, *faces)``, split by points across threads.

    Every point is independent and numpy releases the GIL inside the heavy kernels, so this is
    identical to the serial call, only faster: 4.6x on 4 threads, measured on this workload.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, min(os.cpu_count() or 1, 8))
    if workers == 1 or len(points) < 256:
        return function(points, *faces)
    parts = np.array_split(np.arange(len(points)), workers * 4)
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda index: function(points[index], *faces), parts))
    return np.concatenate(results)


def _episode_penetration(pred, pred_scene, target) -> float:
    """Contact penetration: the participant's hand against the participant's OWN object mesh.

    Hand points are moved into the object's mesh frame by the inverse of its Sim(3) pose, their
    depth inside the mesh is measured, and Bowen's ``sdf_interpenetration_distance_per_frame``
    turns that into centimetres (mean over all 512 points, outside points counting zero). Depth is
    measured in mesh units, so it is multiplied back by the object scale, and then by the scale of
    the frame-0 similarity alignment the other metrics use, which puts it in the reference's
    metric units. The reference enters through that scale only.
    """
    align = float(fit_similarity_transform(pred["mhr_vertices"][0], target["mhr_vertices"][0])[0])
    hands, scale = pred_scene["hands"], pred_scene["scale"]
    local = np.einsum("tpi,tij->tpj", hands - pred_scene["translation"][:, None],
                      pred_scene["rotation"]) / scale
    depth = _penetration_depth(local.reshape(-1, 3), pred_scene["mesh_vertices"],
                               pred_scene["mesh_faces"]).reshape(hands.shape[:2])
    return float(np.mean(sdf_interpenetration_distance_per_frame(-depth * scale * align)))


# --- CORE END -----------------------------------------------------------------------


#: Public aliases. The generated Kaggle metric calls the private names; local tooling these.
to_arrays = _to_arrays
to_frames = _to_frames
episode_metrics = _episode_metrics
episode_penetration = _episode_penetration
penetration_depth = _penetration_depth
stretches = _stretches
sample_mesh_surface = _sample_mesh_surface
EPISODE_FRAME = _EPISODE_FRAME
PARAMETER_COUNTS = _PARAMETER_COUNTS
ARRAYS = _ARRAYS
