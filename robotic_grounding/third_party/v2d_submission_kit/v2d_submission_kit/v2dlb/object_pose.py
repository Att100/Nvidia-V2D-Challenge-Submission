"""The object-pose metric math, transcribed from flash_chord/evaluation/metrics.py.

This module is the single source of truth for the maths. ``metrics/build_kaggle_metrics.py``
inlines the section between the CORE markers verbatim into each generated Kaggle metric file, so
the number a participant computes locally with ``tools/pack_submission.py`` and the number the
host computes on Kaggle come from the same lines of code rather than from two transcriptions
that can drift apart.

Conventions, all matching the evaluator and NOT the released HuggingFace parquets:
  - quaternions are xyzw,
  - positions are metres,
  - poses are (T, W, B, 7) achieved against (T, 1, B, 7) reference, broadcasting over worlds.

``tests/test_metric_equivalence.py`` asserts these functions agree with the real
flash_chord functions on random data, to float tolerance.
"""

from __future__ import annotations

import numpy as np

# --- CORE BEGIN (inlined into every generated Kaggle metric; edit here, never there) ---

# Matched to flash_chord/evaluation/metrics.py.
_ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)
_SPIDER_POSITION_THRESHOLD_M = 0.1
_SPIDER_ORIENTATION_THRESHOLD_RAD = 0.5
_MANIPTRANS_POSITION_THRESHOLD_M = 0.03
_MANIPTRANS_ORIENTATION_THRESHOLD_DEG = 30.0
#: Radius of the six object-frame keypoints MPPE is measured on, in metres.
#: 5 cm, confirmed by Shalin on 2026-09-18 as the authoritative value. It is NOT stated in the
#: CHORD paper, and flash_chord/objectives/keypoints.py uses UNIT vectors for the shaped training
#: reward, where only the relative gradient matters - so re-deriving MPPE from that kernel lands
#: 20x high. The radius is a lever arm on the orientation term, so it sets the scale of the
#: whole number; do not change it without a new decision from the metric owner.
_MPPE_KEYPOINT_RADIUS_M = 0.05

#: The six principal axis directions, in the order flash_chord/objectives/keypoints.py lists
#: them. Order does not affect the mean, but keeping it makes the two readable side by side.
_MPPE_KEYPOINT_DIRECTIONS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)

#: Submitted pose columns, in the evaluator's xyzw order.
_POSE_COLUMNS = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
#: The same seven on the solution side, carrying the reference trajectory.
_REF_COLUMNS = ["ref_" + _c for _c in _POSE_COLUMNS]


# --- Math, transcribed from flash_chord/evaluation/metrics.py -------------------------

def _rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """Rotation matrices ``[..., 3, 3]`` for an xyzw quaternion array."""
    x, y, z, w = (quat_xyzw[..., index] for index in range(4))
    return np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(*quat_xyzw.shape[:-1], 3, 3)


def _quaternion_geodesic_angle(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    """Shortest-arc angle in ``[0, pi]`` radians between broadcastable xyzw quaternion arrays."""
    # sin(theta/2) form: 2*arccos(dot) loses half its digits near identity.
    dot = np.clip(np.abs(np.sum(a_xyzw * b_xyzw, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot)


def _object_add(achieved: np.ndarray, reference: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Average 3D distance ``(T, W, B)`` in metres over each body's object-frame vertices.

    ``achieved`` is (T, W, B, 7); ``reference`` is (T, 1, B, 7); ``vertices`` is (B, V, 3).
    """
    delta_rotation = _rotation_matrix(achieved[..., 3:]) - _rotation_matrix(reference[..., 3:])
    delta_position = achieved[..., :3] - reference[..., :3]
    steps, worlds, bodies, _ = achieved.shape
    add = np.empty((steps, worlds, bodies), dtype=np.float64)
    for body in range(bodies):
        offsets = np.einsum("twij,vj->twvi", delta_rotation[:, :, body], vertices[body])
        offsets += delta_position[:, :, body, None, :]
        add[:, :, body] = np.linalg.norm(offsets, axis=-1).mean(axis=-1)
    return add


def _add_auc(add: np.ndarray) -> float:
    """Mean over bodies of the area under each body's ADD accuracy curve, pooled over (T, W)."""
    axis = np.linspace(0.0, 1.0, _ADD_THRESHOLDS_M.size)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    per_body = [
        float(trapezoid(np.array([np.mean(add[..., body] < t) for t in _ADD_THRESHOLDS_M]), x=axis))
        for body in range(add.shape[2])
    ]
    return float(np.mean(per_body))


def _quaternion_multiply(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternion arrays, broadcasting over leading axes."""
    ax, ay, az, aw = (a_xyzw[..., i] for i in range(4))
    bx, by, bz, bw = (b_xyzw[..., i] for i in range(4))
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _warp_to_reference_start(achieved: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rigidly move the submitted trajectory so its first frame starts where the reference does.

    A submission may express its poses in whatever world frame its simulator happened to use.
    Scoring that directly measures the frame mismatch, not the policy. This removes the mismatch
    and nothing else: ONE rigid transform per world, derived from the first frame, applied to
    every object and every frame.

        q_align = q_ref[0, slot 0] * conj(q_achieved[0, slot 0])
        R_align = rotation(q_align)
        t_align = p_ref[0, slot 0] - R_align @ p_achieved[0, slot 0]

    Two properties make this safe to apply before any metric.

    It is ONE transform for the whole scene, taken from object slot 0. Aligning each object
    independently would zero every object's starting error and destroy the relative geometry that
    RPE and the multi-object gate in MP-SR exist to measure. A single rigid transform leaves every
    inter-object relationship untouched.

    It is rigid, not a similarity. No scale is fitted, so a submission cannot shrink its way to a
    smaller error.

    RPE is invariant to this by construction -- it is computed in object 0's body frame, which a
    global rigid transform does not change -- so applying the warp cannot alter an RPE score. The
    other four metrics are absolute and do change. ``tests/test_object_pose_path.py`` asserts both.

    Object slot 0's first-frame error becomes exactly zero, which is the intended cost: one frame
    of one object, against removing an error the participant had no way to avoid.
    """
    reference_start = reference[0, 0, 0]                      # (7,) first frame, first object
    achieved_start = achieved[0, :, 0]                        # (W, 7)

    conjugate = achieved_start[:, 3:] * np.array([-1.0, -1.0, -1.0, 1.0])
    q_align = _quaternion_multiply(reference_start[None, 3:], conjugate)          # (W, 4)
    q_align = q_align / np.linalg.norm(q_align, axis=-1, keepdims=True)
    rotation = _rotation_matrix(q_align)                                          # (W, 3, 3)
    offset = reference_start[None, :3] - np.einsum("wij,wj->wi", rotation, achieved_start[:, :3])

    warped = np.empty_like(achieved)
    warped[..., :3] = np.einsum("wij,twbj->twbi", rotation, achieved[..., :3]) + offset[None, :, None, :]
    warped[..., 3:] = _quaternion_multiply(q_align[None, :, None, :], achieved[..., 3:])
    return warped


def _per_world_pose_error(achieved: np.ndarray, reference: np.ndarray):
    """Frame-averaged ``(W, B)`` position error in metres and orientation error in radians."""
    position = np.linalg.norm(achieved[..., :3] - reference[..., :3], axis=-1).mean(axis=0)
    orientation = _quaternion_geodesic_angle(achieved[..., 3:], reference[..., 3:]).mean(axis=0)
    return position, orientation


def _spider_success(achieved: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-world success mask. Position error is raw, matching the evaluator."""
    position, orientation = _per_world_pose_error(achieved, reference)
    return (position.mean(axis=-1) <= _SPIDER_POSITION_THRESHOLD_M) & (
        orientation.mean(axis=-1) <= _SPIDER_ORIENTATION_THRESHOLD_RAD
    )


def _maniptrans_success(achieved: np.ndarray, reference: np.ndarray, object_ids: np.ndarray) -> np.ndarray:
    """Per-world success mask. Bodies average within an object; every object must clear both gates."""
    position, orientation = _per_world_pose_error(achieved, reference)
    orientation_deg = np.degrees(orientation)
    success = np.ones(achieved.shape[1], dtype=np.bool_)
    for object_id in np.unique(object_ids):
        members = object_ids == object_id
        success &= (position[:, members].mean(axis=-1) < _MANIPTRANS_POSITION_THRESHOLD_M) & (
            orientation_deg[:, members].mean(axis=-1) < _MANIPTRANS_ORIENTATION_THRESHOLD_DEG
        )
    return success


def _relative_position_error_cm(achieved: np.ndarray, reference: np.ndarray) -> float:
    """Mean multi-object relative position error, in centimetres.

    The relative pose of object 1 expressed in object 0's body frame, compared against the same
    quantity on the reference: ``|| R0a^T (p1a - p0a) - R0r^T (p1r - p0r) ||``. Taking it in
    object 0's frame rather than the world frame is what makes it a *relative* error, coupled to
    the predicted orientation of object 0; a plain world-frame difference would be invariant to
    that orientation entirely.

    Definition transcribed from ``hand_object_commands.py`` at commit 04abe86f (the only place it
    is implemented), which returns metres; the CHORD paper reports centimetres.

    A single-object episode has no object pair and so no relative error to get wrong. It scores
    0.0 rather than NaN, which keeps every episode contributing and makes the metric defined on
    every track. The consequence is that single-object episodes are scored as perfect on this
    metric, pulling a track's mean toward zero in proportion to how many of them it has.
    """
    bodies = achieved.shape[2]
    if bodies < 2:
        return 0.0
    errors = []
    for first in range(bodies):
        for second in range(first + 1, bodies):
            rotation_a = _rotation_matrix(achieved[:, :, first, 3:])
            rotation_r = _rotation_matrix(reference[:, :, first, 3:])
            delta_a = achieved[:, :, second, :3] - achieved[:, :, first, :3]
            delta_r = reference[:, :, second, :3] - reference[:, :, first, :3]
            local_a = np.einsum("twji,twj->twi", rotation_a, delta_a)
            local_r = np.einsum("twji,twj->twi", rotation_r, delta_r)
            errors.append(np.linalg.norm(local_a - local_r, axis=-1))
    return float(np.mean(np.stack(errors)) * 100.0)


def _mean_per_point_error_cm(achieved: np.ndarray, reference: np.ndarray) -> float:
    """Mean per-point position error over six object-frame keypoints, in centimetres.

    Six points are rigidly attached to each object at ``_MPPE_KEYPOINT_RADIUS_M`` along the
    object frame's principal axes, carried into the world by that object's pose, and compared
    point for point against the same construction on the reference::

        kp_achieved = p_a + R_a v        kp_reference = p_r + R_r v

    The mean of ``||kp_achieved - kp_reference||`` over points, bodies, worlds and frames.

    This is the geometry of ``object_keypoints_objective`` in
    ``flash_chord/objectives/keypoints.py``, with two deliberate differences. That kernel is a
    training reward: it squares each distance and passes it through ``shaped_objective``, and it
    places the points at unit radius. A reported error is neither squared nor shaped, and the
    radius is 5 cm. Both matter - squaring changes which errors dominate the average, and the
    radius scales the orientation contribution linearly.

    Unlike RPE this is defined for a single object, so it never returns NaN.
    """
    offsets = _MPPE_KEYPOINT_DIRECTIONS * _MPPE_KEYPOINT_RADIUS_M     # (6, 3)
    rotation_a = _rotation_matrix(achieved[..., 3:])                  # (T, W, B, 3, 3)
    rotation_r = _rotation_matrix(reference[..., 3:])                 # (T, 1, B, 3, 3)
    # (..., 3, 3) @ (6, 3) -> (..., 6, 3), then offset by each pose's own translation.
    keypoints_a = achieved[..., None, :3] + np.einsum("...ij,kj->...ki", rotation_a, offsets)
    keypoints_r = reference[..., None, :3] + np.einsum("...ij,kj->...ki", rotation_r, offsets)
    return float(np.linalg.norm(keypoints_a - keypoints_r, axis=-1).mean() * 100.0)


# --- Reshaping the long CSV back into the evaluator's tensors -------------------------

def _episode_tensors(frame: pd.DataFrame):
    """(achieved (T,W,B,7), reference (T,1,B,7), object_ids (B,), object_names (B,)) for one episode.

    The grid is validated to be complete and rectangular. A missing or duplicated cell would
    otherwise reshape into a silently misaligned tensor, which is the worst possible failure here
    because it produces a plausible wrong number rather than an error.
    """
    frame = frame.sort_values(["frame_index", "world_index", "object_slot"], kind="mergesort")
    frames = np.unique(frame["frame_index"].to_numpy())
    worlds = np.unique(frame["world_index"].to_numpy())
    slots = np.unique(frame["object_slot"].to_numpy())
    expected = len(frames) * len(worlds) * len(slots)
    if len(frame) != expected:
        raise ValueError(
            f"episode grid is not rectangular: {len(frame)} rows for "
            f"{len(frames)} frames x {len(worlds)} worlds x {len(slots)} objects"
        )
    shape = (len(frames), len(worlds), len(slots))
    achieved = frame[_POSE_COLUMNS].to_numpy(dtype=np.float64).reshape(*shape, 7)
    reference = frame[_REF_COLUMNS].to_numpy(dtype=np.float64).reshape(*shape, 7)
    # The reference is shared across worlds; keep world 0 and let it broadcast.
    names = frame["object_name"].to_numpy().reshape(*shape)[0, 0]
    return achieved, reference[:, :1], slots.astype(np.int64), names


def _normalise_quaternions(pose: np.ndarray, who: str, error_type) -> np.ndarray:
    """Unit-normalise the quaternion half of a (..., 7) pose array, rejecting degenerate ones."""
    pose = pose.copy()
    norms = np.linalg.norm(pose[..., 3:], axis=-1)
    if np.any(norms < 1e-8):
        raise error_type(f"{who} contains a degenerate (near-zero-norm) quaternion")
    pose[..., 3:] /= norms[..., None]
    return pose


# --- CORE END -------------------------------------------------------------------------


def episode_metric(metric: str, achieved: np.ndarray, reference: np.ndarray, object_ids, vertices,
                   align_initial_pose: bool = False):
    """One episode's value for a named metric, or NaN when the metric is undefined for it.

    ``align_initial_pose`` mirrors the per-track switch the generated Kaggle metric carries, so a
    participant's local number and the host's come from the same code and the same treatment.
    """
    if align_initial_pose:
        achieved = _warp_to_reference_start(achieved, reference)
    if metric == "add_auc":
        return _add_auc(_object_add(achieved, reference, vertices))
    if metric == "spider_sr":
        return float(np.mean(_spider_success(achieved, reference)))
    if metric == "maniptrans_sr":
        return float(np.mean(_maniptrans_success(achieved, reference, object_ids)))
    if metric == "rpe_cm":
        return _relative_position_error_cm(achieved, reference)
    if metric == "mppe_cm":
        return _mean_per_point_error_cm(achieved, reference)
    raise ValueError(f"unknown metric {metric!r}")
