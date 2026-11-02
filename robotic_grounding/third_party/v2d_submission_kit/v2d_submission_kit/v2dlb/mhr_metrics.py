"""Track 1 reconstruction metrics, vendored from Bowen's CARI4D evaluator.

The body between the CORE markers is Bowen's ``metrics.py`` transcribed with one change: its
``from .rotations import ...`` is replaced by the two functions inlined below, because a Kaggle
metric notebook is a single flat file with no package around it. Everything else is his, so the
number a participant computes locally and the number the host computes come from the same lines.

The five scored metrics are ``mhr_table3_metrics``: cd_h_cm, cd_o_cm, cd_c_cm, acc_h_cm,
acc_o_cm. All are centimetres and all are lower-is-better.

Input contract, per side (``pred`` and ``target``), verified against his code:

    mhr_vertices          [T, V, 3]   frame 0 only is used, for one global similarity alignment
    mhr_joints            [T, J, 3]   J > 113; 22 body joints are selected by index
    human_surface_points  [T, N, 3]
    object_surface_points [T, N, 3]   same N as the human points, so the combined Chamfer
                                      distance weights both meshes equally
    object_translation    [T, 3]

    T >= 3, because acceleration is a second difference.

ALIGNMENT. One similarity transform is fitted from frame 0's ``mhr_vertices`` and applied to
every predicted array. This settles open question 3 in docs/TRACK1_DESIGN.md: alignment is a
single global fit, not the per-frame 25-start ICP that ``evaluate_mesh_quality.py`` used, and not
nothing at all. Pose error after frame 0 is therefore scored, not discarded.

torch is optional. Kaggle's kernel image has torch 2.10 CPU (measured 2026-09-21), but nothing
here requires it: every branch has a numpy path, and ``is_torch_tensor`` returns False when torch
is absent.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# --- CORE BEGIN (inlined into the generated Kaggle metric; edit here, never there) ---


def is_torch_tensor(value: Any) -> bool:
    """True when ``value`` is a torch tensor, without importing torch to find out.

    Inlined from Bowen's ``rotations`` module. Checking sys.modules first means a numpy-only
    caller never pays the torch import, and a host without torch still works.
    """
    torch = __import__("sys").modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def rot6d_to_rotmat(rot6d: Any) -> Any:
    """6D rotation representation to rotation matrices, by Gram-Schmidt on the two columns.

    Inlined from Bowen's ``rotations`` module. Unused by the five Table 3 metrics -- it is only
    reached through ``mhr_metrics``/``_root_rot_error`` -- but kept so the vendored file stays a
    faithful copy that can be diffed against his.
    """
    if is_torch_tensor(rot6d):
        import torch

        a, b = rot6d[..., :3], rot6d[..., 3:6]
        e1 = torch.nn.functional.normalize(a, dim=-1)
        e2 = torch.nn.functional.normalize(b - (e1 * b).sum(-1, keepdim=True) * e1, dim=-1)
        return torch.stack((e1, e2, torch.cross(e1, e2, dim=-1)), dim=-1)
    rot6d = np.asarray(rot6d)
    a, b = rot6d[..., :3], rot6d[..., 3:6]
    e1 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (e1 * b).sum(-1, keepdims=True) * e1
    e2 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack((e1, e2, np.cross(e1, e2)), axis=-1)


MHR_TABLE3_BODY_JOINT_INDICES = (
    1,   # pelvis: root
    2,   # left hip: l_upleg
    18,  # right hip: r_upleg
    35,  # lower spine: c_spine1
    3,   # left knee: l_lowleg
    19,  # right knee: r_lowleg
    36,  # middle spine: c_spine2
    4,   # left ankle: l_foot
    20,  # right ankle: r_foot
    37,  # upper spine: c_spine3
    8,   # left foot: l_ball
    24,  # right foot: r_ball
    110,  # neck: c_neck
    74,  # left collar: l_clavicle
    38,  # right collar: r_clavicle
    113,  # head: c_head
    75,  # left shoulder: l_uparm
    39,  # right shoulder: r_uparm
    76,  # left elbow: l_lowarm
    40,  # right elbow: r_lowarm
    78,  # left wrist: l_wrist
    42,  # right wrist: r_wrist
)

MHR_TABLE3_REQUIRED_KEYS = (
    "mhr_vertices",
    "mhr_joints",
    "human_surface_points",
    "object_surface_points",
    "object_translation",
)

MHR_TABLE3_METRIC_KEYS = ("cd_h_cm", "cd_o_cm", "cd_c_cm", "acc_h_cm", "acc_o_cm")
MHR_INTERPENETRATION_METRIC_KEY = "interpenetration_cm"
MHR_CONTACT_ACCURACY_METRIC_KEY = "contact_accuracy"


def _point_error(pred: Any, target: Any) -> Any:
    if is_torch_tensor(pred):
        import torch

        return torch.linalg.norm(pred - target, dim=-1).mean()
    return np.linalg.norm(np.asarray(pred) - np.asarray(target), axis=-1).mean()


def _root_rot_error(pred_rot6d: Any, target_rot6d: Any) -> Any:
    pred = rot6d_to_rotmat(pred_rot6d)
    target = rot6d_to_rotmat(target_rot6d)
    if is_torch_tensor(pred):
        import torch

        rel = pred.transpose(-1, -2) @ target
        trace = rel.diagonal(dim1=-1, dim2=-2).sum(-1)
        cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
        return torch.acos(cos).mean()
    rel = pred.swapaxes(-1, -2) @ target
    trace = np.trace(rel, axis1=-1, axis2=-2)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos).mean()


def mhr_metrics(pred: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    """Compute native MHR metrics for matching topology predictions."""

    out: dict[str, Any] = {}
    if "mhr_vertices" in pred and "mhr_vertices" in target:
        out["mhr_v2v"] = _point_error(pred["mhr_vertices"], target["mhr_vertices"])
    if "mhr_joints" in pred and "mhr_joints" in target:
        out["mhr_mpjpe"] = _point_error(pred["mhr_joints"], target["mhr_joints"])
    elif "mhr_keypoints" in pred and "mhr_keypoints" in target:
        out["mhr_keypoint_error"] = _point_error(pred["mhr_keypoints"], target["mhr_keypoints"])
    if "mhr_trans" in pred and "mhr_trans" in target:
        out["mhr_trans_error"] = _point_error(pred["mhr_trans"], target["mhr_trans"])
    if "mhr_global_rot6d" in pred and "mhr_global_rot6d" in target:
        out["mhr_root_rot_error"] = _root_rot_error(pred["mhr_global_rot6d"], target["mhr_global_rot6d"])
    return out


def _as_metric_array(value: Any) -> Any:
    return value if is_torch_tensor(value) else np.asarray(value)


def _validate_float_array(name: str, value: Any) -> None:
    if is_torch_tensor(value):
        import torch

        if not value.dtype.is_floating_point:
            raise TypeError(f"{name} must be floating point, got {value.dtype}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
        return
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _validate_same_backend(values: Mapping[str, Any]) -> bool:
    tensor_keys = [key for key, value in values.items() if is_torch_tensor(value)]
    if tensor_keys and len(tensor_keys) != len(values):
        array_keys = [key for key in values if key not in tensor_keys]
        raise TypeError(f"metric inputs cannot mix torch tensors {tensor_keys} with NumPy arrays {array_keys}")
    if not tensor_keys:
        return False

    reference = values[tensor_keys[0]]
    mismatched = [key for key, value in values.items() if value.device != reference.device or value.dtype != reference.dtype]
    if mismatched:
        raise TypeError(f"torch metric inputs must share device and dtype; mismatched keys: {mismatched}")
    return True


def _validate_sequence_points(name: str, value: Any) -> None:
    _validate_float_array(name, value)
    if value.ndim != 3 or value.shape[0] == 0 or value.shape[1] == 0 or value.shape[2] != 3:
        raise ValueError(f"{name} must have non-empty shape [T, N, 3], got {tuple(value.shape)}")


def fit_similarity_transform(source_points: Any, target_points: Any) -> tuple[Any, Any, Any]:
    """Fit ``target = scale * source @ rotation + translation`` without reflection."""

    source = _as_metric_array(source_points)
    target = _as_metric_array(target_points)
    uses_torch = _validate_same_backend({"source_points": source, "target_points": target})
    _validate_float_array("source_points", source)
    _validate_float_array("target_points", target)
    if source.ndim != 2 or source.shape[-1] != 3 or source.shape[0] < 3 or source.shape != target.shape:
        raise ValueError(f"similarity alignment requires matching [N, 3] point sets with N >= 3, got {tuple(source.shape)} and {tuple(target.shape)}")

    source_mean = source.mean(axis=0) if not uses_torch else source.mean(dim=0)
    target_mean = target.mean(axis=0) if not uses_torch else target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    denominator = (source_centered * source_centered).sum()

    if uses_torch:
        import torch

        if float(denominator.detach().cpu()) <= torch.finfo(source.dtype).eps:
            raise ValueError("source_points have zero spatial extent")
        u, singular_values, vh = torch.linalg.svd(source_centered.transpose(0, 1) @ target_centered)
        final_sign = torch.where(torch.linalg.det(u @ vh) < 0, -torch.ones_like(singular_values[-1:]), torch.ones_like(singular_values[-1:]))
        correction = torch.cat((torch.ones_like(singular_values[:-1]), final_sign))
        rotation = (u * correction[None]) @ vh
        scale = (singular_values * correction).sum() / denominator
        translation = target_mean - scale * (source_mean @ rotation)
        if not bool(torch.isfinite(scale)) or float(scale.detach().cpu()) <= 0:
            raise ValueError(f"similarity alignment produced invalid scale {float(scale.detach().cpu())}")
        return scale, rotation, translation

    if float(denominator) <= np.finfo(source.dtype).eps:
        raise ValueError("source_points have zero spatial extent")
    u, singular_values, vh = np.linalg.svd(source_centered.T @ target_centered)
    correction = np.ones_like(singular_values)
    correction[-1] = -1.0 if np.linalg.det(u @ vh) < 0 else 1.0
    rotation = (u * correction[None]) @ vh
    scale = (singular_values * correction).sum() / denominator
    translation = target_mean - scale * (source_mean @ rotation)
    if not np.isfinite(scale) or float(scale) <= 0:
        raise ValueError(f"similarity alignment produced invalid scale {float(scale)}")
    return scale, rotation, translation


def apply_similarity_transform(points: Any, transform: tuple[Any, Any, Any]) -> Any:
    """Apply a transform returned by :func:`fit_similarity_transform` to row-vector points."""

    value = _as_metric_array(points)
    scale, rotation, translation = transform
    rotation = _as_metric_array(rotation)
    translation = _as_metric_array(translation)
    _validate_same_backend({"points": value, "rotation": rotation, "translation": translation})
    _validate_float_array("points", value)
    _validate_float_array("rotation", rotation)
    _validate_float_array("translation", translation)
    if value.ndim < 2 or value.shape[-1] != 3 or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError(f"similarity transform requires points [..., 3], rotation [3, 3], and translation [3], got {tuple(value.shape)}, {tuple(rotation.shape)}, and {tuple(translation.shape)}")
    return scale * (value @ rotation) + translation


def _directed_nearest_distance_mean(source: Any, target: Any, chunk_size: int, uses_torch: bool) -> Any:
    frame_means = []
    for frame_index in range(source.shape[0]):
        source_frame = source[frame_index]
        target_frame = target[frame_index]
        target_squared = (target_frame * target_frame).sum(dim=-1) if uses_torch else np.einsum("ij,ij->i", target_frame, target_frame)
        nearest_chunks = []
        for start in range(0, source.shape[1], chunk_size):
            source_chunk = source_frame[start:start + chunk_size]
            source_squared = (source_chunk * source_chunk).sum(dim=-1) if uses_torch else np.einsum("ij,ij->i", source_chunk, source_chunk)
            squared_distances = source_squared[:, None] + target_squared[None] - 2.0 * (source_chunk @ target_frame.transpose(0, 1) if uses_torch else source_chunk @ target_frame.T)
            nearest_indices = squared_distances.argmin(dim=-1) if uses_torch else squared_distances.argmin(axis=-1)
            nearest_targets = target_frame.index_select(0, nearest_indices) if uses_torch else target_frame[nearest_indices]
            nearest_chunks.append((source_chunk - nearest_targets).norm(dim=-1) if uses_torch else np.linalg.norm(source_chunk - nearest_targets, axis=-1))
        if uses_torch:
            import torch

            nearest = torch.cat(nearest_chunks)
        else:
            nearest = np.concatenate(nearest_chunks)
        frame_means.append(nearest.mean())
    if uses_torch:
        import torch

        return torch.stack(frame_means)
    return np.asarray(frame_means)


def symmetric_chamfer_distance(source_points: Any, target_points: Any, *, chunk_size: int = 512) -> Any:
    """Return per-frame symmetric surface Chamfer distance for ``[T, N, 3]`` point sets."""

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
    source = _as_metric_array(source_points)
    target = _as_metric_array(target_points)
    uses_torch = _validate_same_backend({"source_points": source, "target_points": target})
    _validate_sequence_points("source_points", source)
    _validate_sequence_points("target_points", target)
    if source.shape[0] != target.shape[0]:
        raise ValueError(f"Chamfer point sets have different frame counts: {source.shape[0]} vs {target.shape[0]}")
    return _directed_nearest_distance_mean(source, target, chunk_size, uses_torch) + _directed_nearest_distance_mean(target, source, chunk_size, uses_torch)


def acceleration_error_per_frame(predicted_points: Any, target_points: Any) -> Any:
    """Return paper-compatible finite-difference acceleration error without frame-rate scaling."""

    predicted = _as_metric_array(predicted_points)
    target = _as_metric_array(target_points)
    uses_torch = _validate_same_backend({"predicted_points": predicted, "target_points": target})
    _validate_float_array("predicted_points", predicted)
    _validate_float_array("target_points", target)
    if predicted.shape != target.shape or predicted.ndim < 2 or predicted.shape[0] < 3 or predicted.shape[-1] != 3:
        raise ValueError(f"acceleration error requires matching [T, ..., 3] arrays with T >= 3, got {tuple(predicted.shape)} and {tuple(target.shape)}")

    predicted_acceleration = predicted[:-2] - 2.0 * predicted[1:-1] + predicted[2:]
    target_acceleration = target[:-2] - 2.0 * target[1:-1] + target[2:]
    error = (predicted_acceleration - target_acceleration).norm(dim=-1) if uses_torch else np.linalg.norm(predicted_acceleration - target_acceleration, axis=-1)
    if error.ndim > 1:
        dimensions = tuple(range(1, error.ndim))
        error = error.mean(dim=dimensions) if uses_torch else error.mean(axis=dimensions)
    if uses_torch:
        import torch

        return torch.cat((error[:1], error, error[-1:]))
    return np.concatenate((error[:1], error, error[-1:]))


def sdf_interpenetration_distance_per_frame(signed_distances_m: Any) -> Any:
    """Return mean negative signed-distance depth per frame in centimeters.

    ``signed_distances_m`` has shape ``[T, P]`` and uses the convention that
    query points inside the reference surface have negative signed distance.
    Outside points contribute zero while the denominator remains all ``P``
    query points.
    """

    distances = _as_metric_array(signed_distances_m)
    _validate_float_array("signed_distances_m", distances)
    if distances.ndim != 2 or distances.shape[0] == 0 or distances.shape[1] == 0:
        raise ValueError(f"signed_distances_m must have non-empty shape [T, P], got {tuple(distances.shape)}")
    if is_torch_tensor(distances):
        import torch

        return torch.relu(-distances).mean(dim=1) * 100.0
    return np.maximum(-distances, 0.0).mean(axis=1) * 100.0


def contact_accuracy_counts_from_logits(contact_logits: Any, target_distances_m: Any, *, distance_threshold_m: float = 0.015, valid_mask: Any | None = None) -> tuple[Any, Any]:
    """Return correct and valid hand-label counts for binary contact logits."""

    logits = _as_metric_array(contact_logits)
    targets = _as_metric_array(target_distances_m)
    uses_torch = _validate_same_backend({"contact_logits": logits, "target_distances_m": targets})
    _validate_float_array("contact_logits", logits)
    _validate_float_array("target_distances_m", targets)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != 2 or logits.shape != targets.shape:
        raise ValueError(f"contact logits and target distances must have matching non-empty shape [T, 2], got {tuple(logits.shape)} and {tuple(targets.shape)}")
    if not np.isfinite(distance_threshold_m) or distance_threshold_m <= 0:
        raise ValueError(f"distance_threshold_m must be positive and finite, got {distance_threshold_m}")
    if uses_torch:
        import torch

        if bool((targets < 0).any()):
            raise ValueError("target_distances_m contains negative distances")
        if valid_mask is None:
            valid = torch.ones_like(logits, dtype=torch.bool)
        else:
            if not is_torch_tensor(valid_mask) or valid_mask.device != logits.device or valid_mask.dtype != torch.bool or valid_mask.shape != logits.shape:
                raise TypeError(f"valid_mask must be a boolean torch tensor matching contact logits on {logits.device}, got {type(valid_mask)}")
            valid = valid_mask
        correct = ((logits > 0) == (targets < float(distance_threshold_m))) & valid
        valid_count = valid.sum()
        if int(valid_count.detach().cpu()) == 0:
            raise ValueError("contact accuracy requires at least one valid hand label")
        return correct.sum(), valid_count
    if bool((targets < 0).any()):
        raise ValueError("target_distances_m contains negative distances")
    if valid_mask is None:
        valid = np.ones_like(logits, dtype=bool)
    else:
        valid = np.asarray(valid_mask)
        if valid.dtype != np.bool_ or valid.shape != logits.shape:
            raise TypeError(f"valid_mask must be a boolean NumPy array with shape {tuple(logits.shape)}, got dtype={valid.dtype} shape={tuple(valid.shape)}")
    valid_count = np.count_nonzero(valid)
    if valid_count == 0:
        raise ValueError("contact accuracy requires at least one valid hand label")
    correct = ((logits > 0) == (targets < float(distance_threshold_m))) & valid
    return np.count_nonzero(correct), valid_count


def contact_accuracy_from_logits(contact_logits: Any, target_distances_m: Any, *, distance_threshold_m: float = 0.015, valid_mask: Any | None = None) -> Any:
    """Return pooled left/right contact accuracy using a zero-logit decision boundary."""

    correct_count, valid_count = contact_accuracy_counts_from_logits(contact_logits, target_distances_m, distance_threshold_m=distance_threshold_m, valid_mask=valid_mask)
    if is_torch_tensor(correct_count):
        return correct_count.to(dtype=contact_logits.dtype) / valid_count
    return float(correct_count) / int(valid_count)


def _table3_inputs(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    missing = [key for key in MHR_TABLE3_REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"{name} is missing Table 3 inputs: {missing}")
    return {key: _as_metric_array(data[key]) for key in MHR_TABLE3_REQUIRED_KEYS}


def _select_body_joints(joints: Any, body_joint_indices: Sequence[int] | None) -> Any:
    if body_joint_indices is None:
        return joints
    indices = tuple(int(index) for index in body_joint_indices)
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= joints.shape[1]:
        raise ValueError(f"body_joint_indices must be unique valid indices for {joints.shape[1]} MHR joints, got {indices}")
    if is_torch_tensor(joints):
        import torch

        return joints.index_select(1, torch.as_tensor(indices, device=joints.device))
    return joints[:, indices]


def mhr_table3_metric_series(pred: Mapping[str, Any], target: Mapping[str, Any], *, body_joint_indices: Sequence[int] | None = MHR_TABLE3_BODY_JOINT_INDICES, chamfer_chunk_size: int = 512) -> dict[str, Any]:
    """Compute per-frame CARI4D Table 3 metrics from one native MHR sequence.

    Inputs are metric-space arrays. ``mhr_vertices`` supplies matching-topology
    first-frame correspondences for one global similarity alignment. The human
    and object surface arrays must each contain the same number of uniformly
    sampled points so the combined Chamfer distance weights both meshes equally.
    Predicted object surface points and translations must already be posed.
    """

    pred_values = _table3_inputs(pred, "pred")
    target_values = _table3_inputs(target, "target")
    all_values = {f"pred.{key}": value for key, value in pred_values.items()}
    all_values.update({f"target.{key}": value for key, value in target_values.items()})
    uses_torch = _validate_same_backend(all_values)
    for prefix, values in (("pred", pred_values), ("target", target_values)):
        for key in ("mhr_vertices", "mhr_joints", "human_surface_points", "object_surface_points"):
            _validate_sequence_points(f"{prefix}.{key}", values[key])
        _validate_float_array(f"{prefix}.object_translation", values["object_translation"])
        if values["object_translation"].ndim != 2 or values["object_translation"].shape[1] != 3:
            raise ValueError(f"{prefix}.object_translation must have shape [T, 3], got {tuple(values['object_translation'].shape)}")

    frame_counts = {f"{prefix}.{key}": int(values[key].shape[0]) for prefix, values in (("pred", pred_values), ("target", target_values)) for key in MHR_TABLE3_REQUIRED_KEYS}
    if len(set(frame_counts.values())) != 1 or next(iter(frame_counts.values())) < 3:
        raise ValueError(f"Table 3 inputs require one shared frame count T >= 3, got {frame_counts}")
    if pred_values["mhr_vertices"].shape != target_values["mhr_vertices"].shape:
        raise ValueError(f"first-frame MHR alignment requires matching vertex topology, got {tuple(pred_values['mhr_vertices'].shape)} and {tuple(target_values['mhr_vertices'].shape)}")
    if pred_values["mhr_joints"].shape != target_values["mhr_joints"].shape:
        raise ValueError(f"human acceleration requires matching MHR joints, got {tuple(pred_values['mhr_joints'].shape)} and {tuple(target_values['mhr_joints'].shape)}")
    surface_counts = {f"{prefix}.{key}": int(values[key].shape[1]) for prefix, values in (("pred", pred_values), ("target", target_values)) for key in ("human_surface_points", "object_surface_points")}
    if len(set(surface_counts.values())) != 1:
        raise ValueError(f"Table 3 combined Chamfer requires equal human/object surface sample counts, got {surface_counts}")

    transform = fit_similarity_transform(pred_values["mhr_vertices"][0], target_values["mhr_vertices"][0])
    pred_human_surface = apply_similarity_transform(pred_values["human_surface_points"], transform)
    pred_object_surface = apply_similarity_transform(pred_values["object_surface_points"], transform)
    pred_joints = _select_body_joints(apply_similarity_transform(pred_values["mhr_joints"], transform), body_joint_indices)
    target_joints = _select_body_joints(target_values["mhr_joints"], body_joint_indices)
    pred_object_translation = apply_similarity_transform(pred_values["object_translation"], transform)
    if uses_torch:
        import torch

        pred_combined = torch.cat((pred_human_surface, pred_object_surface), dim=1)
        target_combined = torch.cat((target_values["human_surface_points"], target_values["object_surface_points"]), dim=1)
    else:
        pred_combined = np.concatenate((pred_human_surface, pred_object_surface), axis=1)
        target_combined = np.concatenate((target_values["human_surface_points"], target_values["object_surface_points"]), axis=1)

    return {
        "cd_h_cm": symmetric_chamfer_distance(pred_human_surface, target_values["human_surface_points"], chunk_size=chamfer_chunk_size) * 100.0,
        "cd_o_cm": symmetric_chamfer_distance(pred_object_surface, target_values["object_surface_points"], chunk_size=chamfer_chunk_size) * 100.0,
        "cd_c_cm": symmetric_chamfer_distance(pred_combined, target_combined, chunk_size=chamfer_chunk_size) * 100.0,
        "acc_h_cm": acceleration_error_per_frame(pred_joints, target_joints) * 100.0,
        "acc_o_cm": acceleration_error_per_frame(pred_object_translation, target_values["object_translation"]) * 100.0,
    }


def mhr_table3_metrics(pred: Mapping[str, Any], target: Mapping[str, Any], *, body_joint_indices: Sequence[int] | None = MHR_TABLE3_BODY_JOINT_INDICES, chamfer_chunk_size: int = 512) -> dict[str, Any]:
    """Return sequence means of the five CARI4D Table 3 metrics in centimeters."""

    series = mhr_table3_metric_series(pred, target, body_joint_indices=body_joint_indices, chamfer_chunk_size=chamfer_chunk_size)
    return {key: value.mean() for key, value in series.items()}


# --- CORE END ---------------------------------------------------------------------
