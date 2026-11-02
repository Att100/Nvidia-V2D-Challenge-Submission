"""Kaggle metric scoring code. Reference arrays and embedded model or vertex assets are omitted from this participant copy."""

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class ParticipantVisibleError(Exception):
    pass


                                                                               
                                                                    
def is_torch_tensor(value: Any) -> bool:

    torch = __import__("sys").modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def rot6d_to_rotmat(rot6d: Any) -> Any:

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
    1,                 
    2,                      
    18,                      
    35,                         
    3,                        
    19,                        
    36,                          
    4,                       
    20,                       
    37,                         
    8,                      
    24,                      
    110,                
    74,                           
    38,                            
    113,                
    75,                          
    39,                           
    76,                        
    40,                         
    78,                       
    42,                        
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


    distances = _as_metric_array(signed_distances_m)
    _validate_float_array("signed_distances_m", distances)
    if distances.ndim != 2 or distances.shape[0] == 0 or distances.shape[1] == 0:
        raise ValueError(f"signed_distances_m must have non-empty shape [T, P], got {tuple(distances.shape)}")
    if is_torch_tensor(distances):
        import torch

        return torch.relu(-distances).mean(dim=1) * 100.0
    return np.maximum(-distances, 0.0).mean(axis=1) * 100.0


def contact_accuracy_counts_from_logits(contact_logits: Any, target_distances_m: Any, *, distance_threshold_m: float = 0.015, valid_mask: Any | None = None) -> tuple[Any, Any]:


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


    series = mhr_table3_metric_series(pred, target, body_joint_indices=body_joint_indices, chamfer_chunk_size=chamfer_chunk_size)
    return {key: value.mean() for key, value in series.items()}


                                                                               
                                                                    
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


                                                                            
                                                                    
def _quat_to_mat(q):

    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _euler_xyz_to_mat(e):

    cx, cy, cz = np.cos(e[..., 0]), np.cos(e[..., 1]), np.cos(e[..., 2])
    sx, sy, sz = np.sin(e[..., 0]), np.sin(e[..., 1]), np.sin(e[..., 2])
    return np.stack([
        np.stack([cy * cz, -cx * sz + sx * sy * cz, sx * sz + cx * sy * cz], -1),
        np.stack([cy * sz, cx * cz + sx * sy * sz, -sx * cz + cx * sy * sz], -1),
        np.stack([-sy, sx * cy, cx * cy], -1),
    ], -2)


class _MHRBody:


    _FLIP = np.array([1.0, -1.0, -1.0])

    def __init__(self, asset) -> None:
        import scipy.sparse as sp

        f64 = np.float64
        self.parents = asset["joint_parents"].astype(np.int64)
        self.offsets = asset["joint_offsets"].astype(f64)
        self.pre_rotation = _quat_to_mat(asset["joint_prerotations"].astype(f64))
        self.parameter_transform = asset["parameter_transform"].astype(f64)                  
        bind = asset["inverse_bind_pose"].astype(f64)                                      
        self.bind_t, self.bind_R, self.bind_s = bind[:, :3], _quat_to_mat(bind[:, 3:7]), bind[:, 7]
        self.base_shape = asset["base_shape"].astype(f64).reshape(-1)                    
        self.shape_vectors = asset["shape_vectors"].astype(f64).reshape(45, -1)             
        self.skin = asset["skin_dense"].astype(f64)                                        
        rows, cols = asset["corr_w1_idx"]
        self.w1 = sp.csr_matrix((asset["corr_w1_val"].astype(f64), (rows, cols)),
                                shape=tuple(asset["corr_w1_shape"]))
        self.w2 = sp.csr_matrix((asset["corr_w2_data"].astype(f64), asset["corr_w2_indices"],
                                 asset["corr_w2_indptr"]), shape=tuple(asset["corr_w2_shape"]))
        self.roles = {name: asset[f"role_{name}"].astype(np.int64)
                      for name in ("alignment", "body", "left_hand", "right_hand")}

    def __call__(self, model_params, shape_params, chunk: int = 512):

        model_params = np.atleast_2d(np.asarray(model_params, dtype=np.float64))
        shape_params = np.asarray(shape_params, dtype=np.float64).reshape(45)
        parts = [self._forward(model_params[s:s + chunk], shape_params)
                 for s in range(0, len(model_params), chunk)]
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))

    def _forward(self, model_params, shape_params):
        n = len(model_params)
        joint_params = (model_params @ self.parameter_transform.T).reshape(n, 127, 7)
        local_t = joint_params[..., :3] + self.offsets
        local_R = self.pre_rotation[None] @ _euler_xyz_to_mat(joint_params[..., 3:6])
        local_s = np.exp2(joint_params[..., 6])
        world_t, world_R, world_s = (np.empty_like(local_t), np.empty_like(local_R),
                                     np.empty_like(local_s))
        for j, parent in enumerate(self.parents):
            if parent < 0:
                world_t[:, j], world_R[:, j], world_s[:, j] = local_t[:, j], local_R[:, j], local_s[:, j]
            else:
                world_t[:, j] = world_t[:, parent] + world_s[:, parent, None] * np.einsum(
                    "nab,nb->na", world_R[:, parent], local_t[:, j])
                world_R[:, j] = world_R[:, parent] @ local_R[:, j]
                world_s[:, j] = world_s[:, parent] * local_s[:, j]

                                                                                               
        rot = _euler_xyz_to_mat(joint_params[:, 2:, 3:6])
        features = np.concatenate([rot[..., :, 0], rot[..., :, 1]], -1)
        features[..., 0] -= 1.0
        features[..., 4] -= 1.0
        hidden = np.maximum(self.w1 @ features.reshape(n, -1).T, 0.0)
        rest = self.base_shape + shape_params @ self.shape_vectors + np.asarray(self.w2 @ hidden).T
        rest = rest.reshape(n, -1, 3)

        skin_R = (world_s[..., None, None] * world_R) @ (self.bind_s[:, None, None] * self.bind_R)
        skin_t = world_t + world_s[..., None] * np.einsum("njab,jb->nja", world_R, self.bind_t)
        blend = np.concatenate([skin_R, skin_t[..., None]], -1).reshape(n, 127, 12)
        mixed = (self.skin @ blend.transpose(1, 0, 2).reshape(127, n * 12))
        mixed = mixed.reshape(-1, n, 3, 4).transpose(1, 0, 2, 3)
        vertices = np.einsum("nvab,nvb->nva", mixed[..., :3], rest) + mixed[..., 3]
        return vertices / 100.0 * self._FLIP, world_t / 100.0 * self._FLIP


def _load_body(source) -> _MHRBody:

    import io

    data = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    with np.load(data, allow_pickle=False) as asset:
        return _MHRBody({key: asset[key] for key in asset.files})


                                                                                  
                                                                    
                                                                                            
_EPISODE_FRAME = 999999

                                                        
_PARAMETER_COUNTS = {"mhr_pose": 136, "mhr_scales": 68, "mhr_shape": 45}

                                                                                        
                                                                                                  
_ROTATION_TOLERANCE = 1e-3


def _to_arrays(block, value_columns: list[str]) -> dict[str, np.ndarray]:

    out: dict[str, np.ndarray] = {}
    per_frame = block[block["array_name"].map(lambda n: _ARRAYS.get(n, ("",))[0]) == "frame"]
    frames = np.unique(per_frame["frame_index"].to_numpy())
    for name, (scope, kind) in _ARRAYS.items():
        rows = block[block["array_name"] == name]
        if rows.empty:
            raise ValueError(f"no rows for array {name!r}")
        points = np.unique(rows["point_index"].to_numpy())
                                                                                               
                                                                              
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

    flat = rows.reshape(*rows.shape[:-2], -1)
    if flat.shape[-1] != 3 * -(-count // 3):
        raise ValueError(f"{name} must carry {count} numbers in {-(-count // 3)} rows")
    if np.any(flat[..., count:] != 0):
        raise ValueError(f"{name}: the unused tail of the last row must be 0 (it carries "
                         f"{count} numbers); a nonzero tail means the numbers are misaligned")
    return flat[..., :count]


def _object_scale(rows: np.ndarray) -> float:

    s, y, z = rows[0]
    if not (np.isfinite(s) and s > 0):
        raise ValueError(f"object_scale must be a positive number, got {s}")
    if y != 0 or z != 0:
        raise ValueError("object_scale carries the scale in x; y and z must be 0")
    return float(s)


def _to_frames(block, value_columns: list[str], seed: int, body):

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

    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    edges = [0, *breaks.tolist(), len(frames)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]


def _episode_metrics(pred, target, frames, body_joint_indices) -> dict[str, float]:

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

    try:
        import numba
    except Exception:                                                       
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
                t = d1 / (d1 - d3)
                q0, q1, q2 = a0 + t * ab0, a1 + t * ab1, a2 + t * ab2
            elif d6 >= 0.0 and d5 <= d6:
                q0, q1, q2 = c0, c1, c2
            elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                t = d2 / (d2 - d6)
                q0, q1, q2 = a0 + t * ac0, a1 + t * ac1, a2 + t * ac2
            elif va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
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

    depth = np.zeros(len(points))
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    near = np.flatnonzero(((points >= lo) & (points <= hi)).all(axis=1))
    if near.size == 0:
        return depth
    v0, v1, v2 = (np.ascontiguousarray(vertices[faces[:, k]], dtype=np.float64) for k in range(3))
    if _PENETRATION_KERNELS is not None:
        winding, distance = _PENETRATION_KERNELS
    else:
        winding = lambda q, *f: _in_parallel(_winding_numbers, q, *f)                      
        distance = lambda q, *f: _in_parallel(_point_triangle_distance, q, *f)              
    query = np.ascontiguousarray(points[near], dtype=np.float64)
    inside_local = np.flatnonzero(np.abs(winding(query, v0, v1, v2)) > 0.5)
    if inside_local.size:
        depth[near[inside_local]] = distance(query[inside_local], v0, v1, v2)
    return depth


def _in_parallel(function, points, *faces):

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

    align = float(fit_similarity_transform(pred["mhr_vertices"][0], target["mhr_vertices"][0])[0])
    hands, scale = pred_scene["hands"], pred_scene["scale"]
    local = np.einsum("tpi,tij->tpj", hands - pred_scene["translation"][:, None],
                      pred_scene["rotation"]) / scale
    depth = _penetration_depth(local.reshape(-1, 3), pred_scene["mesh_vertices"],
                               pred_scene["mesh_faces"]).reshape(hands.shape[:2])
    return float(np.mean(sdf_interpenetration_distance_per_frame(-depth * scale * align)))



_METRIC = "acc_o_cm"
_METRIC_DISPLAY = "ACC-O (cm)"
_TRACK_NUMBER = 1

_MHR_NOTICE = ""
_MHR_LICENSE = ""
_MHR_ASSET_B64 = ()
_BODY = None

_ARRAYS = {'mhr_pose': ('frame', 'ordinal'), 'mhr_scales': ('episode', 'ordinal'), 'mhr_shape': ('episode', 'ordinal'), 'object_rotation': ('frame', 'ordinal'), 'object_translation': ('frame', 'ordinal'), 'object_scale': ('episode', 'ordinal'), 'object_mesh_vertices': ('episode', 'ordinal'), 'object_mesh_faces': ('episode', 'ordinal')}
_KEY_COLUMNS = ['episode_index', 'frame_index', 'array_name', 'point_index']
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
        block = block.sort_values(["array_name", "frame_index", "point_index"])
        try:
                                                                                                
                                                                         
            predicted, frames, scene = _to_frames(block, _POINT_COLUMNS, int(episode), _BODY)
            reference, _, _ = _to_frames(block, _REF_COLUMNS, int(episode), _BODY)
        except ValueError as error:
            raise ParticipantVisibleError(f"episode {episode}: {error}") from None
        try:
            if _METRIC == "interpenetration_cm":
                series = {_METRIC: _episode_penetration(predicted, scene, reference)}
            else:
                                                                                             
                series = _episode_metrics(predicted, reference, frames, None)
        except (ValueError, KeyError) as error:
            raise ParticipantVisibleError(f"episode {episode}: {error}") from None
        values.append(float(series[_METRIC]))

    if not values:
        raise ValueError(f"no episode produced a {_METRIC_DISPLAY} value")
    result = float(np.mean(values))
    if not np.isfinite(result):
        raise ValueError(f"computed {_METRIC_DISPLAY} is not finite: {result}")
    return result
