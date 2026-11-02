"""Kaggle metric scoring code. Reference arrays and embedded model or vertex assets are omitted from this participant copy."""

import base64
import io
import re

import numpy as np
import pandas as pd
import pandas.api.types


class ParticipantVisibleError(Exception):
    pass


                                                                               
                                                                    
                                               
_ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)
_SPIDER_POSITION_THRESHOLD_M = 0.1
_SPIDER_ORIENTATION_THRESHOLD_RAD = 0.5
_MANIPTRANS_POSITION_THRESHOLD_M = 0.03
_MANIPTRANS_ORIENTATION_THRESHOLD_DEG = 30.0
                                                                           
                                                                                              
                                                                                                 
                                                                                                
                                                                                           
                                                                               
_MPPE_KEYPOINT_RADIUS_M = 0.05

                                                                                            
                                                                                            
_MPPE_KEYPOINT_DIRECTIONS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)

                                                         
_POSE_COLUMNS = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
                                                                          
_REF_COLUMNS = ["ref_" + _c for _c in _POSE_COLUMNS]


                                                                                        

def _rotation_matrix(quat_xyzw: np.ndarray) -> np.ndarray:

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

                                                                           
    dot = np.clip(np.abs(np.sum(a_xyzw * b_xyzw, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot)


def _object_add(achieved: np.ndarray, reference: np.ndarray, vertices: np.ndarray) -> np.ndarray:

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

    axis = np.linspace(0.0, 1.0, _ADD_THRESHOLDS_M.size)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    per_body = [
        float(trapezoid(np.array([np.mean(add[..., body] < t) for t in _ADD_THRESHOLDS_M]), x=axis))
        for body in range(add.shape[2])
    ]
    return float(np.mean(per_body))


def _quaternion_multiply(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:

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

    reference_start = reference[0, 0, 0]                                                      
    achieved_start = achieved[0, :, 0]                                

    conjugate = achieved_start[:, 3:] * np.array([-1.0, -1.0, -1.0, 1.0])
    q_align = _quaternion_multiply(reference_start[None, 3:], conjugate)                  
    q_align = q_align / np.linalg.norm(q_align, axis=-1, keepdims=True)
    rotation = _rotation_matrix(q_align)                                                     
    offset = reference_start[None, :3] - np.einsum("wij,wj->wi", rotation, achieved_start[:, :3])

    warped = np.empty_like(achieved)
    warped[..., :3] = np.einsum("wij,twbj->twbi", rotation, achieved[..., :3]) + offset[None, :, None, :]
    warped[..., 3:] = _quaternion_multiply(q_align[None, :, None, :], achieved[..., 3:])
    return warped


def _per_world_pose_error(achieved: np.ndarray, reference: np.ndarray):

    position = np.linalg.norm(achieved[..., :3] - reference[..., :3], axis=-1).mean(axis=0)
    orientation = _quaternion_geodesic_angle(achieved[..., 3:], reference[..., 3:]).mean(axis=0)
    return position, orientation


def _spider_success(achieved: np.ndarray, reference: np.ndarray) -> np.ndarray:

    position, orientation = _per_world_pose_error(achieved, reference)
    return (position.mean(axis=-1) <= _SPIDER_POSITION_THRESHOLD_M) & (
        orientation.mean(axis=-1) <= _SPIDER_ORIENTATION_THRESHOLD_RAD
    )


def _maniptrans_success(achieved: np.ndarray, reference: np.ndarray, object_ids: np.ndarray) -> np.ndarray:

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

    offsets = _MPPE_KEYPOINT_DIRECTIONS * _MPPE_KEYPOINT_RADIUS_M             
    rotation_a = _rotation_matrix(achieved[..., 3:])                                   
    rotation_r = _rotation_matrix(reference[..., 3:])                                  
                                                                                      
    keypoints_a = achieved[..., None, :3] + np.einsum("...ij,kj->...ki", rotation_a, offsets)
    keypoints_r = reference[..., None, :3] + np.einsum("...ij,kj->...ki", rotation_r, offsets)
    return float(np.linalg.norm(keypoints_a - keypoints_r, axis=-1).mean() * 100.0)


                                                                                        

def _episode_tensors(frame: pd.DataFrame):

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
                                                                               
    names = frame["object_name"].to_numpy().reshape(*shape)[0, 0]
    return achieved, reference[:, :1], slots.astype(np.int64), names


def _normalise_quaternions(pose: np.ndarray, who: str, error_type) -> np.ndarray:

    pose = pose.copy()
    norms = np.linalg.norm(pose[..., 3:], axis=-1)
    if np.any(norms < 1e-8):
        raise error_type(f"{who} contains a degenerate (near-zero-norm) quaternion")
    pose[..., 3:] /= norms[..., None]
    return pose


                                                                                        
_TRACK = "track_2_tier1"
_ALIGN_INITIAL_POSE = False
_METRIC = "add_auc"
_METRIC_DISPLAY = "AUC"
_ACCEPTED = {'eval_script_sha256': [], 'eval_code_sha256': []}
                                                                                        

_PROVENANCE_COLUMNS = ['eval_script_sha256', 'eval_code_sha256', 'checkpoint_sha256', 'packer_script_sha256', 'uploader_script_sha256', 'code_commit_url']
_UNIFORM_PROVENANCE_COLUMNS = ['eval_script_sha256', 'eval_code_sha256', 'packer_script_sha256', 'uploader_script_sha256', 'code_commit_url']
_PER_EPISODE_PROVENANCE_COLUMNS = ['checkpoint_sha256']
_CODE_COMMIT_PATTERN = '^https://[\\w.-]*github[\\w.-]*/[\\w.-]+/[\\w.-]+/commit/[0-9a-fA-F]{7,40}/?$'
_REQUIRED_SUBMISSION_COLUMNS = _POSE_COLUMNS + _PROVENANCE_COLUMNS
_REQUIRED_SOLUTION_COLUMNS = (
    ["episode_index", "world_index", "frame_index", "object_slot", "object_name"] + _REF_COLUMNS
)
_USAGE_COLUMN = "Usage"

                                                                                         
                                                                                
_VERTICES_B64 = ""


def _load_vertices(solution: pd.DataFrame | None = None) -> dict:

    blob = _VERTICES_B64.strip()
    if not blob and solution is not None and _METRIC == "add_auc":
        column = "auc_vertex_samples_b64"
        if column not in solution.columns:
            raise ValueError(f"solution is missing '{column}'")
        nonempty = solution[column].dropna().astype(str).str.strip()
        nonempty = nonempty[nonempty != ""]
        if len(nonempty) != 1:
            raise ValueError(f"solution must carry exactly one '{column}' payload")
        blob = nonempty.iloc[0]
    if not blob:
        return {}
    try:
        with np.load(io.BytesIO(base64.b64decode("".join(blob.split()), validate=True))) as archive:
            return {name: archive[name].astype(np.float64) for name in archive.files}
    except Exception as error:
        raise ValueError("solution or metric has invalid AUC vertex payload") from error


_VERTICES = _load_vertices()


                                                                                         

def _check_solution(solution: pd.DataFrame, row_id_column_name: str) -> None:
    if row_id_column_name not in solution.columns:
        raise ValueError(f"solution is missing the '{row_id_column_name}' row-id column")
    missing = [c for c in _REQUIRED_SOLUTION_COLUMNS if c not in solution.columns]
    if missing:
        raise ValueError(f"solution is missing required column(s): {missing}")
    if len(solution) == 0:
        raise ValueError("solution has no rows")
    if solution[row_id_column_name].duplicated().any():
        raise ValueError(f"solution has duplicate '{row_id_column_name}' values")
    if solution[row_id_column_name].isnull().any():
        raise ValueError(f"solution has null '{row_id_column_name}' values")
    for column in ["episode_index", "world_index", "frame_index", "object_slot"]:
        if not pandas.api.types.is_numeric_dtype(solution[column]) or solution[column].isnull().any():
            raise ValueError(f"solution column '{column}' must be numeric and non-null")
    reference = solution[_REF_COLUMNS].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(reference)):
        raise ValueError("solution contains a non-finite reference pose value")
    if solution["object_name"].isnull().any():
        raise ValueError("solution contains a null object_name")
    unknown = sorted(set(solution["object_name"].astype(str)) - set(_VERTICES)) if _VERTICES else []
    if unknown:
        raise ValueError(
            f"solution references object(s) with no embedded vertices: {unknown}. "
            "Regenerate the metric with metrics/build_kaggle_metrics.py."
        )


def _check_submission(submission: pd.DataFrame, row_id_column_name: str) -> None:
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"submission is missing the '{row_id_column_name}' column")
    for column in _REQUIRED_SUBMISSION_COLUMNS:
        if column not in submission.columns:
            raise ParticipantVisibleError(f"submission is missing required column '{column}'")
    for column in _POSE_COLUMNS:
        if not pandas.api.types.is_numeric_dtype(submission[column]):
            raise ParticipantVisibleError(f"submission column '{column}' must be numeric")
        values = submission[column].to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.all(np.isfinite(values)):
            raise ParticipantVisibleError(
                f"submission column '{column}' contains a null, NaN or infinite value"
            )
    if submission[row_id_column_name].duplicated().any():
        first = submission.loc[submission[row_id_column_name].duplicated(), row_id_column_name].iloc[0]
        raise ParticipantVisibleError(
            f"submission has duplicate '{row_id_column_name}' values, first is '{first}'"
        )


def _check_provenance(merged: pd.DataFrame) -> None:

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
        per_episode = merged.groupby("episode_index")[column].apply(
            lambda block: block.astype(str).str.strip().nunique()
        )
        offenders = sorted(per_episode[per_episode > 1].index.tolist())
        if offenders:
            raise ParticipantVisibleError(
                f"'{column}' must be identical within each episode; episode(s) "
                f"{offenders[:5]} carry more than one value. One episode is one set of weights, "
                "but different episodes may use different ones."
            )

                                                                                               
                                                                                                 
                                                                                                 
                                
                                                                                            
    commit = merged["code_commit_url"].astype(str).str.strip().iloc[0]
    if not re.match(_CODE_COMMIT_PATTERN, commit):
        raise ParticipantVisibleError(
            f"'code_commit_url' must be a GitHub commit URL, for example "
            f"https://github.com/<owner>/<repo>/commit/<sha>. Got '{commit}'. A branch or tag URL "
            "is not accepted, because it does not pin the code that produced this submission."
        )

    for column, accepted in _ACCEPTED.items():
        if not accepted:
            continue
        reported = merged[column].astype(str).str.strip().iloc[0].lower()
        if reported not in accepted:
            raise ParticipantVisibleError(
                f"{column} '{reported}' is not a released version of the evaluator. "
                "Re-run with the official evaluate_policy.py and resubmit."
            )


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:

    if _USAGE_COLUMN in solution.columns:
        solution = solution.drop(columns=[_USAGE_COLUMN])

    _check_solution(solution, row_id_column_name)
    vertices_by_name = _VERTICES if _VERTICES else _load_vertices(solution)
    if _METRIC == "add_auc":
        missing_names = sorted(set(solution["object_name"].astype(str)) - set(vertices_by_name))
        if missing_names:
            raise ValueError(f"solution has no AUC vertices for object(s): {missing_names}")
    _check_submission(submission, row_id_column_name)

                                                                                              
                                                                                                 
                                                           
    solution = solution.copy()
    submission = submission.copy()
    solution[row_id_column_name] = solution[row_id_column_name].astype(str)
    submission[row_id_column_name] = submission[row_id_column_name].astype(str)

    keep = [row_id_column_name, *_REQUIRED_SUBMISSION_COLUMNS]
    merged = solution.merge(
        submission[keep], on=row_id_column_name, how="left", validate="one_to_one"
    )
    missing = merged["pos_x"].isnull()
    if missing.any():
        first = merged.loc[missing, row_id_column_name].iloc[0]
        raise ParticipantVisibleError(
            f"submission is missing {int(missing.sum())} required row(s); first is '{first}'. "
            "Every scored frame of every scored episode must be present."
        )

    _check_provenance(merged)

    values = []
    for episode, group in merged.groupby("episode_index", sort=True):
        try:
            achieved, reference, object_ids, names = _episode_tensors(group)
        except ValueError as error:
                                                                                             
                                                            
            raise ParticipantVisibleError(f"episode {int(episode)}: {error}") from error

        achieved = _normalise_quaternions(achieved, "submission", ParticipantVisibleError)
        reference = _normalise_quaternions(reference, "solution", ValueError)

                                                                                         
                                                                                            
                                                                                     
        if _ALIGN_INITIAL_POSE:
            achieved = _warp_to_reference_start(achieved, reference)

        if _METRIC == "add_auc":
            vertices = np.stack([vertices_by_name[str(name)] for name in names])
            values.append(_add_auc(_object_add(achieved, reference, vertices)))
        elif _METRIC == "spider_sr":
            values.append(float(np.mean(_spider_success(achieved, reference))))
        elif _METRIC == "maniptrans_sr":
            values.append(float(np.mean(_maniptrans_success(achieved, reference, object_ids))))
        elif _METRIC == "rpe_cm":
                                                                                                  
            values.append(_relative_position_error_cm(achieved, reference))
        elif _METRIC == "mppe_cm":
            values.append(_mean_per_point_error_cm(achieved, reference))
        else:
            raise ValueError(f"unknown metric {_METRIC!r}")

    if not values:
        raise ValueError(f"no episode produced a {_METRIC_DISPLAY} value")
    result = float(np.mean(values))
    if not np.isfinite(result):
        raise ValueError(f"computed {_METRIC_DISPLAY} is not finite: {result}")
    return result
