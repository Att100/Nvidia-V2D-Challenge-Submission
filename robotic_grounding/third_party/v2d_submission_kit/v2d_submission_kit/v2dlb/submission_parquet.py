"""Read the object-trajectory parquet that evaluate_policy.py writes.

Produced by ``flash_chord.evaluation.rollout.write_object_trajectories_parquet`` and identified
by the metadata key ``flash_chord.schema = "flash_chord_object_trajectories_v1"``.

    environment_id        int32    parallel evaluation world
    step                  int32    0-based rollout step
    reference_frame       int32    step + start_frame
    time_s                float64  step / control_fps
    object_id             int32    which tracked object a body belongs to (bodies group by this)
    object_body_id        int32    rigid body, in reference-body order
    object_body_name      dict<str>
    environment_valid     bool     false where the world recorded a non-finite pose
    achieved_position_{x,y,z}, achieved_quaternion_{x,y,z,w}      float32
    reference_position_{x,y,z}, reference_quaternion_{x,y,z,w}    float32

Rows are ordered step, environment, body. Positions are metres and quaternions are xyzw, both
declared in the metadata rather than assumed.

The file carries the REFERENCE as well as the achieved pose. That is not used for scoring -- the
host scores against its own reference, or a participant could hand in a reference matching
whatever they rolled out -- but it is used to check that the two agree, which is the cheapest way
to catch a frame or convention mismatch between the evaluator and the released dataset.

An older, simpler layout (plain ``pos_x``/``quat_x`` columns, or one width-7 list column) is also
accepted so the packer keeps working against hand-made fixtures and any earlier exports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_EPISODE = re.compile(r"episode_(\d+)")
_SCHEMA_V1 = "flash_chord_object_trajectories_v1"
_POSE_SUFFIXES = ("position_x", "position_y", "position_z",
                  "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")
_LEGACY_POSE = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
_LEGACY_ARRAY = ("pose", "achieved_pose_w", "pose_xyzw")


class TrajectoryError(ValueError):
    """The trajectory parquet does not meet the contract."""


@dataclass
class SubmittedTrajectory:
    episode_index: int
    world_index: np.ndarray             # (N,) int64
    frame_index: np.ndarray             # (N,) int64  (reference_frame)
    object_slot: np.ndarray             # (N,) int64  (object_body_id)
    object_id: np.ndarray               # (N,) int64  (which object a body belongs to)
    object_name: np.ndarray | None      # (N,) str
    pose_xyzw: np.ndarray               # (N, 7) float64, achieved
    reference_xyzw: np.ndarray | None   # (N, 7) float64, as recorded by the evaluator
    world_valid: np.ndarray             # (N,) bool
    metadata: dict = field(default_factory=dict)
    source: str = ""

    @property
    def schema_version(self) -> str:
        return self.metadata.get("flash_chord.schema", "legacy")


def _metadata(table) -> dict:
    raw = table.schema.metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}


def _column(table, name):
    return table[name].to_numpy(zero_copy_only=False)


def _stack(table, prefix: str, path: Path) -> np.ndarray:
    """The seven pose columns for one prefix, or a message naming exactly which are absent."""
    missing = [f"{prefix}_{s}" for s in _POSE_SUFFIXES if f"{prefix}_{s}" not in table.column_names]
    if missing:
        raise TrajectoryError(
            f"{path.name}: the flash_chord schema is only partly present; missing {missing}. "
            "All seven position and quaternion columns are required for each of 'achieved' and "
            "'reference'."
        )
    return np.stack([_column(table, f"{prefix}_{s}") for s in _POSE_SUFFIXES], axis=-1).astype(np.float64)


def _episode_from(metadata: dict, path: Path) -> int:
    for candidate in (
        metadata.get("flash_chord.episode_index"),
        metadata.get("episode_index"),
    ):
        if candidate not in (None, ""):
            return int(candidate)
    for candidate in (
        str(metadata.get("flash_chord.reference_parquet", "")),
        str(metadata.get("reference_parquet", "")),
        path.name,
    ):
        match = _EPISODE.search(candidate)
        if match:
            return int(match.group(1))
    raise TrajectoryError(
        f"{path.name}: cannot tell which episode this is. Either name the file "
        "episode_<index>.parquet, or record 'flash_chord.episode_index' (or "
        "'flash_chord.reference_parquet') in the parquet's schema metadata."
    )


def read_trajectory(path: Path) -> SubmittedTrajectory:
    import pyarrow.parquet as pq

    path = Path(path)
    try:
        table = pq.read_table(path)
    except Exception as error:  # noqa: BLE001
        raise TrajectoryError(f"{path.name}: could not be read as parquet ({error})") from error

    metadata = _metadata(table)
    columns = set(table.column_names)
    rows = table.num_rows
    if rows == 0:
        raise TrajectoryError(f"{path.name}: has no rows")

    quaternion_order = (
        metadata.get("flash_chord.quaternion_order")
        or metadata.get("quaternion_layout")
        or "xyzw"
    ).lower()
    if quaternion_order not in ("xyzw", "wxyz"):
        raise TrajectoryError(
            f"{path.name}: unknown quaternion order {quaternion_order!r}; expected xyzw or wxyz"
        )

    if any(c.startswith("achieved_") for c in columns):
        pose = _stack(table, "achieved", path)
        reference = (_stack(table, "reference", path)
                     if any(c.startswith("reference_position") for c in columns) else None)
        for required in ("environment_id", "object_body_id"):
            if required not in columns:
                raise TrajectoryError(
                    f"{path.name}: the flash_chord schema requires a '{required}' column"
                )
        frame = np.asarray(
            _column(table, "reference_frame" if "reference_frame" in columns else "step"), dtype=np.int64
        )
        worlds = np.asarray(_column(table, "environment_id"), dtype=np.int64)
        slots = np.asarray(_column(table, "object_body_id"), dtype=np.int64)
        object_id = (np.asarray(_column(table, "object_id"), dtype=np.int64)
                     if "object_id" in columns else slots.copy())
        names = (np.asarray(table["object_body_name"].to_pylist(), dtype=object)
                 if "object_body_name" in columns else None)
        valid = (np.asarray(_column(table, "environment_valid"), dtype=bool)
                 if "environment_valid" in columns else np.ones(rows, dtype=bool))
    else:
        array_column = next((c for c in _LEGACY_ARRAY if c in columns), None)
        if array_column is not None:
            flat = table[array_column].combine_chunks().flatten().to_numpy()
            pose = np.asarray(flat, dtype=np.float64).reshape(rows, -1)
            if pose.shape[1] != 7:
                raise TrajectoryError(
                    f"{path.name}: '{array_column}' has width {pose.shape[1]}, expected 7"
                )
        else:
            missing = [c for c in _LEGACY_POSE if c not in columns]
            if missing:
                raise TrajectoryError(
                    f"{path.name}: unrecognised layout. Expected the flash_chord schema "
                    f"('achieved_position_x', ...), or the seven columns {_LEGACY_POSE}, or one "
                    f"width-7 list column named one of {list(_LEGACY_ARRAY)}. Missing: {missing}."
                )
            pose = np.stack([_column(table, c) for c in _LEGACY_POSE], axis=-1).astype(np.float64)
        reference = None
        if "frame_index" not in columns:
            raise TrajectoryError(f"{path.name}: missing the required 'frame_index' column")
        frame = np.asarray(_column(table, "frame_index"), dtype=np.int64)
        worlds = (np.asarray(_column(table, "world_index"), dtype=np.int64)
                  if "world_index" in columns else np.zeros(rows, dtype=np.int64))
        if "object_slot" in columns:
            slots = np.asarray(_column(table, "object_slot"), dtype=np.int64)
        elif "object_name" in columns:
            seen, order = {}, []
            for name in np.asarray(_column(table, "object_name"), dtype=object):
                seen.setdefault(name, len(seen))
                order.append(seen[name])
            slots = np.asarray(order, dtype=np.int64)
        else:
            raise TrajectoryError(f"{path.name}: needs an 'object_slot' or an 'object_name' column")
        object_id = slots.copy()
        names = (np.asarray(_column(table, "object_name"), dtype=object)
                 if "object_name" in columns else None)
        valid = np.ones(rows, dtype=bool)

    if quaternion_order == "wxyz":
        def to_xyzw(a):
            return None if a is None else np.concatenate([a[:, :3], a[:, 4:7], a[:, 3:4]], axis=1)

        pose, reference = to_xyzw(pose), to_xyzw(reference)

    if not np.all(np.isfinite(pose)):
        raise TrajectoryError(
            f"{path.name}: contains a NaN or infinite achieved pose. The evaluator sanitises "
            "non-finite worlds to zero and flags them with environment_valid=false; a NaN here "
            "means something else went wrong."
        )

    return SubmittedTrajectory(
        episode_index=_episode_from(metadata, path),
        world_index=worlds,
        frame_index=frame,
        object_slot=slots,
        object_id=object_id,
        object_name=names,
        pose_xyzw=pose,
        reference_xyzw=reference,
        world_valid=valid,
        metadata=metadata,
        source=str(path),
    )


def provenance(trajectory: SubmittedTrajectory) -> dict:
    """Hashes recorded in the parquet metadata, under either the flash_chord or bare key."""
    out = {}
    for field_name in ("eval_script_sha256", "eval_code_sha256", "checkpoint_sha256"):
        value = (
            trajectory.metadata.get(f"flash_chord.{field_name}")
            or trajectory.metadata.get(field_name)
            or ""
        ).strip()
        if value:
            out[field_name] = value
    return out


def write_example(path: Path, columns, *, episode_index: int, worlds: int = 1,
                  eval_hash: str = "", code_hash: str = "", checkpoint_hash: str = "",
                  quaternion_order: str = "xyzw") -> Path:
    """Write a contract-conforming trajectory parquet. Used by the tests as a stand-in export."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({name: np.asarray(values) for name, values in columns.items()})
    table = table.replace_schema_metadata({
        "flash_chord.schema": _SCHEMA_V1,
        "flash_chord.episode_index": str(episode_index),
        "flash_chord.world_count": str(worlds),
        "flash_chord.quaternion_order": quaternion_order,
        "flash_chord.position_unit": "meter",
        "flash_chord.eval_script_sha256": eval_hash,
        "flash_chord.eval_code_sha256": code_hash,
        "flash_chord.checkpoint_sha256": checkpoint_hash,
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return Path(path)
