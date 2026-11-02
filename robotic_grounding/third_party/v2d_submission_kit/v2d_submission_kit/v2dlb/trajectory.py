"""Read reference object trajectories out of the challenge parquets.

Two layouts, one per track family:

  track_1 / track_2   flat: ``observation.object.pose`` is a fixed_size_list<float32, 7> and
                      ``observation.object.visible`` is a bool. Exactly one object per episode,
                      named in the parquet's schema metadata under ``objects``.
  track_3             nested: ``observation.objects`` is a list<struct{name, pose, visible}>,
                      1 or 2 entries, constant membership within an episode, sorted by the
                      producer (which is NOT always alphabetical, so slot order is taken from
                      the struct order and preserved).

QUATERNION LAYOUT. The parquets store ``[x, y, z, qw, qx, qy, qz]`` -- w-first. The evaluator and
therefore every metric in this repo use xyzw. The conversion happens here, once, and nowhere
else.

VISIBILITY. When ``visible`` is false the stored pose is all zeros, including a zero quaternion,
which is not a rotation. Those frames are dropped. A frame is dropped for the whole episode if
any of its objects is invisible, so the per-episode grid stays rectangular: the metric reshapes
the long CSV back into (frames, worlds, objects) and a hole would silently misalign it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpisodeTrajectory:
    """One episode's reference trajectory."""

    episode_index: int
    object_names: list[str]          # index = object_slot
    frame_index: np.ndarray          # (T,) int64, visible frames only
    pose_xyzw: np.ndarray            # (T, B, 7) float64, position metres + quaternion xyzw
    dropped_frames: int


def _wxyz_to_xyzw(pose: np.ndarray) -> np.ndarray:
    """(..., 7) [x,y,z,qw,qx,qy,qz] -> (..., 7) [x,y,z,qx,qy,qz,qw]."""
    out = np.empty_like(pose)
    out[..., :3] = pose[..., :3]
    out[..., 3:6] = pose[..., 4:7]
    out[..., 6] = pose[..., 3]
    return out


def _schema_metadata(table) -> dict:
    raw = table.schema.metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}


def read_episode(parquet_path: Path, episode_index: int) -> EpisodeTrajectory:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    columns = set(table.column_names)
    rows = table.num_rows
    frame_index = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64)

    if "observation.objects" in columns:
        nested = table["observation.objects"].to_pylist()
        names = [entry["name"] for entry in nested[0]]
        bodies = len(names)
        pose = np.empty((rows, bodies, 7), dtype=np.float64)
        visible = np.ones((rows, bodies), dtype=bool)
        for row, entries in enumerate(nested):
            if [e["name"] for e in entries] != names:
                raise ValueError(
                    f"{parquet_path}: object membership changes at row {row}; the metric assumes "
                    "it is constant within an episode"
                )
            for slot, entry in enumerate(entries):
                pose[row, slot] = entry["pose"]
                visible[row, slot] = bool(entry["visible"])
    elif "observation.object.pose" in columns:
        metadata = _schema_metadata(table)
        declared = json.loads(metadata.get("objects", "[]"))
        names = [declared[0]] if declared else ["object"]
        flat = table["observation.object.pose"].combine_chunks().flatten().to_numpy()
        pose = np.asarray(flat, dtype=np.float64).reshape(rows, 1, 7)
        visible = np.asarray(
            table["observation.object.visible"].to_numpy(zero_copy_only=False), dtype=bool
        ).reshape(rows, 1)
    else:
        raise ValueError(f"{parquet_path}: no recognised object-pose column")

    keep = visible.all(axis=1)
    return EpisodeTrajectory(
        episode_index=episode_index,
        object_names=names,
        frame_index=frame_index[keep],
        pose_xyzw=_wxyz_to_xyzw(pose[keep]),
        dropped_frames=int(rows - keep.sum()),
    )


def episode_parquet(data_root: Path, episode_index: int) -> Path:
    """The parquet for one episode, chunked the way LeRobot writes them."""
    data_root = Path(data_root)
    matches = sorted((data_root / "data").rglob(f"episode_{episode_index:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(f"no parquet for episode {episode_index} under {data_root/'data'}")
    return matches[0]


def read_track(data_root: Path, episode_indices: list[int]) -> list[EpisodeTrajectory]:
    return [read_episode(episode_parquet(data_root, index), index) for index in episode_indices]
