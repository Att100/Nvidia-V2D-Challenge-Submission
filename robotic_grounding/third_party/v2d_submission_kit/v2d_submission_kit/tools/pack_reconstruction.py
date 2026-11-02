#!/usr/bin/env python3
"""Pack a reconstruction into a Kaggle submission: Track 1, or Track 3's object meshes (CD-O).

Both write Parquet (``row_id, x, y, z, code_commit_url``) holding exactly the rows of the
competition's ``sample_submission.parquet``, which is where the scored episodes, frames and
objects come from. Object meshes go through the same fixed face/vertex budget the host's reference
meshes do (``v2dlb/mesh_budget.py``), so pass your full-resolution mesh; it is decimated and padded
here.

TRACK 1 -- MHR parameters plus the object's mesh and Sim(3) pose, one file per episode:

    python3 tools/pack_reconstruction.py track1 \\
        --episodes <dir> --sample sample_submission.parquet \\
        --commit https://github.com/<owner>/<repo>/commit/<sha> --out submission.parquet

  ``<dir>/episode_000003.npz`` holds, indexed by VIDEO FRAME (row t is frame t; include at least
  every scored frame):

      pose                [T, 136]   mhr_model_params[:, :136]: root translation, root rotation,
                                     body pose (tools/track1/mesh_to_mhr_params.py writes this)
      scales              [68]       mhr_model_params[:, 136:], constant over the episode
      shape               [45]       identity shape parameters, constant over the episode
      object_rotation     [T, 3, 3]  R in  point_scene = s * R @ point_mesh + t
      object_translation  [T, 3]     t
      object_scale        ()         s, one per episode

  and the object mesh as ``<dir>/episode_000003_object.{glb,obj,ply,stl,off}``.

TRACK 3 CD-O -- one reconstructed mesh per (episode, object):

    python3 tools/pack_reconstruction.py track3-meshes \\
        --meshes <dir> --sample sample_submission.parquet \\
        --commit https://github.com/<owner>/<repo>/commit/<sha> --out submission.parquet

  ``<dir>/episode_000003/<object_name>.{glb,obj,ply,stl,off}``, one file per object in the
  episode. The sample submission's row ids name every object it scores, and each is matched to
  the file of that name.

Needs numpy, pandas, pyarrow, trimesh and fast_simplification.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v2dlb import report_schema as rs
from v2dlb.mesh_budget import budget_mesh
from v2dlb.mesh_quality import report_mesh_quality

MESH_SUFFIXES = (".glb", ".obj", ".ply", ".stl", ".off")


def fail(message: str) -> None:
    raise SystemExit(f"pack_reconstruction: {message}")


def read_sample(path: Path) -> pd.DataFrame:
    sample = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "row_id" not in sample.columns:
        fail(f"{path} has no row_id column; pass the competition's sample submission")
    return sample[["row_id"]].astype({"row_id": str})


def find_mesh(stem: Path) -> Path:
    for suffix in MESH_SUFFIXES:
        if stem.with_suffix(suffix).exists():
            return stem.with_suffix(suffix)
    fail(f"no mesh at {stem}.{{{','.join(s[1:] for s in MESH_SUFFIXES)}}}")


def budget_of(sample_keys: pd.DataFrame, faces_array: int, vertices_array: int) -> tuple[int, int]:
    """The face and vertex budget, read from the sample's own row counts."""
    first = sample_keys.iloc[0]
    one = sample_keys
    for column in [c for c in sample_keys.columns if c not in ("array", "point")]:
        one = one[one[column] == first[column]]
    faces = int((one["array"] == faces_array).sum())
    vertices = int((one["array"] == vertices_array).sum())
    if not faces or not vertices:
        fail("the sample submission carries no object mesh rows")
    return faces, vertices


def write(rows: pd.DataFrame, sample: pd.DataFrame, commit: str, out: Path) -> None:
    if not re.match(rs.CODE_COMMIT_PATTERN, commit):
        fail("--commit must be https://github.com/<owner>/<repo>/commit/<40-hex sha>")
    merged = sample.merge(rows, on="row_id", how="left", validate="one_to_one")
    if merged[["x", "y", "z"]].isna().any().any():
        missing = merged.loc[merged[["x", "y", "z"]].isna().any(axis=1), "row_id"]
        fail(f"{len(missing):,} required row(s) were not produced, e.g. {list(missing[:3])}")
    if not np.isfinite(merged[["x", "y", "z"]].to_numpy()).all():
        fail("some values are not finite")
    merged["code_commit_url"] = commit
    out.parent.mkdir(parents=True, exist_ok=True)
    merged[["row_id", "x", "y", "z", "code_commit_url"]].to_parquet(out, index=False)
    print(f"wrote {out}: {len(merged):,} rows, {out.stat().st_size / 1e6:.1f} MB")


# --- Track 1 ----------------------------------------------------------------------------------


def pack_track1(args) -> None:
    from v2dlb import mhr_submission as ms

    sample = read_sample(args.sample)
    parts = sample["row_id"].str.split("_", expand=True)
    if parts.shape[1] != 5 or not (parts[0] == "t1").all():
        fail("this is not a Track 1 sample submission (row ids should start t1_)")
    keys = pd.DataFrame({"episode": parts[1].astype(int), "frame": parts[2].astype(int),
                         "array": parts[3].astype(int), "point": parts[4].astype(int)})
    order = ms.array_order()
    faces_budget, vertices_budget = budget_of(keys.drop(columns="frame"),
                                              order.index("object_mesh_faces"),
                                              order.index("object_mesh_vertices"))

    values = np.full((len(sample), 3), np.nan)
    for episode, group in keys.groupby("episode"):
        path = args.episodes / f"episode_{episode:06d}.npz"
        if not path.exists():
            fail(f"missing {path}")
        data = dict(np.load(path))
        needed = ("pose", "scales", "shape", "object_rotation", "object_translation", "object_scale")
        absent = [k for k in needed if k not in data]
        if absent:
            fail(f"{path.name} lacks {absent}")
        frames = sorted(group.loc[group["frame"] != ms.EPISODE_FRAME, "frame"].unique())
        if frames[-1] >= len(data["pose"]) or frames[-1] >= len(data["object_rotation"]):
            fail(f"{path.name}: frame {frames[-1]} is scored but the arrays stop at "
                 f"{min(len(data['pose']), len(data['object_rotation'])) - 1}")
        pose = np.asarray(data["pose"], float)
        if pose.shape[1:] != (136,) or np.shape(data["scales"]) != (68,) or np.shape(data["shape"]) != (45,):
            fail(f"{path.name}: expected pose [T,136], scales [68], shape [45]")
        rotation = np.asarray(data["object_rotation"], float)
        try:
            ms._rotations(rotation[frames])
        except ValueError as error:
            fail(f"{path.name}: {error}")
        scale = float(np.asarray(data["object_scale"]).reshape(()))
        if not scale > 0:
            fail(f"{path.name}: object_scale must be positive")
        mesh_v, mesh_f = budget_mesh(find_mesh(args.episodes / f"episode_{episode:06d}_object"),
                                     faces_budget, vertices_budget)
        report_mesh_quality({(episode,0):(mesh_v,mesh_f)})
        per_episode = {
            "mhr_scales": ms_pack(data["scales"], ms.PARAMETER_COUNTS["mhr_scales"]),
            "mhr_shape": ms_pack(data["shape"], ms.PARAMETER_COUNTS["mhr_shape"]),
            "object_scale": np.array([[scale, 0.0, 0.0]]),
            "object_mesh_vertices": mesh_v,
            "object_mesh_faces": mesh_f.astype(float),
        }
        per_frame = {
            "mhr_pose": lambda f: ms_pack(pose[f], ms.PARAMETER_COUNTS["mhr_pose"]),
            "object_rotation": lambda f: rotation[f],
            "object_translation": lambda f: np.asarray(data["object_translation"], float)[f][None],
        }
        for (frame, array), rows in group.groupby(["frame", "array"]):
            name = order[array]
            block = per_episode[name] if frame == ms.EPISODE_FRAME else per_frame[name](frame)
            values[rows.index] = block[rows["point"].to_numpy()]
        print(f"  episode {episode}: {len(frames)} scored frames")
    write(pd.DataFrame({"row_id": sample["row_id"], "x": values[:, 0], "y": values[:, 1],
                        "z": values[:, 2]}), sample, args.commit, args.out)


def ms_pack(values, count: int) -> np.ndarray:
    """``count`` numbers three to a row, the unused tail of the last row zero."""
    flat = np.zeros(3 * -(-count // 3))
    flat[:count] = np.asarray(values, dtype=float).reshape(count)
    return flat.reshape(-1, 3)


# --- Track 3 CD-O -----------------------------------------------------------------------------


def pack_track3_meshes(args) -> None:
    from v2dlb import object_mesh as om

    sample = read_sample(args.sample)
    parts = sample["row_id"].str.extract(om.ROW_ID_PATTERN)
    if parts.isna().any().any():
        fail("this is not a Track 3 CD-O sample submission (row ids should look like "
             "t3m_000003_<object>_<array>_<point>)")
    keys = pd.DataFrame({"episode": parts[0].astype(int), "object": parts[1],
                         "array": parts[2].astype(int), "point": parts[3].astype(int)})
    faces_index = om.ARRAYS.index("object_mesh_faces")
    vertices_index = om.ARRAYS.index("object_mesh_vertices")
    faces_budget, vertices_budget = budget_of(keys, faces_index, vertices_index)

    values = np.full((len(sample), 3), np.nan)
    for (episode, name), group in keys.groupby(["episode", "object"]):
        path = find_mesh(args.meshes / f"episode_{episode:06d}" / name)
        mesh_v, mesh_f = budget_mesh(path, faces_budget, vertices_budget)
        report_mesh_quality({(episode,0):(mesh_v,mesh_f)}, {(episode,0):name})
        for array, block in ((vertices_index, mesh_v), (faces_index, mesh_f.astype(float))):
            rows = group[group["array"] == array]
            values[rows.index] = block[rows["point"].to_numpy()]
    for episode, group in keys.groupby("episode"):
        print(f"  episode {episode}: {', '.join(sorted(group['object'].unique()))}")
    write(pd.DataFrame({"row_id": sample["row_id"], "x": values[:, 0], "y": values[:, 1],
                        "z": values[:, 2]}), sample, args.commit, args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("track1", help="MHR parameters + object mesh and Sim(3) pose")
    one.add_argument("--episodes", type=Path, required=True)
    three = commands.add_parser("track3-meshes", help="one reconstructed mesh per (episode, object)")
    three.add_argument("--meshes", type=Path, required=True)
    for command in (one, three):
        command.add_argument("--sample", type=Path, required=True,
                             help="the competition's sample_submission.parquet")
        command.add_argument("--commit", required=True,
                             help="https://github.com/<owner>/<repo>/commit/<sha> of your code")
        command.add_argument("--out", type=Path, default=Path("submission.parquet"))
    args = parser.parse_args()
    (pack_track1 if args.command == "track1" else pack_track3_meshes)(args)


if __name__ == "__main__":
    main()
