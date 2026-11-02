"""Read episode rosters out of the local HuggingFace dataset mirror.

The mirror is produced by ``tools/mirror_hf_dataset.py`` and holds the parquets, meshes,
urdfs and meta of ``nvidia/video_to_data_challenge`` (no videos).

Episode indices are SPARSE for track_3 and dense for track_1/track_2, so always enumerate
them from ``meta/info.json`` -> ``episode_indices`` (falling back to ``meta/episodes.jsonl``
and then to the parquet filenames) rather than assuming ``range(total_episodes)``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_EPISODE_FILE = re.compile(r"episode_(\d+)\.parquet$")


def participant_reference_root(dataset_root: Path, spec: dict) -> Path:
    """Locate the clean Track 2 reference in either the mirror or participant release layout."""
    root = Path(dataset_root)
    candidates = [root / spec["data_root"]]
    if spec.get("tier"):
        candidates.append(root / "track_2" / "tier_1_multiview_caption")
    return next((path for path in candidates if (path / "meta").exists()), candidates[0])


def participant_input_reference_root(dataset_root: Path, spec: dict) -> Path:
    """Locate the selected Track 2 tier's released policy input."""
    folders = {
        "tier1": ("baseline", "tier_1_multiview_caption"),
        "tier2": ("synthetic_random_walk", "tier_2_synthetic_noise"),
        "tier3": ("cari4d_reconstruction", "tier_3_cari4d"),
    }
    tier = spec.get("tier", "tier1")
    if tier not in folders:
        raise ValueError(f"unknown Track 2 tier: {tier}")
    if tier == "tier1":
        return participant_reference_root(dataset_root, spec)
    candidates = [Path(dataset_root) / "track_2" / folder for folder in folders[tier]]
    return next((path for path in candidates if (path / "meta").exists()), candidates[0])


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def episode_indices(data_root: Path) -> list[int]:
    """Sorted episode_index values present under ``data_root``.

    ``data_root`` is a track root such as ``<mirror>/track_2`` or
    ``<mirror>/track_3/confidential_holdout``.
    """
    data_root = Path(data_root)
    info = data_root / "meta" / "info.json"
    if info.exists():
        meta = _read_json(info)
        declared = meta.get("episode_indices")
        if declared:
            return sorted(int(i) for i in declared)

    episodes = data_root / "meta" / "episodes.jsonl"
    if episodes.exists():
        out = []
        for line in episodes.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if "episode_index" in rec:
                    out.append(int(rec["episode_index"]))
        if out:
            return sorted(out)

    # Last resort: the parquet filenames. Reliable but silent about intent, so it is the
    # fallback rather than the primary path.
    out = []
    for parquet in (data_root / "data").rglob("episode_*.parquet"):
        match = _EPISODE_FILE.search(parquet.name)
        if match:
            out.append(int(match.group(1)))
    if not out:
        raise FileNotFoundError(f"no episodes found under {data_root}")
    return sorted(out)


def episode_lengths(data_root: Path) -> dict[int, int]:
    """episode_index -> frame count, from meta/episodes.jsonl when available."""
    data_root = Path(data_root)
    episodes = data_root / "meta" / "episodes.jsonl"
    lengths: dict[int, int] = {}
    if episodes.exists():
        for line in episodes.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "episode_index" in rec and "length" in rec:
                lengths[int(rec["episode_index"])] = int(rec["length"])
    return lengths


def usage_split(indices: list[int], private_fraction: float) -> dict[int, str]:
    """Assign each EPISODE to the Public or Private leaderboard.

    ``private_fraction == 0`` puts everything on the Public leaderboard. That is the V2D
    Challenge's configuration: every episode on HuggingFace is public data, and the real held-out
    evaluation happens offline on private data after the challenge closes. Kaggle is therefore
    only being asked to host a live public ranking, and a second hidden split inside data the
    participants already have would hide nothing.

    Otherwise the split unit is the episode, never the row: neighbouring frames of a 20-30 fps
    trajectory are millimetres apart, so a row-level split would leak almost perfectly.

    Deterministic and interleaved rather than random: walking the sorted episode list and taking
    every k-th one for Public spreads the two halves evenly across whatever ordering the dataset
    already has (object, subject, sequence length), and reproduces byte for byte on any machine
    without seeding an RNG.
    """
    if not 0.0 <= private_fraction < 1.0:
        raise ValueError(f"private_fraction must be in [0, 1), got {private_fraction}")
    ordered = sorted(indices)
    if private_fraction == 0.0:
        return {index: "Public" for index in ordered}
    n_public = max(1, round(len(ordered) * (1.0 - private_fraction)))
    n_public = min(n_public, len(ordered) - 1)
    stride = len(ordered) / n_public
    public = {ordered[min(len(ordered) - 1, int(i * stride + stride / 2))] for i in range(n_public)}
    return {index: ("Public" if index in public else "Private") for index in ordered}
