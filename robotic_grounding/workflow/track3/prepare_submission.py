#!/usr/bin/env python3
"""Normalize Track 3 trajectories and stage body-frame reconstruction meshes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# These legacy references predate the public object naming contract. Values map
# simulation object_body_id -> public Track 3 object_slot.
LEGACY_SLOT_MAP = {
    8: {0: 1, 1: 0},
    38: {0: 0},
}


def resolve_relative(path: Path, owner: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (owner.parent / expanded)


def checkpoint_reference(checkpoint: Path) -> Path:
    from safetensors import safe_open

    with safe_open(checkpoint, framework="numpy") as handle:
        metadata = handle.metadata() or {}
    if "user.resolved_config_json" not in metadata:
        raise RuntimeError(f"checkpoint has no resolved configuration: {checkpoint}")
    config = json.loads(metadata["user.resolved_config_json"])
    reference = Path(config["task"]["parquet"])
    if reference.exists():
        return reference.resolve()
    candidate = checkpoint.parent / reference
    return candidate.resolve() if candidate.exists() else reference


def resolve_mesh(path: Path, search_roots: list[Path]) -> Path:
    expanded = path.expanduser()
    if expanded.is_file():
        return expanded.resolve()
    matches: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        matches.extend(candidate for candidate in root.rglob(path.name) if candidate.is_file())
    unique = sorted(set(candidate.resolve() for candidate in matches))
    if len(unique) != 1:
        raise RuntimeError(f"cannot uniquely remap missing mesh {path}: {unique}")
    return unique[0]


def reference_contract(
    checkpoint: Path,
    public_names: list[str],
    episode: int,
    search_roots: list[Path],
) -> tuple[list[str], list[Path], dict[int, int]]:
    import pyarrow.parquet as pq

    reference = checkpoint_reference(checkpoint)
    row = pq.read_table(
        reference,
        columns=["object_body_names", "object_mesh_paths"],
    ).slice(0, 1).to_pylist()[0]
    internal_names = list(row["object_body_names"])
    mesh_paths = [resolve_mesh(Path(value), search_roots) for value in row["object_mesh_paths"]]
    if episode in LEGACY_SLOT_MAP:
        slot_map = LEGACY_SLOT_MAP[episode]
    elif internal_names == public_names:
        slot_map = {index: index for index in range(len(public_names))}
    else:
        raise RuntimeError(
            f"episode {episode}: internal object order {internal_names} does not match {public_names}"
        )
    if (
        set(slot_map) != set(range(len(internal_names)))
        or set(slot_map.values()) != set(range(len(public_names)))
    ):
        raise RuntimeError(f"episode {episode}: invalid slot map {slot_map}")
    return internal_names, mesh_paths, slot_map


def stage_meshes(
    output_root: Path,
    episode: int,
    public_names: list[str],
    mesh_paths: list[Path],
    slot_map: dict[int, int],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    episode_root = output_root / f"episode_{episode:06d}"
    episode_root.mkdir(parents=True, exist_ok=True)
    for internal_slot, public_slot in slot_map.items():
        source = mesh_paths[internal_slot]
        destination = episode_root / f"{public_names[public_slot]}{source.suffix.lower()}"
        if destination.is_symlink():
            if destination.resolve() != source:
                raise RuntimeError(f"refusing to replace mismatched mesh link {destination}")
        elif destination.exists():
            raise RuntimeError(f"refusing to replace existing mesh {destination}")
        else:
            destination.symlink_to(source)


def normalize_trajectory(
    source: Path,
    destination: Path,
    public_names: list[str],
    slot_map: dict[int, int],
) -> None:
    table = pq.read_table(source)
    required = {"object_id", "object_body_id", "object_body_name"}
    missing = required - set(table.column_names)
    if missing:
        raise RuntimeError(f"{source}: missing columns {sorted(missing)}")
    new_slots = [slot_map[int(slot)] for slot in table["object_body_id"].to_pylist()]
    new_names = [public_names[slot] for slot in new_slots]
    table = table.set_column(
        table.schema.get_field_index("object_body_id"),
        "object_body_id",
        pa.array(new_slots, type=pa.int32()),
    )
    table = table.set_column(
        table.schema.get_field_index("object_id"),
        "object_id",
        pa.array(new_slots, type=pa.int32()),
    )
    table = table.set_column(
        table.schema.get_field_index("object_body_name"),
        "object_body_name",
        pa.array(new_names).dictionary_encode(),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--roster", required=True, type=Path)
    parser.add_argument("--results-raw", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--meshes", required=True, type=Path)
    parser.add_argument(
        "--asset-search-root",
        action="append",
        type=Path,
        default=[],
        help="root searched by basename when a checkpoint records a stale mesh path; repeatable",
    )
    args = parser.parse_args()

    checkpoint_values = json.loads(args.checkpoints.read_text())
    checkpoints = {
        int(key): resolve_relative(Path(value), args.checkpoints).resolve()
        for key, value in checkpoint_values.items()
    }
    roster = {int(key): value for key, value in json.loads(args.roster.read_text())["episodes"].items()}
    if set(checkpoints) != set(roster):
        raise RuntimeError(
            f"checkpoint roster mismatch: missing={sorted(set(roster) - set(checkpoints))}, "
            f"extra={sorted(set(checkpoints) - set(roster))}"
        )
    search_roots = [path.expanduser().resolve() for path in args.asset_search_root]

    for episode in sorted(roster):
        checkpoint = checkpoints[episode]
        public_names = roster[episode]
        _, mesh_paths, slot_map = reference_contract(
            checkpoint, public_names, episode, search_roots
        )
        stage_meshes(args.meshes, episode, public_names, mesh_paths, slot_map)

        raw_parquet = args.results_raw / f"episode_{episode:06d}.parquet"
        raw_report = args.results_raw / f"episode_{episode:06d}.json"
        if not raw_parquet.is_file():
            raise FileNotFoundError(f"missing evaluated trajectory: {raw_parquet}")
        normalize_trajectory(
            raw_parquet,
            args.results / raw_parquet.name,
            public_names,
            slot_map,
        )
        if raw_report.is_file():
            args.results.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_report, args.results / raw_report.name)

    print(f"prepared {len(roster)} episodes under {args.results} and {args.meshes}")


if __name__ == "__main__":
    main()

