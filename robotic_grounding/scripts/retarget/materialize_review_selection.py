#!/usr/bin/env python3
"""Materialize one explicitly reviewed retarget candidate without silent overwrite."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def one_parquet(partition: Path) -> Path:
    paths = sorted(partition.glob("*.parquet"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one parquet under {partition}, found {paths}")
    return paths[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-partition", type=Path)
    parser.add_argument("--selected-partition", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    args.selected_partition.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.selected_partition.glob("*.parquet"))
    if existing:
        if not args.reuse_existing or len(existing) != 1:
            raise FileExistsError(f"selected partition is not empty: {existing}")
        selected = existing[0]
        source = selected
    else:
        if args.source_partition is None:
            raise ValueError("--source-partition is required when no selection exists")
        source = one_parquet(args.source_partition)
        selected = args.selected_partition / "part-0.parquet"
        shutil.copy2(source, selected)

    report = {
        "status": args.status,
        "rationale": args.rationale,
        "source_parquet": str(source.resolve()),
        "selected_parquet": str(selected.resolve()),
        "automatic_gate_override": args.status == "REVIEW",
        "raw_reconstruction_modified": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists():
        raise FileExistsError(args.report)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
