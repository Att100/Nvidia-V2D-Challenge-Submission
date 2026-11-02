#!/usr/bin/env python3
"""Example batch runner for the official flash_chord policy evaluator.

Use one trained checkpoint per episode. The kit includes evaluate_policy.py and its Hydra
configuration. Run this in an environment with flash_chord and Isaac Lab installed.
This runner writes the episode_XXXXXX.parquet files consumed by pack_submission.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def runtime_root() -> Path | None:
    """Find the checkout that supplies flash_chord's package and relative assets."""
    spec = importlib.util.find_spec("flash_chord")
    if spec is None or spec.origin is None:
        return None
    checkout = Path(spec.origin).resolve().parents[2]
    if (checkout / "src/flash_chord/assets").is_dir():
        return checkout
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-script", type=Path,
                        default=Path(__file__).resolve().parents[1] / "flash_chord/scripts/evaluate_policy.py",
                        help="evaluator entry point (default: the copy included in this kit)")
    parser.add_argument("--checkpoints", required=True, type=Path,
                        help="JSON object mapping episode index to its checkpoint path")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--source-root", type=Path,
                        help="optional path remap for the task parquet stored in each checkpoint")
    parser.add_argument("--runtime-root", type=Path,
                        help="flash_chord checkout for relative robot asset paths; detected from the installed package")
    parser.add_argument("--world-count", type=int, default=1)
    args = parser.parse_args()

    script = args.eval_script.resolve()
    if not script.is_file():
        parser.error(f"evaluator script does not exist: {script}")
    if args.world_count < 1:
        parser.error("--world-count must be positive")
    workdir = args.runtime_root.resolve() if args.runtime_root else runtime_root()
    if workdir is None or not workdir.is_dir():
        parser.error("flash_chord checkout was not found; install it or pass --runtime-root")
    mapping = json.loads(args.checkpoints.read_text())
    if not isinstance(mapping, dict) or not mapping:
        parser.error("--checkpoints must be a nonempty JSON object")
    checkpoints = {}
    for key, value in mapping.items():
        try:
            episode = int(key)
        except ValueError:
            parser.error(f"invalid episode index {key!r}")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            parser.error(f"episode {episode}: checkpoint does not exist: {path}")
        checkpoints[episode] = path

    args.results.mkdir(parents=True, exist_ok=True)
    results = args.results.resolve()
    for episode, checkpoint in sorted(checkpoints.items()):
        stem = f"episode_{episode:06d}"
        command = [
            sys.executable, str(script),
            f"evaluation.checkpoint={checkpoint}",
            f"evaluation.world_count={args.world_count}",
            "evaluation.reset_mode=explicit",
            f"evaluation.metrics_output={results / (stem + '.json')}",
            f"evaluation.object_trajectories_output={results / (stem + '.parquet')}",
            f"hydra.run.dir={results / 'hydra' / stem}",
            "logging.mode=disabled",
        ]
        if args.source_root:
            command.append(f"evaluation.source_root={args.source_root.resolve()}")
        print(f"evaluating episode {episode} from {checkpoint}", flush=True)
        subprocess.run(command, cwd=workdir, check=True)


if __name__ == "__main__":
    main()
