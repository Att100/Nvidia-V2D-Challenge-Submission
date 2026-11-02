"""Zero-argument worker jobs used for isolated foundation-model processes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence


WORKER_JOB_ENVIRONMENT = "EGO_SCENE_HOI_WORKER_JOB"
WORKER_JOB_SCHEMA = "ego-scene-hoi-worker-job-v2"


def write_worker_job(
    path: Path,
    *,
    stage: str,
    task: str,
    arguments: Sequence[str],
    config: dict[str, Any],
) -> Path:
    """Serialize one immutable worker invocation beneath the run directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": WORKER_JOB_SCHEMA,
        "stage": stage,
        "task": task,
        "arguments": list(arguments),
        "config": config,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path.resolve()


def load_worker_job() -> dict[str, Any]:
    """Load the worker job selected by the orchestrator environment."""

    raw = os.environ.get(WORKER_JOB_ENVIRONMENT)
    if not raw:
        raise RuntimeError(
            f"{WORKER_JOB_ENVIRONMENT} is required; workers accept no CLI arguments"
        )
    path = Path(raw).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != WORKER_JOB_SCHEMA:
        raise ValueError(f"Unsupported worker job schema: {path}")
    if not isinstance(payload.get("stage"), str) or not payload["stage"]:
        raise ValueError(f"Worker job has no stage: {path}")
    if not isinstance(payload.get("task"), str) or not payload["task"]:
        raise ValueError(f"Worker job has no task: {path}")
    return payload


def worker_route() -> tuple[str, str]:
    """Return the validated ``(stage, task)`` dispatcher key."""

    payload = load_worker_job()
    return str(payload["stage"]), str(payload["task"])


def worker_arguments() -> list[str]:
    """Return the Hydra-derived internal arguments for a legacy parser."""

    payload = load_worker_job()
    arguments = payload.get("arguments", [])
    if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
        raise TypeError("Worker job arguments must be a list of strings")
    return arguments


def parse_worker_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse a serialized job rather than process command-line arguments."""

    return parser.parse_args(worker_arguments())


def worker_config() -> dict[str, Any]:
    """Return the plain resolved Hydra configuration embedded in the job."""

    config = load_worker_job().get("config", {})
    if not isinstance(config, dict):
        raise TypeError("Worker config must be a JSON object")
    return config
