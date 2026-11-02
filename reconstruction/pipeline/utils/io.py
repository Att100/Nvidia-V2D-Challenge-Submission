"""Small artifact and JSON helpers used only by the pipeline package."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable


def all_artifacts_exist(paths: Iterable[Path]) -> bool:
    """Return true when every expected path exists and is non-empty if a file."""

    for path in paths:
        if not path.exists():
            return False
        if path.is_file() and path.stat().st_size == 0:
            return False
    return True


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, human-readable JSON, creating parent directories."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for an artifact dependency."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha256(value: Any) -> str:
    """Hash a JSON-compatible configuration value deterministically."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
