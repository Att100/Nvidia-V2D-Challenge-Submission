from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


STAGE_MANIFEST_FORMAT = "interaction-4d-stage-manifest-v1"


def _serialize(value: Any, *, base: Path | None = None) -> Any:
    if isinstance(value, Path):
        path = value.expanduser().resolve()
        if base is not None:
            try:
                return str(path.relative_to(base))
            except ValueError:
                pass
        return str(path)
    if isinstance(value, Mapping):
        return {str(key): _serialize(item, base=base) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item, base=base) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _serialize(value.tolist(), base=base)
        except (ValueError, TypeError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def write_stage_manifest(
    output_dir: str | Path,
    *,
    stage: str,
    inputs: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    parameters: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write the common manifest consumed by read-only tools."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    existing_artifacts = {
        key: value
        for key, value in artifacts.items()
        if not isinstance(value, Path) or value.expanduser().exists()
    }
    payload = {
        "format": STAGE_MANIFEST_FORMAT,
        "stage": str(stage),
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(directory),
        "inputs": _serialize(inputs),
        "artifacts": _serialize(existing_artifacts, base=directory),
        "parameters": _serialize(parameters or {}),
        "metrics": _serialize(metrics or {}),
    }
    path = directory / "stage_manifest.json"
    temporary = directory / ".stage_manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return path


def load_stage_manifest(path_or_directory: str | Path) -> dict[str, Any]:
    path = Path(path_or_directory).expanduser()
    if path.is_dir():
        path = path / "stage_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != STAGE_MANIFEST_FORMAT:
        raise ValueError(f"Unsupported stage manifest format in {path}")
    return payload
