"""Hydra-composed configuration helpers for pipeline modules and workers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import os
import sys
from typing import Any


def get_config(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a dotted key from a plain Hydra configuration container."""

    value: Any = config
    for component in dotted_key.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return default
        value = value[component]
    return value


def repository_root(config: Mapping[str, Any]) -> Path:
    """Return the configured repository root as an absolute path."""

    configured = get_config(config, "runtime.repository_root")
    return Path(configured or Path.cwd()).expanduser().resolve()


def resolve_path(config: Mapping[str, Any], dotted_key: str) -> Path | None:
    """Resolve a configured path relative to the repository root."""

    value = get_config(config, dotted_key)
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (repository_root(config) / path).resolve()


def python_executable(config: Mapping[str, Any], stage: str | None = None) -> str:
    """Select a stage-specific interpreter, falling back to the current one."""

    configured = None
    conda_environment = None
    if stage:
        configured = get_config(config, f"runtime.environments.{stage}.python_executable")
        conda_environment = get_config(
            config, f"runtime.environments.{stage}.conda_environment"
        )
    configured = configured or get_config(config, "runtime.python_executable")
    if configured:
        return str(Path(configured).expanduser().resolve())
    if conda_environment:
        conda_executable = os.environ.get("CONDA_EXE")
        if conda_executable:
            conda_root = Path(conda_executable).expanduser().resolve().parent.parent
        else:
            # The active interpreter is normally <conda-root>/envs/<name>/bin/python.
            executable = Path(sys.executable).resolve()
            conda_root = (
                executable.parents[3]
                if len(executable.parents) > 3 and executable.parents[2].name == "envs"
                else executable.parents[1]
            )
        candidate = conda_root / "envs" / str(conda_environment) / "bin" / "python"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Conda environment {conda_environment!r} has no Python at {candidate}"
            )
        return str(candidate)
    return sys.executable

