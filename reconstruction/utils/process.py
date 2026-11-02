"""Safe subprocess execution with captured logs and reproducible commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandRunner:
    """Execute an argv sequence without shell interpolation."""

    working_directory: Path

    def run(
        self,
        command: Sequence[str],
        *,
        log_path: Path,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        """Run a command and tee merged stdout/stderr into ``log_path``."""

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as stream:
            subprocess.run(
                list(command),
                cwd=self.working_directory,
                env=(
                    None
                    if environment is None
                    else {**os.environ, **dict(environment)}
                ),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )
