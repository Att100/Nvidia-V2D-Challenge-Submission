"""Unified data contracts shared by pipeline modules and optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class StageStatus(str, Enum):
    """Execution state recorded for one pipeline stage."""

    COMPLETED = "completed"
    REUSED = "reused"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class PipelineRequest:
    """Input shared by every class-based pipeline stage."""

    sequence_root: Path
    run_root: Path
    frame_count: int
    resume: bool = True
    force: bool = False
    dry_run: bool = False
    device: str = "cuda:0"
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageResult:
    """Normalized result returned by a foundation module or optimizer."""

    stage: str
    status: StageStatus
    artifacts: tuple[Path, ...] = ()
    command: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "stage": self.stage,
            "status": self.status.value,
            "artifacts": [str(path) for path in self.artifacts],
            "command": list(self.command),
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": self.metrics,
            "message": self.message,
        }
