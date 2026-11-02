"""Lazy SAM 3 model construction and process-local lifetime management."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
import sys
from typing import Any


@dataclass(frozen=True)
class SAM3VideoEngineConfig:
    """Process-local SAM 3 video predictor configuration."""

    repository_root: Path
    gpu: int = 0
    checkpoint: Path | None = None
    apply_temporal_disambiguation: bool = False


class SAM3VideoEngine:
    """Own one lazily loaded SAM 3 video predictor and shut it down safely."""

    def __init__(self, config: SAM3VideoEngineConfig) -> None:
        self.config = config
        self.predictor: Any | None = None

    def __enter__(self) -> Any:
        thirdparty = self.config.repository_root.resolve() / "thirdparty"
        if str(thirdparty) not in sys.path:
            sys.path.insert(0, str(thirdparty))
        from sam3.model_builder import build_sam3_video_predictor

        kwargs: dict[str, Any] = {"gpus_to_use": [self.config.gpu]}
        kwargs["apply_temporal_disambiguation"] = (
            self.config.apply_temporal_disambiguation
        )
        if self.config.checkpoint is not None:
            kwargs["checkpoint_path"] = str(self.config.checkpoint.resolve())
        self.predictor = build_sam3_video_predictor(**kwargs)
        return self.predictor

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.predictor is not None:
            self.predictor.shutdown()
            self.predictor = None
        gc.collect()
