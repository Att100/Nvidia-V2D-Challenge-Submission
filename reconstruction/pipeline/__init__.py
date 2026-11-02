"""Public pipeline v2 API with lazy orchestration imports.

Dataset modules import lightweight contracts from :mod:`pipeline.utils`; eager
import of the full orchestrator here would make that path circular because the
orchestrator itself imports dataset preparers.
"""

from typing import Any

from .utils import PipelineRequest, StageResult, StageStatus

__all__ = [
    "EgoSceneHOIPipeline",
    "PipelineRequest",
    "PipelineRunResult",
    "StageResult",
    "StageStatus",
]


def __getattr__(name: str) -> Any:
    """Load the heavyweight orchestrator only when its public symbol is used."""

    if name in {"EgoSceneHOIPipeline", "PipelineRunResult"}:
        from .pipeline import EgoSceneHOIPipeline, PipelineRunResult

        return {
            "EgoSceneHOIPipeline": EgoSceneHOIPipeline,
            "PipelineRunResult": PipelineRunResult,
        }[name]
    raise AttributeError(name)
