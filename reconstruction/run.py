#!/usr/bin/env python3
"""Hydra entry point for the class-based ego-scene/4D-HOI pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from pipeline import EgoSceneHOIPipeline, PipelineRequest


def _plain_config(config: DictConfig) -> dict[str, Any]:
    """Resolve a Hydra configuration into a worker-safe plain container."""

    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("The root Hydra configuration must be a mapping")
    return value


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Compose configuration, execute selected stages, and print the report path."""

    resolved = _plain_config(config)
    pipeline_config = resolved["pipeline"]
    pipeline = EgoSceneHOIPipeline(
        config=resolved,
        contact_iterations=int(pipeline_config["contact_iterations"]),
        device=str(pipeline_config["device"]),
    )
    operation = str(resolved.get("operation", "run"))
    if operation == "list_stages":
        print("\n".join(pipeline.stage_names))
        return
    if operation != "run":
        raise ValueError(f"Unsupported operation: {operation}")

    repository_root = Path(str(resolved["runtime"]["repository_root"])).resolve()
    sequence_root = Path(str(resolved["sequence"]["root"])).expanduser()
    run_root = Path(str(resolved["output"]["root"])).expanduser()
    if not sequence_root.is_absolute():
        sequence_root = repository_root / sequence_root
    if not run_root.is_absolute():
        run_root = repository_root / run_root
    execution = resolved["execution"]
    run_root.mkdir(parents=True, exist_ok=True)
    config_path = run_root / "config/resolved.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")
    request = PipelineRequest(
        sequence_root=sequence_root.resolve(),
        run_root=run_root.resolve(),
        frame_count=int(resolved["sequence"]["frame_count"]),
        resume=bool(execution["resume"]),
        force=bool(execution["force"]),
        dry_run=bool(execution["dry_run"]),
        device=str(pipeline_config["device"]),
        config=resolved,
    )
    result = pipeline.run(
        request,
        selected_stages=execution.get("selected_stages"),
    )
    print(f"[pipeline-v2] report: {result.report_path}", flush=True)


if __name__ == "__main__":
    main()
