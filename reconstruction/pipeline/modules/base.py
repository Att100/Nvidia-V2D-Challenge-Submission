"""Base classes for external foundation-model workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

from pipeline.utils import (
    PipelineRequest,
    StageResult,
    StageStatus,
    WORKER_JOB_ENVIRONMENT,
    all_artifacts_exist,
    read_json,
    write_json,
    write_worker_job,
)
from utils import CommandRunner


WORKER_DISPATCHER = "pipeline.modules.worker"


class FoundationModule(ABC):
    """Class interface shared by every foundation-model invocation."""

    name: str

    @abstractmethod
    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        """Return the minimum artifacts that prove stage completion."""

    @abstractmethod
    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        """Build ``python -m dispatcher TASK`` plus serialized job arguments."""

    def build_commands(self, request: PipelineRequest) -> tuple[tuple[str, ...], ...]:
        """Return one or more isolated worker commands in execution order."""

        command = self.build_command(request)
        return (command,) if command else ()

    def _fingerprint(self, request: PipelineRequest, commands: Sequence[Sequence[str]]) -> str:
        """Hash the Hydra-derived stage job specification."""

        payload = {
            "stage": self.name,
            "commands": [list(command) for command in commands],
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _manifest_path(self, request: PipelineRequest) -> Path:
        return request.run_root / "manifests" / f"{self.name}.json"

    def _cache_matches(
        self,
        request: PipelineRequest,
        artifacts: tuple[Path, ...],
        fingerprint: str,
    ) -> bool:
        if not all_artifacts_exist(artifacts):
            return False
        manifest_path = self._manifest_path(request)
        if manifest_path.is_file():
            if read_json(manifest_path).get("fingerprint") == fingerprint:
                return True
            if bool(
                request.config.get("execution", {}).get(
                    "accept_legacy_artifacts", False
                )
            ):
                self._write_manifest(request, artifacts, fingerprint, adopted=True)
                return True
            return False
        if bool(request.config.get("execution", {}).get("accept_legacy_artifacts", False)):
            self._write_manifest(request, artifacts, fingerprint, adopted=True)
            return True
        return False

    def _write_manifest(
        self,
        request: PipelineRequest,
        artifacts: tuple[Path, ...],
        fingerprint: str,
        *,
        adopted: bool = False,
    ) -> None:
        write_json(
            self._manifest_path(request),
            {
                "schema_version": "ego-scene-hoi-stage-manifest-v1",
                "stage": self.name,
                "fingerprint": fingerprint,
                "artifacts": [str(path) for path in artifacts],
                "adopted_legacy_artifacts": adopted,
            },
        )

    def run_worker_command(
        self,
        request: PipelineRequest,
        command: Sequence[str],
        *,
        job_index: int,
        log_path: Path,
    ) -> tuple[str, ...]:
        """Run a package worker without forwarding any command-line arguments."""

        command = tuple(command)
        if len(command) < 4 or command[1:3] != ("-m", WORKER_DISPATCHER):
            raise ValueError(
                "Worker command must use the shared dispatcher and include a task: "
                f"{command}"
            )
        task = command[3]
        arguments = command[4:]
        python = command[0]
        job_path = write_worker_job(
            request.run_root / "jobs" / self.name / f"{job_index:02d}.json",
            stage=self.name,
            task=task,
            arguments=arguments,
            config=dict(request.config),
        )
        launch = (python, "-m", WORKER_DISPATCHER)
        runner = CommandRunner(Path(__file__).resolve().parents[2])
        runner.run(
            launch,
            log_path=log_path,
            environment={WORKER_JOB_ENVIRONMENT: str(job_path)},
        )
        return launch

    def run(self, request: PipelineRequest) -> StageResult:
        """Reuse validated outputs or execute the worker command."""

        artifacts = self.expected_artifacts(request)
        commands = self.build_commands(request)
        fingerprint = self._fingerprint(request, commands)
        display_command = tuple(
            token
            for command_index, item in enumerate(commands)
            for token in (("<then>",) if command_index else ()) + tuple(item[:4])
        )
        if request.resume and not request.force and self._cache_matches(
            request, artifacts, fingerprint
        ):
            return StageResult(
                stage=self.name,
                status=StageStatus.REUSED,
                artifacts=artifacts,
                command=display_command,
                message="Validated existing artifacts.",
            )
        if request.dry_run:
            return StageResult(
                stage=self.name,
                status=StageStatus.SKIPPED,
                artifacts=artifacts,
                command=display_command,
                message="Dry run; worker was not launched.",
            )
        if not commands:
            missing = [str(path) for path in artifacts if not path.exists()]
            raise FileNotFoundError(f"{self.name} has no executable command; missing {missing}")
        start = time.perf_counter()
        for command_index, item in enumerate(commands):
            log_path = (
                request.run_root
                / "logs/v2"
                / f"{self.name}_{command_index:02d}.log"
            )
            self.run_worker_command(
                request,
                item,
                job_index=command_index,
                log_path=log_path,
            )
        if not all_artifacts_exist(artifacts):
            raise RuntimeError(f"{self.name} finished without all expected artifacts")
        self._write_manifest(request, artifacts, fingerprint)
        return StageResult(
            stage=self.name,
            status=StageStatus.COMPLETED,
            artifacts=artifacts,
            command=display_command,
            elapsed_seconds=time.perf_counter() - start,
        )
