"""Input preparation for synchronized native RGB and grayscale stereo."""

from __future__ import annotations

from pathlib import Path
import time

from datasets import RGBStereoGrayPreparer
from pipeline.config import get_config
from pipeline.utils import PipelineRequest, StageResult, StageStatus, all_artifacts_exist


class InputPreparationModule:
    """Normalize the three Track-3 videos into the shared run layout."""

    name = "prepare_input"

    @staticmethod
    def _artifacts(request: PipelineRequest) -> tuple[Path, ...]:
        return (
            request.run_root / "input/rgb_upright.mp4",
            request.run_root / "input/cameras.npz",
            request.run_root / "input/valid_masks.npz",
            request.run_root / "input/manifest.json",
            request.run_root / "sequence_manifest.json",
            request.run_root / "input/stereo_left_raw.mp4",
            request.run_root / "input/stereo_right_raw.mp4",
            request.run_root / "input/multiview_manifest.json",
        )

    def run(self, request: PipelineRequest) -> StageResult:
        """Validate synchronization and copy all three input streams."""

        if str(get_config(request.config, "dataset.name", "")) != "rgb_stereo_gray":
            raise ValueError("This bundle supports only dataset=rgb_stereo_gray")
        artifacts = self._artifacts(request)
        if request.resume and not request.force and all_artifacts_exist(artifacts):
            return StageResult(self.name, StageStatus.REUSED, artifacts)
        if request.dry_run:
            return StageResult(self.name, StageStatus.SKIPPED, artifacts)

        repository = Path(str(get_config(request.config, "runtime.repository_root"))).resolve()
        left = Path(str(get_config(request.config, "dataset.streams.stereo_left"))).expanduser()
        right = Path(str(get_config(request.config, "dataset.streams.stereo_right"))).expanduser()
        left = left if left.is_absolute() else repository / left
        right = right if right.is_absolute() else repository / right
        started = time.perf_counter()
        produced = RGBStereoGrayPreparer(
            stereo_left=left,
            stereo_right=right,
            horizontal_fov_degrees=float(
                get_config(request.config, "dataset.camera.horizontal_fov_degrees", 60.0)
            ),
        ).prepare(request.sequence_root, request.run_root, request.frame_count)
        if not all_artifacts_exist(produced):
            raise RuntimeError("Track-3 input preparation did not produce all artifacts")
        return StageResult(
            self.name,
            StageStatus.COMPLETED,
            tuple(produced),
            elapsed_seconds=time.perf_counter() - started,
        )
