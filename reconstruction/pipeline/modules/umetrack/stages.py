"""Pipeline adapters for UmeTrack inference and GPU MANO fitting."""

from __future__ import annotations

from pathlib import Path

from pipeline.config import get_config, python_executable, resolve_path
from pipeline.modules.base import FoundationModule, WORKER_DISPATCHER
from pipeline.utils import PipelineRequest


class UmeTrackInferenceModule(FoundationModule):
    """Run two-view grayscale UmeTrack from authoritative left-view masks."""

    name = "umetrack_hand_reconstruction"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        root = request.run_root / "hands/umetrack"
        return root / "umetrack_results.npz", root / "report.json", root / "umetrack_overlay.mp4"

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        return (
            python_executable(request.config, self.name),
            "-m",
            WORKER_DISPATCHER,
            "infer",
            "--run-root",
            str(request.run_root),
            "--output",
            str(request.run_root / "hands/umetrack"),
            "--device",
            request.device,
            "--crop-margin",
            str(get_config(request.config, "pipeline.stereo.umetrack.crop_margin", 1.20)),
            "--min-mask-area",
            str(get_config(request.config, "pipeline.stereo.umetrack.minimum_mask_area", 80)),
            "--max-calibration-samples",
            str(get_config(request.config, "pipeline.stereo.umetrack.maximum_scale_samples", 80)),
            "--mode",
            "stereo",
        )


class UmeTrackMANOFittingModule(FoundationModule):
    """Fit official MANO parameters to the native UmeTrack hand sequence."""

    name = "umetrack_mano_fitting"

    def expected_artifacts(self, request: PipelineRequest) -> tuple[Path, ...]:
        root = request.run_root / "hands/umetrack_mano"
        return root / "mano_fitted_results.npz", root / "report.json"

    def build_command(self, request: PipelineRequest) -> tuple[str, ...]:
        mano_dir = resolve_path(request.config, "models.umetrack.mano_dir")
        if mano_dir is None:
            raise ValueError("models.umetrack.mano_dir is required")
        return (
            python_executable(request.config, self.name),
            "-m",
            WORKER_DISPATCHER,
            "fit_mano",
            "--umetrack-results",
            str(request.run_root / "hands/umetrack/umetrack_results.npz"),
            "--mano-dir",
            str(mano_dir),
            "--output",
            str(request.run_root / "hands/umetrack_mano"),
            "--device",
            request.device,
            "--shape-iterations",
            str(get_config(request.config, "pipeline.stereo.mano_fitting.shape_iterations", 300)),
            "--pose-iterations",
            str(get_config(request.config, "pipeline.stereo.mano_fitting.pose_iterations", 450)),
            "--surface-iterations",
            str(get_config(request.config, "pipeline.stereo.mano_fitting.surface_iterations", 120)),
            "--pose-basis",
            str(get_config(request.config, "pipeline.stereo.mano_fitting.pose_basis", "pca")),
        )
