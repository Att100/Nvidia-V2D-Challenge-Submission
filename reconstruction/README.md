# NVIDIA V2D Challenge Track 3: RGB + Stereo Grayscale HOI

This directory is the frozen challenge hand-off for the egocentric pipeline that
uses one native RGB stream plus a synchronized grayscale stereo pair. It contains
the runnable source, Hydra configuration, third-party inference code, checkpoint
download instructions, Conda environment exports, and Track-3 documentation. Generated
sequence results are intentionally not duplicated from the source project.

Raw challenge videos are not duplicated here. The runtime and viewer are
self-contained and do not import code from the source research workspace.

## Layout

```text
configs/                 active Hydra configuration
pipeline/                class-based pipeline modules and optimizers
datasets/, tools/, utils/
thirdparty/               minimal third-party source and checkpoint destinations
environment/              reproducible Conda exports
docs/                     pipeline, model, environment, and result docs
manifests/                machine-readable bundled-model inventory
scripts/                  bundle setup and validation utilities
```

## Quick start

Create the two environments:

```bash
cd /path/to/v2d_challenge_track3/reconstruction
conda env create -f environment/ego_scene_hoi.yaml
conda env create -f environment/umetrack.yaml
conda activate ego_scene_hoi
python scripts/validate_bundle.py
```

The main environment is named `ego_scene_hoi`; UmeTrack is deliberately isolated
in the `umetrack` environment and is launched by the pipeline worker adapter.

For a synchronized sequence, the RGB stream is `sequence.root`, while `camera_c`
is the left stereo input and `camera_b` is the right stereo input for the released
Track-3 device:

```bash
python run.py \
  dataset=rgb_stereo_gray \
  sequence.root=/data/observation.images.ego_cam_a/episode_000001.mp4 \
  sequence.frame_count=335 \
  dataset.streams.stereo_left=/data/observation.images.ego_cam_c/episode_000001.mp4 \
  dataset.streams.stereo_right=/data/observation.images.ego_cam_b/episode_000001.mp4 \
  output.root=/output/episode_000001 \
  execution.resume=true
```

The default calibration path uses AnyCalib plus the known 75 mm baseline. The
pipeline does not use dataset ground-truth extrinsics. Codex reasoning requires an
authenticated `codex` CLI with access to `gpt-6-astra`. SAM3, AnyCalib, VIPE's
DROID model, FLUX.2, and SAM 3D Objects dependencies download their public weights on first use. The
remaining checkpoints are restored with `scripts/download_checkpoints.sh` and
the MANO files are installed manually according to `docs/CHECKPOINTS.md`.

See [the pipeline description](docs/TRACK3_PIPELINE.md), [environment setup](docs/ENVIRONMENT.md),
[checkpoint downloads](docs/CHECKPOINTS.md), [model inventory](docs/MODELS.md),
and [result status](docs/RESULTS.md).

## Exporting a deliverable

After the complete pipeline finishes, export and validate one self-contained
sequence with:

```bash
conda activate ego_scene_hoi
python tools/export_sequence_deliverable.py \
  --run-root /output/episode_000001 \
  --output /deliverables/episode_000001 \
  --device cuda
```

The exporter automatically selects the final joint-HACO hand and object results
when present. It writes NumPy-array pickle files, scaled reconstructed GLBs,
validity masks, `array_schema.json`, `manifest.json`, a standalone
`render_hand_object.py`, and a numerical/rendered validation report. The command
fails if round-trip validation does not pass.

Use `--skip-render` for a numeric-only export or `--mano-root /path/to/MANO` when
the MANO models are not installed under `thirdparty/MANO`. Native RGB is exported
only as a synchronized reference video; no RGB-to-left registration or RGB camera
trajectory is required or included. All delivered geometry uses the metric VIPE
rectified-left world.

## Results

Results are generated under the configured `output.root`. The copied hand-off does
not include duplicate deliverables; see [result status](docs/RESULTS.md) for the
original validated episode inventory.

## Scope

This hand-off contains only the RGB plus stereo-grayscale Track-3 workflow. The
current entry point is `run.py` with `dataset=rgb_stereo_gray`; earlier monocular
and unrelated dataset workflows are intentionally excluded.
