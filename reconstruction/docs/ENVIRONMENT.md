# Environment and Reproduction

## Tested host

The bundle was produced on Linux under WSL2 with an NVIDIA RTX 4090. The main
environment uses Python 3.12, PyTorch 2.8, and CUDA 12.8 packages. UmeTrack uses a
separate Python 3.9 / PyTorch 2.1 environment because its original dependency set
is older. Substantial RAM, GPU memory, and temporary disk space are required for
FLUX and SAM 3D Objects; run those stages sequentially.

System tools expected on `PATH` include `ffmpeg`, `ffprobe`, a C/C++ toolchain,
CUDA/NVCC for compiled extensions, Git, and the authenticated Codex CLI.

## Conda environments

Exact exports:

```bash
conda env create -f environment/ego_scene_hoi.yaml
conda env create -f environment/umetrack.yaml
```

The `*_from_history.yaml` files are smaller human-readable specifications. Use the
full YAML files when reproducing the frozen environment and the history files when
porting it to a different CUDA stack.

No host-specific Conda `prefix` is embedded in the exports.

Build FoundationPose's local C++ extension inside this checkout. The build script
derives all paths from its own location and the active Conda environment:

```bash
conda activate ego_scene_hoi
bash thirdparty/FoundationPose/build_all_conda.sh
```

Do not copy a pre-existing CMake build directory from another workspace; its
cache contains absolute paths and is intentionally rejected by bundle validation.

## Model acquisition

SAM3, AnyCalib, VIPE's DROID network, FLUX.2, and the MoGe dependency used by
SAM 3D Objects download their public weights into the standard Hugging Face or
Torch cache on first use. Network access is therefore required for a fresh setup.
After following `docs/CHECKPOINTS.md`, Foundation Stereo, FoundationPose, HACO,
UmeTrack, MANO, and the main SAM 3D Objects checkpoints use explicit relative
paths under `thirdparty`.

Codex reasoning also requires network access. Authenticate the Codex CLI
separately and make sure `gpt-6-astra` with high reasoning effort is available.

## Required data layout

Raw challenge videos are not copied. A typical layout is:

```text
/data/
  observation.images.ego_cam_a/episode_000001.mp4  # native RGB
  observation.images.ego_cam_b/episode_000001.mp4  # stereo right
  observation.images.ego_cam_c/episode_000001.mp4  # stereo left
```

All three streams must be synchronized and have matching frame counts. For the
released device, C is the left camera and B is the right camera.

## Running

```bash
conda activate ego_scene_hoi
python run.py \
  dataset=rgb_stereo_gray \
  sequence.root=/data/observation.images.ego_cam_a/episode_000001.mp4 \
  sequence.frame_count=335 \
  dataset.streams.stereo_left=/data/observation.images.ego_cam_c/episode_000001.mp4 \
  dataset.streams.stereo_right=/data/observation.images.ego_cam_b/episode_000001.mp4 \
  output.root=/output/episode_000001 \
  execution.resume=true
```

Hydra writes the resolved configuration under the run root. The UmeTrack stage is
automatically invoked in its own environment using the environment name configured
in `configs/runtime/local.yaml`.

## Validation

Run a non-GPU bundle audit with:

```bash
python scripts/validate_bundle.py
```

Machine-readable checkpoint hashes and result summaries are in `manifests`.
