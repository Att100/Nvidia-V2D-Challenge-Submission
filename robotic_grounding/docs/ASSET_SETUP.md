# Runtime asset and model setup

This repository intentionally contains source code, configuration, and licenses only. Robot
meshes, URDFs, ONNX files, reconstruction model weights, generated object meshes, policy
checkpoints, and all MANO/HACO numerical data are not committed.

The unified downloader is:

```bash
bash robotic_grounding/scripts/download_required_assets.sh <target>
```

Run commands from the directory that contains the sibling `robotic_grounding/` and
`reconstruction/` folders.

## 1. Prerequisites

The minimal robot-asset downloader requires:

```bash
git --version
git lfs version
rsync --version
```

Install Git LFS through the host package manager if `git lfs version` fails. Reconstruction
downloads additionally require the clients documented in `reconstruction/docs/CHECKPOINTS.md`:

```bash
python -m pip install gdown huggingface-hub
hf auth login
```

SAM 3D Objects is gated. Accept its upstream license before authenticating. Do not commit Hugging
Face, MANO, GitHub, or Kaggle credentials to this repository.

## 2. Download the required SharpaWave robot assets

```bash
bash robotic_grounding/scripts/download_required_assets.sh robotic_grounding
```

This uses a sparse Git checkout plus Git LFS and installs about 48 MiB of files. The source is
fixed by default to:

```text
repository: https://github.com/nvidia-isaac/video_to_data.git
commit:     8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef
```

That is the public FlashCHORD revision named by the V2D submission kit. Only these asset families
are restored:

- FlashCHORD left/right SharpaWave URDFs;
- regular SharpaWave visual meshes used by policy rendering;
- convex SharpaWave collision meshes used by training and official evaluation;
- the corresponding legacy retarget URDF and mesh tree.

The downloader intentionally excludes G1, Vega, MotionBricks, SONIC policies, example human
motions, and documentation media because none is used by the selected Track 3 policies.

The sparse checkout and its LFS cache live outside the code tree by default:

```text
${XDG_CACHE_HOME:-/tmp}/v2d-track3-assets/
```

To use scratch storage instead:

```bash
export V2D_ASSET_CACHE_ROOT=/path/to/cache/v2d-track3-assets
bash robotic_grounding/scripts/download_required_assets.sh robotic_grounding
```

## 3. Download reconstruction checkpoints

```bash
bash robotic_grounding/scripts/download_required_assets.sh reconstruction
```

This delegates to `reconstruction/scripts/download_checkpoints.sh`, which downloads the public
FoundationStereo, FoundationPose, HACO, UmeTrack, and SAM 3D Objects checkpoints from their
official sources. SAM 3D Objects alone requires roughly 13 GB and may need about twice that space
temporarily.

Some models use their upstream runtime caches instead of repository-local files, including SAM3,
FLUX.2, MoGe, AnyCalib, and VIPE DROID-SLAM. See:

- `reconstruction/docs/CHECKPOINTS.md`
- `reconstruction/docs/MODELS.md`

## 4. Install licensed MANO models manually

Register at the official MANO website, accept its license, download MANO v1.2, and obtain the
MANO/HACO helper arrays from an authorized upstream installation. Install all files through the
repository helper:

```bash
bash reconstruction/scripts/install_mano_assets.sh \
  /path/to/MANO_LEFT.pkl /path/to/MANO_RIGHT.pkl \
  /path/to/mano_mean_params.npz \
  /path/to/V_regressor_84.npy /path/to/V_regressor_336.npy
```

Do not publish these license-controlled files.

## 5. Complete setup and validation

To perform every automatic download:

```bash
bash robotic_grounding/scripts/download_required_assets.sh all
```

After manually installing MANO, validate all explicit paths:

```bash
bash robotic_grounding/scripts/download_required_assets.sh check
python reconstruction/scripts/validate_bundle.py --require-checkpoints
```

The robot check also rejects unhydrated Git LFS pointer files.

## 6. Episode-specific objects and submission meshes

Reconstructed interaction objects, support surfaces, retarget Parquets, FlashCHORD checkpoints,
and rollout results are generated artifacts, not downloadable repository assets. They are produced
by the reconstruction/grounding pipeline or restored from a private experiment artifact store.
The Track 3 submission packer requires final reconstructed meshes in the same object body frame as
the policy rollout poses.

## 7. Reproducibility and licensing

- Do not change `V2D_ASSET_COMMIT` without recording the new commit in the submission manifest.
- Keep model and dataset licenses from every upstream project.
- Never commit API tokens or account credentials.
- Keep downloaded assets ignored by Git; only source code, configuration, licenses, and manifests
  belong in the submission-code repository.
