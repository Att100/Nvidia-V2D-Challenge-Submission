# Checkpoint Download Guide

Large model files are not committed to this cleaned project. This document
restores every checkpoint to the exact path expected by the Hydra configuration.
Run all commands from the repository root.

## 1. Install download clients

```bash
conda activate ego_scene_hoi
python -m pip install gdown huggingface-hub
```

SAM 3D Objects is a gated model. Request access to
[`facebook/sam-3d-objects`](https://huggingface.co/facebook/sam-3d-objects),
accept its license, create a Hugging Face token, and authenticate:

```bash
hf auth login
```

## 2. Download public checkpoints

The helper script downloads FoundationStereo, FoundationPose, the HACO ViT-B
model, UmeTrack, and SAM 3D Objects:

```bash
bash scripts/download_checkpoints.sh all
```

Individual components can be restored separately:

```bash
bash scripts/download_checkpoints.sh foundation_stereo
bash scripts/download_checkpoints.sh foundation_pose
bash scripts/download_checkpoints.sh haco
bash scripts/download_checkpoints.sh umetrack
bash scripts/download_checkpoints.sh sam3d
```

The script uses the official sources:

- [FoundationStereo Google Drive folder](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf)
- [FoundationPose Google Drive folder](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i)
- [HACO checkpoint repository](https://huggingface.co/datasets/dqj5182/haco-checkpoints/tree/main/NeurIPS2025)
- [UmeTrack pretrained model](https://github.com/facebookresearch/UmeTrack/tree/main/pretrained_models)
- [SAM 3D Objects setup instructions](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)

SAM 3D Objects needs about 13 GB. Its official downloader may temporarily use
roughly twice that amount while copying files into the final directory.

## 3. Install MANO manually

MANO is license-controlled and is not downloaded by the helper script.

1. Register at [the MANO website](https://mano.is.tue.mpg.de/).
2. Accept the model license and download MANO v1.2.
3. Extract `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`.
4. Obtain `mano_mean_params.npz`, `V_regressor_84.npy`, and
   `V_regressor_336.npy` from the licensed MANO/HACO distributions used by your
   authorized upstream installation.
5. Install all license-controlled files with the helper:

```bash
bash scripts/install_mano_assets.sh \
  /path/to/MANO_LEFT.pkl /path/to/MANO_RIGHT.pkl \
  /path/to/mano_mean_params.npz \
  /path/to/V_regressor_84.npy /path/to/V_regressor_336.npy
```

Do not rename or commit these files. The left and right models are not
interchangeable. The installer destinations are covered by `.gitignore`.

## 4. Models downloaded automatically at runtime

The following weights use their standard upstream cache and require no manual
placement:

- SAM3: Hugging Face `facebook/sam3`;
- FLUX.2 klein 9B FP8: `rootlocalghost/FLUX.2-klein-9B-FP8`;
- MoGe ViT-L: `Ruicheng/moge-vitl`, used by SAM 3D Objects;
- AnyCalib `anycalib_gen`: downloaded from its GitHub release through Torch Hub;
- VIPE DROID-SLAM: downloaded through `gdown` on first construction.

These files normally appear under the user's Hugging Face and Torch caches, not
inside this repository.

## 5. Expected files and validation

After downloading and installing MANO, run:

```bash
bash scripts/download_checkpoints.sh check
python scripts/validate_bundle.py --require-checkpoints
```

The important paths are:

```text
thirdparty/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
thirdparty/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth
thirdparty/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth
thirdparty/HACO_RELEASE/release_checkpoint/haco_neurips_vit_b_checkpoint.ckpt
thirdparty/UmeTrack/pretrained_models/pretrained_weights.torch
thirdparty/MANO/mano_data/MANO_LEFT.pkl
thirdparty/MANO/mano_data/MANO_RIGHT.pkl
thirdparty/MANO/mano_data/mano_mean_params.npz
thirdparty/HACO_RELEASE/data/base_data/human_models/mano/V_regressor_84.npy
thirdparty/HACO_RELEASE/data/base_data/human_models/mano/V_regressor_336.npy
thirdparty/sam-3d-objects/checkpoints/hf/*.ckpt
```
