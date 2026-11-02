#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1" >&2
    exit 1
  fi
}

new_temp_dir() {
  mktemp -d "${TMPDIR:-/tmp}/track3-checkpoints.XXXXXX"
}

remove_temp_dir() {
  local path="$1"
  if [[ "$path" == "${TMPDIR:-/tmp}/track3-checkpoints."* && -d "$path" ]]; then
    find "$path" -depth -delete
  fi
}

download_foundation_stereo() {
  require_command gdown
  local temporary checkpoint
  temporary="$(new_temp_dir)"
  gdown --folder \
    "https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf" \
    -O "$temporary"
  checkpoint="$(find "$temporary" -type f -path '*/23-51-11/model_best_bp2.pth' -print -quit)"
  test -n "$checkpoint"
  install -Dm644 "$checkpoint" \
    "$ROOT/thirdparty/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
  remove_temp_dir "$temporary"
}

download_foundation_pose() {
  require_command gdown
  local temporary refiner scorer refiner_config scorer_config
  temporary="$(new_temp_dir)"
  gdown --folder \
    "https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i" \
    -O "$temporary"
  refiner="$(find "$temporary" -type f -path '*/2023-10-28-18-33-37/model_best.pth' -print -quit)"
  scorer="$(find "$temporary" -type f -path '*/2024-01-11-20-02-45/model_best.pth' -print -quit)"
  refiner_config="$(find "$temporary" -type f -path '*/2023-10-28-18-33-37/config.yml' -print -quit)"
  scorer_config="$(find "$temporary" -type f -path '*/2024-01-11-20-02-45/config.yml' -print -quit)"
  test -n "$refiner"
  test -n "$scorer"
  test -n "$refiner_config"
  test -n "$scorer_config"
  install -Dm644 "$refiner" \
    "$ROOT/thirdparty/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth"
  install -Dm644 "$scorer" \
    "$ROOT/thirdparty/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"
  install -Dm644 "$refiner_config" \
    "$ROOT/thirdparty/FoundationPose/weights/2023-10-28-18-33-37/config.yml"
  install -Dm644 "$scorer_config" \
    "$ROOT/thirdparty/FoundationPose/weights/2024-01-11-20-02-45/config.yml"
  remove_temp_dir "$temporary"
}

download_haco() {
  require_command wget
  local destination
  destination="$ROOT/thirdparty/HACO_RELEASE/release_checkpoint/haco_neurips_vit_b_checkpoint.ckpt"
  mkdir -p "$(dirname "$destination")"
  wget -c \
    "https://huggingface.co/datasets/dqj5182/haco-checkpoints/resolve/main/NeurIPS2025/haco_neurips_vit_b_checkpoint.ckpt" \
    -O "$destination"
}

download_umetrack() {
  require_command wget
  local destination
  destination="$ROOT/thirdparty/UmeTrack/pretrained_models/pretrained_weights.torch"
  mkdir -p "$(dirname "$destination")"
  wget -c \
    "https://raw.githubusercontent.com/facebookresearch/UmeTrack/main/pretrained_models/pretrained_weights.torch" \
    -O "$destination"
}

download_sam3d() {
  require_command hf
  local temporary source destination
  temporary="$(new_temp_dir)"
  hf download \
    --repo-type model \
    --local-dir "$temporary" \
    --max-workers 1 \
    facebook/sam-3d-objects
  source="$temporary/checkpoints"
  destination="$ROOT/thirdparty/sam-3d-objects/checkpoints/hf"
  test -d "$source"
  mkdir -p "$destination"
  cp -a "$source/." "$destination/"
  remove_temp_dir "$temporary"
}

check_checkpoints() {
  local missing=0
  local required=(
    "thirdparty/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth"
    "thirdparty/FoundationPose/weights/2023-10-28-18-33-37/config.yml"
    "thirdparty/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth"
    "thirdparty/FoundationPose/weights/2024-01-11-20-02-45/config.yml"
    "thirdparty/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth"
    "thirdparty/HACO_RELEASE/release_checkpoint/haco_neurips_vit_b_checkpoint.ckpt"
    "thirdparty/UmeTrack/pretrained_models/pretrained_weights.torch"
    "thirdparty/MANO/mano_data/MANO_LEFT.pkl"
    "thirdparty/MANO/mano_data/MANO_RIGHT.pkl"
    "thirdparty/MANO/mano_data/mano_mean_params.npz"
    "thirdparty/HACO_RELEASE/data/base_data/human_models/mano/MANO_LEFT.pkl"
    "thirdparty/HACO_RELEASE/data/base_data/human_models/mano/MANO_RIGHT.pkl"
    "thirdparty/HACO_RELEASE/data/base_data/human_models/mano/V_regressor_84.npy"
    "thirdparty/HACO_RELEASE/data/base_data/human_models/mano/V_regressor_336.npy"
    "thirdparty/sam-3d-objects/checkpoints/hf/pipeline.yaml"
    "thirdparty/sam-3d-objects/checkpoints/hf/ss_generator.ckpt"
    "thirdparty/sam-3d-objects/checkpoints/hf/slat_generator.ckpt"
    "thirdparty/sam-3d-objects/checkpoints/hf/ss_decoder.ckpt"
    "thirdparty/sam-3d-objects/checkpoints/hf/slat_decoder_gs.ckpt"
    "thirdparty/sam-3d-objects/checkpoints/hf/slat_decoder_gs_4.ckpt"
    "thirdparty/sam-3d-objects/checkpoints/hf/slat_decoder_mesh.ckpt"
    "thirdparty/sam-3d-objects/checkpoints/hf/slat_encoder.ckpt"
  )
  for relative in "${required[@]}"; do
    if [[ ! -s "$ROOT/$relative" ]]; then
      echo "MISSING $relative"
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "Checkpoint check failed. MANO must be installed manually; see docs/CHECKPOINTS.md." >&2
    return 1
  fi
  echo "All required explicit checkpoints are present."
}

usage() {
  echo "Usage: $0 {all|foundation_stereo|foundation_pose|haco|umetrack|sam3d|check}"
}

target="${1:-all}"
case "$target" in
  all)
    download_foundation_stereo
    download_foundation_pose
    download_haco
    download_umetrack
    download_sam3d
    echo "Automatic downloads finished. Install MANO manually, then run: $0 check"
    ;;
  foundation_stereo) download_foundation_stereo ;;
  foundation_pose) download_foundation_pose ;;
  haco) download_haco ;;
  umetrack) download_umetrack ;;
  sam3d) download_sam3d ;;
  check) check_checkpoints ;;
  *) usage; exit 2 ;;
esac
