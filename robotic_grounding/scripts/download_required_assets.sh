#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROBOTIC_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${ROBOTIC_ROOT}/.." && pwd)
RECON_ROOT=${PROJECT_ROOT}/reconstruction

V2D_ASSET_REPOSITORY=${V2D_ASSET_REPOSITORY:-https://github.com/nvidia-isaac/video_to_data.git}
V2D_ASSET_COMMIT=${V2D_ASSET_COMMIT:-8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef}
V2D_ASSET_CACHE_ROOT=${V2D_ASSET_CACHE_ROOT:-${XDG_CACHE_HOME:-/tmp}/v2d-track3-assets}
ASSET_CHECKOUT=${V2D_ASSET_CACHE_ROOT}/video_to_data-${V2D_ASSET_COMMIT}

UPSTREAM_ASSET_PATHS=(
  robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/sharpawave
  robotic_grounding/flash_chord/src/flash_chord/assets/meshes/sharpawave/left
  robotic_grounding/flash_chord/src/flash_chord/assets/meshes/sharpawave/right
  robotic_grounding/flash_chord/src/flash_chord/assets/meshes/sharpawave/left_cvx
  robotic_grounding/flash_chord/src/flash_chord/assets/meshes/sharpawave/right_cvx
  robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/sharpawave
  robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/meshes/sharpa_wave
)

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    return 1
  fi
}

require_git_lfs() {
  require_command git
  if ! git lfs version >/dev/null 2>&1; then
    echo "Git LFS is required. Install it before downloading robotic assets." >&2
    return 1
  fi
}

initialize_asset_checkout() {
  mkdir -p "${V2D_ASSET_CACHE_ROOT}"
  if [[ -e "${ASSET_CHECKOUT}" && ! -d "${ASSET_CHECKOUT}/.git" ]]; then
    echo "Asset cache exists but is not a Git checkout: ${ASSET_CHECKOUT}" >&2
    return 1
  fi
  if [[ ! -d "${ASSET_CHECKOUT}/.git" ]]; then
    git init "${ASSET_CHECKOUT}"
    git -C "${ASSET_CHECKOUT}" remote add origin "${V2D_ASSET_REPOSITORY}"
    git -C "${ASSET_CHECKOUT}" lfs install --local
  fi
  git -C "${ASSET_CHECKOUT}" sparse-checkout init --cone
  git -C "${ASSET_CHECKOUT}" sparse-checkout set "${UPSTREAM_ASSET_PATHS[@]}"
}

fetch_asset_commit() {
  GIT_LFS_SKIP_SMUDGE=1 git -C "${ASSET_CHECKOUT}" fetch --depth=1 origin "${V2D_ASSET_COMMIT}"
  GIT_LFS_SKIP_SMUDGE=1 git -C "${ASSET_CHECKOUT}" checkout --detach FETCH_HEAD

  local include
  include=$(IFS=,; echo "${UPSTREAM_ASSET_PATHS[*]}")
  git -C "${ASSET_CHECKOUT}" lfs pull --include="${include}" origin "${V2D_ASSET_COMMIT}"

  local actual
  actual=$(git -C "${ASSET_CHECKOUT}" rev-parse HEAD)
  if [[ "${actual}" != "${V2D_ASSET_COMMIT}" ]]; then
    echo "Asset checkout mismatch: expected ${V2D_ASSET_COMMIT}, got ${actual}" >&2
    return 1
  fi
}

install_robotic_assets() {
  require_git_lfs
  require_command rsync
  initialize_asset_checkout
  fetch_asset_commit

  local relative source destination
  for relative in "${UPSTREAM_ASSET_PATHS[@]}"; do
    source=${ASSET_CHECKOUT}/${relative}
    destination=${PROJECT_ROOT}/${relative}
    if [[ ! -d "${source}" ]]; then
      echo "Downloaded asset directory is missing: ${source}" >&2
      return 1
    fi
    mkdir -p "${destination}"
    rsync -a "${source}/" "${destination}/"
  done
  check_robotic_assets
}

check_not_lfs_pointer() {
  local path=$1
  if [[ ! -s "${path}" ]]; then
    echo "MISSING ${path}" >&2
    return 1
  fi
  if head -n 1 "${path}" | grep -q '^version https://git-lfs.github.com/spec/v1'; then
    echo "UNHYDRATED_LFS_POINTER ${path}" >&2
    return 1
  fi
}

check_robotic_assets() {
  local required=(
    "${ROBOTIC_ROOT}/flash_chord/src/flash_chord/assets/urdfs/sharpawave/left_sharpawave.urdf"
    "${ROBOTIC_ROOT}/flash_chord/src/flash_chord/assets/urdfs/sharpawave/right_sharpawave.urdf"
    "${ROBOTIC_ROOT}/flash_chord/src/flash_chord/assets/meshes/sharpawave/left_cvx/left_hand_C_MC_cut.STL"
    "${ROBOTIC_ROOT}/flash_chord/src/flash_chord/assets/meshes/sharpawave/right_cvx/right_hand_C_MC_cut.STL"
    "${ROBOTIC_ROOT}/source/robotic_grounding/robotic_grounding/assets/urdfs/sharpawave/dual_sharpa_wave.urdf"
    "${ROBOTIC_ROOT}/source/robotic_grounding/robotic_grounding/assets/urdfs/meshes/sharpa_wave/left/left_hand_C_MC.STL"
    "${ROBOTIC_ROOT}/source/robotic_grounding/robotic_grounding/assets/urdfs/meshes/sharpa_wave/right/right_hand_C_MC.STL"
  )
  local path missing=0
  for path in "${required[@]}"; do
    if ! check_not_lfs_pointer "${path}"; then
      missing=1
    fi
  done
  if [[ ${missing} -ne 0 ]]; then
    echo "Robotic asset check failed." >&2
    return 1
  fi
  echo "Robotic grounding assets are present and hydrated."
}

download_reconstruction_assets() {
  local downloader=${RECON_ROOT}/scripts/download_checkpoints.sh
  if [[ ! -x "${downloader}" ]]; then
    if [[ ! -f "${downloader}" ]]; then
      echo "Missing reconstruction downloader: ${downloader}" >&2
      return 1
    fi
    chmod +x "${downloader}"
  fi
  "${downloader}" all
  echo "Automatic reconstruction downloads finished. Install licensed MANO models manually."
  echo "See ${RECON_ROOT}/docs/CHECKPOINTS.md"
}

check_reconstruction_assets() {
  local downloader=${RECON_ROOT}/scripts/download_checkpoints.sh
  if [[ ! -f "${downloader}" ]]; then
    echo "Missing reconstruction downloader: ${downloader}" >&2
    return 1
  fi
  bash "${downloader}" check
}

usage() {
  cat <<EOF
Usage: $0 {robotic_grounding|reconstruction|all|check}

  robotic_grounding  Download only SharpaWave assets needed by retarget and FlashCHORD (~48 MiB).
  reconstruction     Download public reconstruction checkpoints; MANO remains manual.
  all                Run both automatic download paths.
  check              Validate robotic assets and all explicit reconstruction checkpoints.

Environment overrides:
  V2D_ASSET_REPOSITORY  Upstream video_to_data Git URL.
  V2D_ASSET_COMMIT      Pinned upstream asset commit.
  V2D_ASSET_CACHE_ROOT  External sparse Git/LFS cache directory.
EOF
}

target=${1:-all}
case "${target}" in
  robotic_grounding)
    install_robotic_assets
    ;;
  reconstruction)
    download_reconstruction_assets
    ;;
  all)
    install_robotic_assets
    download_reconstruction_assets
    ;;
  check)
    check_robotic_assets
    check_reconstruction_assets
    ;;
  *)
    usage
    exit 2
    ;;
esac
