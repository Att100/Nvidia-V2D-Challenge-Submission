#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 REFERENCE_PATH OUTPUT_DIR [HYDRA_OVERRIDE ...]" >&2
  exit 2
fi

REFERENCE_PATH=$1
OUTPUT_DIR=$2
shift 2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROBOTIC_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
FLASH_ROOT=${ROBOTIC_ROOT}/flash_chord
FLASH_CHORD_PYTHON=${FLASH_CHORD_PYTHON:-python}
V2D_CACHE_ROOT=${V2D_CACHE_ROOT:-${XDG_CACHE_HOME:-/tmp/v2d-track3-cache}}
FLASH_WORLD_COUNT=${FLASH_WORLD_COUNT:-4096}
FLASH_TOTAL_STEPS=${FLASH_TOTAL_STEPS:-250003456}
FLASH_CHECKPOINT_INTERVAL=${FLASH_CHECKPOINT_INTERVAL:-10000384}

if [[ ! -e "${REFERENCE_PATH}" ]]; then
  echo "Reference does not exist: ${REFERENCE_PATH}" >&2
  exit 1
fi
if ! command -v "${FLASH_CHORD_PYTHON}" >/dev/null 2>&1 && [[ ! -x "${FLASH_CHORD_PYTHON}" ]]; then
  echo "FlashCHORD Python is not executable: ${FLASH_CHORD_PYTHON}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" "${V2D_CACHE_ROOT}/warp" "${V2D_CACHE_ROOT}/jax"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${FLASH_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${V2D_CACHE_ROOT}}
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-${V2D_CACHE_ROOT}/warp}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-${V2D_CACHE_ROOT}/jax}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}

resume_args=()
if [[ -n "${FLASH_RESUME_CHECKPOINT:-}" ]]; then
  if [[ ! -f "${FLASH_RESUME_CHECKPOINT}" ]]; then
    echo "Resume checkpoint does not exist: ${FLASH_RESUME_CHECKPOINT}" >&2
    exit 1
  fi
  resume_args+=(training.resume=true "training.checkpoint_path=${FLASH_RESUME_CHECKPOINT}")
fi

echo "[train] reference=${REFERENCE_PATH}"
echo "[train] output=${OUTPUT_DIR}"
echo "[train] python=${FLASH_CHORD_PYTHON}"

cd "${FLASH_ROOT}"
"${FLASH_CHORD_PYTHON}" scripts/train_flash_sac.py \
  experiment=sharpa_flash_sac \
  "task.parquet=${REFERENCE_PATH}" \
  task.motion_speed=1.0 \
  task.source_frame_playback=false \
  task.motion_start_frame=0 \
  task.motion_end_frame=-1 \
  "scene.world_count=${FLASH_WORLD_COUNT}" \
  scene.object_scale_min=1.0 \
  scene.object_scale_max=1.0 \
  reset.seed=42 \
  "training.total_environment_steps=${FLASH_TOTAL_STEPS}" \
  "training.checkpoint.save_interval_environment_steps=${FLASH_CHECKPOINT_INTERVAL}" \
  logging.mode=disabled \
  "output_dir=${OUTPUT_DIR}" \
  "${resume_args[@]}" \
  "$@"

