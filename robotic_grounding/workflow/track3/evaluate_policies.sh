#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 CHECKPOINTS_JSON RESULTS_DIR" >&2
  exit 2
fi

CHECKPOINTS_JSON=$1
RESULTS_DIR=$2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROBOTIC_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
FLASH_ROOT=${ROBOTIC_ROOT}/flash_chord
KIT_ROOT=${ROBOTIC_ROOT}/third_party/v2d_submission_kit/v2d_submission_kit
FLASH_CHORD_PYTHON=${FLASH_CHORD_PYTHON:-python}
V2D_CACHE_ROOT=${V2D_CACHE_ROOT:-${XDG_CACHE_HOME:-/tmp/v2d-track3-cache}}
WORLD_COUNT=${WORLD_COUNT:-1}

if [[ ! -f "${CHECKPOINTS_JSON}" ]]; then
  echo "Checkpoint map does not exist: ${CHECKPOINTS_JSON}" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}" "${V2D_CACHE_ROOT}/warp" "${V2D_CACHE_ROOT}/jax"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${FLASH_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HYDRA_FULL_ERROR=1
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${V2D_CACHE_ROOT}}
export WARP_CACHE_PATH=${WARP_CACHE_PATH:-${V2D_CACHE_ROOT}/warp}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-${V2D_CACHE_ROOT}/jax}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

cd "${KIT_ROOT}"
"${FLASH_CHORD_PYTHON}" examples/run_policy_evaluation.py \
  --checkpoints "${CHECKPOINTS_JSON}" \
  --results "${RESULTS_DIR}" \
  --runtime-root "${FLASH_ROOT}" \
  --world-count "${WORLD_COUNT}"

