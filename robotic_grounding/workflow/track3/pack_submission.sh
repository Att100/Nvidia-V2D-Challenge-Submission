#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 RESULTS_DIR MESH_DIR GITHUB_COMMIT_URL OUTPUT_PARQUET" >&2
  exit 2
fi

RESULTS_DIR=$1
MESH_DIR=$2
GITHUB_COMMIT_URL=$3
OUTPUT_PARQUET=$4
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROBOTIC_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
KIT_ROOT=${ROBOTIC_ROOT}/third_party/v2d_submission_kit/v2d_submission_kit
SUBMISSION_PYTHON=${SUBMISSION_PYTHON:-python}

if [[ ! "${GITHUB_COMMIT_URL}" =~ ^https://github.com/[^/]+/[^/]+/commit/[0-9a-fA-F]{7,40}$ ]]; then
  echo "Expected a fixed GitHub /commit/<sha> URL, got: ${GITHUB_COMMIT_URL}" >&2
  exit 1
fi

cd "${KIT_ROOT}"
"${SUBMISSION_PYTHON}" tools/pack_submission.py \
  --track track_3 \
  --results "${RESULTS_DIR}" \
  --meshes "${MESH_DIR}" \
  --code-commit "${GITHUB_COMMIT_URL}" \
  -o "${OUTPUT_PARQUET}"

