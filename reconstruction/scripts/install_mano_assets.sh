#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  cat >&2 <<'EOF'
Usage: install_mano_assets.sh MANO_LEFT.pkl MANO_RIGHT.pkl mano_mean_params.npz V_regressor_84.npy V_regressor_336.npy

All inputs must come from MANO/HACO distributions that you obtained under their
respective licenses. This script copies them into the paths used by reconstruction.
EOF
  exit 2
fi

MANO_LEFT=$1
MANO_RIGHT=$2
MEAN_PARAMS=$3
REGRESSOR_84=$4
REGRESSOR_336=$5
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RECON_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

for source in "${MANO_LEFT}" "${MANO_RIGHT}" "${MEAN_PARAMS}" "${REGRESSOR_84}" "${REGRESSOR_336}"; do
  if [[ ! -s "${source}" ]]; then
    echo "Missing or empty licensed MANO/HACO input: ${source}" >&2
    exit 1
  fi
done

MANO_DIR=${RECON_ROOT}/thirdparty/MANO/mano_data
HACO_MANO_DIR=${RECON_ROOT}/thirdparty/HACO_RELEASE/data/base_data/human_models/mano
install -Dm644 "${MANO_LEFT}" "${MANO_DIR}/MANO_LEFT.pkl"
install -Dm644 "${MANO_RIGHT}" "${MANO_DIR}/MANO_RIGHT.pkl"
install -Dm644 "${MEAN_PARAMS}" "${MANO_DIR}/mano_mean_params.npz"
install -Dm644 "${MANO_LEFT}" "${HACO_MANO_DIR}/MANO_LEFT.pkl"
install -Dm644 "${MANO_RIGHT}" "${HACO_MANO_DIR}/MANO_RIGHT.pkl"
install -Dm644 "${REGRESSOR_84}" "${HACO_MANO_DIR}/V_regressor_84.npy"
install -Dm644 "${REGRESSOR_336}" "${HACO_MANO_DIR}/V_regressor_336.npy"

echo "Installed licensed MANO/HACO data under ${RECON_ROOT}."
echo "These paths are ignored by Git; do not commit them."

