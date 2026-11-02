#!/usr/bin/env bash
#SBATCH --job-name=v2d-track3-train
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${REFERENCE_PATH:?export REFERENCE_PATH before submission}"
: "${OUTPUT_DIR:?export OUTPUT_DIR before submission}"
: "${FLASH_CHORD_PYTHON:?export FLASH_CHORD_PYTHON before submission}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
module load "${CUDA_MODULE:-CUDA/12.8.0}"
bash "${SCRIPT_DIR}/train_policy.sh" "${REFERENCE_PATH}" "${OUTPUT_DIR}"

