#!/usr/bin/env bash
#SBATCH --job-name=v2d-track3-eval
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${CHECKPOINTS_JSON:?export CHECKPOINTS_JSON before submission}"
: "${RESULTS_DIR:?export RESULTS_DIR before submission}"
: "${FLASH_CHORD_PYTHON:?export FLASH_CHORD_PYTHON before submission}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
module load "${CUDA_MODULE:-CUDA/12.8.0}"
bash "${SCRIPT_DIR}/evaluate_policies.sh" "${CHECKPOINTS_JSON}" "${RESULTS_DIR}"

