#!/bin/bash -l
# Eval trained adapters on one shared engine load; results land in each <run>/eval/. Usage: ADAPTERS="outputs/runs/.../final_adapter ..." [TASKS="gsm8k math500"] sbatch slurm/eval.sh
#SBATCH -J eval
#SBATCH -p gpu
#SBATCH -q short
#SBATCH --gpus=1
#SBATCH --constraint=a100|a100-80g|h100
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 04:00:00
#SBATCH -o .slurm-logs/%x-%j.out

set -e
cd "${SLURM_SUBMIT_DIR:?}"
module load cuda/13.1
export HF_HOME="$PWD/.hf-cache"

uv run -m turbolora.eval --adapters ${ADAPTERS:?} --tasks ${TASKS:-gsm8k} ${SHOW:+--show $SHOW} --tp "${TP:-1}" --skip-existing
