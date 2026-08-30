#!/bin/bash -l
# Quick test eval (short QOS). Usage: MODEL=qwen2.5-7b [TASKS="gsm8k math500"] [SHOW=5] sbatch slurm/eval.sh
#SBATCH -J eval
#SBATCH -p gpu-preempt
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=a100|a100-80g|h100
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 01:00:00
#SBATCH -o .slurm-logs/%x-%j.out

set -e
cd "${SLURM_SUBMIT_DIR:?}"
module load cuda/13.1
export HF_HOME="$PWD/.hf-cache"

uv run -m turbolora.eval --model "${MODEL:?}" --tasks ${TASKS:-gsm8k} ${SHOW:+--show $SHOW} --tp "${TP:-1}"
