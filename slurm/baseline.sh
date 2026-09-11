#!/bin/bash -l
# Baseline eval of untrained models (short QOS). Usage: MODEL="qwen2.5-7b qwen2.5-7b-math" [TASKS="gsm8k math500"] [SAMPLES="1 4"] [SHOW=5] sbatch slurm/baseline.sh
#SBATCH -J baseline
#SBATCH -p gpu-preempt
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=a100|a100-80g|h100
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 04:00:00
#SBATCH -o .slurm-logs/%x-%j.out

set -e
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/common.sh

for model in ${MODEL:?}; do
    for samples in ${SAMPLES:-1}; do
        "$HOME/.local/bin/uv" run -m turbolora.eval --model "$model" --tasks ${TASKS:-gsm8k} --samples "$samples" --skip-existing ${SHOW:+--show $SHOW} --tp "${TP:-1}"
        release_gpu
    done
done
