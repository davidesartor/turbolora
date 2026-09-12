#!/bin/bash -l
# Eval snapshots on one shared engine load; results land in each snapshot's eval@K. Usage: ADAPTERS="outputs/runs/.../snapshots/step-000393 ..." [TASKS="gsm8k math500"] [SAMPLES=4] [SPLIT=train] [MAX_TOKENS=1024] sbatch slurm/eval.sh
# Long batches (train-set evals) go preemptable: sbatch -p gpu,gpu-preempt -q normal --requeue -t 12:00:00 ... (--skip-existing makes a requeue resume per adapter)
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
source slurm/common.sh

"$HOME/.local/bin/uv" run -m turbolora.eval --adapters ${ADAPTERS:?} --tasks ${TASKS:-gsm8k} ${SHOW:+--show $SHOW} --tp "${TP:-1}" --samples "${SAMPLES:-1}" --split "${SPLIT:-test}" --max-tokens "${MAX_TOKENS:-4096}" --skip-existing
