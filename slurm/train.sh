#!/bin/bash -l
# Seed array for a full GRPO run. Usage: MODEL=qwen2.5-7b TASK=easy ADAPTER=lora [LOSS=gspo] [LR=5e-6] [CFG=r1] sbatch slurm/train.sh [--rank 1]
# CFG names the run dir (<adapter>-<loss>[-lr<lr>]-<cfg>); single seed: sbatch -a 0 ...
# 1.5B fits an L4/A40: sbatch --constraint='l4|a40|l40s|a100-40g|a100-80g|h100' (resume_sweep.sh does this)
#SBATCH -J grpo
#SBATCH -a 0-2
#SBATCH -p gpu,gpu-preempt
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100-40g|a100-80g|h100
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 24:00:00
#SBATCH -o .slurm-logs/%x-%A-%a.out

set -e
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/common.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LR="${LR:-}"
LOSS="${LOSS:-grpo}"
ADAPTER="${ADAPTER:?}"
MODEL="${MODEL:?}"
TASK="${TASK:?}"
CFG="${CFG:-}"
RUN="${ADAPTER}-${LOSS}/${CFG:?}${LR:+-lr$LR}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
OUT="outputs/runs/$(family "$MODEL")/${MODEL}/${RUN}/seed${SEED}"
# tag the job with its run dir so resume_sweep.sh can see which seeds are already queued
scontrol update job="$SLURM_JOB_ID" comment="$OUT" || true
cmd=(
    "$HOME/.local/bin/uv" run -m "turbolora.train_${ADAPTER}"
    --model "$MODEL"
    --task "$TASK"
    --loss "$LOSS"
    --out "$OUT"
    --seed "$SEED"
)
if [ -n "$LR" ]; then
    cmd+=(--lr "$LR")
fi
run_signalled "${cmd[@]}" "$@"
