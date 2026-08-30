#!/bin/bash -l
# Seed array for a full GRPO run. Usage: MODEL=qwen2.5-7b TASK=gsm8k ADAPTER=lora [LOSS=gspo] [LR=5e-6] sbatch slurm/train.sh
#SBATCH -J grpo
#SBATCH -a 0-2
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100-40g|a100-80g|h100
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 24:00:00
#SBATCH -o .slurm-logs/%x-%A-%a.out

set -e
cd "${SLURM_SUBMIT_DIR:?}"
module load cuda/13.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$PWD/.hf-cache"
export UNSLOTH_COMPILE_LOCATION="$PWD/.unsloth-cache"
export WANDB_PROJECT=turbolora
export WANDB_DIR="$PWD/outputs"
grep -qs api.wandb.ai ~/.netrc || export WANDB_MODE=offline

LR="${LR:-5e-6}"
LOSS="${LOSS:-grpo}"
ADAPTER="${ADAPTER:?}"
MODEL="${MODEL:?}"
TASK="${TASK:?}"
TAG="$MODEL-$TASK"
SEED="${SLURM_ARRAY_TASK_ID:?}"

"$HOME/.local/bin/uv" run -m "turbolora.train_${ADAPTER}" \
    --model "$MODEL" \
    --task "$TASK" \
    --loss "$LOSS" \
    --out "outputs/${LOSS}-${ADAPTER}-${TAG}-lr${LR}-seed${SEED}" \
    --lr "$LR" \
    --seed "$SEED" \
    "$@"
