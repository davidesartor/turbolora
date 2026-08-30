#!/bin/bash -l
# Cheap end-to-end validation: few GRPO steps, production memory profile. Extra args pass to train.py:
#   MODEL=qwen2.5-7b-instruct TASK=gsm8k ADAPTER=tinylora sbatch slurm/smoke.sh --rank 2 --proj-dim 13 --tie 196
#SBATCH -J smoke
#SBATCH -p gpu-preempt
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100-40g|a100-80g|h100
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 01:30:00
#SBATCH -o .slurm-logs/%x-%j.out

set -e
cd "${SLURM_SUBMIT_DIR:?}"
module load cuda/13.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$PWD/.hf-cache"
export UNSLOTH_COMPILE_LOCATION="$PWD/.unsloth-cache"
export WANDB_PROJECT=turbolora
export WANDB_DIR="$PWD/outputs"
grep -qs api.wandb.ai ~/.netrc || export WANDB_MODE=offline
MODEL="${MODEL:?}"
TASK="${TASK:?}"

uv run -m "turbolora.train_${ADAPTER:?}" \
    --model "$MODEL" \
    --task "$TASK" \
    --out "outputs/smoke-${MODEL}-${SLURM_JOB_ID}" \
    --max-steps 3 \
    "$@"
