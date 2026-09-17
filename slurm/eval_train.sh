#!/bin/bash -l
# Greedy eval of adapters on their own train tier at the training budget -> eval@K/<tier>-train.json.gz.
# Usage: TASK=hard ADAPTERS="outputs/runs/.../snapshots/step-000393 ..." [SAMPLES=4] sbatch slurm/eval_train.sh
# Batches: PLAN=<file> of lines `<samples> <task> <adapters...>` instead, one engine load per line; --skip-existing makes a requeue resume.
# tmp/tools/launch_train_eval.py cuts pending adapters into ~18-h plans so nothing needs more than a day.
# 1.5B fits the short QOS: sbatch -p gpu -q short -t 04:00:00 --no-requeue --constraint='l4|a40|l40s|a100-40g|a100-80g|h100'
#SBATCH -J eval_train
#SBATCH -p gpu,gpu-preempt
#SBATCH -q normal
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100-40g|a100-80g|h100
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 24:00:00
#SBATCH -o .slurm-logs/%x-%j.out

set -e
cd "${SLURM_SUBMIT_DIR:?}"
module load cuda/13.1
export HF_HOME="$PWD/.hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1  # everything is cached; mass starts hit the HF 429 rate limit
export UNSLOTH_COMPILE_LOCATION=/tmp/unsloth-cache TRITON_CACHE_DIR=/tmp/triton VLLM_CACHE_ROOT=/tmp/vllm  # job-private: shared over NFS they hit ESTALE

if [ -n "$PLAN" ]; then
    while read -r samples task adapters; do
        "$HOME/.local/bin/uv" run -m turbolora.eval --adapters $adapters --tasks "$task" --samples "$samples" --split train --max-tokens 1024 --skip-existing
    done < "$PLAN"
else
    "$HOME/.local/bin/uv" run -m turbolora.eval --adapters ${ADAPTERS:?} --tasks "${TASK:?}" --samples "${SAMPLES:-1}" --split train --max-tokens 1024 --skip-existing
fi
