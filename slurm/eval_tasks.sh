#!/bin/bash -l
# Benchmark eval of adapters on one shared engine load; results land in each adapter's eval@K. Usage: ADAPTERS="outputs/runs/.../snapshots/step-000393 ..." [TASKS="gsm8k math500"] [SAMPLES=4] sbatch slurm/eval_tasks.sh
#SBATCH -J eval_tasks
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1  # everything is cached; mass starts hit the HF 429 rate limit
export UNSLOTH_COMPILE_LOCATION=/tmp/unsloth-cache TRITON_CACHE_DIR=/tmp/triton VLLM_CACHE_ROOT=/tmp/vllm  # job-private: shared over NFS they hit ESTALE

"$HOME/.local/bin/uv" run -m turbolora.eval --adapters ${ADAPTERS:?} --tasks ${TASKS:-gsm8k} --samples "${SAMPLES:-1}" ${SHOW:+--show $SHOW} --tp "${TP:-1}" --skip-existing
