#!/bin/bash -l
# Seed array for a full GRPO run. Usage: MODEL=qwen2.5-7b TASK=easy ADAPTER=lora [LOSS=gspo] [LR=5e-6] [CFG=r1] sbatch slurm/train.sh [--rank 1]
# CFG names the run dir (<adapter>-<loss>[-lr<lr>]-<cfg>); single seed: sbatch -a 0 ...
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
module load cuda/13.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$PWD/.hf-cache"
# job-private node-local compile caches: concurrent jobs sharing these over NFS hit ESTALE
export UNSLOTH_COMPILE_LOCATION=/tmp/unsloth-cache
export TRITON_CACHE_DIR=/tmp/triton
export VLLM_CACHE_ROOT=/tmp/vllm
LR="${LR:-}"
LOSS="${LOSS:-grpo}"
ADAPTER="${ADAPTER:?}"
MODEL="${MODEL:?}"
TASK="${TASK:?}"
CFG="${CFG:-}"
RUN="${ADAPTER}-${LOSS}${LR:+-lr$LR}${CFG:+-$CFG}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
# preemption sends TERM (900s grace), wall-limit sends USR1: both ask python for a checkpoint
trap 'kill -USR1 "$pid"' USR1 TERM
cmd=(
    "$HOME/.local/bin/uv" run -m "turbolora.train_${ADAPTER}"
    --model "$MODEL"
    --task "$TASK"
    --loss "$LOSS"
    --out "outputs/runs/${MODEL}/${TASK}/${RUN}/seed${SEED}"
    --seed "$SEED"
)
if [ -n "$LR" ]; then
    cmd+=(--lr "$LR")
fi
"${cmd[@]}" "$@" &
pid=$!
# `wait` returns early on a trapped signal (128+sig, which `set -e` would treat as fatal);
# keep waiting until python actually exits, then propagate its real exit code
status=0
while kill -0 "$pid" 2>/dev/null; do wait "$pid" && status=0 || status=$?; done
exit "$status"
