#!/bin/bash -l
# Seed array for a BO (TinyLoRA) run. Usage: MODEL=qwen2.5-7b TASK=gsm8k [CFG=u1-notie] sbatch slurm/bo.sh [--proj-dim 1 --untie ...]
# CFG names the run dir (tinylora-bo[-<cfg>]). Any bf16 card works: 1.5B fits 16G, 7B needs L4/A40+; shrink --n-questions on small cards
#SBATCH -J bo
#SBATCH -a 0-2
#SBATCH -p gpu,gpu-preempt
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --gpus=1
#SBATCH --constraint=l4|a16|a4000|a40|l40s|a100-40g|a100-80g|h100
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 24:00:00
#SBATCH -o .slurm-logs/%x-%A-%a.out

set -e
cd "${SLURM_SUBMIT_DIR:?}"
module load cuda/13.1
export HF_HOME="$PWD/.hf-cache"
# job-private node-local compile caches: concurrent jobs sharing these over NFS hit ESTALE
export UNSLOTH_COMPILE_LOCATION=/tmp/unsloth-cache
export TRITON_CACHE_DIR=/tmp/triton
export VLLM_CACHE_ROOT=/tmp/vllm

MODEL="${MODEL:?}"
TASK="${TASK:?}"
SEED="${SLURM_ARRAY_TASK_ID:?}"

# preemption sends TERM (900s grace), wall-limit sends USR1: python finishes the running trial and exits; trials.json resumes
trap 'kill -USR1 "$pid"' USR1 TERM
"$HOME/.local/bin/uv" run -m turbolora.train_bo \
    --model "$MODEL" \
    --task "$TASK" \
    --out "outputs/runs/${MODEL}/${TASK}/tinylora-bo${CFG:+-$CFG}/seed${SEED}" \
    --seed "$SEED" \
    "$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; done
