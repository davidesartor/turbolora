#!/bin/bash -l
# Seed array for a TuRBO (TinyLoRA) run. Usage: MODEL=qwen2.5-7b TASK=gsm8k [CFG=r2-u1] sbatch slurm/bo.sh [--proj-dim 1 --untie ...]
# CFG names the run dir (tinylora-turbo/<cfg>). Any bf16 card works: 1.5B fits 16G, 7B needs L4/A40+; on small cards lower --batch or --k-rollouts (completions per vLLM call = one GRPO step)
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
source slurm/family.sh
module load cuda/13.1
export HF_HOME="$PWD/.hf-cache"
# job-private node-local compile caches: concurrent jobs sharing these over NFS hit ESTALE
export UNSLOTH_COMPILE_LOCATION=/tmp/unsloth-cache
export TRITON_CACHE_DIR=/tmp/triton
export VLLM_CACHE_ROOT=/tmp/vllm

MODEL="${MODEL:?}"
TASK="${TASK:?}"
SEED="${SLURM_ARRAY_TASK_ID:?}"
OUT="outputs/runs/$(family "$MODEL")/${MODEL}/tinylora-turbo/${CFG:?}/seed${SEED}"
# tag the job with its run dir so resume_sweep.sh can see which seeds are already queued
scontrol update job="$SLURM_JOB_ID" comment="$OUT" || true

# preemption sends TERM (900s grace), wall-limit sends USR1: python finishes the running trial and exits; trials.json resumes
trap 'kill -USR1 "$pid"' USR1 TERM
"$HOME/.local/bin/uv" run -m turbolora.train_turbolora \
    --model "$MODEL" \
    --task "$TASK" \
    --out "$OUT" \
    --seed "$SEED" \
    "$@" &
pid=$!
# `wait` returns early on a trapped signal (128+sig, which `set -e` would treat as fatal and orphan python);
# keep waiting until python actually exits, then propagate its real exit code
status=0
while kill -0 "$pid" 2>/dev/null; do wait "$pid" && status=0 || status=$?; done
exit "$status"
