#!/bin/bash -l
# Seed array for GRPO of LoRA-XS. Usage: MODEL=qwen2.5-7b TASK=hard CFG=r8 sbatch [-a 0] slurm/train_loraxs.sh [extra train_loraxs args]
# CFG=r<rank> names the run dir (loraxs-grpo/<cfg>) and sets --rank. 1.5B fits an L4/A40: sbatch --constraint='l4|a40|l40s|a100-40g|a100-80g|h100'
#SBATCH -J train_loraxs
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
export HF_HOME="$PWD/.hf-cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1  # everything is cached; mass starts hit the HF 429 rate limit
export UNSLOTH_COMPILE_LOCATION=/tmp/unsloth-cache TRITON_CACHE_DIR=/tmp/triton VLLM_CACHE_ROOT=/tmp/vllm  # job-private: shared over NFS they hit ESTALE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ "${CFG:?}" =~ ^r([0-9]+) ]] || { echo "CFG must be r<rank>, got $CFG" >&2; exit 1; }

# run dir tagged into the job comment (sweep.py reads it via squeue %k)
OUT="outputs/runs/$("$HOME/.local/bin/uv" run python -c "from turbolora.models import MODELS; print(MODELS['${MODEL:?}'].family)")/${MODEL}/loraxs-grpo/${CFG}/seed${SLURM_ARRAY_TASK_ID:?}"
scontrol update job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" comment="$OUT" || true

# a run stamped before a requeue (ghost job after a node/DNS hiccup) is done: don't reload the model to rediscover that
grep -q train_hours "$OUT/run.json" 2>/dev/null && { echo "$OUT already finished"; exit 0; }

# forward slurm's signals (preemption TERM with 900s grace, wall-limit USR1) to python as USR1 so it checkpoints; `wait` returns
# early on a trapped signal (128+sig, fatal under set -e), so keep waiting until python actually exits and return its real code
trap 'kill -USR1 "$pid"' USR1 TERM
"$HOME/.local/bin/uv" run -m turbolora.train_loraxs --model "$MODEL" --task "${TASK:?}" --out "$OUT" --seed "$SLURM_ARRAY_TASK_ID" --rank "${BASH_REMATCH[1]}" "$@" &
pid=$! status=0
while kill -0 "$pid" 2>/dev/null; do wait "$pid" && status=0 || status=$?; done
exit "$status"
