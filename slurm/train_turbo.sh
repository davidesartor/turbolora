#!/bin/bash -l
# Seed array for TuRBO search of TinyLoRA (needs the tinylora-grpo twin's final_adapter). Usage: MODEL=qwen2.5-7b TASK=hard CFG=r2-u8-b4-t1 sbatch [-a 0] slurm/train_turbo.sh [--untie ...]
# CFG=r<rank>-u<proj_dim>-b<batch>[-t1] names the run dir (tinylora-turbo/<cfg>) and sets --rank/--proj-dim/--batch, -t1 = --no-greedy (GRPO's T=1 sampler).
# 1.5B also fits the 16 GB cards: sbatch --constraint='l4|a16|a4000|a40|l40s|a100-40g|a100-80g|h100'; 7B on an L4 takes ~18 min/trial, drop it: --constraint='a40|l40s|a100-40g|a100-80g|h100'
#SBATCH -J train_turbo
#SBATCH -a 0-2
#SBATCH -p gpu,gpu-preempt
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --gpus=1
#SBATCH --constraint=l4|a40|l40s|a100-40g|a100-80g|h100
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

[[ "${CFG:?}" =~ ^r([0-9]+)-u([0-9]+)-b([0-9]+)(-t1)? ]] || { echo "CFG must be r<rank>-u<proj_dim>-b<batch>[-t1], got $CFG" >&2; exit 1; }

# run dir tagged into the job comment (sweep.py reads it via squeue %k)
OUT="outputs/runs/$("$HOME/.local/bin/uv" run python -c "from turbolora.models import MODELS; print(MODELS['${MODEL:?}'].family)")/${MODEL}/tinylora-turbo/${CFG}/seed${SLURM_ARRAY_TASK_ID:?}"
scontrol update job="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" comment="$OUT" || true

# forward slurm's signals (preemption TERM with 900s grace, wall-limit USR1) to python as USR1 so it checkpoints; `wait` returns
# early on a trapped signal (128+sig, fatal under set -e), so keep waiting until python actually exits and return its real code
trap 'kill -USR1 "$pid"' USR1 TERM
"$HOME/.local/bin/uv" run -m turbolora.train_turbolora --model "$MODEL" --task "${TASK:?}" --out "$OUT" --seed "$SLURM_ARRAY_TASK_ID" --rank "${BASH_REMATCH[1]}" --proj-dim "${BASH_REMATCH[2]}" --batch "${BASH_REMATCH[3]}" ${BASH_REMATCH[4]:+--no-greedy} "$@" &
pid=$! status=0
while kill -0 "$pid" 2>/dev/null; do wait "$pid" && status=0 || status=$?; done
exit "$status"
