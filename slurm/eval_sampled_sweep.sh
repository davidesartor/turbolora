#!/bin/bash -l
# Sampled (4 completions, T=1, GRPO's rollout setting) six-task eval of every finished run's last snapshot plus each base model,
# one base model per job on a single engine load, resubmitting itself until nothing is pending.
# Usage: sbatch slurm/eval_sampled_sweep.sh (all models, one per hop) or MODEL=qwen2.5-7b sbatch ... (that model only; one job per model runs in parallel)
#SBATCH -J eval-sampled
#SBATCH -p gpu,gpu-preempt
#SBATCH -q normal
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100-40g|a100-80g|h100
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 12:00:00
#SBATCH -o .slurm-logs/%x-%j.out

set -e
cd "${SLURM_SUBMIT_DIR:?}"
source slurm/family.sh
module load cuda/13.1
export HF_HOME="$PWD/.hf-cache"
# job-private node-local compile caches: concurrent jobs sharing these over NFS hit ESTALE
export UNSLOTH_COMPILE_LOCATION=/tmp/unsloth-cache
export TRITON_CACHE_DIR=/tmp/triton
export VLLM_CACHE_ROOT=/tmp/vllm

TASKS="gsm8k math500 aime24 amc23 minerva olympiad"
SAMPLES=4
EVAL="$HOME/.local/bin/uv run -m turbolora.eval --tasks $TASKS --samples $SAMPLES --skip-existing"

complete() { [ "$(ls "$1"/eval@$SAMPLES/*.json.gz 2>/dev/null | wc -l)" -ge 6 ]; }

# last snapshot of every finished run (run.json carries train_hours) that still lacks any sampled task
pending() {
  grep -l train_hours outputs/runs/*/${MODEL:-*}/*/*/seed*/run.json | sort | while read -r r; do
    last=$(ls -d "$(dirname "$r")"/snapshots/step-* 2>/dev/null | sort | tail -1)
    [ -n "$last" ] && ! complete "$last" && echo "$last"
  done
}

next=$(pending | head -1)
[ -z "$next" ] && { echo "nothing pending"; exit 0; }

# one engine load serves a single base model: its untrained baseline first, then every pending adapter of that model
model=$(cut -d/ -f4 <<< "$next")
complete "outputs/runs/$(family "$model")/$model/base" || $EVAL --model "$model"
# the baseline's vLLM engine core releases the GPU a few seconds after the process exits; the next engine wants 90% of it
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ]; do sleep 5; done
adapters=$(pending | grep "^outputs/runs/[^/]*/$model/" | tr '\n' ' ')
echo "evaluating $(wc -w <<< "$adapters") adapters of $model"
$EVAL --adapters $adapters

# normal QOS so the chain is not refused while this job still runs (short allows one submitted job per user)
left=$(pending | wc -l)
[ "$left" -eq 0 ] && exit 0
echo "$left adapters still pending"
sbatch -J "$SLURM_JOB_NAME" slurm/eval_sampled_sweep.sh  # MODEL propagates through the exported environment
exit 0
