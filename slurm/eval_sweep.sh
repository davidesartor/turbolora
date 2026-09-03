#!/bin/bash -l
# Eval every snapshot that still lacks results, one base model per job, resubmitting itself
# until nothing is pending (short QOS allows only one queued job). Usage: sbatch slurm/eval_sweep.sh
#SBATCH -J eval-sweep
#SBATCH -p gpu,gpu-preempt
#SBATCH -q short
#SBATCH --gpus=1
#SBATCH --constraint=l40s|a100|a100-80g|h100
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 04:00:00
#SBATCH -o .slurm-logs/%x-%j.out

set -e
cd "${SLURM_SUBMIT_DIR:?}"
module load cuda/13.1
export HF_HOME="$PWD/.hf-cache"

TASKS="gsm8k math500 aime24 amc23 minerva olympiad"

# snapshots missing any task's results, oldest run first
pending() {
  find outputs/runs -path "*/snapshots/step-*" -name adapter_config.json | sort | while read -r c; do
    a=$(dirname "$c")
    [ "$(ls "$a"/*.json.gz 2>/dev/null | wc -l)" -lt 6 ] && echo "$a"
  done
}

next=$(pending | head -1)
[ -z "$next" ] && { echo "nothing pending"; exit 0; }

# one engine load serves a single base model, so take every pending adapter of that model
model=$(cut -d/ -f3 <<< "$next")
adapters=$(pending | grep "^outputs/runs/$model/" | tr '\n' ' ')
echo "evaluating $(wc -w <<< "$adapters") adapters of $model"

uv run -m turbolora.eval --adapters $adapters --tasks $TASKS --skip-existing

# short QOS allows one submitted job per user, so this resubmit is refused while this job still holds the slot
left=$(pending | wc -l)
[ "$left" -eq 0 ] && exit 0
echo "$left adapters still pending"
sbatch slurm/eval_sweep.sh || echo "resubmit refused; run 'sbatch slurm/eval_sweep.sh' once the short slot frees"
exit 0
