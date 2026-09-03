#!/bin/bash -l
# Seed array for a BO (TinyLoRA) run. Usage: MODEL=qwen2.5-7b TASK=gsm8k [CFG=u1-notie] sbatch slurm/bo.sh [--proj-dim 1 --untie ...]
# CFG names the run dir (tinylora-bo[-<cfg>])
#SBATCH -J bo
#SBATCH -a 0-2
#SBATCH -p gpu-preempt
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

MODEL="${MODEL:?}"
TASK="${TASK:?}"
SEED="${SLURM_ARRAY_TASK_ID:?}"

# preemption sends TERM (900s grace), wall-limit sends USR1: python finishes the running trial and exits; trials.json resumes
trap 'kill -USR1 "$pid"' USR1 TERM
uv run -m turbolora.train_bo \
    --model "$MODEL" \
    --task "$TASK" \
    --out "outputs/runs/${MODEL}/${TASK}/tinylora-bo${CFG:+-$CFG}/seed${SEED}" \
    --seed "$SEED" \
    "$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do wait "$pid"; done
