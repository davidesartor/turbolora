#!/bin/bash -l
# Baseline eval of untrained models (short QOS). Usage: MODEL="qwen2.5-7b qwen2.5-7b-math" [TASKS="gsm8k math500"] [SAMPLES=4] [SHOW=5] sbatch slurm/baseline.sh
#SBATCH -J baseline
#SBATCH -p gpu-preempt
#SBATCH --qos=short
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

# eval.py exits via os._exit, so the vLLM engine core can outlive it and keep the GPU; the next engine wants 90% of it
release_gpu() {
    for _ in $(seq 12); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && return
        sleep 5
    done
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    sleep 5
}

for model in ${MODEL:?}; do
    uv run -m turbolora.eval --model "$model" --tasks ${TASKS:-gsm8k} --samples "${SAMPLES:-1}" --skip-existing ${SHOW:+--show $SHOW} --tp "${TP:-1}"
    release_gpu
done
