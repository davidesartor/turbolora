# Shared job setup, sourced by train.sh, bo.sh, eval.sh and baseline.sh after `cd "$SLURM_SUBMIT_DIR"`.
source slurm/family.sh
module load cuda/13.1
export HF_HOME="$PWD/.hf-cache"
# job-private node-local compile caches: concurrent jobs sharing these over NFS hit ESTALE
export UNSLOTH_COMPILE_LOCATION=/tmp/unsloth-cache
export TRITON_CACHE_DIR=/tmp/triton
export VLLM_CACHE_ROOT=/tmp/vllm

# Run python in the background, forwarding slurm's signals (preemption TERM with 900s grace, wall-limit USR1) as USR1 so it
# checkpoints. `wait` returns early on a trapped signal (128+sig, which `set -e` would treat as fatal and orphan python):
# keep waiting until python actually exits, then return its real exit code.
run_signalled() {
    trap 'kill -USR1 "$pid"' USR1 TERM
    "$@" &
    pid=$!
    local status=0
    while kill -0 "$pid" 2>/dev/null; do wait "$pid" && status=0 || status=$?; done
    return "$status"
}

# eval.py exits via os._exit, so the vLLM engine core can outlive it and keep the GPU; the next engine wants 90% of it
release_gpu() {
    for _ in $(seq 12); do
        [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && return
        sleep 5
    done
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
    sleep 5
}
