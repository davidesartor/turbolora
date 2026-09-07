#!/bin/bash
# Sampled (pass@4) baselines for every single-GPU model that has a greedy baseline but no sampled one: the short QOS holds
# one job per user, so this driver submits one baseline.sh job per model whenever the slot is free and waits for it.
# Usage: sbatch slurm/baseline_watch.sh (or setsid nohup slurm/baseline_watch.sh &)
#SBATCH -J baseline-watch
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 48:00:00
#SBATCH -o .slurm-logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
LOG=.slurm-logs/baseline-watch.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$LOG"; }

TASKS="gsm8k math500 aime24 amc23 minerva olympiad"
SAMPLES=4
MULTI_GPU="qwen2.5-32b qwen2.5-32b-instruct mixtral-8x7b"

pending() {
  for dir in outputs/runs/*/*/base/; do
    model=$(basename "$(dirname "$dir")")
    grep -qw "$model" <<< "$MULTI_GPU" && continue
    [ "$(ls "$dir"/eval@1/*.json.gz 2>/dev/null | wc -l)" -ge 6 ] || continue
    [ "$(ls "$dir"/eval@$SAMPLES/*.json.gz 2>/dev/null | wc -l)" -lt 6 ] && echo "$model"
  done
}
short_busy() { squeue -h -u "$USER" -q short -o %i | grep -q . ; }

log "watcher started on $(hostname) job ${SLURM_JOB_ID:-none}: $(pending | tr '\n' ' ')"
while model=$(pending | head -1) && [ -n "$model" ]; do
  # another short job (a baseline of ours or anything else) holds the slot: wait for it
  while short_busy; do sleep 120; done
  until jid=$(MODEL=$model TASKS=$TASKS SAMPLES=$SAMPLES sbatch --parsable -J "baseline-$model" slurm/baseline.sh 2>>"$LOG"); do sleep 60; done
  log "SUBMITTED $jid $model ($(pending | wc -l) models pending)"
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 120; done
  log "FINISHED $jid $model: $(sacct -j "$jid" -X -n -o State,Elapsed | head -1 | tr -s ' ')"
done
log "watcher done: $(pending | wc -l) models pending"
