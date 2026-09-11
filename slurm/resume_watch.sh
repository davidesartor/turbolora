#!/bin/bash
# Babysit the sweep: every INTERVAL resubmit seeds that died or hit the 24h wall (resume_sweep.sh skips queued
# seeds). Exits once nothing is left to submit and no grpo/bo job is queued.
# Usage: sbatch slurm/resume_watch.sh
#SBATCH -J resume-watch
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 96:00:00
#SBATCH -o .slurm-logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
LOG=.slurm-logs/resume-watch.log
INTERVAL="${INTERVAL:-7200}"
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$LOG"; }

log "watching (interval ${INTERVAL}s)"
while :; do
  submitted=$(GO=1 slurm/resume_sweep.sh 2>&1 | grep -E "Submitted|sbatch")
  [ -n "$submitted" ] && log "$submitted"
  if [ -z "$submitted" ] && ! squeue -u "$USER" -h -o "%j" | grep -qxE "grpo|bo"; then
    log "sweep idle, exiting"
    exit
  fi
  sleep "$INTERVAL"
done
