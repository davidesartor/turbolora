#!/bin/bash
# Keep one eval_sweep job alive until every finished run is evaluated: short QOS holds a single
# job per user, so the chain needs a driver outside Slurm. Idles while training is still going,
# picking up runs as they reach final_adapter.
# Usage: setsid nohup slurm/eval_watch.sh & (or sbatch slurm/eval_watch.sh to outlive the session)
#SBATCH -J eval-watch
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 48:00:00
#SBATCH -o .slurm-logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
LOG=.slurm-logs/eval-watch.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$LOG"; }

pending() {
  find outputs/runs -type d -name final_adapter | sort | while read -r a; do
    [ "$(ls "$(dirname "$a")"/eval/*.json 2>/dev/null | wc -l)" -lt 6 ] && echo "$a"
  done
}
training() { squeue -h -u "$USER" -r -n grpo -o %i | grep -q . ; }

# hand off to a fresh job before the wall limit rather than dying mid-watch
deadline=$(( $(date +%s) + 47 * 3600 ))
requeue_self() {
  [ "$SLURM_JOB_NAME" = eval-watch ] || return 1
  log "wall limit near, chaining to $(sbatch --parsable slurm/eval_watch.sh)"
  exit 0
}

log "watcher started on $(hostname) job ${SLURM_JOB_ID:-none}"
while [ -n "$(pending)" ] || training; do
  [ "$(date +%s)" -ge "$deadline" ] && requeue_self

  # nothing evaluable yet, but runs are still training: wait for one to finish
  if [ -z "$(pending)" ]; then sleep 300; continue; fi

  # adopt the job already in the queue (watcher restart, or the sweep's own resubmit), else submit
  jid=$(squeue -h -u "$USER" -n eval-sweep -o %i | head -1)
  if [ -n "$jid" ]; then
    log "ADOPTED job $jid ($(pending | wc -l) adapters pending)"
  else
    until jid=$(sbatch --parsable slurm/eval_sweep.sh 2>>"$LOG"); do sleep 60; done
    log "SUBMITTED job $jid ($(pending | wc -l) adapters pending)"
  fi
  while squeue -h -j "$jid" 2>/dev/null | grep -q .; do sleep 120; done
  log "FINISHED job $jid: $(sacct -j "$jid" -X -n -o State,Elapsed | head -1 | tr -s ' ')"
done
log "watcher done, nothing pending and no training left"
