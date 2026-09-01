#!/bin/bash
# Keep one eval_sweep job alive until every finished run is evaluated: short QOS holds a single
# job per user, so the chain needs a driver outside Slurm. Usage: setsid nohup slurm/eval_watch.sh &
cd "$(dirname "$0")/.."
LOG=.slurm-logs/eval-watch.log
log() { echo "$(date '+%m-%d %H:%M') $*" >> "$LOG"; }

pending() {
  find outputs/runs -type d -name final_adapter | sort | while read -r a; do
    [ "$(ls "$(dirname "$a")"/eval/*.json 2>/dev/null | wc -l)" -lt 6 ] && echo "$a"
  done
}

log "watcher started on $(hostname)"
while [ -n "$(pending)" ]; do
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
log "watcher done, nothing pending"
