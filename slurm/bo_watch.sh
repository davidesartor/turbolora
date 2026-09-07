#!/bin/bash
# Unstick BO jobs whose grading hung: math-verify's one-shot SIGALRM is lost when it fires inside a __del__, leaving sympy
# unbounded (hang or host OOM) while its handler stays installed, so an external SIGALRM raises the timeout and the trial
# goes on. Every INTERVAL the main python of each running `bo` task is py-spy'd via an overlap step; two consecutive dumps
# in math_verify/sympy frames = hung -> kill -ALRM. Usage: sbatch slurm/bo_watch.sh, or ONCE=1 slurm/bo_watch.sh
#SBATCH -J bo-watch
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 48:00:00
#SBATCH -o .slurm-logs/%x-%j.out
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
LOG=.slurm-logs/bo-watch.log
INTERVAL="${INTERVAL:-600}"
log() { echo "$(date '+%m-%d %H:%M') $*" | tee -a "$LOG"; }

# runs on the job's node: prints "<pid> <state>" for the main train python, state = grading|other|none
probe='
job=$(grep -oE "job_[0-9]+" /proc/self/cgroup | head -1)
for p in $(ps -o pid= -u "$USER"); do
  grep -q "$job" /proc/$p/cgroup 2>/dev/null || continue
  tr "\0" " " </proc/$p/cmdline 2>/dev/null | grep -q "bin/python3 -m turbolora.train_turbolora" || continue
  stack=$(timeout 60 "$HOME/.local/bin/uvx" py-spy dump --pid "$p" 2>/dev/null | sed -n "/MainThread/,/^Thread/p")
  [ -z "$stack" ] && { echo "$p unknown"; exit; }
  grep -qE "math_verify|sympy|latex2sympy" <<< "$stack" && echo "$p grading" || echo "$p other"
  exit
done
echo "0 none"'

declare -A strikes
check() {
  for task in $(squeue -u "$USER" -h -t R -n bo -o %i); do
    raw=$(scontrol show job "$task" 2>/dev/null | grep -oE '^JobId=[0-9]+' | cut -d= -f2)
    [ -n "$raw" ] || continue
    read -r pid state < <(timeout 180 srun --jobid="$raw" --overlap --gpus=0 -n1 -c1 --mem=0 --quiet bash -c "$probe" 2>/dev/null)
    if [ "$state" = grading ]; then
      strikes[$task]=$(( ${strikes[$task]:-0} + 1 ))
      log "$task pid $pid in math-verify (strike ${strikes[$task]})"
      if [ "${strikes[$task]}" -ge 2 ]; then
        srun --jobid="$raw" --overlap --gpus=0 -n1 -c1 --mem=0 --quiet kill -ALRM "$pid" && log "$task sent SIGALRM to $pid"
        strikes[$task]=0
      fi
    else
      strikes[$task]=0
    fi
  done
}

if [ -n "$ONCE" ]; then check; exit; fi
log "watching (interval ${INTERVAL}s)"
while :; do check; sleep "$INTERVAL"; done
