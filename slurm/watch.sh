#!/bin/bash -l
# Requeue watcher, run on a login node: resubmits every seed the sweep is short of, until done or LIFETIME_H runs out.
# Usage: [MAX_SLICE_H=12] [MIN_SLICE_H=2] [WATCH_MIN=10] [LIFETIME_H=48] [TRAINEVAL=.] \
#   setsid nohup slurm/watch.sh > .slurm-logs/sweep_watch-$(hostname)-$(date +%m%d-%H%M).out 2>&1 &
# Each pass picks the longest slice that still fits before the next cluster-wide maintenance window; whatever
# times out comes back as `partial` and this picks it up next pass.
# MAX_SLICE_H trades reload cost against schedulability: a short walltime backfills into gaps a long one cannot fit,
# but every slice re-pays ~10-20 min of init plus a full queue wait, which dominates once only a few seeds are left.

set -e
cd "$(dirname "$0")/.."
UV="$HOME/.local/bin/uv"
MAX_SLICE_H=${MAX_SLICE_H:-12}  # each slice re-pays model load and a fresh queue wait, so keep it long while few seeds are left
MIN_SLICE_H=${MIN_SLICE_H:-2}
WATCH_MIN=${WATCH_MIN:-10}
TRAINEVAL=${TRAINEVAL:-.}  # model regex for the train-set eval backlog; '.' = every model
stop_at=$(($(date +%s) + ${LIFETIME_H:-48} * 3600))
submits=.slurm-logs/sweep_watch-submits.log

# Slurm never starts a job that would run into a cluster-wide maintenance window, so a slice longer than the gap
# just sits pending: take the longest that fits, capped at MAX_SLICE_H and floored at MIN_SLICE_H.
slice_hours() {
    local now next gap soonest=
    now=$(date +%s)
    for next in $(scontrol show res --oneliner 2>/dev/null |
        awk '/Flags=[^ ]*MAINT/ && /ALL_NODES/ {match($0, /StartTime=[^ ]+/); print substr($0, RSTART + 10, RLENGTH - 10)}'); do
        gap=$(($(date -d "$next" +%s) - now))
        [ "$gap" -gt 0 ] && { [ -z "$soonest" ] || [ "$gap" -lt "$soonest" ]; } && soonest=$gap
    done
    gap=$MAX_SLICE_H
    [ -n "$soonest" ] && gap=$(((soonest - 1800) / 3600))  # half an hour of margin for queue wait and model load
    [ "$gap" -gt "$MAX_SLICE_H" ] && gap=$MAX_SLICE_H
    [ "$gap" -lt "$MIN_SLICE_H" ] && gap=$MIN_SLICE_H
    echo "$gap"
}

while :; do
    echo "== $(date -u +%FT%TZ)"
    # a node that hands out an invisible GPU kills jobs in seconds ("No CUDA GPUs are available"); stay off it while it misbehaves
    bad=$(sacct -u "$USER" -X -n -S now-3hours --format=State,Elapsed,NodeList%30 |
        awk '$1 == "FAILED" && $2 ~ /^00:0[01]:/ {print $3}' | sort | uniq -c | awk '$1 >= 2 {print $2}' | paste -sd,)
    if [ -n "$bad" ]; then export EXCLUDE="$bad"; echo "excluding $bad"; else unset EXCLUDE; fi

    JOB_TIME="$(slice_hours):00:00"
    echo "slice $JOB_TIME"
    # the counts line is `done=N` alone only when every grid seed is finished; a status that errored says nothing,
    # so never read it as completion -- that once exited the watcher and left the grid idle for days
    grid_done=
    if ! summary=$("$UV" run slurm/sweep.py status); then
        echo "WARN sweep.py status failed; skipping this pass"
    elif [ -z "$(echo "$summary" | tail -1 | sed 's/done=[0-9]*//' | tr -d ' ')" ]; then
        grid_done=1
        echo "training grid complete"
    else
        "$UV" run slurm/sweep.py submit --go --time "$JOB_TIME" | tee -a "$submits"
        # a pending seed submitted under a shorter slice than we can now afford would idle until the window: stretch it
        # (never the short-QOS seed: past its 4 h cap it pends forever on QOSMaxWallDurationPerJobLimit)
        squeue -u "$USER" -h -t PENDING -n train_tinylora,train_turbo,train_lora,train_loraxs -o "%i %l %q" |
            awk -v want="$JOB_TIME" '$2 != want && $2 !~ /-/ && $3 != "short" {print $1}' |
            while read -r j; do scontrol update JobId="$j" TimeLimit="$JOB_TIME" 2>/dev/null || true; done
        # a seed relaunched many times is failing in minutes, not training: flag it, the log line names the cfg and seed
        grep '^MODEL=' "$submits" | sed 's/ --begin=[^ ]*//' | sort | uniq -c | awk '$1 >= 5 {print "WARN relaunched", $1, "times:", $0}'
    fi

    # training has priority: few eval chunks while seeds still need GPUs (they are nice=300, so pending seeds outrank them anyway)
    if [ -n "$grid_done" ]; then eval_jobs=16; else eval_jobs=${EVAL_JOBS:-6}; fi
    evals=$(GO=1 ONLY="$TRAINEVAL" MAX_JOBS="$eval_jobs" "$UV" run python tmp/tools/launch_train_eval.py | tail -3 || echo "train-eval launcher failed")
    echo "$evals"
    case "$evals" in *"0 adapters over 0 models pending"*) [ -n "$grid_done" ] && { echo "all complete"; exit 0; } ;; esac

    [ "$(date +%s)" -lt "$stop_at" ] || { echo "== lifetime over, restart by hand"; exit 0; }
    sleep $((WATCH_MIN * 60))
done

