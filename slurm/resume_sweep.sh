#!/bin/bash -l
# Resubmit every unfinished run (run.json lacks `steps`), one array per config with only the
# missing seeds. Prints the sbatch lines; set GO=1 to actually submit.
# Usage: [GO=1] [STAGGER=45] slurm/resume_sweep.sh
set -e
cd "$(dirname "$0")/.."

STAGGER="${STAGGER:-45}"
delay=0

for run_dir in $(find outputs/runs -mindepth 3 -maxdepth 3 -type d | sort); do
    IFS=/ read -r _ _ model task run <<< "$run_dir"

    seeds=""
    for s in 0 1 2; do
        grep -q '"steps"' "$run_dir/seed$s/run.json" 2>/dev/null || seeds="$seeds,$s"
    done
    [ -z "$seeds" ] && continue

    # run name is <adapter>-<loss>-<cfg>, cfg is r<rank>[-u<proj_dim>]
    adapter="${run%%-*}"
    rest="${run#*-}"
    loss="${rest%%-*}"
    cfg="${rest#*-}"
    rank="${cfg%%-*}"
    rank="${rank#r}"
    extra=(--rank "$rank")
    case "$cfg" in *-u*) extra+=(--proj-dim "${cfg##*-u}") ;; esac

    # HF 429s and node black-holes when a hundred jobs start at once
    begin=()
    [ "$delay" -gt 0 ] && begin=(--begin="now+${delay}seconds")

    echo MODEL="$model" TASK="$task" ADAPTER="$adapter" LOSS="$loss" CFG="$cfg" \
        sbatch -a "${seeds#,}" "${begin[@]}" slurm/train.sh "${extra[@]}"
    if [ -n "$GO" ]; then
        MODEL="$model" TASK="$task" ADAPTER="$adapter" LOSS="$loss" CFG="$cfg" \
            sbatch -a "${seeds#,}" "${begin[@]}" slurm/train.sh "${extra[@]}"
    fi
    delay=$((delay + STAGGER))
done
