#!/bin/bash -l
# Resubmit every unfinished run (run.json lacks `steps`), one array per config with only the
# missing seeds. Prints the sbatch lines; set GO=1 to actually submit.
# Usage: [GO=1] [STAGGER=45] [SKIP=<run-name regex>] slurm/resume_sweep.sh
set -e
cd "$(dirname "$0")/.."

STAGGER="${STAGGER:-45}"
delay=0

for run_dir in $(find outputs/runs -mindepth 3 -maxdepth 3 -type d | sort); do
    IFS=/ read -r _ _ model task run <<< "$run_dir"
    [ -n "$SKIP" ] && [[ "$run" =~ $SKIP ]] && continue

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

    # tinylora-{bo,turbo}-u<proj_dim> runs go through bo.sh (rank stays its default); 7B needs a 24G+ card
    if [ "$loss" = bo ] || [ "$loss" = turbo ]; then
        extra=(--proj-dim "${cfg#u}")
        case "$model" in qwen2.5-1.5b*) ;; *) begin+=(--constraint='l4|a40|l40s|a100-40g|a100-80g|h100') ;; esac
        script=slurm/bo.sh
    else
        # 1.5B GRPO fits an L4/A40; the script's default constraint keeps the 40G+ cards for 7B
        case "$model" in qwen2.5-1.5b*) begin+=(--constraint='l4|a40|l40s|a100-40g|a100-80g|h100') ;; esac
        script=slurm/train.sh
    fi

    echo MODEL="$model" TASK="$task" ADAPTER="$adapter" LOSS="$loss" CFG="$cfg" \
        sbatch -a "${seeds#,}" "${begin[@]}" "$script" "${extra[@]}"
    if [ -n "$GO" ]; then
        MODEL="$model" TASK="$task" ADAPTER="$adapter" LOSS="$loss" CFG="$cfg" \
            sbatch -a "${seeds#,}" "${begin[@]}" "$script" "${extra[@]}"
    fi
    delay=$((delay + STAGGER))
done
