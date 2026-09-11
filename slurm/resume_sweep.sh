#!/bin/bash -l
# Resubmit every unfinished run (run.json lacks `steps`), one array per config with only the
# missing seeds. Skips seeds already queued (job comment names the run dir).
# Prints the sbatch lines; set GO=1 to actually submit.
# Usage: [GO=1] [STAGGER=45] [SKIP=<adapter-loss/cfg regex>] slurm/resume_sweep.sh
set -e
cd "$(dirname "$0")/.."

STAGGER="${STAGGER:-45}"
delay=0
# array tasks overwrite each other's comment, so trust only the run dir in it and take the seed from %K
queued=$(squeue -u "$USER" -r -h -o "%k %K" | sed "s|/seed[0-9]* | |")

for run_dir in $(find outputs/runs -mindepth 4 -maxdepth 4 -type d -path "*/*-*/r[0-9]*" | sort); do
    IFS=/ read -r _ _ _ model run cfg <<< "$run_dir"
    [ -n "$SKIP" ] && [[ "$run/$cfg" =~ $SKIP ]] && continue

    seeds=""
    for s in 0 1 2; do
        seed_dir="$run_dir/seed$s"
        grep -q '"steps"' "$seed_dir/run.json" 2>/dev/null && continue
        grep -qxF "$run_dir $s" <<< "$queued" && continue
        seeds="$seeds,$s"
    done
    [ -z "$seeds" ] && continue
    task=$(grep -o '"task": "[a-z]*"' "$run_dir"/seed*/run.json 2>/dev/null | head -1 | cut -d'"' -f4)
    # empty seed dirs (a job that died before run.json) carry no task: use the model's tier
    [ -z "$task" ] && case "$model" in qwen2.5-1.5b*) task=easy ;; *) task=hard ;; esac

    # run dir is <adapter>-<loss>/<cfg>, cfg is r<rank>[-u<proj_dim>][-b<batch>][-t1]
    adapter="${run%%-*}"
    loss="${run#*-}"
    rank="${cfg%%-*}"
    rank="${rank#r}"
    extra=(--rank "$rank")
    [[ "$cfg" =~ -u([0-9]+) ]] && extra+=(--proj-dim "${BASH_REMATCH[1]}")

    # HF 429s and node black-holes when a hundred jobs start at once
    opts=(--comment="$run_dir")
    [ "$delay" -gt 0 ] && opts+=(--begin="now+${delay}seconds")

    # tinylora-turbo/r<rank>-u<proj_dim>-b<batch>[-t1] runs go through bo.sh (-t1 = sampled at GRPO's T=1); 7B needs a 24G+ card
    if [ "$loss" = turbo ]; then
        [[ "$cfg" =~ -b([0-9]+) ]] && extra+=(--batch "${BASH_REMATCH[1]}")
        [[ "$cfg" =~ -t1$ ]] && extra+=(--no-greedy)
        case "$model" in qwen2.5-1.5b*) ;; *) opts+=(--constraint='l4|a40|l40s|a100-40g|a100-80g|h100') ;; esac
        script=slurm/bo.sh
    else
        # 1.5B GRPO fits an L4/A40; the script's default constraint keeps the 40G+ cards for 7B
        case "$model" in qwen2.5-1.5b*) opts+=(--constraint='l4|a40|l40s|a100-40g|a100-80g|h100') ;; esac
        script=slurm/train.sh
    fi

    echo MODEL="$model" TASK="$task" ADAPTER="$adapter" LOSS="$loss" CFG="$cfg" \
        sbatch -a "${seeds#,}" "${opts[@]}" "$script" "${extra[@]}"
    if [ -n "$GO" ]; then
        MODEL="$model" TASK="$task" ADAPTER="$adapter" LOSS="$loss" CFG="$cfg" \
            sbatch -a "${seeds#,}" "${opts[@]}" "$script" "${extra[@]}"
    fi
    delay=$((delay + STAGGER))
done
