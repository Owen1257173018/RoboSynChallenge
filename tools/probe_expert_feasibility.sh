#!/bin/bash
# Probe the expert planner's success rate per task/setting: run N episodes without
# saving and count invalid resets vs generated demos.
# Usage: bash tools/probe_expert_feasibility.sh <gpu_id> <episodes> <task...>
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?gpu}"; N="${2:?episodes}"; shift 2
source "$REPO_ROOT/../EmbodiChain/.venv/bin/activate"
export EMBODICHAIN_SIM_EXIT_PROCESS=0
cd "$REPO_ROOT"
printf "%-22s %-7s %8s %8s %8s %10s\n" task setting demos invalid feas% secs
for t in "$@"; do
  for s in random clear; do
    log="/tmp/probe_${t}_${s}.log"
    start=$(date +%s)
    timeout 1500 bash launch/run_task.sh "$t" "$s" 3_0 --max_episodes "$N" --headless --gpu_id "$GPU" --filter_dataset_saving > "$log" 2>&1
    secs=$(( $(date +%s) - start ))
    demos=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -ac "Demo action list created")
    invalid=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -ac "Reset valid flag to True")
    total=$((demos + invalid)); feas="n/a"; [[ $total -gt 0 ]] && feas=$(( 100 * demos / total ))
    printf "%-22s %-7s %8s %8s %8s %10s\n" "$t" "$s" "$demos" "$invalid" "$feas" "$secs"
  done
done
