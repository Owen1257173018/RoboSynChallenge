#!/bin/bash
# Sequential local data-collection queue on one GPU.
#   bash tools/collect_queue.sh <gpu_id> <task:setting:episodes> [...]
# Each job runs launch/run_task.sh <task> <setting> 3_0 --max_episodes N --headless --gpu_id G
# and writes a LeRobot v3.0 dataset under lerobot_dataset/<task>/ (auto-indexed dir).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?gpu}"; shift
source "$REPO_ROOT/../EmbodiChain/.venv/bin/activate"
cd "$REPO_ROOT"
mkdir -p outputs/collect_logs
echo "[$(date)] collect queue on GPU$GPU: $*"
for job in "$@"; do
    IFS=: read -r task setting n <<< "$job"
    log="outputs/collect_logs/${task}_${setting}_$(date +%m%d_%H%M).log"
    before=$(ls -d lerobot_dataset/$task/*/ 2>/dev/null | wc -l)
    echo "[$(date)] >>> COLLECT $task $setting x$n (log: $log)"
    start=$(date +%s)
    bash launch/run_task.sh "$task" "$setting" 3_0 --max_episodes "$n" --headless --gpu_id "$GPU" > "$log" 2>&1
    rc=$?
    newest=$(ls -td lerobot_dataset/$task/*/ 2>/dev/null | head -1)
    eps=$(python3 -c "import json;print(json.load(open('$newest/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
    echo "[$(date)] <<< COLLECT $task $setting exit=$rc episodes=$eps dir=$newest secs=$(( $(date +%s) - start ))"
done
echo "[$(date)] collect queue on GPU$GPU finished"
