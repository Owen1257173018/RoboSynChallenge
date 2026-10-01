#!/bin/bash
# ----------------------------------------------------------------------------
# Hang-tolerant data collection: collect in small batches, each into its own
# dataset directory, with a watchdog.
#
#   bash tools/collect_batches.sh <gpu> <task> <setting> <batches> <episodes_per_batch> [stall_minutes=10]
#
# Why: the simulator occasionally freezes mid-episode (seen 2026-09-28 on
# item_assembly: stuck at one step for 4.5 h at 300% CPU). A killed collection
# leaves its parquet file without a footer, so the whole batch is unreadable.
# Small batches bound the loss; the watchdog kills a batch whose log has not
# changed for <stall_minutes> and moves on. Every finished batch is validated
# (all parquet files readable, episode count > 0); unreadable ones are moved to
# lerobot_dataset/<task>_broken/.
# Summary lines: "<<< BATCH i/N ok|stalled|broken episodes=K dir=..."
#
# COLLECT_STREAM=<name> runs a parallel stream: identical configs except the
# dataset save_path, which becomes lerobot_dataset/<task>_<name>/ (the
# recorder picks "max existing index + 1", so two streams sharing one folder
# could race for the same name). Without it: launch/run_task.sh as usual.
#
# Stop early: touch outputs/collect_logs/stop_<task> -- every stream of that
# task finishes its current batch and exits (stop_<task>_<stream>: only that
# stream). Delete the file afterwards.
# ----------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?gpu}"; TASK="${2:?task}"; SETTING="${3:?setting}"; NB="${4:?batches}"; PER="${5:?episodes per batch}"
STALL_MIN="${6:-10}"
STREAM="${COLLECT_STREAM:-}"
cd "$REPO_ROOT"
source "$REPO_ROOT/../EmbodiChain/.venv/bin/activate"
OUT="lerobot_dataset/$TASK${STREAM:+_$STREAM}"
mkdir -p outputs/collect_logs "$OUT"

if [[ -n "$STREAM" ]]; then
    mkdir -p outputs/collect_cfg
    GYM_CFG="outputs/collect_cfg/${TASK}_${SETTING}_${STREAM}.json"
    python3 - "configs/$TASK/$SETTING/gym_config.json" "$GYM_CFG" "$OUT" <<'PYEOF'
import sys, json
src, dst, out = sys.argv[1:]
cfg = json.load(open(src))
def fix(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "save_path":
                node[k] = out
            else:
                fix(v)
    elif isinstance(node, list):
        for v in node:
            fix(v)
fix(cfg)
json.dump(cfg, open(dst, "w"), indent=4)
PYEOF
    ACTION_CFG="configs/$TASK/action_config.json"
    [[ -f "$ACTION_CFG" ]] || ACTION_CFG="configs/$TASK/$SETTING/action_config.json"
fi

run_batch() {
    if [[ -n "$STREAM" ]]; then
        python -m scripts.run_env --gym_config "$GYM_CFG" --action_config "$ACTION_CFG" --num_envs 1 \
            --max_episodes "$PER" --headless --gpu_id "$GPU"
    else
        bash launch/run_task.sh "$TASK" "$SETTING" 3_0 --max_episodes "$PER" --headless --gpu_id "$GPU"
    fi
}

tree_pids() { local p=$1; echo "$p"; for c in $(ps -o pid= --ppid "$p"); do tree_pids "$c"; done; }

validate() {  # prints "<episodes> <ok|broken>"
    python3 - "$1" <<'PYEOF'
import sys, json, glob
import pyarrow.parquet as pq
d = sys.argv[1]
try:
    n = json.load(open(f"{d}/meta/info.json"))["total_episodes"]
    for f in glob.glob(f"{d}/data/*/*.parquet") + glob.glob(f"{d}/meta/episodes/*/*.parquet"):
        pq.read_metadata(f)
    print(n, "ok" if n > 0 else "broken")
except Exception:
    print(0, "broken")
PYEOF
}

echo "[$(date)] batches on GPU$GPU: $TASK $SETTING, $NB x $PER episodes, stall limit ${STALL_MIN} min, output $OUT"
total=0
for b in $(seq 1 "$NB"); do
    if [[ -e "outputs/collect_logs/stop_$TASK" || ( -n "$STREAM" && -e "outputs/collect_logs/stop_${TASK}_$STREAM" ) ]]; then
        echo "[$(date)] stop file found, not starting batch $b"; break
    fi
    before=$(ls -d "$OUT"/*/ 2>/dev/null | sort)
    log="outputs/collect_logs/${TASK}_${SETTING}${STREAM:+_$STREAM}_b${b}_$(date +%m%d_%H%M).log"
    echo "[$(date)] >>> BATCH $b/$NB (log: $log)"
    run_batch > "$log" 2>&1 &
    pid=$!
    status=done
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        age=$(( $(date +%s) - $(stat -c %Y "$log") ))
        if (( age > STALL_MIN * 60 )); then
            status=stalled
            pids=$(tree_pids "$pid" | tr '\n' ' ')
            echo "[$(date)] batch $b: no log output for ${age}s -> killing $pids"
            kill $pids 2>/dev/null; sleep 10; kill -9 $pids 2>/dev/null
            break
        fi
    done
    wait "$pid" 2>/dev/null
    after=$(ls -d "$OUT"/*/ 2>/dev/null | sort)
    newdir=$(comm -13 <(echo "$before") <(echo "$after") | tail -1); newdir="${newdir%/}"
    if [[ -z "$newdir" ]]; then
        echo "[$(date)] <<< BATCH $b/$NB $status episodes=0 dir=none"; continue
    fi
    read -r eps verdict <<< "$(validate "$newdir")"
    if [[ "$verdict" != ok ]]; then
        mkdir -p "lerobot_dataset/${TASK}_broken"; mv "$newdir" "lerobot_dataset/${TASK}_broken/"
        echo "[$(date)] <<< BATCH $b/$NB broken($status) episodes=0 dir=moved to lerobot_dataset/${TASK}_broken/$(basename "$newdir")"
        continue
    fi
    total=$(( total + eps ))
    echo "[$(date)] <<< BATCH $b/$NB ok($status) episodes=$eps dir=$newdir (total $total)"
done
echo "[$(date)] batches finished: $total episodes"
