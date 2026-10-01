#!/bin/bash
# ----------------------------------------------------------------------------
# Chain for data collected with tools/collect_batches.sh, unattended:
#   wait until the streams have saved <n_new> episodes -> stop them (stop file)
#   -> merge the new batches, then merge <base> + new -> (trim static tails)
#   -> train 50k (tools/exp_queue.sh recipe) -> 100-episode final in submission
#   mode -> paired comparison with the current checkpoint's final.
#
#   bash tools/batches_chain.sh <gpu> <task> <run_name> <n_new> <base_dataset> \
#        <current_final_name> <batch_log>[,<batch_log>...]
#
#   batch_log           logs of the collect_batches.sh streams; their
#                       "<<< BATCH ... ok ... dir=..." lines list the new data
#   current_final_name  result name of the current checkpoint's final, e.g.
#                       final100h500_smolvla_item_assembly_run4_trim_2000eps
#   TRIM=1              apply tools/trim_static_tails.py to the merged set
#   DRY=1               print the plan with the batches available right now
# Whole batches are taken in the order they finished until >= n_new episodes.
# ----------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?gpu}"; TASK="${2:?task}"; RUN="${3:?run name}"; N_NEW="${4:?new episodes}"
BASE="${5:?base dataset}"; CUR="${6:?current final name}"; IFS=',' read -ra LOGS <<< "${7:?batch logs}"
PY="$REPO_ROOT/policy/smolvla/.venv/bin/python"
cd "$REPO_ROOT"
mkdir -p outputs/train/logs "lerobot_dataset/${TASK}_merge"
STOP="outputs/collect_logs/stop_$TASK"

ok_batches() {  # "<finish time> <episodes> <dir>" per finished ok batch, all streams, oldest first
    for f in "${LOGS[@]}"; do
        grep -a "<<< BATCH .* ok(" "$f" 2>/dev/null | sed -n 's/^\[\(.*\)\] <<< BATCH .* episodes=\([0-9]*\) dir=\([^ ]*\).*/\1|\2|\3/p' |
            while IFS='|' read -r t e d; do echo "$(date -d "$t" +%s) $e $d"; done
    done | sort -n
}
total_new() { ok_batches | awk '{s += $2} END {print s + 0}'; }

echo "[$(date)] chain $RUN: waiting for $N_NEW new episodes from ${#LOGS[@]} stream(s)"
if [[ "${DRY:-0}" != 1 ]]; then
    until (( $(total_new) >= N_NEW )); do sleep 300; done
    echo "[$(date)] $(total_new) episodes saved -> stop file $STOP"
    touch "$STOP"
    for f in "${LOGS[@]}"; do
        until grep -aq "batches finished" "$f" 2>/dev/null; do sleep 60; done
    done
    rm -f "$STOP"
    echo "[$(date)] all streams finished: $(total_new) episodes in total"
fi

SRC=(); n=0
while read -r _ e d; do
    (( n >= N_NEW )) && break
    SRC+=("$d"); n=$(( n + e ))
done < <(ok_batches)
NEW_DS="lerobot_dataset/${TASK}_merge/${RUN}_new"
DS="lerobot_dataset/${TASK}_merge/${RUN}_data"
TRAIN_DS="$DS"; [[ "${TRIM:-0}" == 1 ]] && TRAIN_DS="${DS}_trim"
ckpt="outputs/train/$RUN/checkpoints/050000"
echo "[$(date)] using ${#SRC[@]} batches = $n new episodes (+ $BASE)"
echo "  merge new -> $NEW_DS ; merge base + new -> $DS${TRIM:+ ; trim -> $TRAIN_DS}"
echo "  train $RUN on GPU$GPU ; final100_$RUN vs $CUR"
if [[ "${DRY:-0}" == 1 ]]; then
    printf '    %s\n' "${SRC[@]:0:3}" "..."
    compgen -G "eval_result/$TASK/smolvla/random/None/$CUR/*/evaluation_metrics.json" >/dev/null \
        && echo "  ok  $CUR exists" || echo "  MISSING $CUR"
    exit 0
fi
(( n >= N_NEW )) || { echo "[$(date)] FAILED: only $n new episodes"; exit 1; }

log="outputs/train/logs/${RUN}_merge.log"
if [[ ! -d "$NEW_DS/meta" ]]; then
    "$PY" tools/merge_datasets.py "$NEW_DS" "${SRC[@]}" > "$log" 2>&1
    [[ -d "$NEW_DS/meta" ]] || { echo "[$(date)] MERGE (new batches) FAILED, see $log"; exit 1; }
fi
if [[ ! -d "$DS/meta" ]]; then
    "$PY" tools/merge_datasets.py "$DS" "$BASE" "$NEW_DS" >> "$log" 2>&1
    [[ -d "$DS/meta" ]] || { echo "[$(date)] MERGE (base + new) FAILED, see $log"; exit 1; }
    "$PY" tools/validate_dataset.py "$DS" >> "$log" 2>&1 || { echo "[$(date)] VALIDATE FAILED, see $log"; exit 1; }
fi
if [[ "$TRAIN_DS" != "$DS" && ! -d "$TRAIN_DS/meta" ]]; then
    "$PY" tools/trim_static_tails.py "$DS" "$TRAIN_DS" >> "$log" 2>&1 || { echo "[$(date)] TRIM FAILED, see $log"; exit 1; }
fi
echo "[$(date)] training data: $TRAIN_DS ($("$PY" -c "import json;print(json.load(open('$TRAIN_DS/meta/info.json'))['total_episodes'])") episodes)"

bash tools/exp_queue.sh "$GPU" - "$TASK|$RUN||$TRAIN_DS"
[[ -d "$ckpt/pretrained_model" ]] || { echo "[$(date)] FAILED: no 50k checkpoint for $RUN"; exit 1; }
bash tools/final_eval.sh "$GPU" 100 "$TASK|$ckpt"
echo "[$(date)] paired comparison (same scenes, submission mode):"
python3 tools/paired_compare.py --task "$TASK" --a "$CUR" --b "final100_$RUN" --label-a "current" --label-b "$RUN"
echo "[$(date)] chain $RUN finished"
