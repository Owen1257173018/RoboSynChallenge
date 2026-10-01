#!/bin/bash
# ----------------------------------------------------------------------------
# X-VLA queue on one GPU, driven by a queue file that may be edited while it
# runs. Per job: (merge) -> tools/xvla_train.sh (30k steps, resumes after a
# crash, up to 3 attempts) -> 100-episode final in submission mode ->
# paired comparison with the current checkpoint's final -> prune the
# intermediate checkpoints and optimizer state (keeps 030000/pretrained_model).
#
#   bash tools/xvla_queue.sh <gpu> <queue_file>
#
# Queue file: one job per line, "#" starts a comment:
#   task|run_name|src1[,src2,...]|current_final_name
#   src      : dataset roots; one -> used directly, several -> merged into
#              $XVLA_DATA_ROOT/<task>_merge/<run>_data (default /data/robosyn_xvla_data)
#   current_final_name : e.g. final100_smolvla_click_bell_run4_2500eps
# After every job the file is re-read and the first line without a final
# result (and not failed before) runs next, so a line inserted at the top
# jumps the queue. DRY=1 prints the pending jobs and checks their inputs.
# ----------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?gpu}"; QFILE="${2:?queue file}"
PY="$REPO_ROOT/policy/smolvla/.venv/bin/python"
DATA_ROOT="${XVLA_DATA_ROOT:-/data/robosyn_xvla_data}"
cd "$REPO_ROOT"
mkdir -p outputs/train/logs
FAILED=" "

final_done() { compgen -G "eval_result/$1/smolvla/random/None/final100_$2/*/evaluation_metrics.json" >/dev/null; }

next_job() {
    grep -v '^\s*#' "$QFILE" | grep -v '^\s*$' | while IFS='|' read -r task run srcs cur; do
        final_done "$task" "$run" && continue
        [[ "$FAILED" == *" $run "* ]] && continue
        echo "$task|$run|$srcs|$cur"; break
    done
}

if [[ "${DRY:-0}" == 1 ]]; then
    grep -v '^\s*#' "$QFILE" | grep -v '^\s*$' | while IFS='|' read -r task run srcs cur; do
        final_done "$task" "$run" && { echo "done     $run"; continue; }
        echo "pending  $run ($task), compare with $cur"
        IFS=',' read -ra SRC <<< "$srcs"
        for s in "${SRC[@]}"; do [[ -d "$s/meta" ]] && echo "    ok  $s" || echo "    MISSING $s"; done
        (( ${#SRC[@]} > 1 )) && echo "    -> merge into $DATA_ROOT/${task}_merge/${run}_data"
        compgen -G "eval_result/$task/smolvla/random/None/$cur/*/evaluation_metrics.json" >/dev/null \
            && echo "    ok  $cur" || echo "    MISSING $cur"
    done
    exit 0
fi

echo "[$(date)] X-VLA queue on GPU$GPU from $QFILE"
while true; do
    job=$(next_job)
    [[ -z "$job" ]] && break
    IFS='|' read -r task run srcs cur <<< "$job"
    IFS=',' read -ra SRC <<< "$srcs"
    echo "[$(date)] ===== JOB $run ($task), compare with $cur ====="
    DS="${SRC[0]}"
    if (( ${#SRC[@]} > 1 )); then
        DS="$DATA_ROOT/${task}_merge/${run}_data"
        if [[ ! -d "$DS/meta" ]]; then
            echo "[$(date)] merging ${#SRC[@]} sources -> $DS"
            mkdir -p "$(dirname "$DS")"
            "$PY" tools/merge_datasets.py "$DS" "${SRC[@]}" > "outputs/train/logs/${run}_merge.log" 2>&1
            "$PY" tools/validate_dataset.py "$DS" >> "outputs/train/logs/${run}_merge.log" 2>&1 || {
                echo "[$(date)] MERGE/VALIDATE FAILED for $run"; FAILED+="$run "; continue; }
        fi
    fi
    CK="outputs/train/$run/checkpoints/030000"
    for attempt in 1 2 3; do
        [[ -d "$CK/pretrained_model" ]] && break
        echo "[$(date)] >>> TRAIN $run on GPU$GPU (attempt $attempt, data $DS)"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            bash tools/xvla_train.sh "$task" "$DS" "$run" "$GPU" 30000 16 >> "outputs/train/logs/$run.log" 2>&1
        echo "[$(date)] <<< TRAIN $run exit=$?"
        [[ -d "$CK/pretrained_model" ]] || sleep 60
    done
    [[ -d "$CK/pretrained_model" ]] || { echo "[$(date)] FAILED: no 30k checkpoint for $run"; FAILED+="$run "; continue; }
    bash tools/final_eval.sh "$GPU" 100 "$task|$CK"
    final_done "$task" "$run" || { echo "[$(date)] FAILED: final eval of $run"; FAILED+="$run "; continue; }
    echo "[$(date)] paired comparison (same scenes, submission mode):"
    python3 tools/paired_compare.py --task "$task" --a "$cur" --b "final100_$run" --label-a "current" --label-b "$run"
    rm -rf "outputs/train/$run/checkpoints/010000" "outputs/train/$run/checkpoints/020000" "$CK/training_state"
    echo "[$(date)] pruned intermediate checkpoints of $run (kept $CK/pretrained_model)"
done
echo "[$(date)] X-VLA queue on GPU$GPU finished (failed: ${FAILED:- none})"
