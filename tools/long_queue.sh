#!/bin/bash
# ----------------------------------------------------------------------------
# "Train longer" queue: for each spec, (merge) -> train to TOTAL_STEPS with the
# cosine decay stretched to DECAY_STEPS -> 100-episode final in submission mode
# -> paired comparison with the submitted checkpoint's final.
#
#   bash tools/long_queue.sh <gpu> <wait_path|-> 'task|run_name|src1[,src2...]|submitted_run' [...]
#
#   TOTAL_STEPS (default 100000), DECAY_STEPS (default 60000)
#   submitted_run: its final100_<submitted_run> result must exist (tools/final_eval.sh)
#   DRY=1 prints the plan only.
# Safe to re-run: skips finished stages; an interrupted training resumes from
# its latest checkpoint (saved every 10k steps).
# ----------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?gpu}"; WAIT="${2:?wait path or -}"; shift 2
TOTAL="${TOTAL_STEPS:-100000}"; DECAY="${DECAY_STEPS:-60000}"
PY="$REPO_ROOT/policy/smolvla/.venv/bin/python"
cd "$REPO_ROOT"
mkdir -p outputs/train/logs
final_step=$(printf "%06d" "$TOTAL")

echo "[$(date)] long queue on GPU$GPU: $# job(s), steps=$TOTAL decay=$DECAY, waiting for: $WAIT"
if [[ "$WAIT" != "-" && "${DRY:-0}" != 1 ]]; then
    until [[ -e "$WAIT" ]]; do sleep 120; done
    echo "[$(date)] wait satisfied"; sleep 60
fi

for spec in "$@"; do
    IFS='|' read -r task run srcs cur <<< "$spec"
    IFS=',' read -ra SRC <<< "$srcs"
    echo "[$(date)] ===== JOB $run ($task), compare with $cur ====="
    if [[ ${#SRC[@]} -eq 1 ]]; then
        DS="${SRC[0]}"
    else
        DS="lerobot_dataset/${task}_merge/${run}_data"
    fi
    ckpt="outputs/train/$run/checkpoints/$final_step"
    if [[ "${DRY:-0}" == 1 ]]; then
        echo "  dataset: $DS ($( [[ ${#SRC[@]} -gt 1 ]] && echo "merge of ${#SRC[@]} sources" || echo "used directly"))"
        for s in "${SRC[@]}"; do [[ -d "$s/meta" ]] && echo "    ok  $s" || echo "    MISSING $s"; done
        echo "  train -> $ckpt ; final100_$run vs final100_$cur"
        compgen -G "eval_result/$task/smolvla/random/None/final100_$cur/*/evaluation_metrics.json" >/dev/null \
            && echo "    ok  final100_$cur exists" || echo "    MISSING final100_$cur"
        continue
    fi
    if [[ ${#SRC[@]} -gt 1 && ! -d "$DS/meta" ]]; then
        echo "[$(date)] merging ${#SRC[@]} sources -> $DS"
        "$PY" tools/merge_datasets.py "$DS" "${SRC[@]}" > "outputs/train/logs/${run}_merge.log" 2>&1
        [[ -d "$DS/meta" ]] || { echo "[$(date)] MERGE FAILED for $run"; continue; }
        "$PY" tools/validate_dataset.py "$DS" >> "outputs/train/logs/${run}_merge.log" 2>&1 || { echo "[$(date)] VALIDATE FAILED for $run"; continue; }
    fi
    if [[ -d "$ckpt/pretrained_model" ]]; then
        echo "[$(date)] training already finished"
    elif [[ -f "outputs/train/$run/checkpoints/last/pretrained_model/train_config.json" ]]; then
        echo "[$(date)] >>> RESUME $run (from $(readlink "outputs/train/$run/checkpoints/last"))"
        bash tools/smolvla_resume.sh "$run" "$GPU" --save_freq=10000 >> "outputs/train/logs/$run.log" 2>&1
        echo "[$(date)] <<< TRAIN $run exit=$?"
    else
        echo "[$(date)] >>> TRAIN $run on GPU$GPU"
        SMOLVLA_NOHUP=0 SMOLVLA_RENAME_MAP='{}' SMOLVLA_DATASET_REPO_ID="local/$(basename "$DS")" \
            bash tools/smolvla_train.sh "$task" "$DS" "$run" "$GPU" \
            --steps="$TOTAL" --batch_size=32 --num_workers=8 --persistent_workers=true \
            --save_freq=10000 --log_freq=200 --seed=1000 --policy.scheduler_decay_steps="$DECAY" \
            > "outputs/train/logs/$run.log" 2>&1
        echo "[$(date)] <<< TRAIN $run exit=$?"
    fi
    [[ -d "$ckpt/pretrained_model" ]] || { echo "[$(date)] FAILED: no $final_step checkpoint for $run"; continue; }
    bash tools/final_eval.sh "$GPU" 100 "$task|$ckpt"
    echo "[$(date)] paired comparison (same 100 scenes, submission mode):"
    python3 tools/paired_compare.py --task "$task" --a "final100_$cur" --b "final100_$run" --label-a "$cur" --label-b "$run"
done
echo "[$(date)] long queue on GPU$GPU finished"
