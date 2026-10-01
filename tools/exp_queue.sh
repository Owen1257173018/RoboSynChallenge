#!/bin/bash
# ----------------------------------------------------------------------------
# One experiment queue per GPU: merge (if needed) -> train 50k -> evaluate at
# every requested smolvla_steps. Skips any stage whose output already exists, so
# a queue can be re-launched safely after a crash: an interrupted training resumes
# from its latest checkpoint (saved every 10k steps).
#
#   bash tools/exp_queue.sh <gpu_id> <wait_path|-> <spec> [<spec> ...]
#
#   spec = task|run_name|steps_csv|src1[,src2,...]
#     task      : RoboSynChallenge task name
#     run_name  : outputs/train/<run_name>/
#     steps_csv : smolvla_steps values to evaluate at, e.g. 10,20
#     srcN      : dataset roots; one -> used directly, several -> merged
#
#   EXP_TRAIN_EXTRA : extra lerobot-train args for new trainings of this queue,
#                     e.g. EXP_TRAIN_EXTRA='--policy.train_expert_only=false'
#
#   <wait_path> : start only after this path exists (e.g. the 50k checkpoint of
#                 whatever currently occupies the GPU). "-" starts immediately.
#
# Example:
#   bash tools/exp_queue.sh 0 outputs/train/smolvla_x_run1/checkpoints/050000/pretrained_model \
#     'water_pouring|smolvla_water_pouring_run2|10,20|lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_water_pouring,lerobot_dataset/water_pouring/cobotmagic_Sim_water_pouring_000'
# ----------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?gpu id}"; WAIT="${2:?wait path or -}"; shift 2
PY="$REPO_ROOT/policy/smolvla/.venv/bin/python"
cd "$REPO_ROOT"
mkdir -p outputs/train/logs

echo "[$(date)] exp queue on GPU$GPU, $# job(s), waiting for: $WAIT"
if [[ "$WAIT" != "-" ]]; then
    until [[ -e "$WAIT" ]]; do sleep 120; done
    echo "[$(date)] wait satisfied"
    sleep 60   # let the previous job release its VRAM
fi

for spec in "$@"; do
    IFS='|' read -r task run steps srcs <<< "$spec"
    IFS=',' read -ra SRC <<< "$srcs"
    echo "[$(date)] ===== JOB $run ($task) ====="

    if [[ ${#SRC[@]} -eq 1 ]]; then
        DS="${SRC[0]}"
    else
        DS="lerobot_dataset/${task}_merge/${run}_data"
        if [[ -d "$DS/meta" ]]; then
            echo "[$(date)] merged dataset already present: $DS"
        else
            echo "[$(date)] merging ${#SRC[@]} sources -> $DS"
            "$PY" tools/merge_datasets.py "$DS" "${SRC[@]}" > "outputs/train/logs/${run}_merge.log" 2>&1
            if [[ ! -d "$DS/meta" ]]; then
                echo "[$(date)] MERGE FAILED for $run, see outputs/train/logs/${run}_merge.log"; continue
            fi
            "$PY" tools/validate_dataset.py "$DS" >> "outputs/train/logs/${run}_merge.log" 2>&1 || {
                echo "[$(date)] VALIDATE FAILED for $run"; continue; }
        fi
    fi
    eps=$("$PY" -c "import json;print(json.load(open('$DS/meta/info.json'))['total_episodes'])" 2>/dev/null || echo '?')
    echo "[$(date)] dataset: $DS ($eps episodes)"

    if [[ -d "outputs/train/$run/checkpoints/050000/pretrained_model" ]]; then
        echo "[$(date)] training already finished, skipping"
    elif [[ -f "outputs/train/$run/checkpoints/last/pretrained_model/train_config.json" ]]; then
        # interrupted (e.g. power loss): continue from the latest checkpoint
        echo "[$(date)] >>> RESUME $run on GPU$GPU (from $(readlink "outputs/train/$run/checkpoints/last"))"
        bash tools/smolvla_resume.sh "$run" "$GPU" --save_freq=10000 >> "outputs/train/logs/$run.log" 2>&1
        echo "[$(date)] <<< TRAIN $run exit=$?"
        [[ -d "outputs/train/$run/checkpoints/050000/pretrained_model" ]] || {
            echo "[$(date)] no 50k checkpoint for $run, see outputs/train/logs/$run.log"; continue; }
    else
        if [[ -d "outputs/train/$run" ]]; then
            # died before its first checkpoint; lerobot-train refuses an existing output dir
            aside="outputs/train/${run}_aborted_$(date +%m%d_%H%M)"
            echo "[$(date)] moving checkpoint-less $run aside -> $aside"
            mv "outputs/train/$run" "$aside"
            [[ -f "outputs/train/logs/$run.log" ]] && mv "outputs/train/logs/$run.log" "outputs/train/logs/$(basename "$aside").log"
        fi
        echo "[$(date)] >>> TRAIN $run on GPU$GPU"
        SMOLVLA_NOHUP=0 SMOLVLA_RENAME_MAP='{}' SMOLVLA_DATASET_REPO_ID="local/$(basename "$DS")" \
            bash tools/smolvla_train.sh "$task" "$DS" "$run" "$GPU" \
            --steps=50000 --batch_size=32 --num_workers=8 --persistent_workers=true \
            --save_freq=10000 --log_freq=200 --seed=1000 ${EXP_TRAIN_EXTRA:-} > "outputs/train/logs/$run.log" 2>&1
        echo "[$(date)] <<< TRAIN $run exit=$?"
        [[ -d "outputs/train/$run/checkpoints/050000/pretrained_model" ]] || {
            echo "[$(date)] no 50k checkpoint for $run, see outputs/train/logs/$run.log"; continue; }
    fi

    for st in ${steps//,/ }; do
        name="${run}_steps${st}"
        if compgen -G "eval_result/$task/smolvla/random/None/$name/*/evaluation_metrics.json" > /dev/null; then
            echo "[$(date)] eval $name already done, skipping"
        else
            echo "[$(date)] >>> EVAL $name (50 eps, GPU$GPU)"
            SMOLVLA_STEPS=$st SMOLVLA_MODEL_NAME="$name" \
                bash tools/smolvla_eval.sh "$task" random "outputs/train/$run/checkpoints/050000" "$GPU" 50 \
                > "outputs/train/logs/${name}_eval.log" 2>&1
        fi
        d=$(ls -td "eval_result/$task/smolvla/random/None/$name/"*/ 2>/dev/null | head -1)
        echo "[$(date)] <<< EVAL $name: $("$PY" -c "
import json;s=json.load(open('$d/evaluation_metrics.json'))['summary']
print(f\"success {s['success_count']}/{s['episode_count']}, avg steps {s['average_action_steps']:.1f}\")" 2>/dev/null || echo 'metrics missing')"
    done
done
echo "[$(date)] exp queue on GPU$GPU finished"
