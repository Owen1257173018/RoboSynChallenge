#!/bin/bash
# ----------------------------------------------------------------------------
# Sequential training queue for one GPU.
#   bash tools/train_queue.sh <gpu_id> <wait_for_path_or_-> <task1> [task2 ...]
# - Waits until <wait_for_path> exists (e.g. the final checkpoint dir of the run
#   currently occupying the GPU); pass "-" to start immediately.
# - For each task: trains SmolVLA with the official recipe (50k steps, batch 32)
#   on lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_<task> (v3.0, renamed),
#   then evaluates the final checkpoint for 20 episodes on the same GPU.
# - Run name: smolvla_<task>_run1 ; logs under outputs/train/logs/.
# ----------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?gpu_id}"; WAIT_FOR="${2:?wait path or -}"; shift 2
TASKS=("$@")
cd "$REPO_ROOT"
echo "[$(date)] queue on GPU$GPU: ${TASKS[*]} (waiting for: $WAIT_FOR)"
if [[ "$WAIT_FOR" != "-" ]]; then
    until [[ -e "$WAIT_FOR" ]]; do sleep 120; done
    echo "[$(date)] wait condition satisfied: $WAIT_FOR"
    sleep 60
fi
for t in "${TASKS[@]}"; do
    DS="lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_$t"
    RUN="smolvla_${t}_run1"
    if [[ -d "outputs/train/$RUN/checkpoints/050000" ]]; then echo "[$(date)] $RUN already finished, skipping"; continue; fi
    echo "[$(date)] >>> TRAIN $RUN on GPU$GPU"
    SMOLVLA_NOHUP=0 SMOLVLA_RENAME_MAP='{}' bash tools/smolvla_train.sh "$t" "$DS" "$RUN" "$GPU" \
        --steps=50000 --batch_size=32 --num_workers=8 --persistent_workers=true \
        --save_freq=10000 --log_freq=200 --seed=1000 > "outputs/train/logs/$RUN.log" 2>&1
    rc=$?
    echo "[$(date)] <<< TRAIN $RUN exit=$rc"
    if [[ -d "outputs/train/$RUN/checkpoints/050000/pretrained_model" ]]; then
        echo "[$(date)] >>> EVAL $RUN @50k (20 eps, GPU$GPU)"
        bash tools/smolvla_eval.sh "$t" random "outputs/train/$RUN/checkpoints/050000" "$GPU" 20 > "outputs/train/logs/${RUN}_eval50k.log" 2>&1
        d=$(ls -td "eval_result/$t/smolvla/random/None/050000/"*/ 2>/dev/null | head -1)
        echo "[$(date)] <<< EVAL $RUN: $(python3 -c "import json;s=json.load(open('$d/evaluation_metrics.json'))['summary'];print(f\"success {s['success_count']}/{s['episode_count']}, avg steps {s['average_action_steps']:.1f}\")" 2>/dev/null || echo 'metrics missing')"
    else
        echo "[$(date)] $RUN has no 50k checkpoint; see outputs/train/logs/$RUN.log"
    fi
done
echo "[$(date)] queue on GPU$GPU finished"
