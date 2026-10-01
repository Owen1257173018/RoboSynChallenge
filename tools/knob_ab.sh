#!/bin/bash
# ----------------------------------------------------------------------------
# Paired A/B test of the inference-time knobs on one GPU.
#   A = stock behaviour  (queue mode, 1 sample)      with the policy seeded
#   B = candidate        (fresh mode, K samples)     with the same policy seed
# Both arms see the same scenes (eval seed 0) AND the same policy noise
# (SMOLVLA_SEED), so tools/paired_compare.py's McNemar test isolates the knobs.
#
#   bash tools/knob_ab.sh <gpu> <episodes> <K> <task|ckpt_dir|steps> [...]
# Results: eval_result/<task>/smolvla/random/None/ab_<A|B>_<task>_s<steps>/...
# ----------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?gpu}"; N="${2:?episodes}"; K="${3:?samples}"; shift 3
PY="$REPO_ROOT/policy/smolvla/.venv/bin/python"
SEED="${KNOB_SEED:-1000}"
cd "$REPO_ROOT"
mkdir -p outputs/train/logs
for spec in "$@"; do
    IFS='|' read -r task ckpt steps <<< "$spec"
    for arm in A B; do
        name="ab_${arm}_${task}_s${steps}_seed${SEED}"
        [[ $arm == B ]] && name="ab_${arm}k${K}_${task}_s${steps}_seed${SEED}"
        if compgen -G "eval_result/$task/smolvla/random/None/$name/*/evaluation_metrics.json" > /dev/null; then
            echo "[$(date)] $name already done"; continue; fi
        if [[ $arm == A ]]; then mode=queue; k=1; else mode=fresh; k=$K; fi
        echo "[$(date)] >>> $name (mode=$mode K=$k steps=$steps seed=$SEED)"
        SMOLVLA_STEPS=$steps SMOLVLA_CHUNK_MODE=$mode SMOLVLA_ACTION_SAMPLES=$k SMOLVLA_SEED=$SEED \
            SMOLVLA_MODEL_NAME="$name" bash tools/smolvla_eval.sh "$task" random "$ckpt" "$GPU" "$N" \
            > "outputs/train/logs/${name}.log" 2>&1
    done
    A="ab_A_${task}_s${steps}_seed${SEED}"; B="ab_Bk${K}_${task}_s${steps}_seed${SEED}"
    echo "[$(date)] <<< $task (steps=$steps):"
    "$PY" tools/paired_compare.py --task "$task" --a "$A" --b "$B" --label-a "queue,K=1" --label-b "fresh,K=$K" 2>&1 | sed 's/^/      /'
done
echo "[$(date)] knob A/B on GPU$GPU finished"
