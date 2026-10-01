#!/bin/bash
# ----------------------------------------------------------------------------
# Final confirmation evals in *submission mode*: no inference knob is passed on
# the command line, so every task runs with the per-task table in
# policy/smolvla/deploy_policy.yml and no policy seed -- exactly what the
# organizers' plain `policy/smolvla/eval.sh <task> random <ckpt> <gpu>` runs.
#
#   bash tools/final_eval.sh <gpu> <episodes> <task|checkpoint_step_dir> [...]
#
# Result name: final<episodes>_<run_name>. Skips a spec whose result exists.
#
# Candidate settings on top of the submission config (not the official finals):
#   FINAL_TAG=as60 FINAL_EXTRA='--smolvla_antistall_window 60' bash tools/final_eval.sh ...
#   -> result name as60_<run_name>; FINAL_EXTRA is passed to eval.sh as extra args
# The adapter's "[smolvla] inference config: ..." start-up line is saved next to
# the metrics as inference_config.txt (read by tools/summarize_results.py).
# ----------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?gpu}"; N="${2:?episodes}"; shift 2
PY="$REPO_ROOT/policy/smolvla/.venv/bin/python"
cd "$REPO_ROOT"
mkdir -p outputs/train/logs
# submission mode: make sure no knob leaks in from the calling shell
unset SMOLVLA_SEED SMOLVLA_CHUNK_MODE SMOLVLA_ACTION_SAMPLES
for spec in "$@"; do
    IFS='|' read -r task ckpt <<< "$spec"
    run="$(basename "$(dirname "$(dirname "$ckpt")")")"
    name="${FINAL_TAG:-final${N}}_${run}"
    if compgen -G "eval_result/$task/smolvla/random/None/$name/*/evaluation_metrics.json" > /dev/null; then
        echo "[$(date)] $name already done"; continue
    fi
    log="outputs/train/logs/${name}_eval.log"
    echo "[$(date)] >>> $name ($task, $N eps, GPU$GPU, submission mode)"
    read -r -a extra <<< "${FINAL_EXTRA:-}"
    SMOLVLA_STEPS=table SMOLVLA_MODEL_NAME="$name" \
        bash tools/smolvla_eval.sh "$task" random "$ckpt" "$GPU" "$N" "${extra[@]}" > "$log" 2>&1
    d=$(ls -td "eval_result/$task/smolvla/random/None/$name/"*/ 2>/dev/null | head -1)
    cfg_line=$(grep -a "\[smolvla\] inference config" "$log" | tail -1)
    [[ -n "$d" && -n "$cfg_line" ]] && echo "$cfg_line" > "$d/inference_config.txt"
    echo "[$(date)] <<< $name: $cfg_line"
    echo "      $("$PY" -c "
import json;s=json.load(open('$d/evaluation_metrics.json'))['summary']
print(f\"success {s['success_count']}/{s['episode_count']} = {100*s['success_rate']:.0f}%, avg steps {s['average_action_steps']:.1f}\")" 2>/dev/null || echo 'metrics missing')"
done
echo "[$(date)] final eval on GPU$GPU finished"
