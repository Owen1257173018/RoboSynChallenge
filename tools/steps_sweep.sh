#!/bin/bash
# Inference-time sweep of smolvla_steps (actions executed per policy query).
#   bash tools/steps_sweep.sh <gpu> <episodes> <steps_list,comma> <task=ckpt_dir=label> [...]
# Example:
#   bash tools/steps_sweep.sh 1 50 20,30 mixer_operating=outputs/train/smolvla_mixer_operating_run1/checkpoints/050000=run1
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${1:?gpu}"; N="${2:?episodes}"; STEPS_LIST="${3:?steps list}"; shift 3
cd "$REPO_ROOT"
printf "%-22s %-8s %6s %8s %10s %8s\n" task label steps success avg_steps calls
for spec in "$@"; do
  IFS='=' read -r task ckpt label <<< "$spec"
  for st in ${STEPS_LIST//,/ }; do
    name="${label}_steps${st}"
    SMOLVLA_STEPS=$st SMOLVLA_MODEL_NAME="$name" bash tools/smolvla_eval.sh "$task" random "$ckpt" "$GPU" "$N" > "outputs/train/logs/sweep_${task}_${name}.log" 2>&1
    d=$(ls -td "eval_result/$task/smolvla/random/None/$name/"*/ 2>/dev/null | head -1)
    python3 - "$task" "$label" "$st" "$d" <<'PY'
import json,sys
task,label,st,d=sys.argv[1:]
try:
    s=json.load(open(f"{d}/evaluation_metrics.json"))["summary"]
    print(f"{task:22s} {label:8s} {st:>6s} {s['success_count']:>3d}/{s['episode_count']:<3d} {s['average_action_steps']:>9.1f} {s['average_inference_calls_per_episode']:>8.1f}")
except Exception as e:
    print(f"{task:22s} {label:8s} {st:>6s}   FAILED ({e})")
PY
  done
done
