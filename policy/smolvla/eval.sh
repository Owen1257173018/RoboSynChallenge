#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]
# bash eval.sh click_bell random checkpoints/SmolVLA_sim_click_bell 5 \
#   --pytorch_device cuda --headless true --eval_video_log true
# ----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
if [[ -z "${EMBODICHAIN_ROOT:-}" ]]; then
    if [[ -d "$WORKSPACE_ROOT/EmbodiChain" ]]; then
        EMBODICHAIN_ROOT="$WORKSPACE_ROOT/EmbodiChain"
    else
        EMBODICHAIN_ROOT="$WORKSPACE_ROOT/EmbodiChain"
    fi
fi
VENV_DIR="${ROBOSYN_VENV_DIR:-$EMBODICHAIN_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

POLICY_NAME=smolvla

TASK_NAME="${1:?Usage: bash policy/smolvla/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]}"
SETTING="${2:?Usage: bash policy/smolvla/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]}"
CHECKPOINT_PATH="${3:?Usage: bash policy/smolvla/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]}"
GPU_ID="${4:-0}"

shift 4 2>/dev/null || true
EXTRA_ARGS=("$@")

for i in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$i]}" in
        true) EXTRA_ARGS[$i]=True ;;
        false) EXTRA_ARGS[$i]=False ;;
        none|null) EXTRA_ARGS[$i]=None ;;
    esac
done

if [[ -n "${SMOLVLA_PYTHON:-}" ]]; then
    EXTRA_ARGS+=(--smolvla_python "$SMOLVLA_PYTHON")
fi
if [[ -n "${SMOLVLA_LEROBOT_ROOT:-}" ]]; then
    EXTRA_ARGS+=(--lerobot_root "$SMOLVLA_LEROBOT_ROOT")
elif [[ -d "$SCRIPT_DIR/lerobot/src/lerobot" ]]; then
    EXTRA_ARGS+=(--lerobot_root "$SCRIPT_DIR/lerobot")
fi

if [[ -d "$CHECKPOINT_PATH/pretrained_model" ]]; then
    CHECKPOINT_DIR="$CHECKPOINT_PATH/pretrained_model"
else
    CHECKPOINT_DIR="$CHECKPOINT_PATH"
fi

# DexSim uses physical GPU ordinals from --gpu_id. Keep the simulator parent
# unmasked, then mask only the SmolVLA worker via --smolvla_cuda_visible_devices.
unset CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
# EmbodiChain >= 0.2.4 calls os._exit(0) inside env.close() by default, which
# kills the evaluator before it can flush stdout, print tracebacks or write
# evaluation_metrics.json. Let the Python process finish normally instead.
export EMBODICHAIN_SIM_EXIT_PROCESS="${EMBODICHAIN_SIM_EXIT_PROCESS:-0}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================="
echo "  SmolVLA Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  GPU:        $GPU_ID"
echo "  Python:     $PYTHON_BIN"
if [[ -n "${SMOLVLA_PYTHON:-}" ]]; then
    echo "  Worker Py:  $SMOLVLA_PYTHON"
else
    echo "  Worker Py:  current eval Python"
fi
if [[ -n "${SMOLVLA_LEROBOT_ROOT:-}" ]]; then
    echo "  LeRobot:    $SMOLVLA_LEROBOT_ROOT"
elif [[ -d "$SCRIPT_DIR/lerobot/src/lerobot" ]]; then
    echo "  LeRobot:    $SCRIPT_DIR/lerobot"
else
    echo "  LeRobot:    installed package in worker Python"
fi
echo "========================================="

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: cannot find executable Python: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Error: checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
    exit 1
fi

cd "$REPO_ROOT"

# Where this run's evaluation_metrics.json will land (eval_policy.py default:
# eval_result/<task>/.../<model_name>/<timestamp>/), honouring overrides.
MODEL_NAME="$(basename "$(dirname "$CHECKPOINT_DIR")")"
RESULT_ROOT="eval_result"
for i in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$i]}" in
        --model_name) MODEL_NAME="${EXTRA_ARGS[$((i + 1))]:-$MODEL_NAME}" ;;
        --eval_result_dir) RESULT_ROOT="${EXTRA_ARGS[$((i + 1))]:-$RESULT_ROOT}" ;;
    esac
done
[[ "$RESULT_ROOT" = /* ]] || RESULT_ROOT="$REPO_ROOT/$RESULT_ROOT"

# DexSim sometimes dies with SIGSEGV/SIGABRT (exit 139/134) in two places:
#  - while the render engine starts up: nothing was evaluated, a retry is safe;
#  - while the process exits AFTER the evaluation finished and
#    evaluation_metrics.json was written (seen on item_assembly): the run is
#    complete, so report success instead of re-running the whole evaluation.
# `|| rc=$?` is required: with `set -e` a bare failing command would abort
# before we could inspect it.
for _attempt in 1 2 3; do
  rc=0
  _mark="$(mktemp)"
  PYTHONWARNINGS=ignore::UserWarning \
  "$PYTHON_BIN" scripts/eval_policy.py \
      --config policy/$POLICY_NAME/deploy_policy.yml \
      --overrides \
      --task_name "$TASK_NAME" \
      --setting "$SETTING" \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --model_name "$(basename "$(dirname "$CHECKPOINT_DIR")")" \
      --gpu_id "$GPU_ID" \
      --smolvla_cuda_visible_devices "$GPU_ID" \
      "${EXTRA_ARGS[@]}" || rc=$?
  if [[ $rc -eq 139 || $rc -eq 134 ]]; then
    if [[ -d "$RESULT_ROOT/$TASK_NAME" ]] && \
       find "$RESULT_ROOT/$TASK_NAME" -path "*/$MODEL_NAME/*" -name evaluation_metrics.json \
            -newer "$_mark" -print -quit | grep -q .; then
      echo "[eval.sh] simulator crashed on exit (exit $rc) after the metrics were saved; evaluation complete" >&2
      rm -f "$_mark"
      exit 0
    fi
    echo "[eval.sh] simulator crashed before the evaluation finished (exit $rc); retry $_attempt/3 in 30s" >&2
    rm -f "$_mark"
    sleep 30
    continue
  fi
  rm -f "$_mark"
  exit $rc
done
exit 139
