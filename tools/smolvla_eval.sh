#!/bin/bash
# ----------------------------------------------------------------------------
# Evaluate a SmolVLA checkpoint in the RoboSynChallenge simulator with the
# two-venv layout used in this workspace:
#   simulator / eval_policy.py : ../EmbodiChain/.venv   (lerobot 0.4.4, numpy<2)
#   SmolVLA worker             : policy/smolvla/.venv   (lerobot 0.6.x)
#
# Usage: bash tools/smolvla_eval.sh <task_name> <random|clear> <checkpoint_dir> [gpu_id] [episodes] [extra eval args...]
#   checkpoint_dir: HF-style dir (…/pretrained_model contents) or a LeRobot step dir
# Example:
#   bash tools/smolvla_eval.sh click_bell random checkpoints/SmolVLA_sim_click_bell 1 20
# Results: eval_result/<task>/smolvla/<setting>/None/<model_name>/<timestamp>/evaluation_metrics.json
# ----------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK="${1:?usage: smolvla_eval.sh <task> <random|clear> <checkpoint_dir> [gpu_id] [episodes] [extra]}"
SETTING="${2:?setting}"
CKPT="${3:?checkpoint_dir}"
GPU_ID="${4:-0}"
EPISODES="${5:-20}"
shift 5 2>/dev/null || shift $#
EXTRA=("$@")
# Optional distinct model name (default in eval.sh is the checkpoint step, e.g. 050000,
# which collides when two evals of the same task/step start in the same second).
if [[ -n "${SMOLVLA_MODEL_NAME:-}" ]]; then EXTRA+=(--model_name "$SMOLVLA_MODEL_NAME"); fi
# Inference-time knobs:
#   SMOLVLA_STEPS           actions executed per inference call (default 10)
#   SMOLVLA_CHUNK_MODE      queue | fresh (default queue)
#   SMOLVLA_ACTION_SAMPLES  average N flow-matching draws per chunk (default 1)
#   SMOLVLA_SEED            base seed -> reproducible rollouts (default unseeded)
# Two modes:
#   explicit (default)  steps/chunk_mode/action_samples are always passed on the
#                       command line (stock values when unset), so the per-task
#                       table in deploy_policy.yml never changes an experiment.
#   SMOLVLA_STEPS=table nothing is passed unless set, i.e. exactly what the
#                       organizers' plain eval.sh call runs (the submission config).
KNOB_ARGS=()
if [[ "${SMOLVLA_STEPS:-10}" == "table" ]]; then
  if [[ -n "${SMOLVLA_CHUNK_MODE:-}" ]]; then KNOB_ARGS+=(--smolvla_chunk_mode "$SMOLVLA_CHUNK_MODE"); fi
  if [[ -n "${SMOLVLA_ACTION_SAMPLES:-}" ]]; then KNOB_ARGS+=(--smolvla_action_samples "$SMOLVLA_ACTION_SAMPLES"); fi
else
  KNOB_ARGS=(--smolvla_steps "${SMOLVLA_STEPS:-10}"
             --smolvla_chunk_mode "${SMOLVLA_CHUNK_MODE:-queue}"
             --smolvla_action_samples "${SMOLVLA_ACTION_SAMPLES:-1}")
fi
if [[ -n "${SMOLVLA_SEED:-}" ]]; then KNOB_ARGS+=(--smolvla_seed "$SMOLVLA_SEED"); fi

source "$REPO_ROOT/../EmbodiChain/.venv/bin/activate"
export HF_ENDPOINT="${HF_ENDPOINT_OFFICIAL:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export SMOLVLA_PYTHON="$REPO_ROOT/policy/smolvla/.venv/bin/python"
export SMOLVLA_LEROBOT_ROOT="$REPO_ROOT/policy/smolvla/lerobot"
export EMBODICHAIN_SIM_EXIT_PROCESS=0
cd "$REPO_ROOT"
# Result folder name used by eval.sh (checkpoint step dir name unless overridden).
if [[ -n "${SMOLVLA_MODEL_NAME:-}" ]]; then MODEL_NAME="$SMOLVLA_MODEL_NAME"
elif [[ -d "$CKPT/pretrained_model" ]]; then MODEL_NAME="$(basename "$CKPT")"
else MODEL_NAME="$(basename "$(dirname "$CKPT")")"; fi
# dexsim occasionally segfaults (exit 139/134): at engine start-up when several
# simulators start at once (retry), or at process exit after the metrics were
# written (seen on item_assembly; the run is complete -- do not re-run it).
# policy/smolvla/eval.sh already handles both; this outer loop mirrors it.
for attempt in 1 2 3; do
  # `|| rc=$?` is load-bearing: with `set -e` a bare failing command aborts the
  # script before the retry below can run (this made the retry dead code from
  # 2026-09-18 until 2026-09-21).
  rc=0
  mark="$(mktemp)"
  bash policy/smolvla/eval.sh "$TASK" "$SETTING" "$CKPT" "$GPU_ID" \
    --pytorch_device cuda --headless true --renderer auto \
    --max_episodes "$EPISODES" --eval_video_log true \
    "${KNOB_ARGS[@]}" --smolvla_rescale_gripper "${SMOLVLA_RESCALE_GRIPPER:-auto}" \
    "${EXTRA[@]}" || rc=$?
  if [[ $rc -eq 139 || $rc -eq 134 ]]; then
    if [[ -d "eval_result/$TASK" ]] && find "eval_result/$TASK" -path "*/$MODEL_NAME/*" \
         -name evaluation_metrics.json -newer "$mark" -print -quit | grep -q .; then
      echo "[smolvla_eval] simulator crashed on exit (exit $rc) after the metrics were saved; evaluation complete" >&2
      rm -f "$mark"; exit 0
    fi
    echo "[smolvla_eval] simulator crashed before the evaluation finished (exit $rc), retry $attempt/3 in 30s" >&2
    rm -f "$mark"
    sleep 30
    continue
  fi
  rm -f "$mark"
  exit $rc
done
exit 139
