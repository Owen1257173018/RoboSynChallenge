#!/bin/bash
# ----------------------------------------------------------------------------
# Launch SmolVLA finetuning for one task with the dedicated SmolVLA venv.
#
# Usage: bash tools/smolvla_train.sh <task_name> <dataset_root> <run_name> [gpu_id] [extra lerobot-train args...]
# Example (official recipe, 50k steps, batch 32):
#   bash tools/smolvla_train.sh click_bell lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_click_bell run1 0
# Example (smoke test):
#   bash tools/smolvla_train.sh click_bell lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_click_bell smoke 0 --steps=100 --save_freq=100 --log_freq=10
#
# Locally collected datasets (v3.0, new feature names) need the new rename map:
#   SMOLVLA_RENAME_MAP='{"observation.images.cam_high":"observation.images.camera1","observation.images.cam_left_wrist":"observation.images.camera2","observation.images.cam_right_wrist":"observation.images.camera3"}'
# Output: outputs/train/<run_name>/ , log: outputs/train/logs/<run_name>.log
# ----------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK="${1:?usage: smolvla_train.sh <task_name> <dataset_root> <run_name> [gpu_id] [extra args]}"
DATASET_ROOT="${2:?dataset_root}"
RUN_NAME="${3:?run_name}"
GPU_ID="${4:-0}"
shift 4 2>/dev/null || shift $#
EXTRA=("$@")

[[ "$DATASET_ROOT" = /* ]] || DATASET_ROOT="$REPO_ROOT/$DATASET_ROOT"
OUTPUT_DIR="$REPO_ROOT/outputs/train/$RUN_NAME"
source "$REPO_ROOT/policy/smolvla/.venv/bin/activate"
export HF_ENDPOINT="${HF_ENDPOINT_OFFICIAL:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export SMOLVLA_LEROBOT_ROOT="$REPO_ROOT/policy/smolvla/lerobot"
export SMOLVLA_LOG_DIR="$REPO_ROOT/outputs/train/logs"
export SMOLVLA_NOHUP="${SMOLVLA_NOHUP:-1}"
export TOKENIZERS_PARALLELISM=false
cd "$REPO_ROOT"
bash policy/smolvla/finetune.sh "$TASK" "$DATASET_ROOT" "$OUTPUT_DIR" "$GPU_ID" "${EXTRA[@]}"
