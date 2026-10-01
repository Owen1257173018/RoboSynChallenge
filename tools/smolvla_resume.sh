#!/bin/bash
# ----------------------------------------------------------------------------
# Resume an interrupted SmolVLA run from its latest checkpoint (checkpoints/last).
# Optimizer, scheduler, RNG state and step count are restored by lerobot-train;
# the output dir, dataset and every other setting come from the saved
# train_config.json, so only overrides need to be given.
#
# Usage: bash tools/smolvla_resume.sh <run_name> [gpu_id] [extra lerobot-train args...]
# Example (after a power loss):
#   bash tools/smolvla_resume.sh smolvla_item_assembly_run3_2000eps 0 --save_freq=10000
# Runs in the foreground; redirect the output yourself.
# ----------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_NAME="${1:?usage: smolvla_resume.sh <run_name> [gpu_id] [extra args]}"
GPU_ID="${2:-0}"
shift 2 2>/dev/null || shift $#
EXTRA=("$@")

CONFIG="$REPO_ROOT/outputs/train/$RUN_NAME/checkpoints/last/pretrained_model/train_config.json"
if [[ ! -f "$CONFIG" ]]; then
    echo "no checkpoint to resume from: $CONFIG" >&2
    exit 1
fi
STEP=$(python3 -c "import json;print(json.load(open('$(dirname "$CONFIG")/../training_state/training_step.json'))['step'])" 2>/dev/null || echo '?')
echo "[$(date)] resuming $RUN_NAME from step $STEP on GPU$GPU_ID"

source "$REPO_ROOT/policy/smolvla/.venv/bin/activate"
export HF_ENDPOINT="${HF_ENDPOINT_OFFICIAL:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
cd "$REPO_ROOT/policy/smolvla/lerobot"
exec lerobot-train --config_path="$CONFIG" --resume=true "${EXTRA[@]}"
