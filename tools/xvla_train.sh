#!/bin/bash
# ----------------------------------------------------------------------------
# Fine-tune X-VLA (lerobot/xvla-base, ~0.9B) on one task, LeRobot's recommended
# recipe for a new robot: full fine-tuning, bfloat16, action_mode=auto (our
# 14-dim joint actions padded to 20). Camera keys are renamed to the base
# model's image/image2/image3; the rename is saved in the checkpoint's
# preprocessor, so evaluation uses our usual keys. Resumes if interrupted.
#
#   bash tools/xvla_train.sh <task> <dataset_root> <run_name> <gpu> [steps=30000] [batch=16]
# ----------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:?task}"; DS="${2:?dataset_root}"; RUN="${3:?run_name}"; GPU="${4:?gpu}"
STEPS="${5:-30000}"; BATCH="${6:-16}"
[[ "$DS" = /* ]] || DS="$REPO_ROOT/$DS"
OUT="$REPO_ROOT/outputs/train/$RUN"
source "$REPO_ROOT/policy/smolvla/.venv/bin/activate"
export HF_ENDPOINT=https://huggingface.co HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false
if [[ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]]; then
    exec bash "$REPO_ROOT/tools/smolvla_resume.sh" "$RUN" "$GPU"
fi
cd "$REPO_ROOT/policy/smolvla/lerobot"
exec lerobot-train --policy.path=lerobot/xvla-base --policy.dtype=bfloat16 \
  --policy.action_mode=auto --policy.max_action_dim=20 \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --dataset.repo_id="local/$(basename "$DS")" --dataset.root="$DS" \
  --rename_map='{"observation.images.cam_high":"observation.images.image","observation.images.cam_left_wrist":"observation.images.image2","observation.images.cam_right_wrist":"observation.images.image3"}' \
  --output_dir="$OUT" --job_name="$RUN" \
  --batch_size="$BATCH" --steps="$STEPS" --save_freq=10000 --log_freq=200 --num_workers=8 --seed=1000
