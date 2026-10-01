#!/bin/bash
# ----------------------------------------------------------------------------
# Convert an official RoboSynChallenge HF dataset (LeRobot v2.1) to v3.0 in
# place so that LeRobot 0.6.x (policy/smolvla/.venv) can train on it.
#
# Usage: bash tools/smolvla_prepare_dataset.sh <task_name> [Sim|Real]
#   dataset dir: lerobot_dataset/RoboSynChallenge/cobotmagic_<Sim|Real>_<task_name>
# After conversion the original v2.1 tree is kept as <dir>_old.
# ----------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASK="${1:?usage: smolvla_prepare_dataset.sh <task_name> [Sim|Real]}"
KIND="${2:-Sim}"
DS_DIR="$REPO_ROOT/lerobot_dataset/RoboSynChallenge/cobotmagic_${KIND}_${TASK}"
REPO_ID="RoboSynChallenge/cobotmagic_${KIND}_${TASK}"
PY="$REPO_ROOT/policy/smolvla/.venv/bin/python"
export HF_ENDPOINT="${HF_ENDPOINT_OFFICIAL:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"

[[ -d "$DS_DIR/meta" ]] || { echo "dataset not found: $DS_DIR" >&2; exit 1; }
VER=$($PY -c "import json;print(json.load(open('$DS_DIR/meta/info.json'))['codebase_version'])")
echo "[$(date)] $DS_DIR codebase_version=$VER"
if [[ "$VER" == v3* ]]; then
    echo "already v3.0, nothing to do"; exit 0
fi
N_MP4=$(find "$DS_DIR/videos" -name '*.mp4' | wc -l)
N_PQ=$(find "$DS_DIR/data" -name '*.parquet' | wc -l)
echo "files: $N_PQ parquet, $N_MP4 mp4"
# Drop auxiliary [4,4] pose features first; the converter crashes on them.
$PY "$REPO_ROOT/tools/strip_extra_features_v21.py" "$DS_DIR"
rm -rf "${DS_DIR}_v30"
cd "$REPO_ROOT/policy/smolvla/lerobot"
$PY src/lerobot/scripts/convert_dataset_v21_to_v30.py --repo-id="$REPO_ID" --root="$DS_DIR" --push-to-hub=false
echo "[$(date)] converted. new version: $($PY -c "import json;print(json.load(open('$DS_DIR/meta/info.json'))['codebase_version'])")"
du -sh "$DS_DIR" "${DS_DIR}_old" 2>/dev/null || true
