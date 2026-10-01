#!/bin/bash
# Waits for each official dataset download to finish (bulk download log), then
# converts v2.1 -> v3.0 and renames features to the local convention.
# Usage: bash tools/prepare_all_datasets_watcher.sh <bulk_download_log>
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BULK_LOG="${1:?bulk download log path}"
PY="$REPO_ROOT/policy/smolvla/.venv/bin/python"
cd "$REPO_ROOT"
SIM_TASKS="table_rearrangement water_pouring items_handover sample_loading handle_basket drawer_open_place mixer_operating item_assembly manipulate_pipette"
REAL_TASKS="water_pouring open_pan item_assembly manipulate_pipette"
prepare() {  # kind task
    local kind=$1 task=$2 dir="lerobot_dataset/RoboSynChallenge/cobotmagic_${1}_${2}"
    echo "[$(date)] preparing $dir"
    bash tools/smolvla_prepare_dataset.sh "$task" "$kind" && $PY tools/rename_dataset_features.py "$dir" && echo "[$(date)] DONE $dir" || echo "[$(date)] FAILED $dir"
}
for t in $SIM_TASKS; do
    until grep -q "<<< Sim_$t OK" "$BULK_LOG" 2>/dev/null; do
        grep -q "<<< Sim_$t FAILED" "$BULK_LOG" 2>/dev/null && { echo "[$(date)] download FAILED for Sim_$t, skipping"; continue 2; }
        sleep 60
    done
    prepare Sim "$t"
done
for t in $REAL_TASKS; do
    until grep -q "<<< Real_$t OK" "$BULK_LOG" 2>/dev/null; do
        grep -q "<<< Real_$t FAILED" "$BULK_LOG" 2>/dev/null && { echo "[$(date)] download FAILED for Real_$t, skipping"; continue 2; }
        sleep 60
    done
    echo "[$(date)] Real_$t downloaded; converting to v3.0 only (feature names differ: 32-dim aloha format, handled later)"
    bash tools/smolvla_prepare_dataset.sh "$t" Real || echo "[$(date)] FAILED Real_$t"
done
echo "[$(date)] all datasets prepared"
