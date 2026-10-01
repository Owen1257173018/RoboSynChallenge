#!/bin/bash
# ----------------------------------------------------------------------------
# Chain "collection finished -> merge + train -> 100-episode final -> paired
# comparison with the submitted checkpoint" for one task, unattended.
#
#   bash tools/after_collect.sh <collect_queue_log> <task> <setting> <gpu> <run_name> \
#        <min_new_episodes> <src1,src2,...,NEW> <submitted_run_name>
#
#   collect_queue_log   the tools/collect_queue.sh log that will print
#                       "<<< COLLECT <task> <setting> ... dir=<new dataset dir>"
#   NEW                 placeholder in the source list, replaced by that new dir
#   submitted_run_name  run whose final100_* result the new run is paired with
#
# DRY=1 only resolves and prints what it would run (use it to check a chain
# before leaving it unattended).
# ----------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLOG="${1:?collect log}"; TASK="${2:?task}"; SETTING="${3:?setting}"; GPU="${4:?gpu}"
RUN="${5:?run name}"; MIN_EPS="${6:?min new episodes}"; SRCS="${7:?sources}"; CUR="${8:?submitted run}"
cd "$REPO_ROOT"

echo "[$(date)] waiting for '<<< COLLECT $TASK $SETTING' in $CLOG"
until grep -aq "<<< COLLECT $TASK $SETTING " "$CLOG" 2>/dev/null; do
    [[ "${DRY:-0}" == 1 ]] && { echo "(dry run: collection not finished yet, using a placeholder dir)"; break; }
    sleep 120
done
line=$(grep -a "<<< COLLECT $TASK $SETTING " "$CLOG" 2>/dev/null | tail -1)
new_dir=$(sed -n 's/.* dir=\([^ ]*\).*/\1/p' <<< "$line"); new_dir="${new_dir%/}"
eps=$(sed -n 's/.* episodes=\([0-9]*\).*/\1/p' <<< "$line")
[[ "${DRY:-0}" == 1 && -z "$new_dir" ]] && { new_dir="lerobot_dataset/$TASK/<new>"; eps=$MIN_EPS; }
echo "[$(date)] collection line: ${line:-<none>}"
if [[ -z "$new_dir" || -z "$eps" || "$eps" -lt "$MIN_EPS" ]]; then
    echo "[$(date)] FAILED: new dataset '$new_dir' has ${eps:-?} episodes (< $MIN_EPS)"; exit 1
fi
srcs="${SRCS//NEW/$new_dir}"
spec="$TASK|$RUN||$srcs"
ckpt="outputs/train/$RUN/checkpoints/050000"
echo "[$(date)] train spec: $spec"
echo "[$(date)] then: final_eval $GPU 100 '$TASK|$ckpt'; paired with final100_$CUR"
[[ "${DRY:-0}" == 1 ]] && exit 0

bash tools/exp_queue.sh "$GPU" - "$spec"
[[ -d "$ckpt/pretrained_model" ]] || { echo "[$(date)] FAILED: no 50k checkpoint for $RUN"; exit 1; }
bash tools/final_eval.sh "$GPU" 100 "$TASK|$ckpt"
echo "[$(date)] paired comparison (same 100 scenes, submission mode):"
python3 tools/paired_compare.py --task "$TASK" --a "final100_$CUR" --b "final100_$RUN" --label-a "$CUR" --label-b "$RUN"
echo "[$(date)] after_collect chain for $RUN finished"
