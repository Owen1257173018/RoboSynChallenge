# Submission notes

How to set up and run the evaluation is in [README.md](README.md). This file lists what we
changed relative to the starter kit and what each checkpoint was trained on.

## What we changed (relative to EDEM-AI/RoboSynChallenge)

| File | Change |
|---|---|
| `policy/smolvla/smolvla_worker.py` | Loads X-VLA as well as SmolVLA checkpoints (chosen from the `type` field of the checkpoint's `config.json`) and casts X-VLA's bfloat16 actions to float32. Converts uint8 camera frames to float before preprocessing. Loads checkpoints without network access. Implements the `fresh` and K-sample inference modes used by the settings table. |
| `policy/smolvla/deploy_policy.py` | `eval()` returns the 4-tuple that `scripts/eval_policy.py` expects and stops executing an action chunk once the task has succeeded. Applies the per-task settings table `smolvla_task_overrides` from the task name (keys given on the command line take precedence). |
| `policy/smolvla/deploy_policy.yml` | Holds the per-task settings table. |
| `policy/smolvla/eval.sh` | Uses the worker environment `policy/smolvla/.venv` automatically when `SMOLVLA_PYTHON` is not set, and stops with a clear message if the worker Python is older than 3.12. Retries when DexSim crashes while starting (exit 139/134); a crash after `evaluation_metrics.json` has been written counts as a finished run. |
| `scripts/eval_policy.py` | Prints the traceback before closing the environment, records per-episode task metrics and passes the command-line override keys to the adapter. Success and step metrics are unchanged. |
| `policy/smolvla/finetune.sh`, `tools/` | Training and data helpers; not used during evaluation. |

## Checkpoints and training

Every checkpoint is X-VLA (LeRobot `policies/xvla`) fine-tuned from the public
`lerobot/xvla-base`: full fine-tuning, bfloat16, `action_mode=auto` (14-dim actions padded to
20), camera keys mapped to the base model's `image`/`image2`/`image3` (stored in the
checkpoint), 30k steps, batch size 16. Training data per task: the official 1000-episode
simulation dataset plus episodes we generated with the official pipeline
(`launch/run_task.sh`, `random`/`clear` settings). No other data was used.

| Task | Trained run | Hugging Face | Training episodes | Settings (automatic) | Local SR |
|---|---|---|---|---|---:|
| click_bell | `xvla_click_bell_run1_2500eps/checkpoints/030000` | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-click-bell) | official 1000 + 1500 random | steps 10 | 99% |
| table_rearrangement | `xvla_table_rearrangement_run1/checkpoints/030000` | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-table-rearrangement) | official 1000 | steps 10, fresh, K=4 | 94% |
| water_pouring | `xvla_water_pouring_run1_1900eps/checkpoints/030000` | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-water-pouring) | official 1000 + 600 clear + 300 random | steps 10, fresh, K=4 | 96% |
| handle_basket | `xvla_handle_basket_run1_1500eps/checkpoints/030000` | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-handle-basket) | official 1000 + 500 random | steps 20, fresh, K=4 | 85% |
| items_handover | `xvla_items_handover_run1_1500eps/checkpoints/030000` | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-items-handover) | official 1000 + 500 random | steps 20, fresh, K=4 | 86% |
| drawer_open_place | `xvla_drawer_open_place_run1_2000eps/checkpoints/030000` | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-drawer-open-place) | official 1000 + 1000 random | steps 10 | 95% |
| mixer_operating | `xvla_mixer_operating_run1_1500eps/checkpoints/030000` | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-mixer-operating) | official 1000 + 500 random | steps 20, fresh, K=4 | 99% |
| item_assembly | `xvla_item_assembly_run1_trim_2000eps/checkpoints/030000` | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-item-assembly) | official 1000 + 1000 random¹ | steps 20 | 83% |
| manipulate_pipette | `xvla_manipulate_pipette_run1_1500eps/checkpoints/030000` | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-manipulate-pipette) | official 1000 + 500 random | steps 10, fresh, K=4 | 59% |
| sample_loading | `xvla_sample_loading_run1/checkpoints/030000` | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-sample-loading) | official 1000 | steps 10 | 10% |

Settings: *steps* = actions executed per inference call; *fresh* = plan a new action chunk from
the current observation at every call (otherwise the chunk is replanned only when it is used
up); *K=4* = average four sampled action chunks. "Local SR" = 100 episodes in our environment
with the public parameters (`random`, evaluation seed 0) and the plain `eval.sh` command, using
the starter kit before 2026-10-01 (randomized top camera); the README lists 50-episode results
with the current starter kit.

¹ The static end-of-demo hold (arms still after the task is done) is excluded from training
(metadata only, `tools/trim_static_tails.py`).

## Reproduction check (2026-10-02)

From a fresh clone of this repository, with both environments built exactly as in the README
(starter kit of 2026-10-01, EmbodiChain v0.2.4.post1, numpy 2.x) and the ten checkpoints
downloaded from Hugging Face (byte-identical to ours), every task was evaluated with the README
command, 50 episodes each, with an empty Hugging Face cache and `HF_HUB_OFFLINE=1`:
391/500 successes (78.2%). Our earlier 100-episode runs with the previous starter kit (randomized
top camera) averaged 80.6%; no task differs significantly between the two (exact McNemar on the
shared scenes, all p ≥ 0.18).
