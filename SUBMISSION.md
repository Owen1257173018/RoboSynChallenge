# Submission notes

How to set up and run the evaluation is in [README.md](README.md). This file lists what we
changed relative to the starter kit and how each checkpoint was trained.

## What we changed (relative to EDEM-AI/RoboSynChallenge)

| File | Change |
|---|---|
| `policy/smolvla/smolvla_worker.py` | Loads X-VLA as well as SmolVLA checkpoints (chosen from the `type` field of the checkpoint's `config.json`) and casts X-VLA's bfloat16 actions to float32. Converts uint8 camera frames to float before preprocessing. Loads checkpoints without network access. Implements the `fresh` and multi-sample inference modes used by the per-task settings. |
| `policy/smolvla/deploy_policy.py` | `eval()` returns the 4-tuple that `scripts/eval_policy.py` expects and stops executing an action chunk once the task has succeeded. Applies the per-task settings `smolvla_task_overrides` from the task name (keys given on the command line take precedence). |
| `policy/smolvla/deploy_policy.yml` | Holds the per-task settings. |
| `policy/smolvla/eval.sh` | Uses the worker environment `policy/smolvla/.venv` automatically when `SMOLVLA_PYTHON` is not set, and stops with a clear message if the worker Python is older than 3.12. Retries when DexSim crashes while starting (exit 139/134); a crash after `evaluation_metrics.json` has been written counts as a finished run. |
| `scripts/eval_policy.py` | Prints the traceback before closing the environment, records per-episode task metrics and passes the command-line override keys to the adapter. Success and step metrics are unchanged; the adapter also works with the unmodified official script. |
| `policy/smolvla/finetune.sh`, `tools/` | Training and data helpers; not used during evaluation. |

## Checkpoints and training

Every checkpoint is X-VLA (LeRobot `policies/xvla`) fine-tuned from the public
`lerobot/xvla-base`: full fine-tuning, bfloat16, `action_mode=auto` (14-dim actions padded to
20), camera keys mapped to the base model's `image`/`image2`/`image3` (stored in the
checkpoint), 30k steps, batch size 16. Training data per task: the official 1000-episode
simulation dataset plus episodes we generated with the official pipeline
(`launch/run_task.sh`, `random`/`clear` settings). No other data was used.

| Task | Hugging Face | Training episodes | Inference settings |
|---|---|---|---|
| click_bell | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-click-bell) | official 1000 + 1500 random | steps 10 |
| table_rearrangement | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-table-rearrangement) | official 1000 | steps 20, fresh |
| water_pouring | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-water-pouring) | official 1000 + 600 clear + 300 random | steps 10, fresh, K=4 |
| handle_basket | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-handle-basket) | official 1000 + 500 random | steps 20, fresh, K=4 |
| items_handover | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-items-handover) | official 1000 + 500 random | steps 20, fresh |
| drawer_open_place | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-drawer-open-place) | official 1000 + 1000 random | steps 10 |
| mixer_operating | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-mixer-operating) | official 1000 + 500 random | steps 20, fresh, K=4 |
| item_assembly | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-item-assembly) | official 1000 + 1000 random¹ | steps 20 |
| manipulate_pipette | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-manipulate-pipette) | official 1000 + 500 random | steps 20, fresh |
| sample_loading | [link](https://huggingface.co/owen1257173018/robosynchallenge-xvla-sample-loading) | official 1000 | steps 10, fresh |

Inference settings are applied automatically from the task name (`smolvla_task_overrides` in
`policy/smolvla/deploy_policy.yml`): *steps* = actions executed per policy call; *fresh* = a new
action chunk is planned from the current observation at every call (otherwise the planned chunk
is executed until it is used up); *K=4* = the average of four sampled action chunks.

¹ The static end-of-demo hold (arms still after the task is done) is excluded from training
(metadata only, `tools/trim_static_tails.py`).
