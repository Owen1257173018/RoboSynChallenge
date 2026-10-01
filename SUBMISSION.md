# RoboSynChallenge submission — X-VLA policies, SmolVLA fallbacks (team: TBD)

This repository is the official RoboSynChallenge starter kit plus our SmolVLA
training/evaluation workflow. Every task now submits an X-VLA checkpoint (footnote 7);
each was compared with our SmolVLA checkpoint for that task on the same data and scenes,
and the SmolVLA checkpoints remain drop-in fallbacks. The same adapter loads both. Evaluation uses
the unchanged policy-adapter contract (`policy/smolvla/eval.sh`, `deploy_policy.yml`,
`deploy_policy.py`).

## What we changed (relative to the upstream repository)

| File | Change |
|---|---|
| `policy/smolvla/deploy_policy.py` | `eval()` returns `(obs, info, truncated, inference_times)` as `scripts/eval_policy.py` expects, times the observation→action path like the other adapters, and stops executing an action chunk as soon as the task is successful. Adds optional inference knobs (`smolvla_chunk_mode`, `smolvla_action_samples`, `smolvla_seed`; defaults = stock behaviour) and a per-task settings table `smolvla_task_overrides` (see below). |
| `policy/smolvla/deploy_policy.yml` | Documents the knobs above and holds `smolvla_task_overrides`. |
| `policy/smolvla/smolvla_worker.py` | Converts uint8 camera frames to float32 in [0, 1] before LeRobot preprocessing (bilinear resize does not support uint8); seeds the policy noise on `reset` when a seed is given; implements `chunk_mode=fresh` / `n_samples>1` (average K flow-matching draws per chunk). Loads checkpoints **without network access**: the VLM is built from the bundled SmolVLM2 config/processor files in `policy/smolvla/smolvlm2_base/` (no weights, 4.7 MB) and every weight comes from the checkpoint (strict load). This replaces the default path, which re-downloads the full SmolVLM2 weights from the Hub on every load only to overwrite them; verified identical (all 500 tensors equal, same dtypes). Also loads X-VLA checkpoints: the policy class is picked from the `type` field of the checkpoint's `config.json` (`smolvla` or `xvla`), and X-VLA's bfloat16 actions are cast to float32 before they are returned. |
| `policy/smolvla/eval.sh` | Exports `EMBODICHAIN_SIM_EXIT_PROCESS=0` so the evaluator can write `evaluation_metrics.json` (EmbodiChain otherwise calls `os._exit(0)` in `env.close()`); retries up to 3 times when DexSim segfaults (exit 139/134). |
| `policy/smolvla/finetune.sh` | Fixes the `SMOLVLA_RENAME_MAP` override (bash brace-expansion bug). |
| `scripts/eval_policy.py` | Prints the traceback before `env.close()`; records per-episode task-progress metrics (read-only snapshot/restore of the env state); records which config keys were given on the command line (`_cli_override_keys`) so adapters can let the CLI win over per-task defaults. No change to success/step metrics. |
| `tools/` | Dataset conversion/merge, training queue and evaluation helpers (see below). |

Gripper actions: keep `smolvla_rescale_gripper: auto` (the default in
`deploy_policy.yml`). Passing `true` breaks every task whose dataset stores raw
gripper values (e.g. table_rearrangement).

## Environment (two Python environments)

```bash
# 1) simulator + evaluator: EmbodiChain v0.2.4 (lerobot 0.4.4, numpy < 2)
git clone --branch v0.2.4 https://github.com/DexForce/EmbodiChain.git ../EmbodiChain
cd ../EmbodiChain && uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e . --extra-index-url http://pyp.open3dv.site:2345/simple/ --allow-insecure-host pyp.open3dv.site:2345
uv pip install "numpy<2.0" "lerobot[smolvla]==0.4.4" && uv pip install "numpy<2.0"
cd ../RoboSynChallenge && uv pip install -e .

# 2) SmolVLA worker + training: LeRobot 0.6.2 at the exact commit we used (Python 3.12)
git clone https://github.com/huggingface/lerobot.git policy/smolvla/lerobot
git -C policy/smolvla/lerobot checkout 89236ea0f4f81a81ca566081e20dd1ff5f823cbe
cd policy/smolvla && uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e "./lerobot[smolvla,dataset]" && cd ../..
```

Evaluation needs no network access: the worker loads each checkpoint together with the
bundled `policy/smolvla/smolvlm2_base/` files (tested with an empty Hugging Face cache and
`HF_HUB_OFFLINE=1`). The X-VLA checkpoint needs nothing beyond this environment (LeRobot's
`xvla` extra is only `transformers`, already pulled in by `smolvla`) and carries its own
tokenizer, so it also loads offline (tested the same way). Network is only needed to download the checkpoints themselves and,
for training, the SmolVLM2 weights (`HF_ENDPOINT` must then point at https://huggingface.co).

## Evaluate

```bash
# from the repository root; <gpu> is the physical GPU id used by DexSim
export SMOLVLA_PYTHON=$PWD/policy/smolvla/.venv/bin/python SMOLVLA_LEROBOT_ROOT=$PWD/policy/smolvla/lerobot
bash policy/smolvla/eval.sh <task_name> random <checkpoint_dir> <gpu> \
  --pytorch_device cuda --headless true --renderer auto --max_episodes 100
# our helper, submission config (defers to the per-task table exactly like the call above):
SMOLVLA_STEPS=table bash tools/smolvla_eval.sh <task_name> random <checkpoint_dir> <gpu> <episodes>
```

Without `SMOLVLA_STEPS=table` the helper runs in *experiment* mode: it always
passes `--smolvla_steps` (default 10), `--smolvla_chunk_mode` (default `queue`)
and `--smolvla_action_samples` (default 1) on the command line, so the per-task
table never changes an experiment's configuration.

Per-task inference settings live in `deploy_policy.yml` under
`smolvla_task_overrides` and are applied automatically from `task_name`, so the
plain call above already uses the validated settings for every task. Any key
passed explicitly on the command line (e.g. `--smolvla_steps 10`) takes
precedence over the table. The effective settings are printed at start-up as
`[smolvla] inference config: ...`.

`<checkpoint_dir>` is a LeRobot `pretrained_model` directory (or a checkpoint
step directory containing one). Results land in
`eval_result/<task>/smolvla/random/None/<model>/<timestamp>/evaluation_metrics.json`.

## Checkpoints (one per task)

Upload: `policy/smolvla/.venv/bin/python tools/upload_checkpoints.py --namespace <hf_user_or_org>`
prints the plan (one model repo per task, built from the table below); add
`--upload` to push. Replace the local paths below with the printed links.

"Local SR" is a 100-episode run in submission mode (`tools/final_eval.sh`:
the plain `policy/smolvla/eval.sh` call with the per-task table, no policy
seed; sim `random` setting, evaluation seed 0). Single 50-episode runs of the
same configuration vary by about ±8 points, so only these 100-episode numbers
are reported. The organizers' held-out test uses different seeds and
parameters. "Data" counts episodes: official = the organizers' 1000-episode sim
dataset, local = episodes we generated with the official EmbodiChain pipeline
(`launch/run_task.sh`, `random` / `clear` settings).

| Task | Checkpoint (local path → HF link TBD) | Data | Inference settings | Local SR |
|---|---|---|---|---:|
| click_bell | `xvla_click_bell_run1_2500eps/checkpoints/030000` | official 1000 + local 1500 random (X-VLA⁷) | steps 10 | 99%⁷ |
| table_rearrangement | `xvla_table_rearrangement_run1/checkpoints/030000` | official 1000 (X-VLA⁷) | steps 10, `fresh`, K=4 (table) | 94%⁷ |
| water_pouring | `xvla_water_pouring_run1_1900eps/checkpoints/030000` | official 1000 + local 900 (600 clear + 300 random) (X-VLA⁷) | steps 10, `fresh`, K=4 (table) | 96%³ ⁷ |
| handle_basket | `xvla_handle_basket_run1_1500eps/checkpoints/030000` | official 1000 + local 500 random (X-VLA⁷) | steps 20, `fresh`, K=4 (table) | 85%¹ ⁷ |
| items_handover | `xvla_items_handover_run1_1500eps/checkpoints/030000` | official 1000 + local 500 (X-VLA⁷) | steps 20, `fresh`, K=4 (table) | 86%¹ ⁷ |
| drawer_open_place | `xvla_drawer_open_place_run1_2000eps/checkpoints/030000` | official 1000 + local 1000 random (X-VLA⁷) | steps 10 | 95%⁷ |
| mixer_operating | `xvla_mixer_operating_run1_1500eps/checkpoints/030000` | official 1000 + local 500 (X-VLA⁷) | steps 20, `fresh`, K=4 (table) | 99%⁷ |
| item_assembly | `xvla_item_assembly_run1_trim_2000eps/checkpoints/030000` | official 1000 + local 1000, static end-of-demo hold excluded⁴ (X-VLA⁷) | steps 20 (table) | 83%⁶ ⁷ |
| manipulate_pipette | `xvla_manipulate_pipette_run1_1500eps/checkpoints/030000` | official 1000 + local 500 (X-VLA⁷) | steps 10, `fresh`, K=4 (table) | 59%⁷ |
| sample_loading | `xvla_sample_loading_run1/checkpoints/030000` | official 1000 (X-VLA⁷) | steps 10 | 10%² ⁷ |

¹ Why these tasks use `fresh` + K=4: paired A/B runs (50 scenes, `smolvla_seed:
1000` so both arms see identical policy noise). Stock (`queue`, K=1) vs `fresh` + K=4:
items_handover 46% → 76% (exact McNemar p=0.0015), handle_basket 58% → 84%
(p=0.0023; A/B on run1), manipulate_pipette 47% → 69% on 49 shared scenes (p=0.043).
`fresh` re-plans a new action chunk at every inference call; K=4 averages four
flow-matching draws per chunk. The knobs did not help on item_assembly (58% →
48%, p=0.40), click_bell (68% → 64%, p=0.77), drawer_open_place (90% → 82%,
p=0.34), mixer_operating (94% → 82%, p=0.15), table_rearrangement (100% → 94%,
p=0.25) or sample_loading (8% → 4%, p=0.69), so those keep the stock setting. The submission leaves `smolvla_seed`
unset (the seed does not change expected performance).

³ The knobs gave 68% → 82% on run1 (seed 1000) and 74% → 82% on
run2 (seed 2000); pooled over both, 14 vs 25 discordant pairs, exact McNemar
p=0.11 -- consistently better but not significant. We adopt a knob whenever it
is at least as good on paired seeds, because extra inference time does not
change our score (the inference-efficiency term is already 0 for a VLA of this
size).

² Measured with the official configs of 2026-09-24 (narrower tube and rack
spawn ranges); the SmolVLA checkpoint scored 8% with them and 6% with the earlier configs. Stock
inference. The task's expert planner rarely succeeds (about 7% of attempts even
in the `clear` setting; with the new configs 10 episodes took 3 hours), so we
could not generate extra training data for it.

⁴ The official item_assembly demos end with ~51 frames in which the arms stand
still after the task is done; `tools/trim_static_tails.py` ends each training
episode 5 frames after the last arm motion (metadata-only; the cut frames become
padded action targets that the loss ignores). Same data and seed as the untrimmed
run; over two evaluation rounds (136 paired scenes) 62% → 68%, 24 vs 31
discordant pairs, p=0.42 -- not significant, and the same trimming did not help
handle_basket, so we treat it as neutral; item_assembly keeps the trimmed run
only because it scored higher. The task is
noisy: the same checkpoint scored 78% on one 50-episode run and 62% on 100 episodes.

Checkpoints that were tried and not kept: item_assembly X-VLA trained on our 1000
earlier-expert episodes only (65% vs 83% on the same 97 scenes, 9 vs 28 discordant pairs),
item_assembly X-VLA trained on the official
1000 + 1000 episodes we generated with the updated expert configuration instead of our
earlier episodes (48% vs 83% on the same 97 scenes, 7 vs 40 discordant pairs: the
earlier-expert episodes are what makes this task work), drawer_open_place run2 (the same data
trained 50k steps: 80%), water_pouring run3 (the 600 clear
episodes replaced by 600 random ones: 74% vs run2's 84% on the same 100 scenes),
items_handover run3 (+300 clear episodes: 74% vs run2's 86%, p=0.023),
manipulate_pipette run4 (+500 more random episodes: 54% vs run2's 57%),
handle_basket run1 (official 1000 only:
73% vs run2's 84% on the same 100 scenes with the submitted fresh+K4 setting,
p=0.035; with stock inference the two had tied), click_bell run2 (official 1000 + local
500 random + 500 clear: 80% on the same 100 scenes as run4's 84%),
manipulate_pipette run3 (+300 clear episodes: 58% → 38%), manipulate_pipette run6 (same data as run2 trained 100k steps: 54% vs 57% on the same 100 scenes), handle_basket run1_trim (end-of-demo hold excluded; 84% →
80% vs run1 with fresh+K4 on 50 seeded scenes). 

⁶ Measured with the official configs of 2026-09-24 (item_assembly: 500-step
limit and updated expert planner; our earlier runs used the 9/11 configs with a
361-step limit and scored 62%). A 100k-step version of the same data scored
65% vs 66% on the same 97 scenes, so the 50k checkpoint stayed (it is now the
fallback for the X-VLA checkpoint, footnote 7). The 1000 local
item_assembly episodes were generated with the expert configuration that preceded
that update; the official dataset matches the updated expert (same episode length
and motion profile as episodes we generate with it now).

⁷ X-VLA (LeRobot `policies/xvla`) fine-tuned from the public `lerobot/xvla-base`
checkpoint (about 0.9B parameters, pretrained by its authors on public robot datasets):
full fine-tuning, bfloat16, `action_mode=auto` (14-dim actions padded to 20), camera keys
renamed to the base model's `image`/`image2`/`image3` (the rename is stored in the
checkpoint), 30k steps, batch 16. Each X-VLA checkpoint is trained on exactly the same
episodes as the SmolVLA checkpoint it replaces, evaluated with the same task settings from
the table on the same 100 scenes in submission mode, and kept when its success rate is at
least as high (ties included: we prefer one model family where the two are equal). SmolVLA
vs X-VLA, shared scenes, discordant pairs (SmolVLA-only / X-VLA-only successes), exact
McNemar p, mean steps:

| Task | SmolVLA | X-VLA | Shared | Discordant | p | Steps |
|---|---:|---:|---:|---|---:|---|
| click_bell | 84% | 99% | 100 | 0 / 15 | 0.0001 | 99 → 54 |
| item_assembly | 66% | 83% | 97 | 11 / 29 | 0.0064 | 363 → 343 |
| water_pouring | 84% | 96% | 100 | 4 / 16 | 0.012 | 253 → 222 |
| handle_basket | 84% | 85% | 100 | 12 / 13 | 1.0 | 369 → 369 |
| items_handover | 86% | 86% | 100 | 11 / 11 | 1.0 | 312 → 318 |
| drawer_open_place | 87%⁵ | 95% | 95 | 3 / 9 | 0.15 | 424 → 387 |
| mixer_operating | 89% | 99%* | 100 | 1 / 11 | 0.0063 | 307 → 260 |
| table_rearrangement | 94% | 94%* | 100 | 6 / 6 | 1.0 | 158 → 164 |
| manipulate_pipette | 57% | 59% | 98 | 19 / 21 | 0.87 | 569 → 554 |
| sample_loading | 8% | 10% | 98 | 6 / 9 | 0.61 | 490 → 485 |

\* With `fresh`, K=4 (now the task's table entry). With the stock `queue` setting the
SmolVLA run used, the same X-VLA checkpoint scored 52% on mixer_operating (0 vs 47
discordant pairs against `fresh`, K=4: X-VLA's 30-step chunks are only re-planned every 30
steps in `queue` mode, which this task does not tolerate) and 92% on table_rearrangement
(4 vs 6). On item_assembly `fresh`, K=4 was worse (60% vs 83%, 33 vs 10), so it keeps
`queue`. The settings table was tuned on SmolVLA;
for each X-VLA task we check whether `fresh`, K=4 is at least as good and adopt it if so.
Reverting a task to its SmolVLA fallback therefore also means restoring that task's table
entry from before the switch (kept in the repository history).

X-VLA's loss does not use LeRobot's `action_is_pad` mask, so for the trimmed item_assembly
data (footnote 4) the frames after the cut are learned as repeats of the last kept action.
The replaced SmolVLA checkpoints (`smolvla_click_bell_run4_2500eps`,
`smolvla_item_assembly_run4_trim_2000eps`, `smolvla_water_pouring_run2_1900eps`,
`smolvla_handle_basket_run2_1500eps`, `smolvla_items_handover_run2_1500eps`,
`smolvla_manipulate_pipette_run2_1500eps`, `smolvla_mixer_operating_run2_1500eps`,
`smolvla_sample_loading_run1`, `smolvla_table_rearrangement_run1`, all at
`checkpoints/050000`, and `smolvla_drawer_open_place_run3_100k/checkpoints/100000`) are
drop-in fallbacks with the same task settings. The drawer_open_place SmolVLA fallback is the
100k-step run of footnote 5; the X-VLA checkpoint uses the standard 30k steps.

⁵ Same data and seed as the 50k-step run2, trained 100k steps with the cosine
learning-rate decay stretched from 30k to 60k steps: 80% → 87% on the same 100
scenes (13 vs 5 discordant pairs, p=0.096), and successful episodes are shorter
(424 vs 462 steps on average).

Training recipe for every checkpoint (except where noted): LeRobot SmolVLA initialised from
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (`load_vlm_weights=true`,
`train_expert_only=true`), chunk 50, batch 32, 50k steps, AdamW 1e-4 with
cosine decay — i.e. the organizers' released recipe; only the data differs.
All simulated data comes from the official EmbodiChain pipeline
(`launch/run_task.sh`), converted to LeRobot v3.0 with `tools/`.
