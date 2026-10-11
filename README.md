# RoboSynChallenge 2026 — X-VLA policy submission

One X-VLA checkpoint per task (fine-tuned from the public
[`lerobot/xvla-base`](https://huggingface.co/lerobot/xvla-base)), evaluated through the
standard policy adapter in `policy/smolvla/`. That folder is the starter kit's SmolVLA adapter,
extended to load X-VLA checkpoints; it keeps its original name. Per-task inference settings are
applied automatically from the task name, so every task uses the same command.

## 1. Set up (once)

Two Python environments, as in the official SmolVLA tutorial: one for the simulator and
evaluator, and one for the policy worker (LeRobot at a pinned commit, Python 3.12).

```bash
# workspace layout: RoboSynChallenge_ws/{EmbodiChain, RoboSynChallenge}
git clone https://github.com/Owen1257173018/RoboSynChallenge.git && cd RoboSynChallenge

# simulator + evaluator: EmbodiChain v0.2.4.post1 with uv, as in the official installation guide
git clone https://github.com/DexForce/EmbodiChain.git ../EmbodiChain
git -C ../EmbodiChain checkout tags/v0.2.4.post1
cd ../EmbodiChain && uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e . --extra-index-url http://pyp.open3dv.site:2345/simple/ --trusted-host pyp.open3dv.site
cd ../RoboSynChallenge && uv pip install -e . && deactivate

# policy worker: eval.sh finds policy/smolvla/.venv automatically
git clone https://github.com/huggingface/lerobot.git policy/smolvla/lerobot
git -C policy/smolvla/lerobot checkout 89236ea0f4f81a81ca566081e20dd1ff5f823cbe
cd policy/smolvla && uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e "./lerobot[smolvla,dataset]" && deactivate && cd ../..
```

- If your simulator environment lives elsewhere (for example the conda environment of the
  official Docker guide), point `eval.sh` at it with `PYTHON_BIN=/path/to/python`, as in the
  starter kit. A worker environment in another place can be given with `SMOLVLA_PYTHON=...`.
- If `pyp.open3dv.site` cannot reach its PyPI mirror for ordinary packages, install the one
  package it hosts first and the rest from PyPI:
  `uv pip install --index-url http://pyp.open3dv.site:2345/simple/ --allow-insecure-host pyp.open3dv.site:2345 --no-deps "dexsim_engine==0.4.3"`,
  then `uv pip install --index-url https://pypi.org/simple -e .` and continue as above.

## 2. Download the checkpoints

One public Hugging Face model repository per task, downloaded into `checkpoints/<task_name>`
(run from the repository root):

```bash
policy/smolvla/.venv/bin/hf download owen1257173018/robosynchallenge-xvla-click-bell --local-dir checkpoints/click_bell
policy/smolvla/.venv/bin/hf download owen1257173018/robosynchallenge-xvla-table-rearrangement --local-dir checkpoints/table_rearrangement
policy/smolvla/.venv/bin/hf download owen1257173018/robosynchallenge-xvla-water-pouring --local-dir checkpoints/water_pouring
policy/smolvla/.venv/bin/hf download owen1257173018/robosynchallenge-xvla-handle-basket --local-dir checkpoints/handle_basket
policy/smolvla/.venv/bin/hf download owen1257173018/robosynchallenge-xvla-items-handover --local-dir checkpoints/items_handover
policy/smolvla/.venv/bin/hf download owen1257173018/robosynchallenge-xvla-drawer-open-place --local-dir checkpoints/drawer_open_place
policy/smolvla/.venv/bin/hf download owen1257173018/robosynchallenge-xvla-mixer-operating --local-dir checkpoints/mixer_operating
policy/smolvla/.venv/bin/hf download owen1257173018/robosynchallenge-xvla-item-assembly --local-dir checkpoints/item_assembly
policy/smolvla/.venv/bin/hf download owen1257173018/robosynchallenge-xvla-manipulate-pipette --local-dir checkpoints/manipulate_pipette
policy/smolvla/.venv/bin/hf download owen1257173018/robosynchallenge-xvla-sample-loading --local-dir checkpoints/sample_loading
```

## 3. Evaluate

One command per task, from the repository root with the simulator environment active
(`source ../EmbodiChain/.venv/bin/activate`). The `0` after the checkpoint path is the GPU id.

```bash
bash policy/smolvla/eval.sh click_bell          random checkpoints/click_bell          0 --pytorch_device cuda --headless true --renderer auto --max_episodes 100
bash policy/smolvla/eval.sh table_rearrangement random checkpoints/table_rearrangement 0 --pytorch_device cuda --headless true --renderer auto --max_episodes 100
bash policy/smolvla/eval.sh water_pouring       random checkpoints/water_pouring       0 --pytorch_device cuda --headless true --renderer auto --max_episodes 100
bash policy/smolvla/eval.sh handle_basket       random checkpoints/handle_basket       0 --pytorch_device cuda --headless true --renderer auto --max_episodes 100
bash policy/smolvla/eval.sh items_handover      random checkpoints/items_handover      0 --pytorch_device cuda --headless true --renderer auto --max_episodes 100
bash policy/smolvla/eval.sh drawer_open_place   random checkpoints/drawer_open_place   0 --pytorch_device cuda --headless true --renderer auto --max_episodes 100
bash policy/smolvla/eval.sh mixer_operating     random checkpoints/mixer_operating     0 --pytorch_device cuda --headless true --renderer auto --max_episodes 100
bash policy/smolvla/eval.sh item_assembly       random checkpoints/item_assembly       0 --pytorch_device cuda --headless true --renderer auto --max_episodes 100
bash policy/smolvla/eval.sh manipulate_pipette  random checkpoints/manipulate_pipette  0 --pytorch_device cuda --headless true --renderer auto --max_episodes 100
bash policy/smolvla/eval.sh sample_loading      random checkpoints/sample_loading      0 --pytorch_device cuda --headless true --renderer auto --max_episodes 100 --eval_max_seed_attempts_per_episodes 300
```

All tasks use the same adapter; the task's inference settings are applied automatically from the
task name. No network access is needed once the checkpoints are downloaded. The start-up log shows
the worker environment and the task's settings (`[smolvla] inference config: ...`). Results are
written to `eval_result/<task_name>/smolvla/random/None/.../evaluation_metrics.json`.

The extra option for `sample_loading` raises the number of candidate scenes the evaluator's expert
pre-check may try per episode: for some seeds it otherwise rejects 100 candidates in a row and
stops ("Could not find an expert-feasible seed").

## Checkpoints

| Task | Hugging Face model repository | Local path | Local success rate |
|---|---|---|---:|
| click_bell | [owen1257173018/robosynchallenge-xvla-click-bell](https://huggingface.co/owen1257173018/robosynchallenge-xvla-click-bell) | `checkpoints/click_bell` | 99% |
| table_rearrangement | [owen1257173018/robosynchallenge-xvla-table-rearrangement](https://huggingface.co/owen1257173018/robosynchallenge-xvla-table-rearrangement) | `checkpoints/table_rearrangement` | 92% |
| water_pouring | [owen1257173018/robosynchallenge-xvla-water-pouring](https://huggingface.co/owen1257173018/robosynchallenge-xvla-water-pouring) | `checkpoints/water_pouring` | 94% |
| handle_basket | [owen1257173018/robosynchallenge-xvla-handle-basket](https://huggingface.co/owen1257173018/robosynchallenge-xvla-handle-basket) | `checkpoints/handle_basket` | 84% |
| items_handover | [owen1257173018/robosynchallenge-xvla-items-handover](https://huggingface.co/owen1257173018/robosynchallenge-xvla-items-handover) | `checkpoints/items_handover` | 85% |
| drawer_open_place | [owen1257173018/robosynchallenge-xvla-drawer-open-place](https://huggingface.co/owen1257173018/robosynchallenge-xvla-drawer-open-place) | `checkpoints/drawer_open_place` | 91% |
| mixer_operating | [owen1257173018/robosynchallenge-xvla-mixer-operating](https://huggingface.co/owen1257173018/robosynchallenge-xvla-mixer-operating) | `checkpoints/mixer_operating` | 100% |
| item_assembly | [owen1257173018/robosynchallenge-xvla-item-assembly](https://huggingface.co/owen1257173018/robosynchallenge-xvla-item-assembly) | `checkpoints/item_assembly` | 78% |
| manipulate_pipette | [owen1257173018/robosynchallenge-xvla-manipulate-pipette](https://huggingface.co/owen1257173018/robosynchallenge-xvla-manipulate-pipette) | `checkpoints/manipulate_pipette` | 57% |
| sample_loading | [owen1257173018/robosynchallenge-xvla-sample-loading](https://huggingface.co/owen1257173018/robosynchallenge-xvla-sample-loading) | `checkpoints/sample_loading` | 15% |

Local success rate: 100 episodes per task with the commands above and the public parameters
(`random` setting, evaluation seed 0); 79.5% on average.

## More

- `SUBMISSION.md`: the files we changed relative to the starter kit, the training data of each
  checkpoint and the per-task settings.
- `README_starter_kit.md`: the original starter-kit README.
