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

```bash
for t in click_bell table_rearrangement water_pouring handle_basket items_handover \
         drawer_open_place mixer_operating item_assembly manipulate_pipette sample_loading; do
  HF_ENDPOINT=https://huggingface.co policy/smolvla/.venv/bin/hf download \
    "owen1257173018/robosynchallenge-xvla-${t//_/-}" --local-dir "checkpoints/$t"
done
```

## 3. Evaluate

```bash
source ../EmbodiChain/.venv/bin/activate
bash policy/smolvla/eval.sh <task_name> random checkpoints/<task_name> <gpu_id> \
  --pytorch_device cuda --headless true --renderer auto --max_episodes 100
```

No network access is needed once the checkpoints are downloaded. The start-up log shows the
worker environment and the task's settings (`[smolvla] inference config: ...`). Results are
written to `eval_result/<task_name>/smolvla/random/None/.../evaluation_metrics.json`.

## Checkpoints

| Task | Hugging Face checkpoint | Local success rate* |
|---|---|---:|
| click_bell | [owen1257173018/robosynchallenge-xvla-click-bell](https://huggingface.co/owen1257173018/robosynchallenge-xvla-click-bell) | 99% |
| table_rearrangement | [owen1257173018/robosynchallenge-xvla-table-rearrangement](https://huggingface.co/owen1257173018/robosynchallenge-xvla-table-rearrangement) | 94% |
| water_pouring | [owen1257173018/robosynchallenge-xvla-water-pouring](https://huggingface.co/owen1257173018/robosynchallenge-xvla-water-pouring) | 96% |
| handle_basket | [owen1257173018/robosynchallenge-xvla-handle-basket](https://huggingface.co/owen1257173018/robosynchallenge-xvla-handle-basket) | 85% |
| items_handover | [owen1257173018/robosynchallenge-xvla-items-handover](https://huggingface.co/owen1257173018/robosynchallenge-xvla-items-handover) | 86% |
| drawer_open_place | [owen1257173018/robosynchallenge-xvla-drawer-open-place](https://huggingface.co/owen1257173018/robosynchallenge-xvla-drawer-open-place) | 95% |
| mixer_operating | [owen1257173018/robosynchallenge-xvla-mixer-operating](https://huggingface.co/owen1257173018/robosynchallenge-xvla-mixer-operating) | 99% |
| item_assembly | [owen1257173018/robosynchallenge-xvla-item-assembly](https://huggingface.co/owen1257173018/robosynchallenge-xvla-item-assembly) | 83% |
| manipulate_pipette | [owen1257173018/robosynchallenge-xvla-manipulate-pipette](https://huggingface.co/owen1257173018/robosynchallenge-xvla-manipulate-pipette) | 59% |
| sample_loading | [owen1257173018/robosynchallenge-xvla-sample-loading](https://huggingface.co/owen1257173018/robosynchallenge-xvla-sample-loading) | 10% |

\* 100 episodes with the public parameters (`random` setting, evaluation seed 0) and the
command above, for reference when checking a setup.

## More

- `SUBMISSION.md`: the files we changed relative to the starter kit, the training data of each
  checkpoint and the per-task settings.
- `README_starter_kit.md`: the original starter-kit README.
