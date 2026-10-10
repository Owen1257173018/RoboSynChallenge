#!/usr/bin/env python
"""Upload the per-task checkpoints listed in tools/submission_checkpoints.json to the Hugging Face Hub.

Each task gets its own model repo, <namespace>/<prefix>-<policy type>-<task>
(policy type from the checkpoint's config.json: smolvla or xvla), containing the
LeRobot ``pretrained_model`` folder (config, weights, pre/post-processors,
tokenizer, train_config.json). The task -> checkpoint mapping is read from
tools/submission_checkpoints.json (paths under outputs/train/).

Dry run by default: it only prints what it would upload. Nothing is sent to the
Hub unless --upload is given.

  # see the plan
  policy/smolvla/.venv/bin/python tools/upload_checkpoints.py --namespace <hf_user_or_org>
  # upload (needs `hf auth login` or HF_TOKEN with write access)
  policy/smolvla/.venv/bin/python tools/upload_checkpoints.py --namespace <hf_user_or_org> --upload
"""

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_JSON = REPO_ROOT / "tools" / "submission_checkpoints.json"


def submitted_checkpoints(mapping_json: Path = CHECKPOINTS_JSON) -> dict[str, Path]:
    """task -> outputs/train/<run>/checkpoints/<step>/pretrained_model, from tools/submission_checkpoints.json."""
    mapping = json.loads(mapping_json.read_text(encoding="utf-8"))
    return {task: REPO_ROOT / "outputs" / "train" / ckpt / "pretrained_model" for task, ckpt in mapping.items()}


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def policy_type(path: Path) -> str:
    return json.loads((path / "config.json").read_text()).get("type", "smolvla")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", required=True, help="Hugging Face user or organization")
    ap.add_argument("--prefix", default="robosynchallenge")
    ap.add_argument("--private", action="store_true", help="create private repos (organizers must be able to read them)")
    ap.add_argument("--upload", action="store_true", help="actually upload (default: dry run)")
    ap.add_argument("--tasks", nargs="*", help="only these tasks")
    ap.add_argument("--endpoint", default="https://huggingface.co",
                    help="Hub endpoint; the shell profile may point HF_ENDPOINT at a mirror, which cannot host uploads")
    args = ap.parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    table = submitted_checkpoints()
    if args.tasks:
        table = {t: p for t, p in table.items() if t in args.tasks}
    if not table:
        raise SystemExit(f"no checkpoints listed in {CHECKPOINTS_JSON}")

    missing = [str(p) for p in table.values() if not (p / "model.safetensors").exists()]
    if missing:
        raise SystemExit("missing checkpoints:\n  " + "\n  ".join(missing))

    total = 0
    print(f"{'dry run' if not args.upload else 'UPLOAD'} -> https://huggingface.co/{args.namespace}")
    for task, path in table.items():
        size = folder_size(path)
        total += size
        repo_id = f"{args.namespace}/{args.prefix}-{policy_type(path)}-{task.replace('_', '-')}"
        print(f"  {task:22s} {path.relative_to(REPO_ROOT)}  ({size / 1e9:.2f} GB) -> {repo_id}")
    print(f"  total {total / 1e9:.1f} GB, {'private' if args.private else 'public'} repos")

    if not args.upload:
        print("\nNothing uploaded (dry run). Re-run with --upload to push.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    urls = {}
    for task, path in table.items():
        repo_id = f"{args.namespace}/{args.prefix}-{policy_type(path)}-{task.replace('_', '-')}"
        api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(path),
            commit_message=f"{policy_type(path)} checkpoint for {task} ({path.parent.parent.parent.name})",
        )
        urls[task] = f"https://huggingface.co/{repo_id}"
        print(f"  uploaded {task}: {urls[task]}")

    print("\nHugging Face repos:")
    for task, url in urls.items():
        print(f"| {task} | {url} |")


if __name__ == "__main__":
    main()
