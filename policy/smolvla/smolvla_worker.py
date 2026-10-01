#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--lerobot-root", default="")
    parser.add_argument("--device", default="cuda")
    # flow-matching integration steps at inference (0 = keep the checkpoint's num_steps)
    parser.add_argument("--denoise-steps", type=int, default=0)
    return parser.parse_args()


LOCAL_VLM_DIR = Path(__file__).resolve().parent / "smolvlm2_base"


def _checkpoint_policy_type(checkpoint_dir: str) -> str:
    """LeRobot policy type recorded in the checkpoint's config.json ("smolvla", "xvla", ...)."""
    try:
        return str(json.loads((Path(checkpoint_dir) / "config.json").read_text()).get("type", "smolvla"))
    except Exception:
        return "smolvla"


def _load_policy(policy_cls, checkpoint_dir: str):
    """Load a SmolVLA checkpoint without network access.

    Checkpoints trained with load_vlm_weights=true would otherwise download the
    full SmolVLM2 weights from the Hugging Face Hub on every load, only to
    overwrite them with the checkpoint's own weights. Instead the VLM is built
    from the bundled config/processor files in smolvlm2_base/ and every weight
    comes from the checkpoint (strict load). Verified identical to the Hub path:
    all 500 tensors equal, same dtypes. Set SMOLVLA_VLM_FROM_HUB=1 to use the Hub.
    """
    if not LOCAL_VLM_DIR.is_dir() or os.environ.get("SMOLVLA_VLM_FROM_HUB") == "1":
        return policy_cls.from_pretrained(checkpoint_dir)
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
    cfg.vlm_model_name = str(LOCAL_VLM_DIR)
    cfg.load_vlm_weights = False
    policy = policy_cls.from_pretrained(checkpoint_dir, config=cfg, strict=True)
    # The Hub path builds the VLM in bfloat16 before the checkpoint is loaded.
    policy.model.vlm_with_expert.vlm.to(torch.bfloat16)
    return policy


def _load_runtime(args: argparse.Namespace):
    if args.lerobot_root:
        lerobot_src = Path(args.lerobot_root).expanduser().resolve() / "src"
        if lerobot_src.exists():
            sys.path.insert(0, str(lerobot_src))
        else:
            raise FileNotFoundError(
                f"LeRobot source path does not exist: {lerobot_src}. "
                "Install lerobot in this Python environment or set SMOLVLA_LEROBOT_ROOT."
            )

    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor import (
            PolicyProcessorPipeline,
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Cannot import lerobot. Install LeRobot in the SmolVLA Python env, "
            "or run policy/smolvla/setup_lerobot.sh and set SMOLVLA_LEROBOT_ROOT."
        ) from exc

    checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    policy_type = _checkpoint_policy_type(checkpoint_dir)
    if policy_type == "xvla":
        # X-VLA checkpoints build their Florence-2 backbone from config.json and ship
        # their tokenizer, so a plain from_pretrained is already offline.
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        policy = XVLAPolicy.from_pretrained(checkpoint_dir)
    else:
        policy = _load_policy(SmolVLAPolicy, checkpoint_dir)
    print(f"[smolvla-worker] policy type: {policy_type}", file=sys.stderr, flush=True)
    policy.to(args.device)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_preprocessor.json",
        overrides={"device_processor": {"device": args.device}},
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint_dir,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


def _obs_to_tensor(key: str, value: np.ndarray) -> torch.Tensor:
    """Convert a raw observation array into the dtype the policy pipeline expects.

    Images arrive from the adapter as uint8 CHW arrays. LeRobot's SmolVLA
    preprocessing (resize-with-pad via bilinear interpolation, normalization)
    only works on float tensors scaled to [0, 1], exactly like frames decoded
    from a LeRobotDataset.
    """
    tensor = torch.from_numpy(np.ascontiguousarray(value))
    if key.startswith("observation.images"):
        if tensor.dtype == torch.uint8:
            tensor = tensor.to(torch.float32) / 255.0
        elif not tensor.is_floating_point():
            tensor = tensor.to(torch.float32)
        elif float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
    elif tensor.is_floating_point() and tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    return tensor


def _load_obs(obs_path: str) -> tuple[dict[str, np.ndarray], str]:
    with np.load(obs_path, allow_pickle=False) as data:
        obs = {key: data[key] for key in data.files if key != "task"}
        task = str(data["task"].item()) if "task" in data else ""
    return obs, task


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        policy, preprocessor, postprocessor = _load_runtime(args)
        if args.denoise_steps > 0:
            # read by the policy on every sampling call (SmolVLA: num_steps, X-VLA: num_denoising_steps)
            field = "num_steps" if hasattr(policy.config, "num_steps") else "num_denoising_steps"
            setattr(policy.config, field, int(args.denoise_steps))
            print(f"[smolvla-worker] flow-matching {field} = {args.denoise_steps}", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        request = json.loads(line)
        cmd = request.get("cmd")

        try:
            if cmd == "reset":
                policy.reset()
                # SmolVLA is a flow-matching policy: every chunk starts from freshly
                # drawn Gaussian noise, so two rollouts of the same checkpoint on the
                # same scene differ. Seeding here makes an episode reproducible.
                seed = request.get("seed")
                if seed is not None:
                    torch.manual_seed(int(seed))
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(seed))
                _send({"ok": True})
                continue

            if cmd != "infer":
                raise ValueError(f"Unsupported command: {cmd}")

            with contextlib.redirect_stdout(sys.stderr):
                obs, task = _load_obs(request["obs_path"])
                raw_state = np.asarray(obs.get("observation.state", []), dtype=np.float32).reshape(-1)
                obs = {key: _obs_to_tensor(key, value) for key, value in obs.items()}
                obs["task"] = task
                proc_obs = preprocessor(obs)

                num_actions = int(request.get("n_action_steps", 1))
                n_samples = max(1, int(request.get("n_samples", 1)))
                chunk_mode = request.get("chunk_mode", "queue")

                actions = []
                if request.get("select") == "max_motion":
                    # Anti-stall: the adapter saw the arms motionless for longer than
                    # any pause in the demonstrations. Draw n_samples chunks from the
                    # policy and keep the one whose arm targets move furthest from the
                    # current joint state, i.e. the policy's own "move on" mode. The
                    # queue is cleared so the next normal call re-plans from scratch.
                    skip = set(int(i) for i in request.get("exclude_dims", []))
                    arm = [i for i in range(raw_state.shape[0]) if i not in skip]
                    best, best_disp = None, -1.0
                    for _ in range(n_samples):
                        chunk = policy.predict_action_chunk(proc_obs)
                        cand = [
                            postprocessor(chunk[:, i]).squeeze(0).float().cpu().numpy().astype(np.float32)
                            for i in range(min(num_actions, chunk.shape[1]))
                        ]
                        disp = float(np.abs(np.stack(cand)[:, arm] - raw_state[arm]).max()) if arm else 0.0
                        if disp > best_disp:
                            best, best_disp = cand, disp
                    policy.reset()
                    actions = [a.tolist() for a in best]
                elif chunk_mode == "fresh" or n_samples > 1:
                    # Re-plan from the current observation on every call and take the
                    # first `num_actions` steps of the chunk. `policy.select_action`
                    # instead pops from a persistent queue and only re-plans once the
                    # queue drains (every n_action_steps=50 executed steps), so the
                    # observation it plans from is not the one we just handed it.
                    # With n_samples > 1 we average that many independent noise draws:
                    # the training demonstrations come from a deterministic expert, so
                    # the action distribution at a given state should be close to
                    # unimodal and averaging cuts the sampling variance.
                    chunks = [policy.predict_action_chunk(proc_obs) for _ in range(n_samples)]
                    chunk = torch.stack(chunks, dim=0).mean(dim=0) if n_samples > 1 else chunks[0]
                    for i in range(min(num_actions, chunk.shape[1])):
                        action = postprocessor(chunk[:, i])
                        actions.append(action.squeeze(0).float().cpu().numpy().astype(np.float32).tolist())
                else:
                    for _ in range(num_actions):
                        action = policy.select_action(proc_obs)
                        action = postprocessor(action)
                        actions.append(action.squeeze(0).float().cpu().numpy().astype(np.float32).tolist())

            _send({"ok": True, "actions": actions})
        except Exception as exc:  # noqa: BLE001
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
