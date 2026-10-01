#!/usr/bin/env python
"""Make a copy of a LeRobot v3.0 dataset whose episodes end shortly after the
last robot motion, without rewriting any data or video files.

Why: some expert demos (item_assembly, handle_basket) end with ~50 frames of the
robot holding perfectly still after the task is done. The policy learns a
"hold still" mode from them and can fall into it before the task is finished.
Evaluation stops as soon as the task succeeds, so the hold phase never needs to
be executed.

How: only meta/episodes/*.parquet is rewritten -- ``dataset_to_index`` and
``length`` are reduced to (last moving frame + 1 + margin). LeRobot's sampler
never picks frames past ``dataset_to_index`` and marks any action target past
it as padding (``action_is_pad``), which SmolVLA's loss ignores. ``data/`` and
``videos/`` are symlinked to the source dataset.

Motion is measured on the arm joints only (gripper dims ignored by default):
in the official item_assembly demos the arms stop ~50 frames before the end,
the grippers stay closed for ~40 frames and only open during the last ~10.
Success is judged on the object poses, so neither the hold nor the release is
needed by the evaluator.

Usage:
  python tools/trim_static_tails.py <src_root> <dst_root> [--margin 5] [--thresh 1e-3] [--ignore-dims 6,13]
"""

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def last_moving_frame(actions: np.ndarray, thresh: float, dims: list[int]) -> int:
    """Index of the last frame whose action differs from the next one by >= thresh on any of dims."""
    delta = np.abs(np.diff(actions[:, dims], axis=0)).max(axis=1)
    moving = np.nonzero(delta >= thresh)[0]
    return int(moving[-1]) + 1 if len(moving) else len(actions) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--margin", type=int, default=5)
    ap.add_argument("--thresh", type=float, default=1e-3)
    ap.add_argument("--ignore-dims", default="6,13", help="action dims ignored for motion (grippers)")
    args = ap.parse_args()
    ignore = {int(x) for x in args.ignore_dims.split(",") if x != ""}
    src, dst = Path(args.src).resolve(), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} already exists")

    # per-episode new length from the action stream
    # an episode can span two data files, so collect all its frames first
    per_ep = {}
    for f in sorted(glob.glob(str(src / "data/*/*.parquet"))):
        t = pq.read_table(f, columns=["episode_index", "index", "action"]).to_pandas()
        for ep, df in t.groupby("episode_index", sort=False):
            per_ep.setdefault(int(ep), []).append(df)
    new_len = {}
    for ep, parts in per_ep.items():
        df = parts[0] if len(parts) == 1 else pd.concat(parts).sort_values("index")
        a = np.stack(df["action"].values)
        dims = [d for d in range(a.shape[1]) if d not in ignore]
        new_len[ep] = min(len(a), last_moving_frame(a, args.thresh, dims) + 1 + args.margin)

    dst.mkdir(parents=True)
    for sub in ("data", "videos"):
        if (src / sub).exists():
            os.symlink(src / sub, dst / sub)
    shutil.copytree(src / "meta", dst / "meta")

    removed = 0
    for f in sorted(glob.glob(str(dst / "meta/episodes/*/*.parquet"))):
        table = pq.read_table(f)
        df = table.to_pandas()
        for i, row in df.iterrows():
            ep = int(row["episode_index"])
            keep = new_len[ep]
            old = int(row["length"])
            assert int(row["dataset_to_index"]) - int(row["dataset_from_index"]) == old, f"episode {ep} inconsistent"
            df.at[i, "length"] = keep
            df.at[i, "dataset_to_index"] = int(row["dataset_from_index"]) + keep
            removed += old - keep
        # keep the original schema (dtypes) exactly
        out = table.from_pandas(df, schema=table.schema, preserve_index=False)
        pq.write_table(out, f)

    lens = np.array(list(new_len.values()))
    info = {
        "source": str(src),
        "margin": args.margin,
        "thresh": args.thresh,
        "ignore_dims": sorted(ignore),
        "episodes": len(lens),
        "frames_removed_from_sampling": removed,
        "new_length_median": float(np.median(lens)),
    }
    (dst / "meta/trim_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
