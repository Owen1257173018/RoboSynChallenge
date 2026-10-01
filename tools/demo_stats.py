#!/usr/bin/env python
"""Summarize the expert demonstrations in one or more LeRobot v3.0 datasets.

Per dataset: episode length, static end-of-demo hold ("static tail": frames
after the last arm motion), first contact frame (item_assembly's
guijiao_contact sensor, if recorded) and the arm-motion profile in 40-frame
windows (mean over episodes of the max per-step arm joint change). Two demos
from the same expert config have near-identical profiles; a different expert
shows up as a different length / profile. Also prints feature keys that differ
from the first dataset, which would block merging.

  policy/smolvla/.venv/bin/python tools/demo_stats.py <dataset_dir> [...] [--episodes 50]
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ARM = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # joint dims of both arms (6 = left gripper, 13 = right gripper)
WINDOW = 40


def episode_stats(df):
    a = np.stack(df["action"].values)
    v = np.abs(np.diff(a[:, ARM], axis=0)).max(axis=1)
    moving = np.nonzero(v >= 1e-3)[0]
    tail = len(v) - 1 - moving[-1] if len(moving) else len(v)
    contact = None
    if "guijiao_contact.distance" in df:
        d = np.stack(df["guijiao_contact.distance"].values)
        ok = np.stack(df["guijiao_contact.is_valid"].values)
        dd = np.where(ok, d, np.inf).min(axis=1)
        contact = int(np.argmax(dd <= 0.003)) if (dd <= 0.003).any() else -1
    prof = [float(v[i:i + WINDOW].mean()) for i in range(0, len(v), WINDOW)]
    return len(a), tail, contact, prof


def summarize(root: Path, max_eps: int):
    info = json.loads((root / "meta" / "info.json").read_text())
    files = sorted(glob.glob(str(root / "data" / "*" / "*.parquet")))
    cols = pq.read_schema(files[0]).names
    use = ["episode_index", "action"] + [c for c in ("guijiao_contact.distance", "guijiao_contact.is_valid") if c in cols]
    rows = []
    for f in files:
        t = pq.read_table(f, columns=use).to_pandas()
        for e in t["episode_index"].unique():
            rows.append(episode_stats(t[t.episode_index == e]))
            if len(rows) >= max_eps:
                break
        if len(rows) >= max_eps:
            break
    lens = np.array([r[0] for r in rows])
    tails = np.array([r[1] for r in rows])
    contacts = np.array([r[2] for r in rows if r[2] is not None])
    width = max(len(r[3]) for r in rows)
    prof = np.nanmean([r[3] + [np.nan] * (width - len(r[3])) for r in rows], axis=0)
    print(f"\n== {root}  ({info['total_episodes']} episodes, {info['total_frames']} frames; {len(rows)} analysed)")
    print(f"   length      : median {np.median(lens):.0f}  min {lens.min()}  max {lens.max()}")
    print(f"   static tail : median {np.median(tails):.0f}  min {tails.min()}  max {tails.max()}")
    if len(contacts):
        hit = contacts[contacts >= 0]
        print(f"   first contact: median {np.median(hit) if len(hit) else float('nan'):.0f}  "
              f"never {int((contacts < 0).sum())}/{len(contacts)}")
    print(f"   motion/{WINDOW}f : {[round(float(x), 3) for x in prof]}")
    return set(info["features"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--episodes", type=int, default=50, help="max episodes analysed per dataset")
    args = ap.parse_args()
    ref = None
    for d in args.datasets:
        feats = summarize(Path(d), args.episodes)
        if ref is None:
            ref = feats
        elif feats != ref:
            print(f"   features differ from the first dataset: only here {sorted(feats - ref)}, missing {sorted(ref - feats)}")


if __name__ == "__main__":
    main()
