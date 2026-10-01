#!/usr/bin/env python
"""Rename feature keys of a LeRobot **v3.0** dataset in place.

Why: LeRobot >= 0.6 refuses `--rename_map` when a policy is initialised from
scratch, so the dataset itself must carry the feature names the policy should
learn. The official RoboSynChallenge HF datasets (converted from v2.1) use the
old EmbodiChain names (`observation.qpos`, `cam_high.color`, ...), while data
collected locally with EmbodiChain 0.2.4 already uses the new names
(`observation.state`, `observation.images.cam_high`, ...). This script makes
the official data match the local convention so both can be trained/merged
together and the SmolVLA adapter's default image-key mapping applies.

Usage:
    python tools/rename_dataset_features.py <dataset_root> [--dry-run]
    python tools/rename_dataset_features.py <dataset_root> --map '{"old":"new",...}'
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_MAP = {
    "observation.qpos": "observation.state",
    "cam_high.color": "observation.images.cam_high",
    "cam_left_wrist.color": "observation.images.cam_left_wrist",
    "cam_right_wrist.color": "observation.images.cam_right_wrist",
}


def _rename_col(col: str, mapping: dict[str, str]) -> str:
    if col in mapping:
        return mapping[col]
    for prefix in ("videos/", "stats/"):
        if col.startswith(prefix):
            rest = col[len(prefix):]
            for old, new in mapping.items():
                if rest == old or rest.startswith(old + "/"):
                    return prefix + new + rest[len(old):]
    return col


def _rename_parquet(path: Path, mapping: dict[str, str], dry_run: bool) -> int:
    schema_names = pq.read_schema(path).names
    new_names = [_rename_col(c, mapping) for c in schema_names]
    changed = sum(1 for a, b in zip(schema_names, new_names) if a != b)
    if changed and not dry_run:
        df = pd.read_parquet(path)
        df.columns = [_rename_col(c, mapping) for c in df.columns]
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--map", default=None, help="JSON object old->new (default: RoboSyn old->new names)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    mapping = json.loads(args.map) if args.map else DEFAULT_MAP
    dry = args.dry_run

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    assert str(info["codebase_version"]).startswith("v3"), f"expected v3.x dataset, got {info['codebase_version']}"
    present = [k for k in mapping if k in info["features"]]
    if not present:
        print("nothing to rename (no mapped keys present):", list(info["features"]))
        return
    print(f"{'[dry-run] ' if dry else ''}renaming in {root}: { {k: mapping[k] for k in present} }")

    # 1) info.json features (preserve order)
    info["features"] = {mapping.get(k, k): v for k, v in info["features"].items()}
    # 2) stats.json
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    if stats is not None:
        stats = {mapping.get(k, k): v for k, v in stats.items()}
    # 3) parquet files (episodes metadata + data)
    n_files = 0
    for pattern in ("meta/episodes/**/*.parquet", "data/**/*.parquet"):
        for p in sorted(root.glob(pattern)):
            n_files += 1 if _rename_parquet(p, mapping, dry) else 0
    # 4) video directories
    moved = []
    for old, new in mapping.items():
        src = root / "videos" / old
        if src.is_dir():
            dst = root / "videos" / new
            moved.append((src.name, new))
            if not dry:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
    if not dry:
        info_path.write_text(json.dumps(info, indent=4))
        if stats is not None:
            stats_path.write_text(json.dumps(stats, indent=4))
    print(f"parquet files changed: {n_files}; video dirs moved: {moved}")
    print("features now:", list(info["features"]))


if __name__ == "__main__":
    main()
