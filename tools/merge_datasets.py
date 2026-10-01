#!/usr/bin/env python
"""Merge several LeRobot v3.0 datasets of the same task into one (for run2-style
"more data" training). Handles the two incompatibilities seen between official
and locally collected RoboSynChallenge data:
  * extra per-object pose / contact features only present in some sources
    (dropped via a temporary remove_feature copy),
  * differing feature metadata (`names` casing, `fps` key) -> aligned to the
    first source before merging.

Usage:
  python tools/merge_datasets.py <out_root> <src_root> [<src_root> ...]
Example:
  python tools/merge_datasets.py lerobot_dataset/items_handover_merge/cobotmagic_Sim_items_handover_1500 \
      lerobot_dataset/RoboSynChallenge/cobotmagic_Sim_items_handover lerobot_dataset/items_handover/cobotmagic_Sim_items_handover_000
Runs inside policy/smolvla/.venv (needs lerobot-edit-dataset).
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

KEEP_PREFIXES = ("observation.", "action")
BOOKKEEPING = {"timestamp", "frame_index", "episode_index", "index", "task_index"}

def load_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())

def _data_files(root: Path):
    return sorted(root.glob("data/**/*.parquet"))


def _arrow_types(root: Path) -> dict:
    import pyarrow.parquet as pq
    files = _data_files(root)
    if not files:
        return {}
    sch = pq.read_schema(files[0])
    return {name: sch.field(name).type for name in sch.names}


def _normalize_dtypes(root: Path, ref: dict, label: str) -> None:
    """Rewrite `root`'s data parquet columns so their element dtype matches `ref`.

    EmbodiChain writes float32 list columns; the official datasets that went
    through the v2.1 -> v3.0 conversion carry float64 ones. `lerobot-edit-dataset
    merge` concatenates the frames with pandas and pyarrow refuses to mix
    `list<double>` with `fixed_size_list<float>`, so align them before merging.
    """
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    mine = _arrow_types(root)
    mismatched = [c for c, t in mine.items() if c in ref and str(ref[c]) != str(t)]
    if not mismatched:
        return
    print(f"[{label}] normalizing dtypes for {mismatched}")
    for f in _data_files(root):
        table = pq.read_table(f)
        cols, names = [], []
        for name in table.schema.names:
            col = table.column(name)
            target = ref.get(name)
            if target is not None and str(target) != str(col.type):
                col = col.cast(target)
            cols.append(col)
            names.append(name)
        out = pa.Table.from_arrays([c.combine_chunks() if hasattr(c, "combine_chunks") else c for c in cols], names=names)
        tmp = f.with_suffix(".parquet.tmp")
        pq.write_table(out, tmp)
        tmp.replace(f)


def _edit_dataset_cmd() -> str:
    """Absolute path of `lerobot-edit-dataset` next to the running interpreter.

    Resolving it via PATH only works when the venv is activated; tools/exp_queue.sh
    calls this script with the venv's python directly, so PATH lookup failed and
    every queued merge died with FileNotFoundError (2026-09-22, ~50 idle GPU-hours).
    """
    import shutil
    cand = Path(sys.executable).parent / "lerobot-edit-dataset"
    if cand.exists():
        return str(cand)
    found = shutil.which("lerobot-edit-dataset")
    if found:
        return found
    raise FileNotFoundError("lerobot-edit-dataset not found next to %s nor on PATH" % sys.executable)


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_root = Path(sys.argv[1]).resolve()
    srcs = [Path(p).resolve() for p in sys.argv[2:]]
    repo_root = Path(__file__).resolve().parents[1]
    # Unique per output dataset: two merges running concurrently must not share
    # (and wipe) the same scratch directory.
    tmp_root = repo_root / "lerobot_dataset" / f".merge_tmp_{out_root.name}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_ENDPOINT="https://huggingface.co", HF_HOME=str(repo_root / ".cache" / "huggingface"))

    prepared = []
    for i, src in enumerate(srcs):
        info = load_info(src)
        assert str(info["codebase_version"]).startswith("v3"), f"{src} is {info['codebase_version']}, convert to v3.0 first"
        extra = [k for k, f in info["features"].items()
                 if not k.startswith(KEEP_PREFIXES) and k not in BOOKKEEPING and f["dtype"] not in ("image", "video")]
        work = src
        if extra:
            work = tmp_root / f"src{i}_{src.name}"
            if work.exists():
                shutil.rmtree(work)
            print(f"[{src.name}] removing extra features {extra} into temporary copy")
            subprocess.run([_edit_dataset_cmd(), "--repo_id", f"local/{src.name}", "--root", str(src),
                            "--new_repo_id", f"local/{work.name}", "--new_root", str(work),
                            "--operation.type", "remove_feature", "--operation.feature_names", json.dumps(extra)],
                           check=True, env=env)
        prepared.append(work)

    # align feature metadata (names / fps) with the first source; never mutate
    # an original source in place -> work on a temporary copy when needed.
    ref = load_info(prepared[0])["features"]
    for i in range(1, len(prepared)):
        work = prepared[i]
        info = load_info(work)
        changed = [k for k, f in info["features"].items() if k in ref and f != ref[k]]
        if not changed:
            continue
        if work in srcs:
            copy = tmp_root / f"meta{i}_{work.name}"
            if copy.exists():
                shutil.rmtree(copy)
            print(f"[{work.name}] copying to align metadata for {changed}")
            shutil.copytree(work, copy)
            work = copy
            prepared[i] = work
        for k in changed:
            info["features"][k] = ref[k]
        (work / "meta" / "info.json").write_text(json.dumps(info, indent=4))
        print(f"[{work.name}] aligned metadata: {changed}")
    # align on-disk arrow dtypes with the first source (see _normalize_dtypes)
    ref_types = _arrow_types(prepared[0])
    for i in range(1, len(prepared)):
        work = prepared[i]
        if not ref_types:
            break
        mine = _arrow_types(work)
        if any(c in ref_types and str(ref_types[c]) != str(t) for c, t in mine.items()):
            if work in srcs:
                copy = tmp_root / f"dtype{i}_{work.name}"
                if copy.exists():
                    shutil.rmtree(copy)
                print(f"[{work.name}] copying to normalize dtypes")
                shutil.copytree(work, copy)
                work = copy
                prepared[i] = work
            _normalize_dtypes(work, ref_types, work.name)

    # ensure identical feature dicts
    feats = [load_info(w)["features"] for w in prepared]
    for w, f in zip(prepared, feats):
        if f != feats[0]:
            diff = {k for k in set(f) | set(feats[0]) if f.get(k) != feats[0].get(k)}
            sys.exit(f"feature mismatch between {prepared[0].name} and {w.name}: {sorted(diff)}")

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    repo_ids = [f"local/{w.name}" for w in prepared]
    print("merging", [str(w) for w in prepared], "->", out_root)
    subprocess.run([_edit_dataset_cmd(), "--new_repo_id", f"local/{out_root.name}", "--new_root", str(out_root),
                    "--operation.type", "merge", "--operation.repo_ids", json.dumps(repo_ids),
                    "--operation.roots", json.dumps([str(w) for w in prepared])], check=True, env=env)
    info = load_info(out_root)
    print(f"DONE {out_root}: {info['total_episodes']} episodes, {info['total_frames']} frames")
    shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
