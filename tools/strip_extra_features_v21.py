#!/usr/bin/env python
"""Remove auxiliary multi-dimensional features (e.g. `bottle_pose` [4,4],
`right_ee_pose` [4,4]) from a LeRobot **v2.1** dataset before converting it to
v3.0. Those object-pose matrices are not used by any policy here, and the
v2.1->v3.0 converter crashes on them (pandas cannot compare the HF
`PandasArrayExtensionDtype` of such columns across episode files:
`AttributeError: 'PandasArrayExtensionDtype' object has no attribute 'v'`).

Usage: python tools/strip_extra_features_v21.py <dataset_root> [--dry-run]
Keeps: every image/video feature, every `observation.*`/`action*` feature and
all scalar bookkeeping columns. Drops: features with >1 dims that are not
images/videos and do not start with `observation.` or `action`.
"""
import argparse, json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
root = Path(a.root).resolve()
info_p = root / "meta" / "info.json"
info = json.loads(info_p.read_text())
assert str(info["codebase_version"]).startswith("v2"), f"expected v2.x, got {info['codebase_version']}"
drop = [k for k, f in info["features"].items()
        if f["dtype"] not in ("image", "video") and len(f["shape"]) > 1
        and not (k.startswith("observation.") or k.startswith("action"))]
if not drop:
    print("nothing to strip"); raise SystemExit(0)
print(f"{'[dry-run] ' if a.dry_run else ''}stripping {drop} from {root}")
files = sorted(root.glob("data/**/*.parquet"))
n = 0
for p in files:
    cols = pq.read_schema(p).names
    present = [c for c in drop if c in cols]
    if not present:
        continue
    n += 1
    if a.dry_run:
        continue
    df = pd.read_parquet(p).drop(columns=present)
    tmp = p.with_suffix(".parquet.tmp"); df.to_parquet(tmp, index=False); tmp.replace(p)
if not a.dry_run:
    for k in drop:
        info["features"].pop(k, None)
    info_p.write_text(json.dumps(info, indent=4))
    st = root / "meta" / "stats.json"
    if st.exists():
        s = json.loads(st.read_text()); [s.pop(k, None) for k in drop]; st.write_text(json.dumps(s, indent=4))
    es = root / "meta" / "episodes_stats.jsonl"
    if es.exists():
        lines = []
        for line in es.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for k in drop:
                rec.get("stats", {}).pop(k, None)
            lines.append(json.dumps(rec))
        es.write_text("\n".join(lines) + "\n")
print(f"parquet files touched: {n}/{len(files)}; features now: {list(info['features'])}")
