#!/usr/bin/env python
"""Validate a LeRobot v3.0 dataset (load + decode a few frames) and optionally
remove the `<root>_old` backup left by the v2.1->v3.0 converter.

Usage: python tools/validate_dataset.py <dataset_root> [--cleanup-old] [--samples N]
"""
import argparse, shutil, sys, time
from pathlib import Path
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--cleanup-old", action="store_true")
ap.add_argument("--samples", type=int, default=5)
a = ap.parse_args()
root = Path(a.root).resolve()
t = time.time()
ds = LeRobotDataset(repo_id=f"local/{root.name}", root=str(root))
n = ds.num_frames
idxs = [0, n // 2, n - 1] + [int(x) for x in torch.randint(0, n, (max(0, a.samples - 3),))]
required = {"observation.state", "action"}
cams = set(ds.meta.camera_keys)
assert required <= set(ds.features), f"missing {required - set(ds.features)}"
assert len(cams) == 3, f"expected 3 cameras, got {cams}"
for i in idxs:
    it = ds[i]
    for k in cams:
        img = it[k]
        assert tuple(img.shape) == (3, 480, 640), (k, img.shape)
        assert img.dtype == torch.float32 and 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert it["observation.state"].shape[-1] == 14 and it["action"].shape[-1] == 14
print(f"OK {root.name}: {ds.num_episodes} eps, {n} frames, fps {ds.fps}, cams {sorted(cams)}, "
      f"task='{ds[0]['task']}', checked {len(idxs)} frames in {time.time()-t:.1f}s")
old = root.parent / f"{root.name}_old"
if a.cleanup_old and old.is_dir():
    shutil.rmtree(old)
    print(f"removed backup {old}")
