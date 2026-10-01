#!/usr/bin/env python
"""Paired comparison of two evaluation runs on the same scenes.

Episodes are matched by their scene seed (evaluation_metrics.json -> episodes[].seed),
so the comparison removes scene-to-scene variance. With the policy seeded as well
(SMOLVLA_SEED), the only remaining difference between the arms is the config
under test. Significance uses the exact two-sided McNemar test on the discordant
pairs, which is what mydocs/explain/05 (5.3) prescribes.

Usage:
    python tools/paired_compare.py <run_dir_A> <run_dir_B> [--label-a A] [--label-b B]
    python tools/paired_compare.py --task items_handover --a arm1_queue_k1_seeded --b arm3_fresh_k4_seeded
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path


def resolve(task: str | None, name_or_dir: str) -> Path:
    p = Path(name_or_dir)
    if (p / "evaluation_metrics.json").exists():
        return p
    if task is None:
        raise SystemExit(f"not a run dir and no --task given: {name_or_dir}")
    cands = sorted(glob.glob(f"eval_result/{task}/smolvla/*/*/{name_or_dir}/*/evaluation_metrics.json"))
    if not cands:
        raise SystemExit(f"no evaluation found for {task}/{name_or_dir}")
    return Path(cands[-1]).parent


def load(d: Path) -> dict[int, dict]:
    m = json.loads((d / "evaluation_metrics.json").read_text())
    return {int(e["seed"]): e for e in m["episodes"]}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: binomial test of b vs Bin(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--task")
    ap.add_argument("--a", dest="a_name")
    ap.add_argument("--b", dest="b_name")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()
    a_ref, b_ref = args.a_name or args.a, args.b_name or args.b
    if not (a_ref and b_ref):
        ap.error("need two runs")
    da, db = resolve(args.task, a_ref), resolve(args.task, b_ref)
    ea, eb = load(da), load(db)
    common = sorted(set(ea) & set(eb))
    la = args.label_a or a_ref
    lb = args.label_b or b_ref
    if not common:
        raise SystemExit("no shared scene seeds -> not a paired comparison")
    sa = sum(ea[s]["success"] for s in common)
    sb = sum(eb[s]["success"] for s in common)
    b = sum(1 for s in common if ea[s]["success"] and not eb[s]["success"])
    c = sum(1 for s in common if eb[s]["success"] and not ea[s]["success"])
    n = len(common)
    p = mcnemar_exact(b, c)
    verdict = ("B 更好" if c > b else "A 更好") if p < 0.05 else "无法区分"
    print(f"配对场景 {n} 个（A 共 {len(ea)}、B 共 {len(eb)}）")
    print(f"  A {la}: {sa}/{n} = {100*sa/n:.0f}%")
    print(f"  B {lb}: {sb}/{n} = {100*sb/n:.0f}%")
    print(f"  不一致对：A成B败 {b}，A败B成 {c}；McNemar 精确双侧 p = {p:.4f} → {verdict}")


if __name__ == "__main__":
    main()
