#!/usr/bin/env python
"""Regenerate the results table from eval_result/ so it can never go stale.

Reads every eval_result/<task>/smolvla/<setting>/*/<model>/<ts>/evaluation_metrics.json,
picks the best run per task by the competition's own scoring formula, and prints a
Markdown report.

Usage:
    python tools/summarize_results.py                 # print to stdout
    python tools/summarize_results.py -o mydocs/04_成绩汇总.md
    python tools/summarize_results.py --min-episodes 50
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

# max_episode_steps per task, from configs/<task>/random/gym_config.json
HORIZON = {
    "click_bell": 361, "table_rearrangement": 361, "water_pouring": 500,
    "handle_basket": 500, "items_handover": 350, "drawer_open_place": 900,
    "mixer_operating": 500, "item_assembly": 500, "manipulate_pipette": 1000,
    "sample_loading": 500,
}
# official published SmolVLA numbers (100 episodes), for reference only
OFFICIAL = {
    "table_rearrangement": 82, "click_bell": 35, "water_pouring": 65, "handle_basket": 40,
    "items_handover": 30, "drawer_open_place": 62, "mixer_operating": 58, "item_assembly": 24,
    "manipulate_pipette": 36, "sample_loading": 6,
}


# Evaluations started before this moment used `--smolvla_rescale_gripper true`, which
# jams the gripper shut on every task whose dataset stores 0-1 gripper actions. They are
# not comparable with anything later; keep them visible but out of the "best" selection.
LEGACY_GRIPPER_BEFORE = "2026-09-16_06-12"

# Early runs all wrote their results under the checkpoint step (e.g. "050000"), so run1 and
# run2 are indistinguishable from the path alone. Disambiguate by timestamp. Everything
# evaluated from 2026-09-18 on carries an explicit SMOLVLA_MODEL_NAME and needs no entry.
ALIAS = {
    ("click_bell", "2026-09-16_01-03-14"): "run1@10k", ("click_bell", "2026-09-16_02-47-15"): "run1@20k",
    ("click_bell", "2026-09-16_04-27-52"): "run1@30k", ("click_bell", "2026-09-16_06-10-43"): "run1@40k",
    ("click_bell", "2026-09-16_07-54-53"): "run1@50k", ("click_bell", "2026-09-16_09-41-29"): "run2@10k",
    ("click_bell", "2026-09-16_12-03-22"): "run2@20k", ("click_bell", "2026-09-16_13-44-32"): "run2@30k",
    ("click_bell", "2026-09-16_15-22-27"): "run2@40k", ("click_bell", "2026-09-16_17-01-12"): "run2@50k",
    ("click_bell", "2026-09-15_23-07-56"): "官方ckpt", ("click_bell", "2026-09-16_07-13-27"): "官方ckpt",
    ("click_bell", "2026-09-16_08-58-26"): "官方ckpt",
    ("table_rearrangement", "2026-09-16_02-25-48"): "run1@10k", ("table_rearrangement", "2026-09-16_05-03-05"): "run1@20k",
    ("table_rearrangement", "2026-09-16_06-46-07"): "run1@20k", ("table_rearrangement", "2026-09-16_07-30-53"): "run1@30k",
    ("table_rearrangement", "2026-09-16_09-53-44"): "run1@40k", ("table_rearrangement", "2026-09-16_12-21-08"): "run1@50k",
    ("table_rearrangement", "2026-09-16_06-11-51"): "官方ckpt",
    ("mixer_operating", "2026-09-17_01-54-45"): "run1@50k", ("mixer_operating", "2026-09-17_02-06-58"): "官方ckpt",
    ("sample_loading", "2026-09-16_04-37-50"): "官方ckpt", ("sample_loading", "2026-09-18_12-05-18"): "run1@50k",
    ("items_handover", "2026-09-17_00-55-21"): "run1@50k",
    ("drawer_open_place", "2026-09-17_10-28-11"): "run1@50k",
    ("handle_basket", "2026-09-17_15-01-48"): "run1@50k",
    ("item_assembly", "2026-09-17_18-57-01"): "run1@50k",
    ("manipulate_pipette", "2026-09-18_03-25-28"): "run1@50k",
    ("water_pouring", "2026-09-18_05-10-23"): "run1@50k",
}


def score(sr_pct: float, ratio: float) -> float:
    """Competition score, inference-efficiency term counted as 0 (SmolVLA is slower than the ACT baseline)."""
    return 0.75 * sr_pct + 0.20 * (1.0 - ratio) * 100.0


ARM_STEPS = {"arm1_queue_k1_seeded": 20, "arm2_fresh_k1_seeded": 20,
             "arm3_fresh_k4_seeded": 20, "arm4_queue_k1_seeded_s10": 10}


def steps_of(model: str, cfg: dict) -> str:
    """smolvla_steps used for a run, decoded from our model_name conventions."""
    if model in ARM_STEPS:
        return str(ARM_STEPS[model])
    for pat in (r"steps(\d+)", r"_s(\d+)_seed", r"_s(\d+)$"):
        m = re.search(pat, model)
        if m:
            return m.group(1)
    return str(cfg.get("smolvla_steps", 10))


def knobs_of(model: str) -> tuple[str, int, bool]:
    """(chunk_mode, action_samples, policy_seeded) decoded from the run name."""
    mode = "fresh" if ("fresh" in model or "_Bk" in model) else "queue"
    m = re.search(r"_Bk(\d+)_|_k(\d+)", model)
    k = int(next(g for g in m.groups() if g)) if m else 1
    seeded = ("seed" in model) and ("unseeded" not in model)
    return mode, k, seeded


def load() -> list[dict]:
    rows = []
    for p in glob.glob("eval_result/*/smolvla/*/*/*/*/evaluation_metrics.json"):
        try:
            m = json.load(open(p))
        except Exception:
            continue
        s, cfg = m.get("summary", {}), m.get("config", {})
        if not s:
            continue
        parts = Path(p).parts
        task, ts = parts[1], parts[6]
        label = ALIAS.get((task, ts), parts[5])
        policy_steps, knobs = steps_of(parts[5], cfg), knobs_of(parts[5])
        # tools/final_eval.sh stores the adapter's "[smolvla] inference config" line
        side = Path(p).parent / "inference_config.txt"
        if side.exists():
            mm = re.search(r"steps=(\d+) chunk_mode=(\w+) action_samples=(\d+) seed=(\S+)", side.read_text())
            if mm:
                policy_steps = mm.group(1)
                knobs = (mm.group(2), int(mm.group(3)), mm.group(4) != "None")
        rows.append({
            "task": task, "setting": parts[3], "model": label, "ts": ts,
            "legacy": ts < LEGACY_GRIPPER_BEFORE,
            "eps": s["episode_count"], "sr": s["success_rate"] * 100,
            "steps": s["average_action_steps"], "ratio": s.get("average_action_steps_ratio", 0.0),
            "calls": s.get("average_inference_calls_per_episode", 0.0),
            "infer_ms": s.get("average_inference_time_seconds", 0.0) * 1000,
            "policy_steps": policy_steps, "path": p,
            "knobs": knobs,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output")
    ap.add_argument("--min-episodes", type=int, default=50)
    ap.add_argument("--setting", default="random")
    a = ap.parse_args()

    rows = [r for r in load() if r["setting"] == a.setting]
    rows.sort(key=lambda r: (r["task"], -r["sr"]))

    best, total, counted = {}, 0.0, 0
    for r in rows:
        if r["eps"] < a.min_episodes or r["legacy"]:
            continue
        r["score"] = score(r["sr"], r["ratio"])
        cur = best.get(r["task"])
        if cur is None or r["score"] > cur["score"]:
            best[r["task"]] = r

    out = ["# 成绩总表（由 tools/summarize_results.py 自动生成）", ""]
    out.append(f"口径：仿真 `{a.setting}` 设置、`smolvla_rescale_gripper=auto`、固定随机种子、"
               f"至少 {a.min_episodes} 回合；估分 = 0.75×成功率 + 0.20×(1−步数比例)×100（推理效率项按 0 计）。")
    out.append("")
    out.append("## 每个任务当前最好的一次评测")
    out.append("")
    out.append("| 任务 | H | checkpoint | steps | 推理配置 | 回合 | 成功率 | 平均步数 | 步数比 | 估分 | 官方公布 SmolVLA |")
    out.append("|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for t in HORIZON:
        r = best.get(t)
        if not r:
            cand = [x for x in rows if x["task"] == t and not x["legacy"]]
            if not cand:
                out.append(f"| {t} | {HORIZON[t]} | — | — | — | — | 未评测 | — | — | — | {OFFICIAL.get(t,'—')}% |")
                continue
            r = max(cand, key=lambda x: x["sr"])
            note = f"（仅 {r['eps']} 回合）"
        else:
            note = ""
            total += r["sr"]
            counted += 1
        mode, k, seeded = r["knobs"]
        cfg_txt = f"{mode}, K={k}" + ("，定种子" if seeded else "")
        out.append(f"| {t} | {HORIZON[t]} | {r['model']}{note} | {r['policy_steps']} | {cfg_txt} | {r['eps']} | "
                   f"{r['sr']:.0f}% | {r['steps']:.1f} | {r['ratio']:.3f} | {score(r['sr'], r['ratio']):.1f} | "
                   f"{OFFICIAL.get(t,'—')}% |")
    mean_all = sum((best[t]["sr"] if t in best else max((x["sr"] for x in rows if x["task"] == t and not x["legacy"]), default=0.0))
                   for t in HORIZON) / len(HORIZON)
    out.append("")
    out.append(f"**10 任务平均成功率：{mean_all:.1f}%**"
               f"（其中 {counted}/{len(HORIZON)} 个任务有 ≥{a.min_episodes} 回合的结果；"
               f"官方公布的 SmolVLA 10 任务均值为 43.8%，种子不同仅供参考）")
    out.append("")
    out.append(f"⚠️ 时间戳早于 {LEGACY_GRIPPER_BEFORE} 的评测用的是 `--smolvla_rescale_gripper true`，"
               "对夹爪动作为 0-1 归一化值的任务会把夹爪锁死，不可与其余结果比较，已排除在选优之外。")
    out.append("")
    # the checkpoint each task submits: tools/submission_checkpoints.json ({task: "run/checkpoints/<step>"})
    submitted = {}
    sub = Path("tools/submission_checkpoints.json")
    if sub.exists():
        submitted = {task: ckpt.split("/")[0] for task, ckpt in json.loads(sub.read_text()).items()}
    finals = {}
    for r in rows:
        if not r["model"].startswith("final") or r["legacy"]:
            continue
        want = submitted.get(r["task"])
        if want and not r["model"].endswith("_" + want):
            continue  # final eval of a candidate that is not the submitted checkpoint
        if r["task"] not in finals or r["ts"] > finals[r["task"]]["ts"]:
            finals[r["task"]] = r
    out.append("## 提交配置确认评测（tools/final_eval.sh：主办方原样调用、按任务表、不定策略种子；checkpoint 以 SUBMISSION.md 为准）")
    out.append("")
    out.append("上面那张表取的是每个任务\"最好的一次\"，系统性偏高（同一配置单次 50 回合可差 ±8 个点）；"
               "这张表才是对外可报的数字。")
    out.append("")
    out.append("| 任务 | checkpoint | steps | 推理配置 | 回合 | 成功率 | 平均步数 | 估分 | 时间戳 |")
    out.append("|---|---|---:|---|---:|---:|---:|---:|---|")
    for t in HORIZON:
        r = finals.get(t)
        if not r:
            out.append(f"| {t} | 未做 | — | — | — | — | — | — | — |")
            continue
        mode, k, seeded = r["knobs"]
        out.append(f"| {t} | {r['model']} | {r['policy_steps']} | {mode}, K={k}{'，定种子' if seeded else ''} | "
                   f"{r['eps']} | {r['sr']:.0f}% | {r['steps']:.1f} | {score(r['sr'], r['ratio']):.1f} | {r['ts']} |")
    if finals:
        m = sum(r["sr"] for r in finals.values()) / len(finals)
        out.append("")
        out.append(f"**已完成 {len(finals)}/{len(HORIZON)} 个任务，这些任务的平均成功率 {m:.1f}%**")
    out.append("")
    out.append("## 全部评测记录")
    out.append("")
    out.append("| 任务 | checkpoint | steps | 回合 | 成功率 | 平均步数 | 推理/回合 | 时间戳 |")
    out.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        flag = " ⚠️旧夹爪口径" if r["legacy"] else ""
        out.append(f"| {r['task']} | {r['model']}{flag} | {r['policy_steps']} | {r['eps']} | {r['sr']:.0f}% | "
                   f"{r['steps']:.1f} | {r['calls']:.1f} | {r['ts']} |")
    text = "\n".join(out) + "\n"
    if a.output:
        Path(a.output).write_text(text)
        print(f"wrote {a.output} ({len(rows)} evaluations, {len(best)} tasks with >={a.min_episodes} episodes)")
    else:
        print(text)


if __name__ == "__main__":
    main()
