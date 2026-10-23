#!/usr/bin/env python3
"""把 gold 里的重复/矛盾文本喂给现有（零样本）laya-multilingual 判定，出对比报告。

目的：看现有模型在这些"多 sighting / 多标注"文本上的倾向与置信度——
零样本能力接近随机的警告仍然适用，本报告是基线观察而非准确率评估。

用法:
  python scripts/laya_judge_conflicts.py
输出:
  outputs/laya_judge_conflicts.jsonl   逐条明细（含 laya P(ad)）
  stdout                               汇总交叉表
"""
import json
import os
import time
from collections import Counter

os.environ.setdefault("USE_TF", "0")

SRC = "data/raw/dataset/converted/dataset_conflicts.jsonl"
OUT = "outputs/laya_judge_conflicts.jsonl"
MODEL = "models/laya-multilingual"
CHUNK = 128  # 每次 forward 的 state 数，控制显存

QUESTIONS = {"is_ad": {"type": "noul",
                       "instructions": "判断这条 Telegram 群消息是否为广告。推销、导流、拉人、博彩、诈骗、色情引流都算广告。",
                       "criteria": {"true": "是广告", "false": "不是广告，是正常聊天内容"}}}


def bucket(p):
    if p >= 0.9: return "≥0.9"
    if p >= 0.7: return "0.7-0.9"
    if p >= 0.5: return "0.5-0.7"
    if p >= 0.3: return "0.3-0.5"
    return "<0.3"


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    print(f"载入重复/矛盾文本: {len(rows)} 条（唯一文本）")

    import laya
    t0 = time.perf_counter()
    agent = laya.load(MODEL)
    print(f"模型加载: {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    results = []
    for i in range(0, len(rows), CHUNK):
        chunk = rows[i:i + CHUNK]
        outs = agent.predict_batch([r["text"] for r in chunk], QUESTIONS, batch_size=CHUNK)
        for r, o in zip(chunk, outs):
            ans = o["answers"]["is_ad"]
            p_ad = float(ans["noul"])
            results.append({**{k: r[k] for k in ("chosen_verdict", "chosen_quality",
                                                 "conflict", "sightings")},
                            "laya_is_ad": round(p_ad, 4),
                            "laya_label": "ad" if p_ad >= 0.5 else "clean",
                            "answer_confidence": ans.get("answer_confidence")})
        if (i // CHUNK) % 10 == 0:
            print(f"  进度 {i + len(chunk)}/{len(rows)}")
    print(f"判定耗时: {time.perf_counter() - t0:.1f}s")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def crosstab(subset, title):
        n = len(subset)
        if n == 0:
            print(f"{title}: 无样本"); return
        agree = sum(1 for r in subset if r["laya_label"] == r["chosen_verdict"])
        ct = Counter((r["chosen_verdict"], r["laya_label"]) for r in subset)
        ps = [r["laya_is_ad"] for r in subset]
        print(f"\n{title} (n={n})")
        print(f"  laya 与 chosen 裁决一致率: {agree}/{n} = {agree / n:.1%}")
        print(f"  P(ad) 均值={sum(ps) / n:.3f}  中位={sorted(ps)[n // 2]:.3f}")
        print(f"  交叉表 (chosen → laya): {dict(sorted(ct.items()))}")
        print(f"  P(ad) 分桶: {dict(sorted(Counter(bucket(p) for p in ps).items()))}")

    crosstab(results, "全部重复/矛盾文本")
    crosstab([r for r in results if r["conflict"]], "矛盾子集（同文本 ad+clean 并存）")
    crosstab([r for r in results if r["chosen_verdict"] == "ad"], "chosen=ad 子集")
    crosstab([r for r in results if r["chosen_verdict"] == "clean"], "chosen=clean 子集")
    by_q = Counter((r["chosen_quality"], r["laya_label"]) for r in results)
    print(f"\n按 chosen_quality 的 laya 倾向: {dict(sorted(by_q.items()))}")
    print(f"[OK] 逐条明细: {OUT}")


if __name__ == "__main__":
    main()
