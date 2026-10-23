#!/usr/bin/env python3
"""head_max_len 预算提升实验：生产格式下 instructions 预算 256 → 1024 → 3200 三档对比。

机制：head_max_len/max_len 是 laya 推理时参数（predict_batch 按 调用覆盖，非烧进权重）；
mmBERT 底座支持 8192 位置。风险：决策头在 head≤256 分布上训练，超预算属外推，效果实测。

每档测: ① is_ad instructions 实际保留 token（白盒 build_sequence）
        ② 请求示例样例 state 的 is_ad 判定
        ③ gold 100 ad + 100 clean 探针的 is_ad 准确率 / P(ad) 均值 / ad_kind 命中 / 耗时

用法:
  python scripts/test_head_budget.py
"""
import json
import os
import random
import time
from collections import Counter

os.environ.setdefault("USE_TF", "0")

MODEL = "models/laya-multilingual"
SYSPROMPT = "data/raw/dataset/系统提示词.txt"
REQUEST = "data/raw/dataset/请求示例.json"
GOLD = "data/raw/dataset/converted/gold.binary.jsonl"
OUT = "outputs/head_budget_sweep.jsonl"

CONFIGS = [
    ("A-默认256", dict(max_len=1024, head_max_len=256), 32),
    ("B-head1024", dict(max_len=2048, head_max_len=1024), 32),
    ("C-全量3200", dict(max_len=4096, head_max_len=3200), 8),  # 4096 序列，行数调小控显存
]


def load_probe():
    pool = {"ad": [], "normal": []}
    for line in open(GOLD, encoding="utf-8"):
        row = json.loads(line)
        pool[row["label"]].append(row)
    random.Random(42).shuffle(pool["ad"])
    random.Random(42).shuffle(pool["normal"])
    probe = pool["ad"][:100] + pool["normal"][:100]
    random.Random(43).shuffle(probe)
    return probe


def probe_metrics(results):
    tp = sum(1 for r in results if r["true"] == "ad" and r["pred_ad"] == "ad")
    tn = sum(1 for r in results if r["true"] == "normal" and r["pred_ad"] != "ad")
    ads = [r for r in results if r["true"] == "ad"]
    cls = [r for r in results if r["true"] == "normal"]
    return {
        "acc": round((tp + tn) / len(results), 4),
        "p_ad_ad": round(sum(r["p_ad"] for r in ads) / len(ads), 4),
        "p_ad_clean": round(sum(r["p_ad"] for r in cls) / len(cls), 4),
        "kind_hit": sum(1 for r in ads if r["pred_kind"] == r["ad_kind_true"]),
        "kind_dist": dict(Counter(r["pred_kind"] for r in ads).most_common(5)),
    }


def main():
    from transformers import AutoTokenizer
    from laya.common import build_sequence

    sysprompt = open(SYSPROMPT, encoding="utf-8").read()
    req = json.load(open(REQUEST, encoding="utf-8"))
    questions = req["questions"]
    sample_state = req["state"]
    questions["is_ad"]["instructions"] = sysprompt
    tok = AutoTokenizer.from_pretrained(f"{MODEL}/tokenizer")
    qi = {"t": "choice", "ins": sysprompt, "crit": questions["is_ad"]["criteria"]}
    probe = load_probe()

    import laya
    agent = laya.load(MODEL)

    all_rows = []
    print(f"{'配置':<12}{'ins保留tok':>11}{'样例is_ad':>12}{'探针acc':>9}{'P(ad|ad)':>10}"
          f"{f'P(ad|clean)':>12}{'kind命中':>9}{'耗时s':>7}")
    for name, budget, chunk in CONFIGS:
        # ① 白盒：is_ad instructions 实际保留量
        ids, markers, stats, tstats = build_sequence(
            tok, sample_state, qi, return_stats=True, return_truncation_stats=True, **budget)
        head_len = len(ids) - tstats["state_tokens_used"] - 1
        ins_kept = head_len - len(markers) - 6  # 粗扣 [CLS]/type前缀/SEP 槽位
        # ② 样例判定
        r = agent.predict(sample_state, questions, **budget)
        is_ad = r["answers"]["is_ad"]
        sample_verdict = f"{is_ad['choice']}@{is_ad['probabilities']['ad']:.2f}"
        # ③ 探针
        results = []
        t0 = time.perf_counter()
        for i in range(0, len(probe), chunk):
            rows = probe[i:i + chunk]
            outs = agent.predict_batch(
                [{"message": {"text": x["text"]}} for x in rows], questions,
                batch_size=chunk, **budget)
            for x, o in zip(rows, outs):
                a = o["answers"]
                results.append({"true": x["label"], "ad_kind_true": x.get("ad_kind"),
                                "pred_ad": a["is_ad"]["choice"],
                                "p_ad": a["is_ad"]["probabilities"]["ad"],
                                "pred_kind": a["ad_kind"]["choice"]})
        dt = time.perf_counter() - t0
        m = probe_metrics(results)
        print(f"{name:<12}{ins_kept:>11}{sample_verdict:>16}{m['acc']:>9.1%}"
              f"{m['p_ad_ad']:>10.3f}{m['p_ad_clean']:>12.3f}"
              f"{m['kind_hit']:>7}/100{dt:>7.1f}")
        for x in results:
            all_rows.append({"config": name, **budget, **x})

    with open(OUT, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n[OK] 逐条明细: {OUT}")
    print("注: ins保留tok 为粗估(扣标记槽位); kind命中=预测ad_kind与数据集ad_kind一致的真广告条数")


if __name__ == "__main__":
    main()
