#!/usr/bin/env python3
"""用生产请求格式（系统提示词注入 + 富 state + 4 连问）测试现有 laya-multilingual。

测三件事:
 1. 截断计量：生产格式下 4 个问题各自的 instructions/options/state token 预算实况
    （系统提示词 ~13.7KB 远超 instructions 预算，ad_kind 8 选项也要吃预算）
 2. 请求示例.json 的样例 state 端到端判定（4 问全输出）
 3. 真实数据探针：gold 子集 100 ad + 100 clean，以生产 4 问格式零样本判定，
    对照数据集标注出准确率（注意：数据集只有正文留底，探针 state 为最小结构，
    不含生产完整 sender 资料，测的是"格式+指令"效果而非完整生产输入）

用法:
  python scripts/test_production_format.py
输出:
  stdout 报告；outputs/production_format_probe.jsonl 逐条明细
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
OUT = "outputs/production_format_probe.jsonl"
CHUNK = 32      # states/forward；4 问 × 32 = 128 行 × 1024 token，控显存
PROBE_PER_CLASS = 100


def main():
    from transformers import AutoTokenizer

    sysprompt = open(SYSPROMPT, encoding="utf-8").read()
    req = json.load(open(REQUEST, encoding="utf-8"))
    questions = req["questions"]
    sample_state = req["state"]
    questions["is_ad"]["instructions"] = sysprompt  # {{系统提示词}} 注入
    cfg = json.load(open(f"{MODEL}/rl_agent_config.json", encoding="utf-8"))
    max_len, head_max = cfg["max_len"], cfg["head_max_len"]
    tok = AutoTokenizer.from_pretrained(f"{MODEL}/tokenizer")

    print("=" * 76)
    print(f"[1] 生产格式截断计量  (max_len={max_len}, head_max_len={head_max})")
    print("=" * 76)
    from laya.common import build_sequence, serialize_state

    state_json = serialize_state(sample_state)
    state_ids = tok(state_json, add_special_tokens=False)["input_ids"]
    print(f"系统提示词: {len(sysprompt)} 字符 ≈ {len(tok(sysprompt, add_special_tokens=False)['input_ids'])} token")
    print(f"样例 state JSON: {len(state_json)} 字符 ≈ {len(state_ids)} token\n")
    print(f"{'question':<10}{'ins原始tok':>12}{'options数':>10}{'head终长':>10}{'state用':>9}{'state丢':>9}")
    for qid, q in questions.items():
        qi = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
        n_opt = len(q["criteria"]) if isinstance(q["criteria"], (dict, list)) else 0
        ins_toks = len(tok(q["instructions"], add_special_tokens=False)["input_ids"])
        ids, markers, stats, tstats = build_sequence(
            tok, sample_state, qi, max_len=max_len, head_max_len=head_max,
            return_stats=True, return_truncation_stats=True)
        head_len = len(ids) - tstats["state_tokens_used"] - 1
        print(f"{qid:<10}{ins_toks:>12}{n_opt:>10}{head_len:>10}"
              f"{tstats['state_tokens_used']:>9}{tstats['state_tokens_dropped']:>9}"
              f"   (options 独立span {stats['options_distinct']}/{stats['options']})")

    print()
    print("=" * 76)
    print("[2] 请求示例样例 state 端到端判定")
    print("=" * 76)
    import laya
    agent = laya.load(MODEL)
    t0 = time.perf_counter()
    r = agent.predict(sample_state, questions)
    dt = (time.perf_counter() - t0) * 1000
    for qid, ans in r["answers"].items():
        keep = {k: v for k, v in ans.items() if k in ("choice", "noul", "score", "probabilities", "confidence")}
        print(f"  {qid:<10} {json.dumps(keep, ensure_ascii=False)}")
    print(f"  首次前向 {dt:.0f}ms（含预热）")

    print()
    print("=" * 76)
    print(f"[3] 真实数据探针: gold 子集各 {PROBE_PER_CLASS} 条 ad/clean，生产 4 问格式零样本")
    print("=" * 76)
    pool = {"ad": [], "normal": []}
    for line in open(GOLD, encoding="utf-8"):
        row = json.loads(line)
        pool[row["label"]].append(row)
    random.Random(42).shuffle(pool["ad"])
    random.Random(42).shuffle(pool["normal"])
    probe = pool["ad"][:PROBE_PER_CLASS] + pool["normal"][:PROBE_PER_CLASS]
    random.Random(43).shuffle(probe)
    print(f"探针样本: {len(probe)} (ad={len(pool['ad'][:PROBE_PER_CLASS])}, normal={len(pool['normal'][:PROBE_PER_CLASS])})")

    t0 = time.perf_counter()
    results = []
    for i in range(0, len(probe), CHUNK):
        chunk = probe[i:i + CHUNK]
        outs = agent.predict_batch(
            [{"message": {"text": row["text"]}} for row in chunk], questions, batch_size=CHUNK)
        for row, o in zip(chunk, outs):
            a = o["answers"]
            results.append({
                "id": row["id"], "true": row["label"], "ad_kind_true": row.get("ad_kind"),
                "pred_ad": a["is_ad"]["choice"],
                "p_ad": a["is_ad"]["probabilities"]["ad"],
                "pred_kind": a["ad_kind"]["choice"],
                "pred_scope": a["ad_scope"]["choice"],
                "pred_sev": a["severity"]["score"],
            })
    print(f"判定耗时: {time.perf_counter() - t0:.1f}s")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tp = sum(1 for r in results if r["true"] == "ad" and r["pred_ad"] == "ad")
    fp = sum(1 for r in results if r["true"] == "normal" and r["pred_ad"] == "ad")
    fn = sum(1 for r in results if r["true"] == "ad" and r["pred_ad"] != "ad")
    tn = sum(1 for r in results if r["true"] == "normal" and r["pred_ad"] != "ad")
    p_ad_true = [r["p_ad"] for r in results if r["true"] == "ad"]
    p_ad_clean = [r["p_ad"] for r in results if r["true"] == "normal"]
    print(f"\nis_ad 混淆: TP={tp} FP={fp} FN={fn} TN={tn}  acc={(tp + tn) / len(results):.1%}")
    print(f"  P(ad|真广告)   均值={sum(p_ad_true) / len(p_ad_true):.3f}")
    print(f"  P(ad|真正常)   均值={sum(p_ad_clean) / len(p_ad_clean):.3f}")
    ads = [r for r in results if r["true"] == "ad"]
    kind_hit = sum(1 for r in ads if r["pred_kind"] == r["ad_kind_true"])
    print(f"ad_kind 命中(真广告子集): {kind_hit}/{len(ads)}")
    print(f"  真实kind分布: {dict(Counter(r['ad_kind_true'] for r in ads).most_common())}")
    print(f"  预测kind分布: {dict(Counter(r['pred_kind'] for r in ads).most_common())}")
    print(f"severity 均值: 真广告={sum(r['pred_sev'] for r in ads) / len(ads):.2f}  "
          f"真正常={sum(r['pred_sev'] for r in results if r['true'] == 'normal') / (len(results) - len(ads)):.2f}")
    print(f"[OK] 逐条明细: {OUT}")


if __name__ == "__main__":
    main()
