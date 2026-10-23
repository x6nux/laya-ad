#!/usr/bin/env python3
"""内部反广告数据集清洗转换 → 微调管线格式（{"text","label"}+元数据）。

修正口径（依据数据集 README + 2026-10-24 实测体检）:
 1. verdict=ad 且 ad_kind=none/缺失 → 按生产兜底口径转 clean
 2. 同文本多标注（gold 内实测 157 个矛盾文本）→ 按 quality 优先级取强标注
    （llm_reviewed > manual > rule > hash > so_initial > other），平级取最新
 3. kind="manual"（23 条，quality 值串入 kind 字段）→ 二分类下照常为 ad；
    kind 方案下标为 unspecified
 4. "类型: " 残留复验：实测 3 条均为正常用户内容（对话类型/消息类型等），非标签泄漏
 5. 空文本剔除；保留 path/quality/ad_kind 元数据供分层采样

用法:
  python scripts/convert_dataset.py --subset gold                       # 二分类（默认，生产网关口径）
  python scripts/convert_dataset.py --subset gold --label-scheme kind   # 细粒度 kind 类别
  python scripts/convert_dataset.py --subset dedup --label-scheme kind
"""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

QUALITY_RANK = {"llm_reviewed": 5, "manual": 4, "rule": 3, "hash": 2, "so_initial": 1, "other": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/dataset/ft")
    ap.add_argument("--subset", default="gold", choices=["gold", "dedup", "full"])
    ap.add_argument("--label-scheme", default="binary", choices=["binary", "kind"],
                    help="binary: normal/ad（生产网关口径）; kind: normal + 广告细类")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.src) / f"antiad_ft_{args.subset}.jsonl.gz"
    if not src.exists():
        raise SystemExit(f"[ERR] 找不到 {src}")
    out = args.out or f"data/raw/dataset/converted/{args.subset}.{args.label_scheme}.jsonl"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    stat = Counter()
    best = {}  # text -> (quality_rank, time, row)
    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            stat["读入"] += 1
            text = (r.get("text") or "").strip()
            if not text:
                stat["丢弃:空文本"] += 1
                continue
            verdict, kind = r["verdict"], r.get("ad_kind")
            if verdict == "ad":
                if not kind or kind == "none":
                    stat["修正:ad+kind_none→clean"] += 1
                    verdict = "clean"
                elif kind == "manual":
                    stat["修正:kind_manual→unspecified"] += 1
                    kind = "unspecified"
            if verdict == "clean":
                label = "normal"
            elif args.label_scheme == "binary":
                label = "ad"
            else:
                label = kind or "unspecified"
            row = {"text": text, "label": label, "path": r["path"],
                   "quality": r["quality"], "ad_kind": kind, "id": r["id"]}
            rank = QUALITY_RANK.get(r["quality"], 0)
            prev = best.get(text)
            if prev is None:
                best[text] = (rank, r["time"], row)
            else:
                stat["去重:同文本取强标注"] += 1
                if (rank, r["time"]) > (prev[0], prev[1]):
                    best[text] = (rank, r["time"], row)

    rows = sorted((b[2] for b in best.values()), key=lambda x: x["id"])
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    labels = Counter(r["label"] for r in rows)
    paths = Counter(r["path"] for r in rows)
    print(f"[OK] {src.name} -> {out}  ({len(rows)} 行)")
    print("清洗统计:", dict(sorted(stat.items())))
    print("label 分布:", dict(labels.most_common()))
    print("path  分布:", dict(paths.most_common()))


if __name__ == "__main__":
    main()
