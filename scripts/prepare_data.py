#!/usr/bin/env python3
"""把原始 jsonl({text,label,...}) 切分 train/val 并输出到 data/processed/。

标签体系数据驱动：label 支持任意字符串（如 normal/ad、广告细类），
自动汇总生成 label_map.json（label→id 按名称排序），训练/评估脚本据此对齐。
真实语料放 data/raw/ 下任意多个 jsonl，用 --input 一次传入。

用法:
  python scripts/prepare_data.py --input data/raw/dataset/converted/gold.binary.jsonl
  python scripts/prepare_data.py --input data/raw/*.jsonl --val_ratio 0.1
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", default=["data/raw/demo_sample.jsonl"],
                    help="训练语料 jsonl（默认指向本地演示样例；真实语料在 data/raw/ 下，均不入库）")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    for pattern in args.input:
        paths = sorted(Path().glob(pattern)) if any(c in pattern for c in "*?[") else [Path(pattern)]
        if not paths:
            print(f"[WARN] 无匹配文件: {pattern}")
            continue
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    lab = r["label"]
                    if lab is None or (isinstance(lab, str) and not lab.strip()):
                        raise ValueError(f"{p}:{ln} label 为空")
                    text = (r["text"] or "").strip()
                    if text:
                        rows.append({"text": text, "label": str(lab)})

    if not rows:
        raise SystemExit("[ERR] 没读到任何有效数据")

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio))
    val, train = rows[:n_val], rows[n_val:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("train.jsonl", train), ("val.jsonl", val)):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    label2id = {name: i for i, name in enumerate(sorted({r["label"] for r in rows}))}
    (out / "label_map.json").write_text(
        json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"total={len(rows)}  train={len(train)}  val={len(val)}")
    print("label 体系:", label2id)
    print("train 分布:", dict(Counter(r["label"] for r in train).most_common()))
    print("val   分布:", dict(Counter(r["label"] for r in val).most_common()))
    print(f"[OK] 已写入 {out}/train.jsonl, val.jsonl, label_map.json")


if __name__ == "__main__":
    main()
