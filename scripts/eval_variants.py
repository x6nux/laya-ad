#!/usr/bin/env python3
"""本地精度评测：quantize_variants 产出的各变体 vs fp32 基准，同一 val 集同口径对比。

用法:
  python scripts/eval_variants.py --dirs /tmp/qtest/fp16 /tmp/qtest/bf16 /tmp/qtest/fp8 /tmp/qtest/int8 /tmp/qtest/int4
  python scripts/eval_variants.py --dirs outputs/ad-clf/best            # fp32 基准
输出: 每变体 accuracy / macro-F1 / 与 fp32 预测一致率；汇总写 release/variant_eval.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from load_quantized import load_variant  # noqa: E402


def evaluate(model_dir: str, tok_dir: str, rows, y, label2id, batch_size=64):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    model = load_variant(model_dir, device="cuda", dtype=torch.bfloat16)
    preds = []
    with torch.inference_mode():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            enc = tok([r["text"] for r in chunk], truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to("cuda")
            preds.extend(model(**enc).logits.float().argmax(-1).cpu().tolist())
    preds = np.array(preds)
    acc = float((preds == y).mean())
    f1 = float(f1_score(y, preds, average="macro"))
    return preds, acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--tok", default=None, help="tokenizer 目录（缺省用第一个变体内的）")
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--out", default="release/variant_eval.json")
    args = ap.parse_args()

    label2id = json.loads(Path("data/processed/label_map.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in open(args.val, encoding="utf-8") if l.strip()]
    y = np.array([label2id[str(r["label"])] for r in rows])
    tok_dir = args.tok or args.dirs[0]

    results = {}
    ref_preds = None
    for d in args.dirs:
        name = Path(d).name or d
        preds, acc, f1 = evaluate(d, tok_dir, rows, y, label2id)
        entry = {"accuracy": round(acc, 4), "f1_macro": round(f1, 4), "n": len(rows)}
        if ref_preds is not None:
            entry["agree_with_ref"] = round(float((preds == ref_preds).mean()), 4)
        else:
            ref_preds = preds
        results[name] = entry
        print(f"[{name:>10}] acc={acc:.4f}  f1={f1:.4f}"
              + (f"  与基准一致率={entry['agree_with_ref']:.2%}" if "agree_with_ref" in entry else "  (基准)"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
