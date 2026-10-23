#!/usr/bin/env python3
"""在 jsonl({text,label}) 上评估已训练的广告分类模型。

输出 accuracy / per-class P-R-F1 / 混淆矩阵，并把逐条概率落盘便于挑坏例。

用法:
  python scripts/eval_ad.py                                  # 默认评估 val 集
  python scripts/eval_ad.py --data data/raw/my_test.jsonl
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def to_id(lab, label2id):
    """标签兼容：字符串名 / 旧数字 id 均可。"""
    if isinstance(lab, int):
        return lab
    if lab in label2id:
        return label2id[lab]
    return int(lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/ad-clf/best")
    ap.add_argument("--data", default="data/processed/val.jsonl")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, attn_implementation="sdpa", dtype=torch.bfloat16).eval().cuda()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    y = np.array([to_id(r["label"], model.config.label2id) for r in rows])

    preds, probs_all = [], []
    with torch.inference_mode():
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i:i + args.batch_size]
            enc = tok([r["text"] for r in chunk], truncation=True,
                      max_length=args.max_len, padding=True,
                      return_tensors="pt").to("cuda")
            probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
            preds.extend(probs.argmax(1).tolist())
            probs_all.extend(probs.tolist())

    names = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(classification_report(y, preds, target_names=names, digits=4))
    print(f"混淆矩阵（行=真实，列=预测，顺序 = {names}）:")
    print(confusion_matrix(y, preds))

    dump = args.model.rstrip("/\\") + "_val_probs.jsonl"
    with open(dump, "w", encoding="utf-8") as f:
        for r, p in zip(rows, probs_all):
            f.write(json.dumps({"text": r["text"], "label": r["label"], "probs": p},
                               ensure_ascii=False) + "\n")
    print("逐条概率已存:", dump)


if __name__ == "__main__":
    main()
