#!/usr/bin/env python3
"""广告识别预测演示：单条文本 / jsonl / 纯文本文件，输出各类别概率。"""
import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEMO_TEXTS = [
    "加微信 xxFREE 免费领取涨粉秘籍，名额有限！",
    "今晚合唱团排练改到七点半，别走错了",
    "这款面霜绝了，用了一个月肉眼可见变好，链接放评论区",
    "明天降温10度，记得给孩子加衣服",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/ad-clf/best")
    ap.add_argument("--text", default=None, help="单条文本")
    ap.add_argument("--file", default=None, help="jsonl({text}) 或 txt，每行一条")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, attn_implementation="sdpa", dtype=torch.bfloat16).eval().cuda()
    names = [model.config.id2label[i] for i in range(model.config.num_labels)]

    if args.text:
        texts = [args.text]
    elif args.file:
        if args.file.endswith(".jsonl"):
            texts = [json.loads(l)["text"]
                     for l in open(args.file, encoding="utf-8") if l.strip()]
        else:
            texts = [l.rstrip("\n") for l in open(args.file, encoding="utf-8") if l.strip()]
    else:
        texts = DEMO_TEXTS

    with torch.inference_mode():
        for i in range(0, len(texts), args.batch_size):
            chunk = texts[i:i + args.batch_size]
            enc = tok(chunk, truncation=True, max_length=512, padding=True,
                      return_tensors="pt").to("cuda")
            probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu()
            for t, p in zip(chunk, probs):
                top = int(p.argmax())
                dist = "  ".join(f"{n}={p[j]:.3f}" for j, n in enumerate(names))
                print(f"[{names[top]:>7}] {t[:60]}   ({dist})")


if __name__ == "__main__":
    main()
