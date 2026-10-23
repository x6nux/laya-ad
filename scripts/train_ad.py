#!/usr/bin/env python3
"""Laya 底座（mmBERT/ModernBERT）中文广告识别微调。

默认启用吞吐优化：bf16 混合精度 + SDPA 注意力 + 动态 padding +
group_by_length（按长度聚桶减少 padding 浪费）+ fused AdamW。

用法:
  python scripts/train_ad.py --config configs/train_ad.yaml     # 推荐超参
  python scripts/train_ad.py --max_steps 6                      # 快速冒烟
  python scripts/train_ad.py --batch_size 64 --epochs 3
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainerCallback, TrainingArguments)

from train_lock import TrainLock


class Collator:
    """动态 padding：本批最长为准，短文本批省大量计算。"""

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        labels = torch.tensor([f["label_ids"] for f in feats], dtype=torch.long)
        batch = self.tok.pad(
            [{k: f[k] for k in ("input_ids", "attention_mask")} for f in feats],
            padding=True, return_tensors="pt",
        )
        batch["labels"] = labels
        return batch


class LockStatusCallback(TrainerCallback):
    """把训练进度实时写进锁文件（外部可轮询 JSON 观察 phase/epoch/step/loss）。"""

    def __init__(self, lock):
        self.lock = lock
        self.last_logs = {}

    def _gpu_mb(self):
        if torch.cuda.is_available():
            return round(torch.cuda.memory_reserved() / 2**20, 1)
        return None

    def _push(self, phase, state=None, **extra):
        d = {}
        if state is not None:
            d = {"epoch": round(state.epoch or 0, 3),
                 "global_step": state.global_step,
                 "max_steps": state.max_steps}
        gpu = self._gpu_mb()
        if gpu is not None:
            d["gpu_mem_mb"] = gpu
        self.lock.update(phase, **{**self.last_logs, **d, **extra})

    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            self.last_logs.update({k: v for k, v in logs.items()
                                   if isinstance(v, (int, float, str))})
        self._push("training", state)

    def on_step_end(self, args, state, control, **kw):
        self._push("training", state)

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if metrics:
            self.last_logs.update({f"eval_{k}": v for k, v in metrics.items()
                                   if isinstance(v, (int, float))})
        self._push("evaluating", state)

    def on_save(self, args, state, control, **kw):
        self._push("saving", state)

    def on_train_end(self, args, state, control, **kw):
        self._push("finalizing", state)


def parse_args():
    # config 先读、作为 argparse 默认值：显式命令行参数优先于 yaml（此前顺序反了，CLI 会被 yaml 覆盖）
    import yaml
    cfg = {}
    argv = __import__("sys").argv
    if "--config" in argv:
        cfg_path = argv[argv.index("--config") + 1]
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="yaml 提供默认超参，命令行显式参数优先")
    ap.add_argument("--model", default="models/mmbert-ad")
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--output", default="outputs/ad-clf")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.06,
                    help="按比例换算成 warmup_steps（transformers v5 已移除 warmup_ratio）")
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--fp32", action="store_true", help="禁用 bf16（排障用）")
    ap.add_argument("--compile", action="store_true", help="torch.compile（首步编译慢，正式长训再开）")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=-1, help="冒烟测试，如 --max_steps 6")
    ap.add_argument("--lock_file", default="outputs/train.lock.json",
                    help="微调互斥锁文件（防同时训练，结束自动删除）")
    ap.add_argument("--force", action="store_true", help="强制接管已存在的活锁")
    for k, v in cfg.items():
        k = k.replace("-", "_")
        if v is not None and k != "config":
            ap.set_defaults(**{k: v})
    args = ap.parse_args()
    return args


def _run(args, lock):
    tok = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset("json", data_files={"train": args.train, "validation": args.val})

    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                batched=True, remove_columns=["text"])

    # 标签体系数据驱动：优先 train 同目录的 label_map.json，否则从数据推导并落盘
    map_path = Path(args.train).parent / "label_map.json"
    if map_path.exists():
        label2id = json.loads(map_path.read_text(encoding="utf-8"))
    else:
        names = sorted({str(l) for l in ds["train"]["label"]} |
                       {str(l) for l in ds["validation"]["label"]})
        label2id = {n: i for i, n in enumerate(names)}
        map_path.write_text(json.dumps(label2id, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"[INFO] 无 label_map.json，已从数据推导并写入 {map_path}: {label2id}")
    id2label = {int(i): n for n, i in label2id.items()}
    ds = ds.map(lambda b: {"label_ids": [label2id[str(l)] for l in b["label"]]},
                batched=True, remove_columns=["label"])

    # 权重保持 fp32，由 Trainer 的 bf16 自动混合精度负责加速（更稳）
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        attn_implementation="sdpa",
    )
    if args.compile:
        model = torch.compile(model)

    acc = load_metric("accuracy")
    f1 = load_metric("f1")

    def compute_metrics(p):
        pred = np.argmax(p.predictions, axis=-1)
        return {
            "accuracy": acc.compute(predictions=pred, references=p.label_ids)["accuracy"],
            "f1_macro": f1.compute(predictions=pred, references=p.label_ids,
                                   average="macro")["f1"],
        }

    steps_per_epoch = math.ceil(len(ds["train"]) / max(1, args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(steps_per_epoch * args.epochs))
    smoke = args.max_steps > 0  # 冒烟模式：不落 checkpoint、不做 epoch 评估
    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=max(1, int(args.warmup_ratio * total_steps)),
        lr_scheduler_type="cosine",
        bf16=not args.fp32,
        optim=args.optim,
        logging_steps=5,
        eval_strategy="no" if smoke else "epoch",
        save_strategy="no" if smoke else "epoch",
        load_best_model_at_end=not smoke,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_num_workers=args.workers,
        report_to=["tensorboard"],
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tok,
        data_collator=Collator(tok),
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(LockStatusCallback(lock))
    trainer.train()

    lock.update("saving")
    best_dir = str(Path(args.output) / "best")
    trainer.save_model(best_dir)
    tok.save_pretrained(best_dir)
    lock.update("eval")
    final = trainer.evaluate()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "final_metrics.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    lock = TrainLock(
        args.lock_file,
        meta={"model": args.model, "output": args.output,
              "command": " ".join(sys.argv[1:])},
        force=args.force,
    )
    try:
        lock.acquire()
    except RuntimeError as e:
        print(f"[lock] {e}")
        sys.exit(1)
    try:
        _run(args, lock)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
