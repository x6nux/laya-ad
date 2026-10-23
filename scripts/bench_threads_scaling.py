#!/usr/bin/env python3
"""AMD CPU 线程 scaling + bf16 实验（seq128 batch8 前向吞吐口径）。

口径（一次一个变量）：
  - 输入：合成 token（纯算力上限，不计时 tokenize），shape=(batch, seq)=(8, 128)，attention 全 1；
  - 吞吐：预热 2 次前向不计，计时 rounds(=3) 轮 × iters 次前向，
    samples/s = batch*iters/elapsed，取各轮中位数；
  - 正确率（--acc）：data/processed/val.jsonl 全量，真实文本、batch=8、max_len=128
    （同 bench_cpu.py 量化正确率口径），不计时，只取 acc；
  - 线程：OMP/MKL 等 env 在 import torch 之前设为 --threads，再 torch.set_num_threads；
    每个线程档位/精度形态用独立进程跑（干净状态）。

form:
  int8     outputs/ad-clf-int8/quantized.pt（torch.load，需先 import transformers）
  fp32     AutoModelForSequenceClassification.from_pretrained(outputs/ad-clf/best)
  bf16w    fp32 模型整体 .to(torch.bfloat16)（bf16 权重）
  autocast fp32 权重 + torch.autocast("cpu", bf16) 包住前向

用法:
  python scripts/bench_threads_scaling.py --threads 8 --form int8
  python scripts/bench_threads_scaling.py --threads 8 --form fp32 --acc
输出: stdout 最后一行 JSON（RESULT: {...}），便于脚本汇总。
"""
import argparse
import json
import os
import statistics
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--form", choices=["int8", "fp32", "bf16w", "autocast"], required=True)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--acc", action="store_true",
                    help="附加跑 val 全量正确率（batch=8 max_len=128，真实文本）")
    ap.add_argument("--model", default="outputs/ad-clf/best")
    ap.add_argument("--int8_model", default="outputs/ad-clf-int8/quantized.pt")
    ap.add_argument("--val", default="data/processed/val.jsonl")
    args = ap.parse_args()

    # ---- 线程 env：必须先于 import torch / transformers（同 bench_cpu.py:47-53）----
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_var] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch  # noqa: E402  env 已就位
    torch.set_num_threads(args.threads)

    if hasattr(sys.stdout, "reconfigure"):  # Windows 管道下避免 GBK 编码报错
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(ROOT)

    import transformers  # noqa: E402  必须先于 torch.load（ModernBert* 反序列化注册）
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

    if args.form == "int8":
        model = torch.load(args.int8_model, map_location="cpu", weights_only=False)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model)
        if args.form == "bf16w":
            model = model.to(torch.bfloat16)
    model.eval()
    print(f"loaded form={args.form} threads={torch.get_num_threads()}", flush=True)

    # ---- 吞吐：合成 seq128 batch8（纯算力上限）----
    x = torch.randint(1000, 200000, (args.batch, args.seq), dtype=torch.long)
    attn = torch.ones_like(x)

    def forward():
        if args.form == "autocast":
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                return model(input_ids=x, attention_mask=attn)
        return model(input_ids=x, attention_mask=attn)

    with torch.inference_mode():
        for _ in range(2):  # 预热不计
            forward()
        sps_list, sec_list = [], []
        for r in range(args.rounds):
            w_start = time.perf_counter()
            for _ in range(args.iters):
                forward()
            elapsed = time.perf_counter() - w_start
            sps_list.append(args.batch * args.iters / elapsed)
            sec_list.append(round(elapsed, 3))

    tok_s = args.batch * args.iters * args.seq / statistics.median(sec_list)
    print(f"rounds sps={[round(s, 2) for s in sps_list]} sec={sec_list}", flush=True)

    result = {
        "threads": args.threads,
        "form": args.form,
        "seq": args.seq,
        "batch": args.batch,
        "iters": args.iters,
        "sps_median": round(statistics.median(sps_list), 3),
        "sps_rounds": [round(s, 2) for s in sps_list],
        "tok_s": round(tok_s, 1),
    }

    # ---- 正确率：val 全量真实文本（batch=8 max_len=128，同 bench_cpu.py 口径）----
    if args.acc:
        tok = AutoTokenizer.from_pretrained(args.model)
        with open(args.val, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        label2id = model.config.label2id

        def to_id(lab):
            if isinstance(lab, int):
                return lab
            if lab in label2id:
                return label2id[lab]
            return int(lab)

        y_true = [to_id(r["label"]) for r in rows]
        texts = [r["text"] for r in rows]
        preds = []
        a_start = time.perf_counter()
        with torch.inference_mode():
            for i in range(0, len(texts), args.batch):
                enc = tok(texts[i:i + args.batch], truncation=True,
                          max_length=args.seq, padding=True, return_tensors="pt")
                preds.extend(model(**enc).logits.argmax(-1).tolist())
        acc = sum(int(p == t) for p, t in zip(preds, y_true)) / len(y_true)
        result["acc"] = round(acc, 4)
        result["acc_wall_s"] = round(time.perf_counter() - a_start, 1)
        result["n_acc"] = len(y_true)
        print(f"acc={acc:.4f} n={len(y_true)}", flush=True)

    print("RESULT: " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
