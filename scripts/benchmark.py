#!/usr/bin/env python3
"""吞吐基准：测不同 batch × 序列长度 × 精度下的前向吞吐。

输出 samples/s、tokens/s、延迟 p50 与显存峰值，用于选训练 batch 和部署配置。
输入为随机 token（测的是纯算力上限）；真实短文本的收益还要叠加动态 padding，
见 README「吞吐优化」一节。

用法:
  python scripts/benchmark.py                      # seq=128/512 × batch=8..128, bf16
  python scripts/benchmark.py --fp32               # fp32 对照
  python scripts/benchmark.py --compile            # 额外测 torch.compile 版本
"""
import argparse
import time

import torch
from transformers import AutoModelForSequenceClassification


def bench_once(model, x, attn, iters):
    for _ in range(3):
        with torch.inference_mode():
            model(input_ids=x, attention_mask=attn)
    torch.cuda.synchronize()
    lat = []
    t0 = time.perf_counter()
    for _ in range(iters):
        t1 = time.perf_counter()
        with torch.inference_mode():
            model(input_ids=x, attention_mask=attn)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t1) * 1000)
    total = time.perf_counter() - t0
    return total, lat


def run(model, args, dtype_name, results):
    for seq in args.seqs:
        for batch in args.batches:
            x = torch.randint(1000, 200000, (batch, seq), device="cuda")
            attn = torch.ones_like(x)
            torch.cuda.reset_peak_memory_stats()
            try:
                total, lat = bench_once(model, x, attn, args.iters)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[{dtype_name}] seq={seq:>4} batch={batch:>4} | OOM，跳过")
                continue
            row = {
                "dtype": dtype_name, "seq": seq, "batch": batch,
                "samples_per_s": batch * args.iters / total,
                "tokens_per_s": batch * args.iters * seq / total,
                "p50_ms": sorted(lat)[len(lat) // 2],
                "vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
            results.append(row)
            print(f"[{row['dtype']}] seq={seq:>4} batch={batch:>4} | "
                  f"{row['samples_per_s']:>9.1f} 样本/s | "
                  f"{row['tokens_per_s'] / 1e6:>6.2f}M tok/s | "
                  f"p50={row['p50_ms']:>7.1f}ms | VRAM={row['vram_gb']:.2f}GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/mmbert-ad")
    ap.add_argument("--seqs", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 32, 64, 128])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--fp32", action="store_true", help="只测 fp32")
    ap.add_argument("--compile", action="store_true", help="追加测 torch.compile")
    args = ap.parse_args()

    results = []
    dtypes = ["fp32"] if args.fp32 else ["bf16"] + (["fp32"] if False else [])
    for dt in dtypes:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, attn_implementation="sdpa",
            dtype=torch.float32 if dt == "fp32" else torch.bfloat16).eval().cuda()
        run(model, args, dt, results)
        if args.compile and dt == "bf16":
            compiled = torch.compile(model)
            print("（torch.compile：每个新 shape 都要重编译，首次会慢）")
            run(compiled, args, dt + "-compile", results)
        del model
        torch.cuda.empty_cache()

    if results:
        best = max(results, key=lambda r: r["samples_per_s"])
        print(f"\n最佳吞吐配置: {best['dtype']} seq={best['seq']} batch={best['batch']} "
              f"-> {best['samples_per_s']:.0f} 样本/s ({best['tokens_per_s'] / 1e6:.2f}M tok/s)")


if __name__ == "__main__":
    main()
