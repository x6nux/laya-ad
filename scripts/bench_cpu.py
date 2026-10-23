#!/usr/bin/env python3
"""CPU fp32 vs int8 动态量化基准（目标：保正确率、最大提性能）。

流程（同机同条件，一次一个变量）：
  1. CPU 加载 outputs/ad-clf/best（fp32），val.jsonl 前 N(=500) 条测正确率 + 吞吐；
  2. torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)；
  3. 同口径重测 int8 正确率与吞吐，量化模型落盘 outputs/ad-clf-int8/quantized.pt。

口径：
  - 正确率：data/processed/val.jsonl 前 500 条，label_map=data/processed/label_map.json；
  - 吞吐：val 真实文本，tokenizer max_len=128、batch=8，计时含 tokenize（部署真实链路），
    样本/秒 = N / elapsed；
  - 线程：torch.set_num_threads(物理核数一半)；OMP/MKL 等 env 在 import torch 之前设置；
  - 稳态：先跑 1 轮预热（不计吞吐），再计时 --rounds(=3) 轮取中位数；
    正确率确定性强，各轮应一致，不一致会打 warn。
采纳标准：int8_acc >= fp32_acc - 0.01 且 speedup(int8_sps/fp32_sps) >= 1.3。

用法:
  python scripts/bench_cpu.py                      # 完整基准
  python scripts/bench_cpu.py --n 16 --rounds 1    # 快速冒烟

quantized.pt 加载方式（必须先 import transformers 让 ModernBert* 类可反序列化；
weights_only=False 才允许加载完整 pickle 的 nn.Module）:
    import transformers, torch
    model = torch.load("outputs/ad-clf-int8/quantized.pt", map_location="cpu", weights_only=False)
    model.eval()
"""
import argparse
import json
import os
import statistics
import sys
import time


def _physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)  # 无 psutil 时按超线程折半兜底


# ---- 线程环境变量：必须先于 import torch / transformers（它们会拉起线程池）----
PHYS_CORES = _physical_cores()
THREADS = max(1, PHYS_CORES // 2)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = str(THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402  env 已就位
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

torch.set_num_threads(THREADS)

if hasattr(sys.stdout, "reconfigure"):  # Windows 管道下避免 GBK 编码报错
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run_pass(model, tok, texts, y_true, batch_size, max_len):
    """一遍全量前向，返回 (acc, elapsed)。"""
    preds = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True,
                      max_length=max_len, padding=True, return_tensors="pt")
            preds.extend(model(**enc).logits.argmax(-1).tolist())
    elapsed = time.perf_counter() - t0
    acc = sum(int(p == t) for p, t in zip(preds, y_true)) / len(y_true)
    return acc, elapsed


def bench(tag, model, tok, texts, y_true, batch_size, max_len, rounds):
    """预热 1 轮 + 计时 rounds 轮取中位数；正确率取计时轮第一轮。"""
    acc0, _ = run_pass(model, tok, texts, y_true, batch_size, max_len)
    print(f"  [{tag}] warmup: acc={acc0:.4f}（不计吞吐）", flush=True)
    sps_list, sec_list, accs = [], [], []
    for r in range(rounds):
        acc, sec = run_pass(model, tok, texts, y_true, batch_size, max_len)
        accs.append(acc)
        sec_list.append(round(sec, 3))
        sps_list.append(len(texts) / sec)
        print(f"  [{tag}] round {r + 1}: acc={acc:.4f}  {sec:.1f}s  "
              f"{len(texts) / sec:.2f} 样本/s", flush=True)
    if len(set(accs + [acc0])) != 1:
        print(f"  [warn] {tag} 各轮正确率不一致: warmup={acc0} rounds={accs}", flush=True)
    return {
        "acc": accs[0],
        "sps": statistics.median(sps_list),
        "sps_rounds": [round(s, 2) for s in sps_list],
        "sec_rounds": sec_list,
        "warmup_acc": acc0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="outputs/ad-clf/best")
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--label_map", default="data/processed/label_map.json")
    ap.add_argument("--out", default="outputs/cpu_quant_bench.json")
    ap.add_argument("--quant_out", default="outputs/ad-clf-int8/quantized.pt")
    ap.add_argument("--n", type=int, default=500, help="val 前 N 条")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--rounds", type=int, default=3, help="计时轮数（另有 1 轮预热）")
    args = ap.parse_args()

    print(f"物理核={PHYS_CORES}  torch.set_num_threads={THREADS}  "
          f"OMP/MKL_NUM_THREADS={os.environ['OMP_NUM_THREADS']}", flush=True)
    print(f"torch={torch.__version__}  transformers={__import__('transformers').__version__}",
          flush=True)

    # ---- 数据：val 前 N 条真实文本 ----
    with open(args.val, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()][:args.n]
    with open(args.label_map, encoding="utf-8") as f:
        label2id = json.load(f)
    texts = [r["text"] for r in rows]
    try:
        y_true = [label2id[r["label"]] for r in rows]
    except KeyError as e:
        raise SystemExit(f"label_map({args.label_map}) 缺少标签 {e}；"
                         f"数据标签集合={sorted({r['label'] for r in rows})}")
    print(f"样本={len(texts)}  标签分布={ {k: sum(r['label'] == k for r in rows) for k in label2id} }",
          flush=True)

    # ---- fp32 基准 ----
    print(f"[1/3] 加载 fp32 模型 {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()
    assert next(model.parameters()).device.type == "cpu", "必须在 CPU 上"
    print("[1/3] fp32 基准测量", flush=True)
    fp32 = bench("fp32", model, tok, texts, y_true, args.batch_size, args.max_len, args.rounds)

    # ---- int8 动态量化（只量化 nn.Linear）----
    print("[2/3] quantize_dynamic -> qint8 ...", flush=True)
    t0 = time.perf_counter()
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8)
    print(f"[2/3] 量化完成 {time.perf_counter() - t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.quant_out) or ".", exist_ok=True)
    torch.save(model_int8, args.quant_out)
    print(f"[2/3] 量化模型已存 {args.quant_out}", flush=True)

    print("[3/3] int8 同口径测量", flush=True)
    int8 = bench("int8", model_int8, tok, texts, y_true, args.batch_size, args.max_len,
                 args.rounds)

    speedup = int8["sps"] / fp32["sps"]
    acc_drop = fp32["acc"] - int8["acc"]
    accepted = (int8["acc"] >= fp32["acc"] - 0.01) and (speedup >= 1.3)

    result = {
        "model": args.model,
        "device": "cpu",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "physical_cores": PHYS_CORES,
        "torch_num_threads": THREADS,
        "env": {k: os.environ[k] for k in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TOKENIZERS_PARALLELISM")},
        "n_samples": len(texts),
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "rounds": args.rounds,
        "sps_agg": f"median of {args.rounds} timed rounds (1 warmup excluded)",
        "fp32_acc": round(fp32["acc"], 4),
        "int8_acc": round(int8["acc"], 4),
        "fp32_sps": round(fp32["sps"], 2),
        "int8_sps": round(int8["sps"], 2),
        "speedup": round(speedup, 3),
        "acc_drop": round(acc_drop, 4),
        "fp32_detail": fp32,
        "int8_detail": int8,
        "quantized_model": args.quant_out,
        "criteria": {"max_acc_drop": 0.01, "min_speedup": 1.3},
        "accepted": accepted,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 60, flush=True)
    print(f"fp32: acc={fp32['acc']:.4f}  {fp32['sps']:.2f} 样本/s", flush=True)
    print(f"int8: acc={int8['acc']:.4f}  {int8['sps']:.2f} 样本/s", flush=True)
    print(f"speedup={speedup:.3f}x  acc_drop={acc_drop:+.4f}  "
          f"accepted={accepted}", flush=True)
    print(f"结果已写 {args.out}", flush=True)


if __name__ == "__main__":
    main()
