#!/usr/bin/env python3
"""从 Laya 完整 checkpoint 中抽取 mmBERT(ModernBERT) encoder 底座。

Laya 的 safetensors 里 encoder 权重带 `encoder.` 前缀（其余是决策头）。
本脚本剥掉前缀后存成独立目录，使其可直接用 AutoModelForSequenceClassification
做标准分类微调；分类头（score.*）是新增的，训练时随机初始化属正常现象。

用法:
  python scripts/extract_encoder.py                       # 默认 models/laya-multilingual -> models/mmbert-ad
  python scripts/extract_encoder.py --inspect             # 只打印键前缀分布，不写文件
"""
import argparse
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya_dir", default="models/laya-multilingual")
    ap.add_argument("--out_dir", default="models/mmbert-ad")
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.laya_dir) / "model.safetensors"
    if not ckpt.exists():
        sys.exit(f"[ERR] 找不到 {ckpt}，请先下载模型（见 README 快速开始）")

    state = load_file(str(ckpt))
    prefixes = {}
    for k in state:
        top = ".".join(k.split(".")[:2])
        prefixes[top] = prefixes.get(top, 0) + 1
    print("键前缀分布（top2 级）:")
    for p, n in sorted(prefixes.items()):
        print(f"  {p:<40} {n} tensors")

    if args.inspect:
        return

    enc_keys = [k for k in state if k.startswith("encoder.")]
    if not enc_keys:
        sys.exit(f"[ERR] checkpoint 中没有 encoder.* 前缀的键，请根据上面的分布手动调整")
    stripped = {k[len("encoder."):]: state[k] for k in enc_keys}
    # ModernBertForSequenceClassification 的主体键以 model. 开头；
    # 若 encoder 是裸 ModernBertModel（键为 embeddings./layers. 开头），补上 model. 前缀
    if not any(k.startswith("model.") for k in stripped):
        stripped = {f"model.{k}": v for k, v in stripped.items()}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(stripped, str(out / "model.safetensors"), metadata={"format": "pt"})
    shutil.copy(Path(args.laya_dir) / "encoder" / "config.json", out / "config.json")
    tok_dir = Path(args.laya_dir) / "tokenizer"
    for f in ("tokenizer.json", "tokenizer_config.json"):
        if (tok_dir / f).exists():
            shutil.copy(tok_dir / f, out / f)

    n_params = sum(v.numel() for v in stripped.values())
    print(f"[OK] encoder 已抽取 -> {out}  (tensors={len(stripped)}, params={n_params / 1e6:.1f}M)")


if __name__ == "__main__":
    main()
