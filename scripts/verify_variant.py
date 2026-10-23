#!/usr/bin/env python3
"""CI 结构校验：加载量化变体 → 哑输入前向 → 断言输出形状与有限性 → 写 VERIFICATION.txt。

注意：这是结构校验（格式可加载、前向不炸），不是精度校验——CI 里没有数据。
精度数字以本地 VERIFICATION / MODEL_CARD 记录为准。
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from load_quantized import load_variant  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    model_dir = Path(args.model)

    model = load_variant(model_dir, device="cpu", dtype=torch.bfloat16)
    tok_len = model.config.max_position_embeddings
    x = torch.randint(0, model.config.vocab_size, (2, 64))
    with torch.inference_mode():
        logits = model(input_ids=x).logits
    assert logits.shape == (2, model.config.num_labels), f"输出形状异常: {logits.shape}"
    assert torch.isfinite(logits.float()).all(), "logits 含非有限值"

    lines = [f"CI 结构校验  {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             "loader: scripts/load_quantized.py (torch + transformers)",
             f"logits_shape: {list(logits.shape)}  finite: true  dtype: {next(model.parameters()).dtype}",
             "", "sha256:"]
    for p in sorted(model_dir.iterdir()):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"  {h}  {p.name}  ({p.stat().st_size} bytes)")
    (model_dir / "VERIFICATION.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {model_dir.name}: shape={list(logits.shape)} finite=true")


if __name__ == "__main__":
    main()
