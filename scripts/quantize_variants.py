#!/usr/bin/env python3
"""从 fp32 safetensors 一次性产出 FP16 / BF16 / FP8 / INT8 / INT4 五种量化变体。

全部落为 safetensors：
- fp16 / bf16：直接 dtype 转换，transformers 原生可加载
- fp8：float8_e4m3fn 逐张量转换（存储格式；推理前需 cast 回 bf16/fp32，见 load_quantized.py）
- int8：per-channel 对称 weight-only 量化（int8 权重 + fp16 per-output-channel scale）
- int4：group-wise（group=64）对称量化，两值打包进 uint8 + fp16 group scale

int8/int4 的 scale 以 `<tensor名>.__scale__` 伴生张量存储，加载器 scripts/load_quantized.py
负责反量化成标准 ModernBertForSequenceClassification。fp16/bf16/fp8 精度无争议，
int8/int4 属有损量化（仓库内提供加载器与校验方式，但精度需用户以自有数据验证）。

用法:
  python scripts/quantize_variants.py --src <fp32模型目录> --out <输出目录> [--variants fp16,bf16,fp8,int8,int4]
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}
INT4_GROUP = 64


def cast_variant(state: dict, name: str) -> dict:
    dt = DTYPE_MAP[name]
    out = {}
    for k, v in state.items():
        out[k] = v.to(dt) if v.is_floating_point() else v
    return out


def quant_int8(state: dict) -> dict:
    """per-output-channel 对称量化：w_int8 = clamp(round(w / scale)), scale = max|w|/127。
    只量化 2D 权重矩阵；1D 参数（bias/LayerNorm）保留 fp16。"""
    out = {}
    for k, v in state.items():
        if not v.is_floating_point() or v.numel() == 0:
            out[k] = v
            continue
        if v.dim() < 2:
            out[k] = v.to(torch.float16)
            continue
        w = v.float()
        scale = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True) / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
        out[k] = q
        out[k + ".__scale__"] = scale.flatten(1).squeeze(-1).to(torch.float16)
    return out


def quant_int4(state: dict) -> dict:
    """group-wise 对称量化（group=64）：每 group 一个 fp16 scale，两 4-bit 打包进一个 uint8。
    只量化 2D 权重矩阵（打包后存扁平字节流）；1D 参数保留 fp16。"""
    out = {}
    for k, v in state.items():
        if not v.is_floating_point() or v.numel() == 0:
            out[k] = v
            continue
        if v.dim() < 2:
            out[k] = v.to(torch.float16)
            continue
        w = v.float().reshape(-1)
        pad = (-w.numel()) % INT4_GROUP
        if pad:
            w = torch.cat([w, torch.zeros(pad)])
        g = w.view(-1, INT4_GROUP)
        scale = g.abs().amax(dim=1) / 7.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(g / scale.unsqueeze(1)), -7, 7).to(torch.int8)  # [-7,7]
        nib = (q + 8).to(torch.uint8)            # 移到 [1,15]，0 保留给填充
        packed = (nib[:, 0::2] << 4) | nib[:, 1::2]
        out[k] = packed.reshape(-1)              # 扁平字节流；加载器按参数形状重塑
        out[k + ".__scale__"] = scale.to(torch.float16)
        if pad:
            out[k + ".__pad__"] = torch.tensor(pad, dtype=torch.int32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="fp32 模型目录（含 model.safetensors）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="fp16,bf16,fp8,int8,int4")
    args = ap.parse_args()

    src = Path(args.src)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    state = load_file(str(src / "model.safetensors"))
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    for name in variants:
        dst = out_root / name
        dst.mkdir(parents=True, exist_ok=True)
        if name in DTYPE_MAP:
            new_state = cast_variant(state, name)
        elif name == "int8":
            new_state = quant_int8(state)
        elif name == "int4":
            new_state = quant_int4(state)
        else:
            raise SystemExit(f"[ERR] 未知变体 {name}（可选 fp16/bf16/fp8/int8/int4）")
        save_file(new_state, str(dst / "model.safetensors"), metadata={"format": "pt"})
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if (src / f).exists():
                (dst / f).write_bytes((src / f).read_bytes())
        n_params = sum(v.numel() for v in new_state.values())
        size_mb = (dst / "model.safetensors").stat().st_size / 2**20
        print(f"[{name:>4}] tensors={len(new_state)} size={size_mb:.1f}MB -> {dst}")
    json.dump({"source_sha_note": "fp32 基准", "variants": variants},
              open(out_root / "variants.json", "w"), indent=2)


if __name__ == "__main__":
    main()
