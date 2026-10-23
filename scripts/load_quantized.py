#!/usr/bin/env python3
"""量化变体统一加载器：fp16 / bf16 / fp8 / int8 / int4 → 可推理的 transformers 模型。

- fp16 / bf16：transformers 原生加载
- fp8（float8_e4m3fn 存储）：cast 回 bf16 加载
- int8（per-channel 对称 weight-only）：按 `__scale__` 伴生张量反量化
- int4（group=64 对称，双 nibble 打包 uint8）：解包 + 按 group scale 反量化

用法:
    from load_quantized import load_variant
    model = load_variant("variants/int4")           # 自动识别格式，返回 eval 模式模型
"""
from pathlib import Path

import torch
from safetensors import safe_open

INT4_GROUP = 64


def _dtypes(sd_path: Path) -> set:
    dtypes = set()
    with safe_open(str(sd_path), framework="pt") as f:
        for k in f.keys():
            dtypes.add(f.get_slice(k).get_dtype())
    return dtypes


def load_variant(model_dir, device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
    """自动识别量化格式并加载，返回 eval 模式模型（fp8/int8/int4 权重已反量化到 dtype）。"""
    from transformers import AutoConfig, AutoModelForSequenceClassification

    model_dir = Path(model_dir)
    sd_path = model_dir / "model.safetensors"
    dts = _dtypes(sd_path)

    if not ({"I8", "U8", "F8_E4M3"} & dts):
        # fp16 / bf16：原生
        kw = {"dtype": dtype} if dtype else {}
        return AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), attn_implementation="sdpa", **kw).to(device).eval()

    model = AutoModelForSequenceClassification.from_config(
        AutoConfig.from_pretrained(str(model_dir)))
    param_shapes = {k: v.shape for k, v in model.state_dict().items()}
    state = {}
    with safe_open(str(sd_path), framework="pt") as f:
        keys = list(f.keys())
        raw = {k: f.get_tensor(k) for k in keys}

    for k, v in raw.items():
        if k.endswith(".__scale__") or k.endswith(".__pad__"):
            continue
        if v.dtype == torch.float8_e4m3fn:  # fp8 存储 → cast
            state[k] = v.to(dtype)
        elif v.dtype == torch.int8 and k + ".__scale__" in raw:  # int8 per-channel
            scale = raw[k + ".__scale__"].float().view(-1, *([1] * (v.dim() - 1)))
            state[k] = (v.float() * scale).to(dtype)
        elif v.dtype == torch.uint8 and k + ".__scale__" in raw:  # int4 打包
            hi = (v >> 4).to(torch.int16) - 8
            lo = (v & 0xF).to(torch.int16) - 8
            q = torch.stack([hi, lo], dim=-1).reshape(-1).float()
            scale = raw[k + ".__scale__"].float()
            s_exp = scale.repeat_interleave(INT4_GROUP)[:q.numel()]
            w = q * s_exp
            pad = int(raw[k + ".__pad__"].item()) if k + ".__pad__" in raw else 0
            if pad:
                w = w[:-pad]
            state[k] = w.reshape(param_shapes[k]).to(dtype)
        else:  # 未量化的透传张量（1D bias/norm 等保留精度）
            state[k] = v.to(dtype)

    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


if __name__ == "__main__":
    import sys
    m = load_variant(sys.argv[1] if len(sys.argv) > 1 else "variants/bf16")
    print("加载成功:", type(m).__name__, "| dtype:", next(m.parameters()).dtype)
