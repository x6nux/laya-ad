"""量化变体加载器（laya-ad 发布格式）。

支持 laya-ad Release 的五种 safetensors 变体并统一加载为可推理模型：
- fp16 / bf16：transformers 原生加载
- fp8（float8_e4m3fn 存储）：cast 回 bf16
- int8（per-channel 对称 weight-only，`__scale__` 伴生 fp16）
- int4（group=64 对称、双 nibble 打包 uint8、`__scale__`/`__pad__` 伴生）

由 scripts/quantize_variants.py 产出；格式与 laya-ad 仓库 scripts/load_quantized.py 一致。
"""
from pathlib import Path

import torch
from safetensors import safe_open

INT4_GROUP = 64


def detect_precision(model_dir: Path) -> str:
    """优先读 PRECISION.txt（CI 注入），否则从权重 dtype 推断。"""
    marker = Path(model_dir) / "PRECISION.txt"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip() or "unknown"
    sd = Path(model_dir) / "model.safetensors"
    dts = set()
    with safe_open(str(sd), framework="pt") as f:
        for k in f.keys():
            dts.add(f.get_slice(k).get_dtype())
    if "F8_E4M3" in dts:
        return "fp8"
    if "U8" in dts:
        return "int4"
    if "I8" in dts:
        return "int8"
    if "BF16" in dts:
        return "bf16"
    if "F16" in dts:
        return "fp16"
    return "fp32"


def load_ad_model(model_dir, device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
    """加载广告分类模型（自动识别量化格式），返回 (model, tokenizer)，eval 模式。"""
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    sd_path = model_dir / "model.safetensors"
    dts = set()
    with safe_open(str(sd_path), framework="pt") as f:
        for k in f.keys():
            dts.add(f.get_slice(k).get_dtype())

    if not ({"I8", "U8", "F8_E4M3"} & dts):
        kw = {"dtype": dtype}
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), attn_implementation="sdpa", **kw)
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        return model.to(device).eval(), tok

    model = AutoModelForSequenceClassification.from_config(
        AutoConfig.from_pretrained(str(model_dir)))
    param_shapes = {k: v.shape for k, v in model.state_dict().items()}
    with safe_open(str(sd_path), framework="pt") as f:
        raw = {k: f.get_tensor(k) for k in f.keys()}

    state = {}
    for k, v in raw.items():
        if k.endswith(".__scale__") or k.endswith(".__pad__"):
            continue
        if v.dtype == torch.float8_e4m3fn:
            state[k] = v.to(dtype)
        elif v.dtype == torch.int8 and k + ".__scale__" in raw:
            scale = raw[k + ".__scale__"].float().view(-1, *([1] * (v.dim() - 1)))
            state[k] = (v.float() * scale).to(dtype)
        elif v.dtype == torch.uint8 and k + ".__scale__" in raw:
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
        else:  # 未量化透传（1D bias/norm 等保留精度）
            state[k] = v.to(dtype)

    model.load_state_dict(state, strict=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    return model.to(device).eval(), tok
