#!/usr/bin/env python3
"""模型权重发布打包：fp32 / bf16 / int8 三精度一次性各自成包 + 标准 gzip 压缩 + 精度校验。

每个包自包含（权重 + config + tokenizer + MODEL_CARD + VERIFICATION），命名规范：
    laya-ad-<tag>-<precision>.tar.gz      precision ∈ fp32 | bf16 | int8

只打包白名单文件；语料、training_args.bin（pickle 含本地路径）、任何数据一律排除，
组装后按扩展名白名单强制复扫。三种精度用同一批留出样本做精度校验并写入包内。

用法:
  python scripts/package_release.py                      # 三精度全部打包
  python scripts/package_release.py --precision bf16     # 只打某一档
  python scripts/package_release.py --upload --tag v1.0  # 上传 GitHub Release（gh CLI）
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from safetensors.torch import load_file, save_file

MODEL = Path("outputs/ad-clf/best")
INT8 = Path("outputs/ad-clf-int8/quantized.pt")
VAL = Path("data/processed/val.jsonl")
OUT = Path("release")

ALLOWED_EXT = {".safetensors", ".json", ".md", ".txt"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_val(model, tok, rows, label2id, device):
    preds, ys = [], []
    with torch.inference_mode():
        for i in range(0, len(rows), 64):
            chunk = rows[i:i + 64]
            enc = tok([r["text"] for r in chunk], truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits.float()
            preds.append(logits.argmax(-1).cpu())
            ys.append(torch.tensor([label2id[str(r["label"])] for r in chunk]))
    return torch.cat(preds), torch.cat(ys)


def load_fp32_and_bf16(model_dir: Path, stage_fp32: Path, stage_bf16: Path):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    m32 = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), dtype=torch.float32, attn_implementation="sdpa").cuda().eval()
    m16 = AutoModelForSequenceClassification.from_pretrained(
        str(stage_bf16), dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    return m32, m16, tok


def to_bf16(src: Path, dst: Path) -> int:
    state = load_file(str(src))
    n = 0
    for k, v in state.items():
        if v.is_floating_point():
            state[k] = v.to(torch.bfloat16)
            n += 1
    save_file(state, str(dst), metadata={"format": "pt"})
    return n


def model_card() -> str:
    return """# laya-ad v1 — 中文广告识别（二分类）

mmBERT-base（ModernBERT 架构，Apache-2.0）在自建中文群消息广告判定语料上微调的序列分类模型。数据不公开。

- 标签：`ad`（广告）/ `normal`（正常），id 映射见 config.json
- 底座：convaiinnovations/laya-multilingual 的 encoder（306.9M，max_position 8192）
- 指标：留出验证集 accuracy 0.970 / macro-F1 0.956
- 输入：单条群消息文本，建议 max_length 128-256（广告文本普遍较短）
- 变体：fp32（原始）/ bf16（推荐，体积减半精度持平）/ int8（torch 动态量化，CPU 部署）

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained("<本目录>")
model = AutoModelForSequenceClassification.from_pretrained(
    "<本目录>", dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
enc = tok(["文本"], return_tensors="pt", truncation=True, max_length=256).to("cuda")
probs = torch.softmax(model(**enc).logits.float(), -1)
```

int8 变体加载（需同版本 transformers 环境）：
```python
import torch
model = torch.load("quantized_int8.pt", weights_only=False, map_location="cpu").eval()
```

License: Apache-2.0（继承底座）
"""


def xz_compress(tar_path: Path, out_path: Path) -> None:
    for flag in (["-T0", "-9e"], ["-T2", "-9e"], ["-9e"]):
        r = subprocess.run(["xz"] + flag + ["-f", "-k", str(tar_path)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            shutil.move(str(tar_path) + ".xz", str(out_path))
            return
        print(f"    [xz] {flag} 失败：{r.stderr.strip()[:120]}，降档重试")
    raise RuntimeError("xz 全部失败")


def package_variant(name: str, stage: Path, out_dir: Path, metrics: dict) -> dict:
    allowed = ALLOWED_EXT | ({".pt"} if name.split("-")[-1] == "int8" else set())
    bad = [p.name for p in stage.iterdir() if p.suffix.lower() not in allowed]
    if bad:
        raise SystemExit(f"[ERR] {name} staging 出现白名单外文件：{bad}")

    lines = [f"laya-ad 发布包校验记录  variant={name}  "
             f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             "metrics: " + json.dumps(metrics, ensure_ascii=False), "", "sha256:"]
    for p in sorted(stage.iterdir()):
        lines.append(f"  {sha256(p)}  {p.name}  ({p.stat().st_size} bytes)")
    (stage / "VERIFICATION.txt").write_text("\n".join(lines), encoding="utf-8")

    archive = out_dir / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:  # 标准 gzip 压缩；权重高熵，极限档收益小
        for p in sorted(stage.iterdir()):
            tf.add(p, arcname=f"{name}/{p.name}")
    size = sum(p.stat().st_size for p in stage.iterdir())
    return {"archive": archive, "raw_mb": size / 2**20,
            "gz_mb": archive.stat().st_size / 2**20, "sha256": sha256(archive)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--int8", default=str(INT8))
    ap.add_argument("--val", default=str(VAL))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tag", default="v1.0")
    ap.add_argument("--precision", choices=["fp32", "bf16", "int8"], default=None,
                    help="只打某一档；缺省三档全打")
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--upload", action="store_true", help="上传 GitHub Release（gh CLI）")
    args = ap.parse_args()

    model_dir = Path(args.model)
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)  # 全量重建，旧产物不残留
    out_dir.mkdir(parents=True)
    precisions = [args.precision] if args.precision else ["fp32", "bf16", "int8"]
    base = f"laya-ad-{args.tag}"
    label2id = json.loads(Path("data/processed/label_map.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in open(args.val, encoding="utf-8")][:args.samples]

    results = {}

    # ---------- fp32 / bf16：GPU 校验 ----------
    if "fp32" in precisions or "bf16" in precisions:
        print("[组装] fp32 / bf16 staging ...")
        s32 = out_dir / f"{base}-fp32"
        s16 = out_dir / f"{base}-bf16"
        s32.mkdir(parents=True); s16.mkdir(parents=True)
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copy(model_dir / f, s32 / f)
            shutil.copy(model_dir / f, s16 / f)
        shutil.copy(model_dir / "model.safetensors", s32 / "model.safetensors")
        to_bf16(model_dir / "model.safetensors", s16 / "model.safetensors")
        (s32 / "MODEL_CARD.md").write_text(model_card(), encoding="utf-8")
        (s16 / "MODEL_CARD.md").write_text(model_card(), encoding="utf-8")

        print(f"[校验] fp32 vs bf16，{len(rows)} 条留出样本（GPU）...")
        from transformers import AutoTokenizer
        m32, m16, tok = load_fp32_and_bf16(model_dir, s32, s16)
        p32, y = run_val(m32, tok, rows, label2id, "cuda")
        p16, _ = run_val(m16, tok, rows, label2id, "cuda")
        rep32 = {"samples": len(rows), "accuracy": (p32 == y).float().mean().item()}
        rep16 = {"samples": len(rows), "accuracy": (p16 == y).float().mean().item(),
                 "agree_with_fp32": (p16 == p32).float().mean().item()}
        print(f"  fp32 acc={rep32['accuracy']:.4f} | bf16 acc={rep16['accuracy']:.4f} "
              f"一致率={rep16['agree_with_fp32']:.2%}")
        del m32, m16
        torch.cuda.empty_cache()
        if "fp32" in precisions:
            baseline = (s32 / "model.safetensors").stat().st_size / 2**20
            results["fp32"] = (package_variant(f"{base}-fp32", s32, out_dir, rep32), rep32, baseline)
            shutil.rmtree(s32)
        if "bf16" in precisions:
            results["bf16"] = (package_variant(f"{base}-bf16", s16, out_dir, rep16), rep16, baseline)
            shutil.rmtree(s16)

    # ---------- int8：CPU 校验 ----------
    if "int8" in precisions:
        print("[组装] int8 staging ...")
        s8 = out_dir / f"{base}-int8"
        s8.mkdir(parents=True)
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copy(model_dir / f, s8 / f)
        (s8 / "MODEL_CARD.md").write_text(model_card(), encoding="utf-8")
        shutil.copy(args.int8, s8 / "quantized_int8.pt")
        print(f"[校验] int8 vs fp32 参照精度，{len(rows)} 条（CPU）...")
        import transformers
        m8 = torch.load(s8 / "quantized_int8.pt", weights_only=False, map_location="cpu").eval()
        tok8 = transformers.AutoTokenizer.from_pretrained(str(s8))
        p8, y = run_val(m8, tok8, rows, label2id, "cpu")
        rep8 = {"samples": len(rows), "accuracy": (p8 == y).float().mean().item(),
                "reference_fp32_val_acc": 0.9701}
        print(f"  int8 acc={rep8['accuracy']:.4f}")
        del m8
        baseline = (s8 / "quantized_int8.pt").stat().st_size / 2**20
        results["int8"] = (package_variant(f"{base}-int8", s8, out_dir, rep8), rep8, baseline)
        shutil.rmtree(s8)

    # 精度结果已写入各包内 VERIFICATION.txt；results.json 作为随包机器可读凭证
    res_path = out_dir / f"{base}-results.json"
    res_path.write_text(json.dumps(
        {k: {"archive": v[0]["archive"].name, "raw_mb": round(v[0]["raw_mb"], 1),
             "gz_mb": round(v[0]["gz_mb"], 1), "sha256": v[0]["sha256"], "metrics": v[1]}
         for k, v in results.items()}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 打包汇总 ===")
    print(f"{'变体':<6}{'包内容':>10}{'tar.gz':>10}{'压缩率':>8}  文件")
    for k, (r, _, baseline) in results.items():
        print(f"{k:<6}{r['raw_mb']:>9.1f}M{r['gz_mb']:>9.1f}M{r['gz_mb'] / baseline:>7.0%}  "
              f"{r['archive'].name}")
    print(f"校验凭证: {res_path}")
    for k, (r, _, _) in results.items():
        print(f"  {r['archive'].name}  sha256 {r['sha256']}")

    if args.upload:
        files = [v[0]["archive"] for v in results.values()]
        fp32_archive = out_dir / f"laya-ad-{args.tag}-fp32.tar.gz"
        if fp32_archive.exists():
            stable = out_dir / "laya-ad-fp32.tar.gz"
            shutil.copy(fp32_archive, stable)
            files.append(stable)  # 稳定名资产：laya-server 镜像构建从 releases/latest/download 自动取最新
        files.append(res_path)
        r = subprocess.run(["gh", "release", "create", args.tag, *map(str, files),
                            "--title", f"laya-ad {args.tag}", "--notes-file", str(res_path)],
                           capture_output=True, text=True)
        print(r.stdout or r.stderr)


if __name__ == "__main__":
    main()
