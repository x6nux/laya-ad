# laya-server（laya-ad 独立维护分支）

- 基线来源：https://github.com/1Panel-dev/laya-server @ `19471b2d377ec29e85ac50ad65877f2937c5340b`（2026-10-10，Apache-2.0）
- **自本版本起完全独立维护，不跟随上游。** 上游 SHA 仅作出身记录。

## 本分支相对上游的改动

1. **ad-clf 后端**（`LAYA_BACKEND=ad-clf` 启用）：接入 laya-ad 微调的 mmBERT 中文广告二分类模型，
   通过 `/v1/systemone` Jev 兼容 API 提供判定；`is_ad` 形状题目（choice{ad,clean}、
   与广告相关的 noul）由分类器回答，其余题目返回 `unsupported: true` 的中性答案，
   调用方可走自有兜底，绝不冒充判断。
2. **全精度变体**：fp32/fp16/bf16/fp8/int8/int4 safetensors 变体（`quant_loader.py`
   统一加载，int8/int4 自动反量化）。镜像 tag 格式 **`<版本>-<精度>`**（如 `v1.0-fp32`）；
   **不带精度的 `latest` 与 `<版本>` tag = fp8**。原上游 Laya 模型权重机制
   （download-models / 多模型 Router）已移除；镜像构建时自动从
   `releases/latest/download/laya-ad-fp32.tar.gz` 拉取 laya-ad 最新基准权重并在镜像内量化。
3. **CPU 优化**（来自 laya-ad 工作流实测）：请求动态微批（`LAYA_AD_BATCH`/`LAYA_AD_BATCH_WINDOW_MS`）、
   CPU 线程数 `LAYA_CPU_THREADS`（默认逻辑核一半，Zen5 实测最优档）、bf16 推理。
   100 并发整批 ≤1s SLO 实测通过（Ryzen 7 9700X 半核）。

模型文件由 CI 构建时注入 `models/ad-clf/`（来源：laya-ad Release 的 fp32 基准资产 + 量化管线）。
