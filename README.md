# laya-ad — 中文区广告识别微调

基于 [Laya](https://huggingface.co/convaiinnovations/laya-multilingual)（mmBERT/ModernBERT 底座，322M，Apache-2.0）微调的中文广告识别模型，生产口径二分类（`ad` / `normal`）。

## 底座情况（公开模型事实，决定技术路线）

- Laya **不是生成式 LLM**，是非自回归分类决策模型：一次前向输出带概率的答案（实测 ~15-40ms/决策）
- 中文用 `convaiinnovations/laya-multilingual`（mmBERT-base，官方支持 `zh`，max_len=1024）
- 官方明说 zero-shot 接近随机、**能力来自微调**，但未提供微调脚本——本仓库自建管线
- 其 encoder 是标准 ModernBERT → 抽出后直接 `AutoModelForSequenceClassification` 标准分类头微调（本仓库主路线）；typed-questions 决策头形态保留为可选路线

## 环境

- GPU: RTX 5060 Ti 16GB（sm_120）→ **torch 必须 cu128 轮子**（本机 torch 2.11.0+cu128）
- Python 3.12 + uv venv（`.venv/`），transformers 5.20.2 / datasets 5.1.0 / laya 0.4.14

## 目录结构

```
configs/train_ad.yaml          # 推荐超参（CLI 显式参数优先于 yaml）
data/raw/                      # 本地（不入库）：内部数据集原始包、清洗产物、生产系统提示词
data/processed/                # 本地（不入库）：train/val/label_map（prepare_data 再生）
scripts/
  convert_dataset.py           # 内部数据集清洗转换（口径修正/去重裁决/双标签方案）
  prepare_data.py              # 语料切分 train/val + label_map 生成
  dataset_conflicts.py         # 多 sighting/矛盾文本明细导出
  laya_judge_conflicts.py      # 零样本 laya 对照判定
  extract_encoder.py           # 从 Laya checkpoint 抽取 encoder 底座
  train_ad.py                  # 微调主脚本（bf16 AMP + SDPA + 动态padding + 互斥锁）
  train_lock.py                # 微调互斥锁模块
  eval_ad.py                   # accuracy / P-R-F1 / 混淆矩阵 / 逐条概率
  predict.py                   # 推理演示（单条/jsonl/txt）
  benchmark.py                 # GPU 吞吐基准
  bench_cpu.py                 # CPU int8 量化基准
  bench_threads_scaling.py     # CPU 线程 scaling
  serve_cpu.py                 # 并发微批推理服务（HTTP）
  bench_serve.py               # 服务压测（SLO 突发 + 持续 RPS + 内存）
  test_instructions_limits.py  # instructions 字段边界测试
  test_production_format.py    # 生产请求格式端到端测试
  test_head_budget.py          # head 预算三档对比实验
models/  outputs/              # 本地（不入库）：权重与训练/压测产物
```

## 快速开始

```bash
source .venv/Scripts/activate   # Git Bash；PowerShell 用 .venv\Scripts\Activate.ps1

# 1. 数据（内部数据集已转换就位；换新语料时重跑转换与切分）
python scripts/prepare_data.py --input data/raw/dataset/converted/<子集>.jsonl --val_ratio 0.1

# 2. 微调（互斥锁防并发，进度实时写 outputs/train.lock.json）
python scripts/train_ad.py --config configs/train_ad.yaml

# 3. 评估 + 预测
python scripts/eval_ad.py --data data/processed/val.jsonl
python scripts/predict.py --text "加微信免费领取涨粉秘籍，名额有限！"
```

标签体系数据驱动（`label_map.json` 对齐），当前版本二分类 `ad` / `normal`。

### 量化变体（Release v1.0，CI 自动产出）

fp32 基准权重随 Release 发布（866MB），FP16/BF16/FP8/INT8/INT4 由 GitHub Actions 从 fp32 自动量化回传（`.github/workflows/quantize-release.yml`）。本地 val 集精度实测（bf16 计算，详见 Release 内 `variant-eval.json`）：

| 变体 | accuracy | macro-F1 | 与基准一致率 | 体积（tar.gz） |
|---|---|---|---|---|
| fp32（基准权重） | 97.01% | 0.956 | — | 865MB |
| fp16 | 97.06% | 0.957 | 100% | 545MB |
| bf16 | 97.06% | 0.957 | 100% | 468MB |
| fp8 (e4m3) | 97.06% | 0.957 | 100% | 228MB |
| int8 (per-channel) | 97.11% | 0.957 | 99.95% | 273MB |
| int4 (group=64) | 96.82% | 0.954 | 99.28% | 141MB |

fp8/int8/int4 通过 `scripts/load_quantized.py` 的 `load_variant()` 加载（自动识别格式并反量化）；int8/int4 属有损量化，精度以自有数据验证为准。

### 服务化镜像（laya-server 分支，ghcr）

`laya-server/` 为独立维护分支（基线 1Panel-dev/laya-server @ 19471b2d）：Jev 兼容 `/v1/systemone` API + Web 控制台，后端接 ad-clf 分类器，内置动态微批 / CPU 半核线程 / bf16 推理（100 并发整批 ≤1s SLO 那套）。镜像构建时自动从 Release 最新权重在镜像内量化：

```bash
# latest 与 v1.0 = fp8；其余 tag：v1.0-fp32 / fp16 / bf16 / fp8 / int8 / int4
docker run -d -p 8080:8080 -e LAYA_ADMIN_USERNAME=admin -e LAYA_ADMIN_PASSWORD='改成你的密码' \
  ghcr.io/x6nux/laya-server:latest

curl -X POST http://127.0.0.1:8080/v1/systemone \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"state":{"message":{"text":"加微信免费领取涨粉秘籍，名额有限！"}},"questions":{"is_ad":{"type":"choice","instructions":"判断是否广告","criteria":{"ad":"广告","clean":"正常"}}}}'
```

发新 Release 后 CI 自动重建全部精度镜像（`.github/workflows/docker-release.yml`）。

## v1 性能对比

| 项 | 微调后 | 对照 |
|---|---|---|
| val 泛化 acc / macro-F1（训练未见样本） | **97.0% / 0.956** | 零样本 laya：接近随机（抛硬币水平） |
| 多 sighting 一致性子集 | 93.3% | 零样本 laya 同集 66.7% |
| 生产格式零样本探针（含长政策指令） | — | 49%（抛硬币），且多问之间互相矛盾 |
| 训练成本 | 全量 3 epochs 约 55 分钟（单卡） | — |

微调后对生产格式明摆着的广告样本高置信判对，零样本则判错且各问打架——**微调路线是唯一可用路线**。

### GPU 吞吐（RTX 5060 Ti 16GB，bf16 + SDPA）

| 配置 | 吞吐 | p50 延迟 | VRAM |
|---|---|---|---|
| seq=128, batch=32（峰值） | **685 样本/s** | 46.6ms | 0.68GB |
| seq=128, batch=8 | 559 样本/s | 13.9ms | 0.61GB |
| seq=512, batch=8 | 152 样本/s | 52.6ms | 0.68GB |
| seq=512, batch=128 | 134 样本/s | 957.9ms | 2.03GB |

显存余量巨大（峰值 <2.1GB / 16GB），瓶颈在 kernel 启动开销；`max_len` 降到 128/256 是最直接的提速手段。推理侧官方 `predict_batch` 可多 state 合并前向。

### CPU 部署（int8 量化 + 并发微批服务 + AMD 调优）

| 项 | 实测 |
|---|---|
| int8 动态量化（Linear→qint8） | acc 仅 -0.20pt，**加速 4.13x**（采纳；产物 `outputs/ad-clf-int8/`） |
| 并发服务 SLO | **100 并发整批 929ms（≤1s 达标）**，无失败请求 |
| 持续压测 / 内存 | RPS 120；峰值内存 2431MB（目标 ≤3GB） |
| AMD（Ryzen 7 9700X，Zen 5，8C16T） | 最优 **8 线程 = 一半逻辑核**（4 线程档慢 37%）；线程曲线 1→14 / 2→23 / 4→37 / 8→59 样本/s；**bf16 在 Zen 5 有效**；最优配置已写回服务默认值 |

服务用法：`python scripts/serve_cpu.py`（127.0.0.1:8765）→ `POST /predict` body `{"texts": [...]}` 支持多文本单请求，`GET /health` 探活；压测复现 `python scripts/bench_serve.py`。

## 指令预算实测（Laya 输入格式机制）

- typed-question 的 instructions 有效预算 = `head_max_len`(256) − 选项渲染 token，下限 8 token；空串直接报错；超长**静默截断**且按 token 计（5k 与 20k 字结果逐位一致、延迟不变）
- 生产请求格式的系统提示词约 3k token → **~91% 被截**，且每次截留片段不一致——**判定政策必须学进权重，不要塞 instructions**
- `head_max_len`/`max_len` 是推理时参数可放大（256→1024→3200 实测）：零样本表现纹丝不动、延迟 17~43 倍——**放大预算只有配合重训才有意义**
- 富 state 的剩余空间按 `max_len − head − 1` 计，超长静默丢弃，工程上用 `build_sequence(return_truncation_stats=True)` 做 token 级校验

## 微调互斥锁（scripts/train_lock.py）

- 启动创建 `outputs/train.lock.json`（原子刷新 JSON：phase/epoch/step/loss/lr/gpu_mem），**防同时微调**：第二实例 exit 1，`--force` 强占
- 结束（正常/异常/Ctrl+C/atexit）自动删除；taskkill 硬杀靠 PID 存活检查在下次启动自动接管
- Windows 注意：查存活不能用 `os.kill(pid, 0)`（会 TerminateProcess），已用 `OpenProcess` 实现

## 已知限制与下一步

- 一致性子集含训练集文本，93.3% 是一致性指标，泛化看 val 的 97.0%
- 资料审计类样本与普通消息分布不同，数据侧保留 path 字段可分层采样
- 未做（NOT RUN）：GPU 侧量化、Route B（决策头微调，需把判定政策学进权重）、服务鉴权与 TLS（当前仅本机联调）、矛盾标注组人工终审
