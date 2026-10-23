#!/usr/bin/env python3
"""mmBERT 广告识别 CPU 推理服务（仅标准库 HTTP + 请求队列 + 微批 worker）。

模型：outputs/ad-clf-int8/quantized.pt（int8 动态量化；采纳结论 outputs/cpu_quant_bench.json：
int8 acc=0.966 vs fp32 0.968、speedup=4.13x、accepted=true）。
加载方法沿 scripts/bench_cpu.py:22-26 注释：必须先 import transformers 让 ModernBert* 类
可反序列化，再 torch.load(..., weights_only=False) 加载完整 pickle 的 nn.Module。

接口（默认 127.0.0.1:8765）：
  GET  /health    -> {"status": "ok", ...}   服务与模型就绪状态
  POST /predict   body {"texts": ["..."]}    一次可传多条文本（一次性多判断），
                  -> {"results": [{"label": "ad"|"normal", "probs": [p_ad, p_normal]}, ...]}
                  结果按 texts 输入顺序返回；单条文本在服务内部合并进微批前向。

微批设计：
  - HTTP 线程只做 tokenize（不 padding），把"单条文本任务"推入内部队列后等待结果；
  - 微批 worker 凑批：拿到第一个任务后开 ≤20ms 凑批窗（--batch-wait-ms），
    窗内凑满 max_batch（--max-batch，默认 32）或窗关即执行；
  - 长度分桶（--short-bucket-max，默认 64 token）：短桶/长桶各一个微批 worker——
    动态 padding 的批长由批内最长文本决定，混桶时长文本会把整批拖长
    （实测 batch=32: seq≈26 -> 154ms vs seq≈114 -> 867ms）；
  - 批内动态 padding 到批内最长（截断上限 --max-length，默认 128 = bench_cpu.py 量化
    正确率口径 acc=0.966；val n=2077 分布 median=21/p90=78/p95=111），不固定 512 ——
    广告文本平均很短（median=21），长截断会白付数倍 seq 算力；
  - --no-microbatch：关闭合并，每条文本单独 batch=1 前向（压测对照模式）。

线程：torch.set_num_threads(物理核数)；OMP/MKL 等 env 必须在 import torch 之前设置
（同 scripts/bench_cpu.py:47-58）。依据 scripts/bench_threads_scaling.py 实测
（AMD 9700X 8 物理核，int8 形态、seq128/batch8，3 轮中位）：
1/2/4/8 线程 = 14.2/23.1/37.1/58.7 样本/s，满物理核 8 比半核 4 快 +58%，
半核档差 36.8%；12/16（SMT 逻辑核）超过物理核数，不在有效档。
注意：端到端 serve 压测（bench_serve.py，100 并发持续）4 与 8 线程持平（118 vs 119 RPS）——
持续高并发下 3 个微批 worker 已饱和 CPU；8 线程收益在单 worker 前向
（seq128/batch8 +58%，低并发/串行大文本受益），代价是串行小请求 p50 约 +3.5ms
（小 GEMM 的线程同步开销）。两者 SLO 均达标。
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)  # 无 psutil 时按超线程折半兜底


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="outputs/ad-clf-int8/quantized.pt")
    ap.add_argument("--tokenizer", default="outputs/ad-clf/best")
    ap.add_argument("--max-batch", type=int, default=48, help="微批最大条数")
    ap.add_argument("--batch-wait-ms", type=float, default=20.0, help="凑批等待窗（ms）")
    ap.add_argument("--max-length", type=int, default=128,
                    help="token 截断上限（= bench_cpu.py 量化正确率口径 acc=0.966；"
                         "val n=2077: median=21/p90=78/p95=111，128 保住 96% 文本完整；"
                         "padding 始终为批内最长，见 --short-bucket-max 与排序出批）")
    ap.add_argument("--max-chars", type=int, default=20000, help="单条文本字符数上限（超限 400）")
    ap.add_argument("--max-texts", type=int, default=128, help="单请求 texts 条数上限（超限 400）")
    ap.add_argument("--short-bucket-max", type=int, default=64,
                    help="长度分桶阈值（token）：≤该值的进短桶、>进长桶，长桶 1 个微批 worker，"
                         "避免长文本把整批 padding 拖长；0 = 不分桶（单队列）")
    ap.add_argument("--short-workers", type=int, default=2,
                    help="短桶微批 worker 数（默认 2：与长桶合计 3 个 worker，"
                         "burst 瞬时并行换整批 wall time；每个 worker 的前向用满 "
                         "THREADS=物理核数个 intra-op 线程，持续吞吐由 CPU 总量决定）")
    ap.add_argument("--no-microbatch", action="store_true",
                    help="逐条前向对照模式：不合并批，每条文本单独前向")
    return ap.parse_args()


ARGS = parse_args()

# ---- 线程环境变量：必须先于 import torch / transformers ----
# 实测（scripts/bench_threads_scaling.py，int8 seq128/batch8）：8 物理核满开 58.7 sps
# 优于半核 4 线程 37.1 sps（+58%），scaling 在物理核内单调上升，无拐点回落。
PHYS_CORES = _physical_cores()
THREADS = PHYS_CORES
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = str(THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

torch.set_num_threads(THREADS)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def log(msg):
    print(f"[serve] {time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---- 模型加载 ----
t0 = time.perf_counter()
import transformers  # noqa: E402  必须先于 torch.load（ModernBert* 反序列化注册）

tok = AutoTokenizer.from_pretrained(ARGS.tokenizer)
model = torch.load(ARGS.model, map_location="cpu", weights_only=False)
model.eval()
ID2LABEL = {int(k): v for k, v in model.config.id2label.items()}
NUM_LABELS = model.config.num_labels
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else 0
log(f"模型加载完成 {time.perf_counter() - t0:.1f}s  model={ARGS.model}  "
    f"labels={ID2LABEL}  threads={torch.get_num_threads()}  "
    f"max_batch={ARGS.max_batch}  batch_wait_ms={ARGS.batch_wait_ms}  "
    f"max_length={ARGS.max_length}  microbatch={not ARGS.no_microbatch}")

# ---- 微批核心 ----
# 任务: (input_ids: list[int], slot: dict) ；slot["done"]=Event, slot["probs"]=list[float]
# 分桶：短桶（≤SHORT_BUCKET_MAX token）/ 长桶两个队列，各一个微批 worker——
# 动态 padding 的批长由批内最长文本决定，混桶时长文本会把短文本的整批拖长，
# 实测 batch=32: seq≈26 -> 154ms vs seq≈114 -> 867ms，分桶后短批吞吐差 5 倍以上。
TASKQ_SHORT: "queue.Queue[tuple[list[int], dict]]" = queue.Queue()
TASKQ_LONG: "queue.Queue[tuple[list[int], dict]]" = queue.Queue()

SHORT_BUCKET_MAX = ARGS.short_bucket_max


def _forward_batch(tasks):
    """一批任务合并前向，批内动态 padding 到最长，就地写回 slot。"""
    ids_list = [t[0] for t in tasks]
    maxl = max(len(x) for x in ids_list)
    input_ids = torch.tensor([x + [PAD_ID] * (maxl - len(x)) for x in ids_list],
                             dtype=torch.long)
    attention_mask = torch.tensor([[1] * len(x) + [0] * (maxl - len(x)) for x in ids_list],
                                  dtype=torch.long)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    probs = torch.softmax(logits.float(), dim=-1)
    for (_, slot), p in zip(tasks, probs):
        slot["probs"] = p.tolist()
        slot["done"].set()


def _forward_single(task):
    """逐条前向（--no-microbatch 对照）。"""
    ids, slot = task
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    slot["probs"] = torch.softmax(logits.float(), dim=-1)[0].tolist()
    slot["done"].set()


def worker(q):
    """微批 worker：凑批窗 ≤batch_wait_ms；缓冲池按 token 长度排序出批——
    动态 padding 的批长由批内最长文本决定，排序出批让同批长度相近，padding 损耗最小；
    2s 老化优先防长任务被短流饿死。"""
    # inference_mode 是线程本地的 RAII，包住 worker 整个生命周期
    with torch.inference_mode():
        buf = []
        while True:
            if ARGS.no_microbatch:
                _forward_single(q.get())
                continue
            if not buf:
                buf.append(q.get())
            deadline = time.perf_counter() + ARGS.batch_wait_ms / 1000.0
            while len(buf) < ARGS.max_batch:
                remain = deadline - time.perf_counter()
                if remain <= 0:
                    break
                try:
                    buf.append(q.get(timeout=remain))
                except queue.Empty:
                    break
            oldest = min(buf, key=lambda t: t[1]["arrival"])
            if time.perf_counter() - oldest[1]["arrival"] > 2.0:
                # 老化：以最老任务长度为中心取最接近的 max_batch 条（必含最老）
                anchor = len(oldest[0])
                buf.sort(key=lambda t: abs(len(t[0]) - anchor))
                take = buf[:ARGS.max_batch]
                for t in take:
                    buf.remove(t)
            else:
                buf.sort(key=lambda t: len(t[0]))
                take = buf[:ARGS.max_batch]
                del buf[:ARGS.max_batch]
            _forward_batch(take)


def start_workers():
    if ARGS.no_microbatch or SHORT_BUCKET_MAX <= 0:
        # 不分桶/逐条：单队列单 worker
        threading.Thread(target=worker, args=(TASKQ_SHORT,), daemon=True,
                         name="mbatch-worker").start()
        return
    for i in range(max(1, ARGS.short_workers)):
        threading.Thread(target=worker, args=(TASKQ_SHORT,), daemon=True,
                         name=f"mbatch-worker-short{i}").start()
    threading.Thread(target=worker, args=(TASKQ_LONG,), daemon=True,
                     name="mbatch-worker-long").start()


start_workers()


# ---- HTTP 层 ----
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive，压测可复用连接
    server_version = "laya-ad-serve/1.0"

    def log_message(self, fmt, *args):  # 静默逐请求日志，压测不刷屏
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "pid": os.getpid(),  # 真实解释器 pid（venv python.exe 是 trampoline，Popen.pid 不是它）
                "model": ARGS.model,
                "labels": [ID2LABEL[i] for i in range(NUM_LABELS)],
                "torch_threads": torch.get_num_threads(),
                "physical_cores": PHYS_CORES,
                "max_batch": ARGS.max_batch,
                "batch_wait_ms": ARGS.batch_wait_ms,
                "max_length": ARGS.max_length,
                "short_bucket_max": SHORT_BUCKET_MAX,
                "microbatch": not ARGS.no_microbatch,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/predict":
            self._send_json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            texts = req["texts"]
            if not isinstance(texts, list):
                raise ValueError("texts 必须是列表")
            texts = [str(t) for t in texts]
            if not texts:
                raise ValueError("texts 不能为空")
            if len(texts) > ARGS.max_texts:
                raise ValueError(f"texts 超过单请求上限 {ARGS.max_texts}")
            if any(len(t) > ARGS.max_chars for t in texts):
                raise ValueError(f"单条文本超过 {ARGS.max_chars} 字符上限")
        except Exception as e:
            self._send_json(400, {"error": str(e)})
            return

        # 请求线程 tokenize 后逐条入队（按长度分桶），等各自 slot 完成，按原顺序组装
        bucketed = (not ARGS.no_microbatch) and SHORT_BUCKET_MAX > 0
        slots = []
        for t in texts:
            ids = tok(t, truncation=True, max_length=ARGS.max_length)["input_ids"]
            slot = {"done": threading.Event(), "probs": None,
                    "arrival": time.perf_counter()}
            if bucketed and len(ids) > SHORT_BUCKET_MAX:
                TASKQ_LONG.put((ids, slot))
            else:
                TASKQ_SHORT.put((ids, slot))
            slots.append(slot)
        for s in slots:
            s["done"].wait()
        results = []
        for s in slots:
            p = s["probs"]
            results.append({"label": ID2LABEL[max(range(len(p)), key=p.__getitem__)],
                            "probs": p})
        self._send_json(200, {"results": results})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256  # socketserver 默认 backlog=5，100 并发瞬时连接会被拒


def main():
    srv = Server((ARGS.host, ARGS.port), Handler)
    log(f"监听 http://{ARGS.host}:{ARGS.port}  (/health, /predict)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
