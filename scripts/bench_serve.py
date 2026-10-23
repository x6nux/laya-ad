#!/usr/bin/env python3
"""serve_cpu.py 压测：串行基线 / SLO 突发 / 100 并发持续 / 峰值内存 / 微批增益对照。

流程：
  1. 启动 scripts/serve_cpu.py 子进程，轮询 /health 至就绪，预热 3 条；
  2. 100 条串行基线（单条文本/请求，逐个发）：总耗时、串行 RPS、延迟；
  3. SLO 突发：100 条同时发出（Barrier 对齐），量"整批总完成时间"（从发起到最慢一条
     响应，即 max(单请求延迟)），重复 5 轮取中位与最差；硬性 SLO：≤1000ms 且零失败；
  4. 100 并发持续压 1000 条（100 线程 × 100 文本 × 10 轮）：RPS、p50/p95、失败数；
  5. 服务进程峰值内存：powershell Get-Process 的 PeakWorkingSet64（进程生命周期峰值，
     含模型权重与全部压测期间），单位 MB；SLO ≤ 3GB；
  6. 微批增益对照：关闭服务，以 --no-microbatch 重启（逐条前向），同口径再压 1000 条，
     batch_gain = 微批 RPS / 逐条 RPS；
  7. 关闭全部服务进程。

用法:
  python scripts/bench_serve.py                 # 完整压测
  python scripts/bench_serve.py --quick         # 快速冒烟（50 并发 200 条、突发 1 轮）
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def log(msg):
    print(f"[bench] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def start_server(extra_args, port):
    cmd = [PY, os.path.join(HERE, "serve_cpu.py"), "--port", str(port)] + extra_args
    log_path = os.path.join(tempfile.gettempdir(), f"serve_bench_{port}.log")
    logf = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT)
    proc.real_pid = None  # venv python.exe 是 trampoline，真实解释器 pid 从 /health 读
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 300:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                proc.real_pid = r.json().get("pid")
                log(f"服务就绪 (trampoline pid={proc.pid}, 解释器 pid={proc.real_pid}, "
                    f"{time.perf_counter() - t0:.1f}s) args={extra_args or '(默认 微批)'}")
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.terminate()
    tail = open(log_path, encoding="utf-8").read()[-2000:]
    raise RuntimeError(f"服务 300s 未就绪，日志 {log_path} 尾部:\n{tail}")


def stop_server(proc):
    if proc and proc.poll() is None:
        # taskkill /T 杀整棵进程树（trampoline + 真实解释器），避免孤儿占端口
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, timeout=30)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    log(f"服务进程已关闭 (pid={getattr(proc, 'real_pid', None) or (proc.pid if proc else '?')})")


def peak_working_set_mb(pid):
    """powershell Get-Process: PeakWorkingSet64 = 进程启动以来 WorkingSet 峰值（bytes）。"""
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-Process -Id {pid}).PeakWorkingSet64 / 1MB"],
        capture_output=True, text=True, timeout=30)
    return float(r.stdout.strip())


class Client:
    """每线程一个 Session（HTTP keep-alive）。"""

    def __init__(self, base):
        self.base = base
        self._local = threading.local()

    @property
    def session(self):
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
            self._local.s.headers["Content-Type"] = "application/json"
        return self._local.s

    def predict_one(self, text, timeout=30):
        """返回 (latency_ms, ok, err)。"""
        a = time.perf_counter()
        try:
            r = self.session.post(self.base + "/predict", json={"texts": [text]},
                                  timeout=timeout)
            ok = (r.status_code == 200 and len(r.json().get("results", [])) == 1)
            return (time.perf_counter() - a) * 1000, ok, None if ok else f"http {r.status_code}"
        except Exception as e:
            return (time.perf_counter() - a) * 1000, False, str(e)[:120]

    def predict_batch(self, texts, timeout=60):
        a = time.perf_counter()
        try:
            r = self.session.post(self.base + "/predict", json={"texts": texts},
                                  timeout=timeout)
            ok = (r.status_code == 200 and len(r.json().get("results", [])) == len(texts))
            return (time.perf_counter() - a) * 1000, ok, None if ok else f"http {r.status_code}"
        except Exception as e:
            return (time.perf_counter() - a) * 1000, False, str(e)[:120]


def pct(sorted_lat, p):
    return sorted_lat[min(len(sorted_lat) - 1, max(0, int(len(sorted_lat) * p) - 1))]


def summarize(lats, fails):
    s = sorted(lats)
    return {"p50_ms": round(pct(s, 0.5), 1), "p95_ms": round(pct(s, 0.95), 1),
            "fail": fails}


def serial_baseline(client, texts):
    log(f"[1/5] 串行基线：{len(texts)} 条逐个发 ...")
    lats, fails = [], 0
    t0 = time.perf_counter()
    for t in texts:
        lat, ok, _ = client.predict_one(t)
        lats.append(lat)
        fails += 0 if ok else 1
    total = time.perf_counter() - t0
    out = {"total_ms": round(total * 1000, 1), "rps": round(len(texts) / total, 2),
           **summarize(lats, fails)}
    log(f"      串行: {out['total_ms']}ms  RPS={out['rps']}  p50={out['p50_ms']}ms  "
        f"p95={out['p95_ms']}ms  fail={out['fail']}")
    return out


def burst_test(texts, client, rounds):
    """100 条同时发出，整批总完成时间 = max(单请求延迟)。
    线程池 + 每线程 Session keep-alive 连接复用（先跑 1 轮预热建连，不计入）。"""
    n = len(texts)
    log(f"[2/5] SLO 突发：{n} 条同时发出 × {rounds} 轮（另有 1 轮建连预热不计）...")
    lat = [[None] * n for _ in range(rounds)]
    ok = [[False] * n for _ in range(rounds)]
    ex = ThreadPoolExecutor(max_workers=n)
    bar = {}

    def one(i, rd):
        b0 = time.perf_counter()
        bar["barrier"].wait()  # 全员对齐后同时开跑
        l, o, _ = client.predict_one(texts[i])
        lat[rd][i] = (time.perf_counter() - b0) * 1000
        ok[rd][i] = o

    def run_round(rd):
        t0_box = {}
        bar["barrier"] = threading.Barrier(
            n + 1, action=lambda: t0_box.update(t0=time.perf_counter()))
        futs = [ex.submit(one, i, rd) for i in range(n)]
        bar["barrier"].wait()  # 主线程最后到，全员同时释放
        for f in futs:
            f.result()
        return (time.perf_counter() - t0_box["t0"]) * 1000

    run_round(-1)  # 建连预热轮
    results = []
    for rd in range(rounds):
        total_ms = run_round(rd)
        nf = sum(1 for o in ok[rd] if not o)
        results.append({"total_ms": round(total_ms, 1),
                        "slowest_ms": round(max(lat[rd]), 1), "fail": nf})
        log(f"      轮 {rd + 1}/{rounds}: 整批总完成={results[-1]['total_ms']}ms  "
            f"最慢单条={results[-1]['slowest_ms']}ms  fail={nf}")
        time.sleep(0.3)  # 轮间让服务队列彻底清空
    ex.shutdown()
    return results


def sustained_test(texts, client, total_reqs, concurrency):
    """concurrency 并发共 total_reqs 条（单文本/请求），返回 RPS 与延迟分布。"""
    log(f"[3/5] 持续压测：{concurrency} 并发共 {total_reqs} 条 ...")
    lat, fails = [], 0
    lock = threading.Lock()

    def job(i):
        nonlocal fails
        t = texts[i % len(texts)]
        l, ok, _ = client.predict_one(t)
        with lock:
            lat.append(l)
            if not ok:
                fails += 1

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(job, range(total_reqs)))
    elapsed = time.perf_counter() - t0
    out = {"elapsed_s": round(elapsed, 2), "rps": round(total_reqs / elapsed, 2),
           **summarize(lat, fails)}
    log(f"      持续: {elapsed:.1f}s  RPS={out['rps']}  p50={out['p50_ms']}ms  "
        f"p95={out['p95_ms']}ms  fail={out['fail']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--n-texts", type=int, default=100, help="压测文本条数（val 前 N 条）")
    ap.add_argument("--burst-rounds", type=int, default=5)
    ap.add_argument("--sustained-total", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=100)
    ap.add_argument("--out", default="outputs/serve_bench.json")
    ap.add_argument("--quick", action="store_true",
                    help="快速冒烟：文本50条、突发2轮、持续200条")
    args = ap.parse_args()
    if args.quick:
        args.n_texts, args.burst_rounds, args.sustained_total = 50, 2, 200

    with open(os.path.join(ROOT, args.val), encoding="utf-8") as f:
        texts = [json.loads(l)["text"] for l in f if l.strip()][:args.n_texts]
    log(f"压测文本: {args.val} 前 {len(texts)} 条（真实长度分布）")
    base = f"http://127.0.0.1:{args.port}"

    result = {"val": args.val, "n_texts": len(texts), "concurrency": args.concurrency,
              "sustained_total": args.sustained_total, "burst_rounds": args.burst_rounds}

    # ---- 第一个服务：默认微批 ----
    proc = start_server([], args.port)
    client = Client(base)
    client.predict_batch(["预热1", "预热2"])  # 触发首次前向
    try:
        result["serial_100"] = serial_baseline(client, texts)
        bursts = burst_test(texts, client, args.burst_rounds)
        result["burst"] = {"rounds": bursts,
                           "median_ms": round(statistics.median(b["total_ms"] for b in bursts), 1),
                           "worst_ms": max(b["total_ms"] for b in bursts),
                           "total_fail": sum(b["fail"] for b in bursts)}
        log(f"      突发汇总: 中位={result['burst']['median_ms']}ms  "
            f"最差={result['burst']['worst_ms']}ms  总失败={result['burst']['total_fail']}")
        result["sustained_1000"] = sustained_test(texts, client,
                                                  args.sustained_total, args.concurrency)
        mem_mb = peak_working_set_mb(proc.real_pid or proc.pid)
        result["mem_peak_working_set_mb"] = round(mem_mb, 1)
        log(f"[4/5] 服务进程峰值内存 (PeakWorkingSet64): {mem_mb:.0f} MB")
    finally:
        stop_server(proc)

    # ---- 第二个服务：--no-microbatch 对照 ----
    log("[5/5] 微批对照：--no-microbatch 重启服务 ...")
    proc = start_server(["--no-microbatch"], args.port + 1)
    client = Client(f"http://127.0.0.1:{args.port + 1}")
    try:
        client.predict_batch(["预热1", "预热2"])
        result["no_microbatch"] = sustained_test(texts, client,
                                                 args.sustained_total, args.concurrency)
    finally:
        stop_server(proc)

    result["batch_gain"] = round(result["sustained_1000"]["rps"] /
                                 max(result["no_microbatch"]["rps"], 1e-9), 3)

    b = result["burst"]
    result["slo"] = {
        "burst_median_ms": b["median_ms"],
        "burst_worst_ms": b["worst_ms"],
        "total_fail": b["total_fail"] + result["sustained_1000"]["fail"] + result["serial_100"]["fail"],
        "mem_mb": result["mem_peak_working_set_mb"],
        "burst_le_1000ms": b["worst_ms"] <= 1000,
        "mem_le_3g": result["mem_peak_working_set_mb"] <= 3072,
    }
    result["slo"]["met"] = (result["slo"]["burst_le_1000ms"] and
                            result["slo"]["total_fail"] == 0 and
                            result["slo"]["mem_le_3g"])
    log(f"SLO: burst最差={b['worst_ms']}ms(≤1000? {result['slo']['burst_le_1000ms']})  "
        f"失败={result['slo']['total_fail']}  内存={result['slo']['mem_mb']}MB(≤3072? "
        f"{result['slo']['mem_le_3g']})  达标={result['slo']['met']}  "
        f"微批增益={result['batch_gain']}x")

    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"结果已写 {args.out}")


if __name__ == "__main__":
    main()
