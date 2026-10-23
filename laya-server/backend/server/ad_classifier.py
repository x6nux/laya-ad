"""广告分类 Predictor：把 laya-ad 微调的 mmBERT 二分类模型接入 Jev 兼容 API。

包含此前工作流验证的 CPU 优化：
- 动态微批：并发请求在 15ms 窗口内合批前向（单 worker 多线程下 100 并发整批 ≤1s SLO 的关键）
- CPU 线程数：LAYA_CPU_THREADS（默认物理核数 ≈ 逻辑核一半，Zen5 实测最优档）
- bf16 推理：Zen5 AVX-512 BF16 有效

支持的题目（其余标记 unsupported 并返回无信息中性答案，绝不冒充判断）：
- choice 且标签恰为 {ad, clean}（大小写不敏感）→ 分类器直接输出
- noul 且 instructions 提及广告（ad/advert/广告）→ "noul" 字段输出 P(is_ad)

usage.input_tokens 为真实 token 数；output_tokens = 题目数（每题一个决策位）。
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch

from .quant_loader import detect_precision, load_ad_model

_AD_RE = re.compile(r"ad|advert|广告", re.IGNORECASE)
_AD_LABELS = {"ad", "clean"}


def _extract_text(state: Any) -> str:
    """从 Jev state 中提取分类文本：字符串直取；dict 按 message.text/text/body/content 顺序；
    其余结构 JSON 序列化。"""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        msg = state.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("text"), str):
            return msg["text"]
        for key in ("text", "body", "content"):
            if isinstance(state.get(key), str):
                return state[key]
        return json.dumps(state, ensure_ascii=False)
    if isinstance(state, list):
        parts = []
        for turn in state:
            if isinstance(turn, dict) and isinstance(turn.get("text"), str):
                parts.append(turn["text"])
        return "\n".join(parts) if parts else json.dumps(state, ensure_ascii=False)
    return str(state)


def _labels(criteria: Any) -> list[str]:
    if isinstance(criteria, dict):
        return [str(k) for k in criteria]
    if isinstance(criteria, list):
        return [str(x) for x in criteria]
    return []


class _MicroBatcher:
    """请求合批器：把并发单条请求在窗口内合并成一次 padding 前向。"""

    def __init__(self, forward_fn: Callable[[list[str], int], list[float]],
                 max_batch: int, window_ms: int):
        self.forward_fn = forward_fn
        self.max_batch = max(1, max_batch)
        self.window_s = max(0, window_ms) / 1000.0
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._loop, daemon=True,
                                        name="ad-clf-microbatch")
        self._worker.start()

    def submit(self, text: str, seq_len: int) -> tuple[float, float]:
        item = {"text": text, "seq_len": seq_len, "event": threading.Event()}
        self._queue.put(item)
        item["event"].wait()
        if item.get("error") is not None:
            raise item["error"]
        return item["probs"]

    def _loop(self) -> None:
        while True:
            batch = [self._queue.get()]
            deadline = time.monotonic() + self.window_s
            while len(batch) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                seq_len = min(512, max(b["seq_len"] for b in batch))
                probs = self.forward_fn([b["text"] for b in batch], seq_len)
                for b, p in zip(batch, probs):
                    b["probs"] = p
                    b["event"].set()
            except Exception as exc:  # 单批失败只影响本批请求
                for b in batch:
                    b["error"] = exc
                    b["event"].set()


class AdClassifierPredictor:
    """实现与 LayaAdapter 相同的 Predictor 协议，后端为 laya-ad 二分类模型。"""

    def __init__(self, model_dir: Path, device: str | None):
        threads = int(os.environ.get("LAYA_CPU_THREADS",
                                     str(max(1, (os.cpu_count() or 2) // 2))))
        if torch.cuda.is_available() and (device or "").startswith("cuda"):
            self.device = "cuda"
        else:
            self.device = device or "cpu"
        torch.set_num_threads(threads if self.device == "cpu" else max(4, threads // 2))
        self.model_dir = Path(model_dir)
        self.model, self.tokenizer = load_ad_model(self.model_dir, self.device)
        self.precision = detect_precision(self.model_dir)
        id2label = {int(k): str(v).casefold() for k, v in self.model.config.id2label.items()}
        self.ad_id = next((i for i, name in id2label.items() if name == "ad"), None)
        self.clean_id = next((i for i, name in id2label.items() if name == "normal"), None)
        if self.ad_id is None or self.clean_id is None:
            raise RuntimeError(f"模型 id2label 缺少 ad/normal: {id2label}")
        self.model_name = f"laya-ad-clf-{self.precision}"
        self._batcher = _MicroBatcher(self._forward_batch,
                                      int(os.environ.get("LAYA_AD_BATCH", "32")),
                                      int(os.environ.get("LAYA_AD_BATCH_WINDOW_MS", "15")))

    def _forward_batch(self, texts: list[str], seq_len: int) -> list[tuple[float, float]]:
        enc = self.tokenizer(texts, truncation=True, max_length=seq_len,
                             padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1)
        return [(float(p[self.ad_id]), float(p[self.clean_id])) for p in probs]

    def predict(self, state: Any, questions: dict[str, Any], model: str | None = None,
                *, max_len: int | None = None) -> dict[str, Any]:
        text = _extract_text(state)
        seq_len = min(int(max_len) if max_len else 512, 512)
        p_ad, p_clean = self._batcher.submit(text, seq_len)
        pred_label = "ad" if p_ad >= p_clean else "clean"

        answers: dict[str, Any] = {}
        for qid, q in questions.items():
            qtype = q.get("type")
            labels = _labels(q.get("criteria"))
            if qtype == "choice" and {l.casefold() for l in labels} == _AD_LABELS:
                prob_map = {}
                for label in labels:
                    prob_map[label] = round(p_ad if label.casefold() == "ad" else p_clean, 4)
                answers[qid] = {
                    "type": "choice",
                    "choice": pred_label if pred_label in [l.casefold() for l in labels] else labels[0],
                    "probabilities": prob_map,
                    "confidence": round(max(p_ad, p_clean), 4),
                    "answer_confidence": round(max(p_ad, p_clean), 4),
                    "unsupported": False,
                }
            elif qtype == "noul" and _AD_RE.search(str(q.get("instructions", ""))):
                # noul 约定：true = 是广告（仅当题目语义与广告相关时才有意义）
                answers[qid] = {
                    "type": "noul",
                    "noul": round(p_ad, 4),
                    "confidence": round(max(p_ad, 1.0 - p_ad), 4),
                    "answer_confidence": round(max(p_ad, 1.0 - p_ad), 4),
                    "unsupported": False,
                }
            else:
                answers[qid] = self._unsupported(q)

        usage = {"input_tokens": self._count_tokens(text, seq_len) * len(questions),
                 "output_tokens": len(questions)}
        return {"answers": answers, "model": self.model_name, "usage": usage}

    def _count_tokens(self, text: str, seq_len: int) -> int:
        enc = self.tokenizer(text, truncation=True, max_length=seq_len)
        return int(len(enc["input_ids"]))

    @staticmethod
    def _unsupported(q: dict[str, Any]) -> dict[str, Any]:
        """无信息中性答案：均匀概率 + 低置信 + unsupported 标记，绝不冒充判断。"""
        qtype = q.get("type")
        if qtype == "choice":
            labels = _labels(q.get("criteria")) or ["unknown"]
            n = len(labels)
            return {"type": "choice", "choice": labels[0],
                    "probabilities": {l: round(1.0 / n, 4) for l in labels},
                    "confidence": round(1.0 / n, 4), "answer_confidence": round(1.0 / n, 4),
                    "unsupported": True}
        if qtype == "score":
            n = len(q.get("criteria") or [])
            uniform = 1.0 / max(1, n)
            return {"type": "score", "score": round(sum(i * uniform for i in range(n)), 4),
                    "legend": {str(i): str(c) for i, c in enumerate(q.get("criteria") or [])},
                    "probabilities": {str(i): round(uniform, 4) for i in range(n)},
                    "confidence": round(uniform, 4), "answer_confidence": round(uniform, 4),
                    "unsupported": True}
        # noul（与广告无关的是非题）：0.5 = 无信息
        return {"type": "noul", "noul": 0.5, "confidence": 0.5, "answer_confidence": 0.5,
                "unsupported": True}
