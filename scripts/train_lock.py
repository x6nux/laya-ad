#!/usr/bin/env python3
"""微调互斥锁：防止同时训练 + 实时状态落盘 + 结束自动清理。

锁文件（默认 outputs/train.lock.json）是一个实时原子刷新的 JSON：

    {
      "phase": "training",            # starting/training/evaluating/saving/finalizing/interrupted
      "pid": 12345, "host": "...",
      "started_at": "...", "updated_at": "...",
      ...训练元信息与实时进度（epoch/global_step/loss/lr/gpu_mem_mb 等）
    }

规则：
- 启动时锁已存在且持锁 PID 存活 → RuntimeError 拒绝启动（调用方可用 --force 强占）
- 持锁进程已死 → 视为陈旧锁，自动接管
- 正常结束 / 异常 / Ctrl+C / 解释器退出（atexit）都会删除锁文件；
  taskkill 硬杀兜不住，靠下次启动的 PID 存活检查自动接管
"""
import atexit
import ctypes
import json
import os
import signal
import socket
import sys
import tempfile
from datetime import datetime

DEFAULT_LOCK = "outputs/train.lock.json"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _pid_alive(pid: int) -> bool:
    """检查进程是否存活。Windows 不能用 os.kill(pid, 0)——在 Windows 上
    os.kill 对非 CTRL 信号会直接 TerminateProcess，必须走 OpenProcess 查询。"""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        try:
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if k32.GetExitCodeProcess(h, ctypes.byref(exit_code)):
                    return exit_code.value == STILL_ACTIVE
                return False
            finally:
                k32.CloseHandle(h)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class TrainLock:
    """用法：lock = TrainLock(meta={...}); lock.acquire(); try: ... finally: lock.release()"""

    def __init__(self, path: str = DEFAULT_LOCK, meta: dict | None = None, force: bool = False):
        self.path = path
        self.meta = dict(meta or {})
        self.force = force
        self._held = False
        self._started_at = ""

    # ---------- 内部 ----------

    def _render(self, phase: str, extra: dict | None = None) -> str:
        doc = {
            "phase": phase,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_at": self._started_at,
            "updated_at": _now_iso(),
            **self.meta,
            **(extra or {}),
        }
        return json.dumps(doc, ensure_ascii=False, indent=2)

    def _write(self, text: str) -> None:
        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, self.path)  # 原子替换，轮询方不会读到半截 JSON
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    # ---------- 生命周期 ----------

    def acquire(self) -> "TrainLock":
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    old = json.load(f)
            except Exception:
                old = {}
            old_pid = old.get("pid", 0)
            alive = _pid_alive(old_pid)
            if alive and not self.force:
                raise RuntimeError(
                    f"已有微调在运行：pid={old_pid} phase={old.get('phase')} "
                    f"started={old.get('started_at')}（锁文件 {self.path}）。确认要强占请加 --force。"
                )
            reason = "--force 强占存活进程的锁" if alive else f"持锁进程 pid={old_pid} 已不存在（陈旧锁）"
            print(f"[lock] 发现旧锁，{reason}，接管。")
        self._started_at = _now_iso()
        self._write(self._render("starting"))
        self._held = True
        atexit.register(self._cleanup_atexit)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                pass  # 非主线程注册信号会 ValueError，跳过即可
        return self

    def release(self) -> None:
        """结束（含异常路径）删除锁文件。"""
        if self._held:
            self._held = False
            try:
                os.remove(self.path)
            except OSError:
                pass

    def update(self, phase: str, **extra) -> None:
        """实时状态刷新；写失败不中断训练。"""
        if not self._held:
            return
        try:
            self._write(self._render(phase, extra))
        except OSError:
            pass

    # ---------- 退出兜底 ----------

    def _on_signal(self, signum, frame):
        self.update("interrupted", signal=int(signum))
        raise KeyboardInterrupt  # 交给上层 finally 统一走 release

    def _cleanup_atexit(self):
        if self._held:
            self.release()
