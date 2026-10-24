"""多精度模型下载管理（multi 镜像用）。

镜像内不带权重；管理员从模型目录（catalog）中选择精度下载：
- fp32 基准从 laya-ad Release 最新稳定名资产拉取（laya-ad-fp32.tar.gz，缓存复用）
- 在容器内按所选精度量化，落盘到 LAYA_MODEL_DIR/<变体目录>（含 MODEL_ID.txt/PRECISION.txt）
- 模型 id 规范：laya-<版本>-<精度>，版本取自 fp32 包内层目录名（laya-ad-v1.0 → v1.0）

状态查询：status()；下载在后台线程执行，同一时间一个任务。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import torch  # noqa: F401  # 提前导入确保量化子进程环境一致

BASE_URL = "https://github.com/x6nux/laya-ad/releases/latest/download/laya-ad-fp32.tar.gz"
CATALOG = ["fp32", "fp16", "bf16", "fp8", "int8", "int4"]
_QUANT_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "quantize_variants.py"


class ModelManager:
    def __init__(self, model_dir: Path, device: str | None):
        self.root = Path(model_dir)
        self.device = device or "cpu"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._status = {"state": "idle", "precision": None,
                        "started_at": None, "error": None}

    # ---------- 查询 ----------

    def status(self) -> dict:
        return dict(self._status)

    def installed(self) -> list[str]:
        out = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / "MODEL_ID.txt").is_file() and (d / "model.safetensors").is_file():
                out.append((d / "MODEL_ID.txt").read_text(encoding="utf-8").strip())
        return out

    def catalog(self) -> dict:
        return {"catalog": CATALOG, "installed": self.installed(),
                "status": self.status()}

    # ---------- 下载 ----------

    def download(self, precision: str) -> dict:
        if precision not in CATALOG:
            return {"accepted": False, "error": f"unknown precision {precision!r}"}
        with self._lock:
            if self._status["state"] == "downloading":
                return {"accepted": False, "error": "another download is running"}
            self._set("downloading", precision)
        threading.Thread(target=self._run, args=(precision,), daemon=True).start()
        return {"accepted": True, "precision": precision}

    def _set(self, state: str, precision: str | None = None, error: str | None = None):
        with self._lock:
            self._status = {"state": state, "precision": precision,
                            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "error": error}

    def _run(self, precision: str) -> None:
        try:
            base_dir = self._ensure_base()
            out_tmp = self.root / ".quantize-tmp"
            shutil.rmtree(out_tmp, ignore_errors=True)
            subprocess.run(
                [sys.executable, str(_QUANT_SCRIPT), "--src", str(base_dir),
                 "--out", str(out_tmp), "--variants", precision],
                check=True, capture_output=True, text=True)
            src = out_tmp / precision
            version = self._version
            variant_dir = self.root / f"laya-{version}-{precision}"
            shutil.rmtree(variant_dir, ignore_errors=True)
            shutil.move(str(src), str(variant_dir))
            (variant_dir / "MODEL_ID.txt").write_text(f"laya-{version}-{precision}", encoding="utf-8")
            (variant_dir / "PRECISION.txt").write_text(precision, encoding="utf-8")
            shutil.rmtree(out_tmp, ignore_errors=True)
            self._set("done", precision)
        except Exception as exc:  # 后台任务失败必须可见
            self._set("error", precision, f"{type(exc).__name__}: {exc}")

    def _ensure_base(self) -> Path:
        cache = self.root / ".cache"
        cache.mkdir(parents=True, exist_ok=True)
        tar_path = cache / "base.tar.gz"
        extract_root = cache / "base"
        if not tar_path.exists():
            self._log("downloading fp32 base from release")
            urllib.request.urlretrieve(BASE_URL, tar_path)
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract_root)
        base = next(p for p in extract_root.iterdir() if (p / "config.json").is_file())
        self._version = base.name.removeprefix("laya-ad-")
        self._log(f"base ready: {base} (version {self._version})")
        return base

    @staticmethod
    def _log(message: str) -> None:
        print(f"[model-manager] {message}", flush=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
