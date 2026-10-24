#!/usr/bin/env python3
"""multi 镜像入口：默认直接起服务（镜像不带权重，管理员经 /internal/models/catalog
自选精度下载）；设 LAYA_PRELOAD=fp8,int4 可在启动前预载指定精度，随后 exec uvicorn。"""
import os
import time

from server.model_manager import ModelManager


def main():
    preload = [p.strip() for p in os.environ.get("LAYA_PRELOAD", "").split(",") if p.strip()]
    if preload:
        mm = ModelManager(os.environ.get("LAYA_MODEL_DIR", "/data/models/ad-clf"),
                          os.environ.get("LAYA_DEVICE") or None)
        for p in preload:
            r = mm.download(p)
            print(f"[entrypoint] preload {p}: {r}", flush=True)
            if not r.get("accepted"):
                continue
            while mm.status()["state"] == "downloading":
                time.sleep(1)
            st = mm.status()
            print(f"[entrypoint] preload {p} -> {st['state']}", flush=True)
            if st.get("error"):
                print(f"[entrypoint] error: {st['error']}", flush=True)
    os.execvp("uvicorn", ["uvicorn", "server.main:app", "--host", "0.0.0.0",
                          "--port", "8080", "--workers", "1", "--proxy-headers"])


if __name__ == "__main__":
    main()
