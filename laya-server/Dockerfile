# syntax=docker/dockerfile:1
FROM node:24-bookworm-slim AS frontend-build
WORKDIR /app/frontend
RUN npm install -g pnpm@11.19.0
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN CI=true pnpm install --frozen-lockfile
COPY frontend/ ./
RUN CI=true pnpm build

FROM python:3.12-slim AS app
# PRECISION: fp32 | fp16 | bf16 | fp8 | int8 | int4 —— 镜像内权重精度（latest 默认 fp8）
ARG PRECISION=fp8
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    LAYA_DATABASE_PATH=/data/laya.sqlite3 \
    LAYA_MODEL_DIR=/opt/models/ad-clf \
    LAYA_FRONTEND_DIR=/app/frontend/dist
WORKDIR /app
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers==5.20.2 "safetensors>=0.6.0"
COPY backend/ /app/backend/
RUN pip install --no-cache-dir --no-deps /app/backend && \
    pip install --no-cache-dir fastapi==0.141.1 uvicorn==0.53.0 argon2-cffi==25.1.0 pydantic==2.13.5
COPY scripts/quantize_variants.py /app/scripts/quantize_variants.py
# 权重自动从 laya-ad Release 最新版下载（稳定名资产 laya-ad-fp32.tar.gz），镜像内按 PRECISION 量化
RUN python - <<'PY'
import pathlib
import tarfile
import urllib.request

url = "https://github.com/x6nux/laya-ad/releases/latest/download/laya-ad-fp32.tar.gz"
print("downloading", url)
urllib.request.urlretrieve(url, "/tmp/base.tar.gz")
with tarfile.open("/tmp/base.tar.gz") as tf:
    tf.extractall("/tmp/base")
src = next(p for p in pathlib.Path("/tmp/base").iterdir() if (p / "config.json").is_file())
pathlib.Path("/tmp/src.txt").write_text(str(src))
print("src:", src)
PY
RUN python scripts/quantize_variants.py --src "$(cat /tmp/src.txt)" --out /tmp/qv \
      --variants "${PRECISION}" \
    && mkdir -p /opt/models \
    && cp -r "/tmp/qv/${PRECISION}" /opt/models/ad-clf \
    && echo "${PRECISION}" > /opt/models/ad-clf/PRECISION.txt
COPY --from=frontend-build /app/frontend/dist/ /app/frontend/dist/
RUN mkdir -p /data && python -c "from server.ad_classifier import AdClassifierPredictor; p = AdClassifierPredictor('/opt/models/ad-clf', device='cpu'); r = p.predict('测试广告文本 加微信领取福利，名额有限', {'is_ad': {'type': 'choice', 'instructions': '判断是否广告', 'criteria': {'ad': '广告', 'clean': '正常'}}}); assert r['answers']['is_ad']['unsupported'] is False; print('smoke ok:', r['model'], r['answers']['is_ad']['choice'])"
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health/live', timeout=3)" || exit 1
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]
