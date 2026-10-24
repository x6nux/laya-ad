"""ad-clf 后端端到端测试：微型随机权重模型走完整 API 流程（不依赖真实权重/数据）。"""
from pathlib import Path

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from server.config import Settings
from server.main import create_app

PASSWORD = "correct horse battery staple"
ORIGIN = {"Origin": "http://testserver"}


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    """微型 ModernBERT（2 层 × 32 hidden）+ 真实 tokenizer，落成 fp32 safetensors 目录。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, ModernBertConfig

    d = tmp_path_factory.mktemp("tiny-model")
    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    tok.save_pretrained(d)
    cfg = ModernBertConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=2,
                           max_position_embeddings=128,
                           pad_token_id=tok.pad_token_id or 0,
                           bos_token_id=tok.cls_token_id or 0,
                           eos_token_id=tok.sep_token_id or 0,
                           cls_token_id=tok.cls_token_id or 0,
                           mask_token_id=None)
    model = AutoModelForSequenceClassification.from_config(cfg)
    model.config.id2label = {0: "ad", 1: "normal"}
    model.config.label2id = {"ad": 0, "normal": 1}
    model.save_pretrained(d, safe_serialization=True)
    (d / "PRECISION.txt").write_text("fp8", encoding="utf-8")
    (d / "MODEL_ID.txt").write_text("laya-test-fp8", encoding="utf-8")
    return d


def _settings(model_dir: Path, tmp_path: Path) -> Settings:
    frontend = tmp_path / "frontend-dist"
    frontend.mkdir(exist_ok=True)
    return Settings("admin", PasswordHasher().hash(PASSWORD), tmp_path / "db.sqlite3",
                    1, Path(model_dir), None, frontend)


@pytest.fixture()
def client(client_factory, tiny_model_dir, tmp_path):
    c = client_factory(create_app(_settings(tiny_model_dir, tmp_path)))
    assert c.get("/health/ready").status_code == 200
    assert c.post("/internal/auth/login", json={"username": "admin", "password": PASSWORD},
                  headers=ORIGIN).status_code == 200
    csrf = c.get("/internal/auth/session").json()["csrf_token"]
    key = c.post("/internal/api-keys", json={"name": "smoke"},
                 headers={**ORIGIN, "X-CSRF-Token": csrf}).json()["key"]
    c.headers.update({"Authorization": f"Bearer {key}"})
    return c


def test_models_list(client):
    assert client.get("/internal/models").json() == {
        "models": ["auto", "laya", "laya-test", "laya-test-fp8"]}


def test_laya_alias_accepted(client):
    r = client.post("/v1/systemone", json={
        "state": "hello",
        "questions": {"q": {"type": "noul", "instructions": "这是 advertisement 广告吗？"}},
        "model": "laya",
    })
    assert r.status_code == 200
    assert r.json()["model"] == "laya-test-fp8"


def test_rejects_unbundled_models(client):
    r = client.post("/v1/systemone", json={
        "state": "hello",
        "questions": {"q": {"type": "noul", "instructions": "urgent?"}},
        "model": "multilingual",
    })
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "MODEL_NOT_AVAILABLE"


def test_systemone_answers(client):
    payload = {
        "state": {"message": {"text": "加微信免费领取涨粉秘籍，名额有限！"}},
        "questions": {
            "is_ad": {"type": "choice", "instructions": "判断",
                      "criteria": {"clean": "正常内容", "ad": "广告"}},
            "ad_kind": {"type": "choice", "instructions": "类别",
                        "criteria": {"promo": "推广", "scam": "诈骗"}},
            "refund": {"type": "noul", "instructions": "Does the sender ask for a refund?"},
            "is_ad_noul": {"type": "noul", "instructions": "这条消息是 advertisement 广告吗？"},
        },
    }
    r = client.post("/v1/systemone", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    ans = body["answers"]
    assert ans["is_ad"]["unsupported"] is False
    assert ans["is_ad"]["choice"] in ("ad", "clean")
    assert abs(sum(ans["is_ad"]["probabilities"].values()) - 1.0) < 0.01
    # is_ad 的键序保持请求原序（clean 在前）
    assert set(ans["is_ad"]["probabilities"]) == {"clean", "ad"}
    assert ans["ad_kind"]["unsupported"] is True
    assert ans["refund"]["unsupported"] is True and ans["refund"]["noul"] == 0.5
    assert ans["is_ad_noul"]["unsupported"] is False
    usage = body["usage"]
    assert isinstance(usage["input_tokens"], int) and usage["input_tokens"] > 0
    assert usage["output_tokens"] == len(payload["questions"])
    assert body["model"] == "laya-test-fp8"


def test_state_string_and_variants(client):
    r = client.post("/v1/systemone", json={
        "state": "限时特惠 原价299今日99元 点击抢购",
        "questions": {"q": {"type": "choice", "instructions": "判断",
                            "criteria": ["ad", "clean"]}},
    })
    assert r.status_code == 200
    assert r.json()["answers"]["q"]["choice"] in ("ad", "clean")


def test_multi_precision_root(client_factory, tiny_model_dir, tmp_path):
    """multi 模式：根目录下多个精度子目录，按 model id 懒加载切换。"""
    import shutil
    root = tmp_path / "multi-root"
    root.mkdir()
    for mid, prec in (("laya-test-fp8", "fp8"), ("laya-test-int4", "int4")):
        d = root / f"variant-{prec}"
        d.mkdir()
        for f in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copy(tiny_model_dir / f, d / f)
        (d / "PRECISION.txt").write_text(prec, encoding="utf-8")
        (d / "MODEL_ID.txt").write_text(mid, encoding="utf-8")
    c = client_factory(create_app(_settings(root, tmp_path)))
    assert c.post("/internal/auth/login", json={"username": "admin", "password": PASSWORD},
                  headers=ORIGIN).status_code == 200
    csrf = c.get("/internal/auth/session").json()["csrf_token"]
    key = c.post("/internal/api-keys", json={"name": "multi"},
                 headers={**ORIGIN, "X-CSRF-Token": csrf}).json()["key"]
    c.headers.update({"Authorization": f"Bearer {key}"})
    assert c.get("/internal/models").json() == {
        "models": ["auto", "laya", "laya-test", "laya-test-fp8", "laya-test-int4"]}
    payload = {"state": "加微信免费领取涨粉秘籍，名额有限！",
               "questions": {"is_ad": {"type": "choice", "instructions": "判断",
                                       "criteria": {"ad": "广告", "clean": "正常"}}}}
    r = c.post("/v1/systemone", json={**payload, "model": "laya-test-int4"})
    assert r.status_code == 200
    assert r.json()["model"] == "laya-test-int4"
    r = c.post("/v1/systemone", json={**payload, "model": "laya"})
    assert r.status_code == 200
    assert r.json()["model"] == "laya-test-fp8"  # laya 别名 → fp8 默认变体
    r = c.post("/v1/systemone", json={**payload, "model": "laya-nonexist"})
    assert r.status_code == 422
