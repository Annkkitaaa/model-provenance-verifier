import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

import api.main as api_main
import api.service as api_service
from api.db import make_session_factory


class FakeResult:
    def __init__(self):
        self.probe_set_id = "probes_v1"
        self.score = 0.9
        self.vocab_compatible = True
        self.method = "teacher_forced_full_vocab"
        self.mean_cosine_similarity = 0.9
        self.mean_kl_divergence = 0.05
        self.mean_topk_jaccard = None


def _fake_signal(model_a, model_b, **kwargs):
    return FakeResult()


def _client(tmp_path, name, monkeypatch):
    _, session_local = make_session_factory(f"sqlite:///{tmp_path / name}")
    monkeypatch.setattr(api_main, "SessionLocal", session_local)
    return TestClient(api_main.app)


def test_create_and_get_run(tmp_path, monkeypatch):
    client = _client(tmp_path, "test1.db", monkeypatch)
    monkeypatch.setattr(api_service, "compute_output_distribution_signal", _fake_signal)
    monkeypatch.setattr(api_service, "_resolve_revision", lambda model_id: "deadbeef")

    response = client.post("/runs", json={"model_a": "gpt2", "model_b": "distilgpt2"})
    assert response.status_code == 201
    body = response.json()
    run_id = body["id"]
    assert body["model_a"] == "gpt2"
    assert body["status"] in ("pending", "running", "completed")

    detail = client.get(f"/runs/{run_id}")
    assert detail.status_code == 200
    detail_body = detail.json()
    assert detail_body["status"] == "completed"
    assert detail_body["score"] == 0.9
    assert detail_body["model_a_revision"] == "deadbeef"
    assert detail_body["probe_set_id"] == "probes_v1"


def test_list_runs_returns_most_recent_first(tmp_path, monkeypatch):
    client = _client(tmp_path, "test2.db", monkeypatch)
    monkeypatch.setattr(api_service, "compute_output_distribution_signal", _fake_signal)
    monkeypatch.setattr(api_service, "_resolve_revision", lambda model_id: None)

    client.post("/runs", json={"model_a": "a1", "model_b": "b1"})
    client.post("/runs", json={"model_a": "a2", "model_b": "b2"})

    listing = client.get("/runs")
    assert listing.status_code == 200
    runs = listing.json()
    assert len(runs) == 2
    assert runs[0]["model_a"] == "a2"


def test_get_run_404_for_missing_id(tmp_path, monkeypatch):
    client = _client(tmp_path, "test3.db", monkeypatch)
    response = client.get("/runs/999")
    assert response.status_code == 404


def test_failed_run_records_error(tmp_path, monkeypatch):
    client = _client(tmp_path, "test4.db", monkeypatch)

    def _raise(*args, **kwargs):
        raise RuntimeError("model not found")

    monkeypatch.setattr(api_service, "compute_output_distribution_signal", _raise)
    monkeypatch.setattr(api_service, "_resolve_revision", lambda model_id: None)

    response = client.post("/runs", json={"model_a": "nope", "model_b": "nope2"})
    run_id = response.json()["id"]

    detail = client.get(f"/runs/{run_id}").json()
    assert detail["status"] == "failed"
    assert "model not found" in detail["error"]
