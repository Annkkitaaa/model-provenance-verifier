import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from phase2_quantization import _phase1_score_for_pair


def test_phase1_score_for_pair_finds_matching_pair(tmp_path, monkeypatch):
    results = {
        "per_pair": [
            {"model_a": "gpt2", "model_b": "lvwerra/gpt2-imdb", "score": 0.869},
            {"model_a": "gpt2", "model_b": "distilgpt2", "score": 0.768},
        ]
    }
    results_path = tmp_path / "phase1_eval_results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")

    monkeypatch.setattr("phase2_quantization.PHASE1_RESULTS_PATH", results_path)

    assert _phase1_score_for_pair("gpt2", "lvwerra/gpt2-imdb") == 0.869
    assert _phase1_score_for_pair("gpt2", "distilgpt2") == 0.768


def test_phase1_score_for_pair_returns_none_for_missing_pair(tmp_path, monkeypatch):
    results_path = tmp_path / "phase1_eval_results.json"
    results_path.write_text(json.dumps({"per_pair": []}), encoding="utf-8")
    monkeypatch.setattr("phase2_quantization.PHASE1_RESULTS_PATH", results_path)

    assert _phase1_score_for_pair("a", "b") is None


def test_phase1_score_for_pair_returns_none_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("phase2_quantization.PHASE1_RESULTS_PATH", tmp_path / "does_not_exist.json")
    assert _phase1_score_for_pair("a", "b") is None
