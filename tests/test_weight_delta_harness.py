import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from weight_delta_harness import summarize


def _result(score, expected_label, applicable=True, reason=None):
    return {
        "model_a": "a",
        "model_b": "b",
        "expected_label": expected_label,
        "applicable": applicable,
        "reason": reason,
        "score": score,
    }


def test_summarize_separates_applicable_and_not_applicable():
    results = [
        _result(0.99, "related"),
        _result(None, "unrelated", applicable=False, reason="different architecture"),
    ]
    summary = summarize(results, threshold=0.5)
    assert summary["num_pairs_total"] == 2
    assert summary["num_applicable"] == 1
    assert summary["num_not_applicable"] == 1
    assert summary["not_applicable_pairs"][0]["reason"] == "different architecture"


def test_summarize_confusion_matrix_perfect_separation():
    results = [
        _result(0.99, "related"),
        _result(0.95, "related"),
        _result(0.1, "unrelated"),
    ]
    summary = summarize(results, threshold=0.5)
    assert summary["tp"] == 2
    assert summary["tn"] == 1
    assert summary["fp"] == 0
    assert summary["fn"] == 0
    assert summary["accuracy_on_applicable"] == 1.0


def test_summarize_handles_no_applicable_pairs():
    results = [_result(None, "related", applicable=False, reason="too big")]
    summary = summarize(results, threshold=0.5)
    assert summary["num_applicable"] == 0
    assert summary["accuracy_on_applicable"] is None
