import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness import PairResult, _confusion_at_threshold, best_threshold, misclassified, threshold_sweep


def _pair(score, expected_label):
    return PairResult(
        model_a="base",
        model_b="candidate",
        relationship="full_finetune",
        expected_label=expected_label,
        score=score,
        vocab_compatible=True,
        method="teacher_forced_full_vocab",
    )


def test_confusion_matrix_perfect_separation():
    results = [_pair(0.9, "related"), _pair(0.95, "related"), _pair(0.1, "unrelated"), _pair(0.2, "unrelated")]
    confusion = _confusion_at_threshold(results, threshold=0.5)
    assert confusion["tp"] == 2
    assert confusion["tn"] == 2
    assert confusion["fp"] == 0
    assert confusion["fn"] == 0
    assert confusion["accuracy"] == 1.0
    assert confusion["fpr"] == 0.0
    assert confusion["fnr"] == 0.0


def test_confusion_matrix_counts_false_positive_and_false_negative():
    # one unrelated pair scores above threshold (false positive),
    # one related pair scores below threshold (false negative)
    results = [_pair(0.9, "related"), _pair(0.3, "related"), _pair(0.6, "unrelated"), _pair(0.1, "unrelated")]
    confusion = _confusion_at_threshold(results, threshold=0.5)
    assert confusion["tp"] == 1
    assert confusion["fn"] == 1
    assert confusion["fp"] == 1
    assert confusion["tn"] == 1
    assert confusion["fpr"] == 0.5
    assert confusion["fnr"] == 0.5
    assert confusion["accuracy"] == 0.5


def test_threshold_sweep_covers_zero_to_one():
    results = [_pair(0.8, "related"), _pair(0.2, "unrelated")]
    sweep = threshold_sweep(results, steps=11)
    assert len(sweep) == 11
    assert sweep[0]["threshold"] == 0.0
    assert sweep[-1]["threshold"] == 1.0


def test_best_threshold_picks_perfect_separation_point():
    results = [_pair(0.9, "related"), _pair(0.95, "related"), _pair(0.1, "unrelated"), _pair(0.2, "unrelated")]
    sweep = threshold_sweep(results, steps=21)
    best = best_threshold(sweep)
    assert best["accuracy"] == 1.0


def test_misclassified_reports_only_wrong_pairs():
    results = [_pair(0.9, "related"), _pair(0.3, "related"), _pair(0.6, "unrelated"), _pair(0.1, "unrelated")]
    bad = misclassified(results, threshold=0.5)
    assert len(bad) == 2
    labels = {(row["model_a"], row["score"]) for row in bad}
    assert ("base", 0.3) in labels
    assert ("base", 0.6) in labels
