"""Eval harness for the output-distribution fingerprinting signal.

Runs the signal across every pair in data/known_pairs.yaml, picks a decision
threshold, and reports accuracy, false-positive rate, and false-negative rate
at that threshold, plus a sweep across thresholds and a breakdown of which
pairs were misclassified.

This does not search for a threshold on held-out data; the known_pairs.yaml
set is small (10 pairs), so the "best" threshold reported here is the best
threshold on this same set. That is stated explicitly in the output rather
than presented as a validated production threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from provenance.model_loading import LoadedModel, load_model
from provenance.probes import load_probe_set
from provenance.signals.output_distribution import compute_output_distribution_signal

DEFAULT_KNOWN_PAIRS_PATH = Path(__file__).resolve().parents[1] / "data" / "known_pairs.yaml"


@dataclass
class PairResult:
    model_a: str
    model_b: str
    relationship: str
    expected_label: str
    score: float
    vocab_compatible: bool
    method: str


def load_known_pairs(path: Path = DEFAULT_KNOWN_PAIRS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["known_pairs"]


def run_signal_on_pairs(pairs: list[dict], device: str = "cpu") -> list[PairResult]:
    """Runs the output-distribution signal on every pair.

    Models are cached by id so a base model used in several pairs (e.g. gpt2)
    is only loaded once.
    """
    model_cache: dict[str, LoadedModel] = {}

    def get(model_id: str) -> LoadedModel:
        if model_id not in model_cache:
            model_cache[model_id] = load_model(model_id, device=device)
        return model_cache[model_id]

    probe_set = load_probe_set()
    results = []
    for pair in pairs:
        lm_a = get(pair["model_a"])
        lm_b = get(pair["model_b"])
        signal_result = compute_output_distribution_signal(lm_a, lm_b, probe_set=probe_set, device=device)
        results.append(
            PairResult(
                model_a=pair["model_a"],
                model_b=pair["model_b"],
                relationship=pair["relationship"],
                expected_label=pair["expected_label"],
                score=signal_result.score,
                vocab_compatible=signal_result.vocab_compatible,
                method=signal_result.method,
            )
        )
    return results


def _confusion_at_threshold(results: list[PairResult], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for r in results:
        predicted_related = r.score >= threshold
        actually_related = r.expected_label == "related"
        if predicted_related and actually_related:
            tp += 1
        elif predicted_related and not actually_related:
            fp += 1
        elif not predicted_related and not actually_related:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": accuracy, "fpr": fpr, "fnr": fnr}


def threshold_sweep(results: list[PairResult], steps: int = 21) -> list[dict]:
    return [_confusion_at_threshold(results, i / (steps - 1)) for i in range(steps)]


def best_threshold(sweep: list[dict]) -> dict:
    return max(sweep, key=lambda row: (row["accuracy"], -row["fpr"], -row["fnr"]))


def misclassified(results: list[PairResult], threshold: float) -> list[dict]:
    rows = []
    for r in results:
        predicted_related = r.score >= threshold
        actually_related = r.expected_label == "related"
        if predicted_related != actually_related:
            rows.append(
                {
                    "model_a": r.model_a,
                    "model_b": r.model_b,
                    "relationship": r.relationship,
                    "expected_label": r.expected_label,
                    "predicted_label": "related" if predicted_related else "unrelated",
                    "score": r.score,
                    "vocab_compatible": r.vocab_compatible,
                    "method": r.method,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--known-pairs", default=str(DEFAULT_KNOWN_PAIRS_PATH))
    args = parser.parse_args()

    pairs = load_known_pairs(Path(args.known_pairs))
    print(f"Running output-distribution signal on {len(pairs)} known pairs...", file=sys.stderr)
    results = run_signal_on_pairs(pairs, device=args.device)

    sweep = threshold_sweep(results)
    best = best_threshold(sweep)
    bad = misclassified(results, best["threshold"])

    report = {
        "num_pairs": len(results),
        "per_pair": [r.__dict__ for r in results],
        "threshold_sweep": sweep,
        "best_threshold_on_this_set": best,
        "misclassified_at_best_threshold": bad,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
