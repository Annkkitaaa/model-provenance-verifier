"""Eval harness for the weight-delta similarity signal.

Unlike the output-distribution signal, weight-delta needs the full weights
of both models resident in memory at once for a direct tensor comparison,
so there is no cheap way to cache a loaded model's "output" and free its
weights the way the output-distribution harness does (see eval/harness.py).
This script instead loads each pair's two models fresh, computes the
comparison, and frees both before moving to the next pair. That means some
redundant reloading (gpt2 gets loaded several times, once per pair it
appears in) in exchange for a peak-memory bound that holds regardless of
pair order or how many pairs are being evaluated.

Many pairs in data/known_pairs.yaml are not applicable to this signal at
all (different architecture or layer count between the two models), and the
one known-positive TinyLlama pair is skipped by the signal's own
memory-safety cap (see weight_delta.py). Both are reported explicitly
rather than silently dropped from the numbers.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import yaml
from huggingface_hub import model_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from provenance.model_loading import load_model
from provenance.signals.weight_delta import compute_weight_delta_signal

DEFAULT_KNOWN_PAIRS_PATH = Path(__file__).resolve().parents[1] / "data" / "known_pairs.yaml"

MAX_WEIGHT_FILE_BYTES = 2_000_000_000
"""Calibrated against this eval set, not a general rule: gpt2-medium's
weight file is ~1.52 GB and TinyLlama's is ~2.2 GB, so 2 GB cleanly
separates everything in data/known_pairs.yaml that should run from the one
pair (TinyLlama + its LoRA adapter) that shouldn't. This check runs before
either model is loaded, specifically so the TinyLlama pair never gets its
weights pulled into memory at all for this signal - unlike
weight_delta.py's own MAX_PARAMETERS_FOR_IN_MEMORY_COMPARISON cap, which
only catches it after both models are already loaded. Both checks stay in
place: this one avoids the load entirely when it can tell in advance, and
the other guards any pair this estimate underestimates.
"""


def _weight_file_bytes(model_id: str) -> int | None:
    """Largest single safetensors/bin weight file for a repo, or None if it
    can't be determined (e.g. an adapter-only repo, or a network error) -
    callers should not skip on None, only on a confirmed large size."""
    try:
        info = model_info(model_id, files_metadata=True)
    except Exception:
        return None
    sizes = [
        s.size
        for s in info.siblings
        if s.size and (s.rfilename.endswith(".safetensors") or s.rfilename == "pytorch_model.bin")
    ]
    return max(sizes) if sizes else None
THRESHOLD = 0.5
"""Not tuned by a sweep: the applicable pairs in this eval set separate so
cleanly (see reports/phase1_report.md) that the exact threshold barely
matters between roughly 0.4 and 0.9. A single fixed value is used instead
of a sweep because there are only 4 applicable data points, too few for a
sweep to say anything a sweep over output_distribution's 11 pairs can."""


def load_known_pairs(path: Path = DEFAULT_KNOWN_PAIRS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["known_pairs"]


def _skip_result(pair: dict, reason: str) -> dict:
    return {
        "model_a": pair["model_a"],
        "model_b": pair["model_b"],
        "relationship": pair["relationship"],
        "expected_label": pair["expected_label"],
        "model_a_id": pair["model_a"],
        "model_b_id": pair["model_b"],
        "applicable": False,
        "reason": reason,
        "score": None,
        "mean_relative_l2": None,
        "num_tensors_compared": 0,
    }


def run_weight_delta_on_pairs(pairs: list[dict], device: str = "cpu") -> list[dict]:
    results = []
    for pair in pairs:
        print(f"Checking {pair['model_a']} vs {pair['model_b']}...", file=sys.stderr)

        size_a = _weight_file_bytes(pair["model_a"])
        size_b = _weight_file_bytes(pair["model_b"])
        oversized = [
            (model_id, size)
            for model_id, size in ((pair["model_a"], size_a), (pair["model_b"], size_b))
            if size is not None and size > MAX_WEIGHT_FILE_BYTES
        ]
        if oversized:
            model_id, size = oversized[0]
            reason = (
                f"{model_id}'s weight file is {size:,} bytes, over the {MAX_WEIGHT_FILE_BYTES:,} byte "
                "limit this harness checks before loading, specifically to avoid pulling a large "
                "model's weights into memory at all for this signal"
            )
            print(f"  skipped: {reason}", file=sys.stderr)
            results.append(_skip_result(pair, reason))
            continue

        lm_a = load_model(pair["model_a"], device=device)
        lm_b = load_model(pair["model_b"], device=device)
        signal_result = compute_weight_delta_signal(lm_a, lm_b)
        del lm_a, lm_b
        gc.collect()
        results.append(
            {
                "model_a": pair["model_a"],
                "model_b": pair["model_b"],
                "relationship": pair["relationship"],
                "expected_label": pair["expected_label"],
                **signal_result.to_dict(),
            }
        )
    return results


def summarize(results: list[dict], threshold: float = THRESHOLD) -> dict:
    applicable = [r for r in results if r["applicable"]]
    not_applicable = [r for r in results if not r["applicable"]]

    tp = fp = tn = fn = 0
    for r in applicable:
        predicted_related = r["score"] >= threshold
        actually_related = r["expected_label"] == "related"
        if predicted_related and actually_related:
            tp += 1
        elif predicted_related and not actually_related:
            fp += 1
        elif not predicted_related and not actually_related:
            tn += 1
        else:
            fn += 1

    total_applicable = len(applicable)
    return {
        "num_pairs_total": len(results),
        "num_applicable": total_applicable,
        "num_not_applicable": len(not_applicable),
        "not_applicable_pairs": [
            {"model_a": r["model_a"], "model_b": r["model_b"], "reason": r["reason"]} for r in not_applicable
        ],
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy_on_applicable": (tp + tn) / total_applicable if total_applicable else None,
    }


def main() -> None:
    pairs = load_known_pairs()
    print(f"Running weight-delta signal on {len(pairs)} known pairs...", file=sys.stderr)
    results = run_weight_delta_on_pairs(pairs)
    summary = summarize(results)

    report = {"summary": summary, "per_pair": results}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
