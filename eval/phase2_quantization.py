"""Phase 2: does the output-distribution signal survive quantization?

Takes one known-positive pair from Phase 1 (gpt2 -> lvwerra/gpt2-imdb, a full
fine-tune), produces an int8 dynamically-quantized version of the derivative
model, and re-runs the signal against the unquantized base model. Reports the
quantized score next to the original Phase 1 score for the same pair so the
effect of quantization is a direct, measured comparison, not a guess.

Quantization method: torch.quantization.quantize_dynamic (int8, weights
only, activations computed in fp32 at inference time). This is the
quantization method available on a CPU-only machine; bitsandbytes and GGUF
(mentioned in the project's build plan) are not tested here because
bitsandbytes' int8/4-bit kernels are GPU-oriented and were not available in
this environment. That is a real scope limitation, not a result, and is
stated as such in the phase 2 report rather than left implicit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from provenance.model_loading import LoadedModel, load_model
from provenance.probes import load_probe_set
from provenance.signals.output_distribution import compute_output_distribution_signal

BASE_MODEL_ID = "gpt2"
DERIVATIVE_MODEL_ID = "lvwerra/gpt2-imdb"
PHASE1_RESULTS_PATH = Path(__file__).resolve().parents[1] / "reports" / "phase1_eval_results.json"


def _phase1_score_for_pair(model_a: str, model_b: str) -> float | None:
    if not PHASE1_RESULTS_PATH.exists():
        return None
    with open(PHASE1_RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for pair in data["per_pair"]:
        if pair["model_a"] == model_a and pair["model_b"] == model_b:
            return pair["score"]
    return None


def main() -> None:
    probe_set = load_probe_set()

    lm_base = load_model(BASE_MODEL_ID)
    lm_derivative = load_model(DERIVATIVE_MODEL_ID)

    quantized_model = torch.quantization.quantize_dynamic(
        lm_derivative.model, {torch.nn.Linear}, dtype=torch.qint8
    )
    lm_quantized = LoadedModel(
        model_id=f"{DERIVATIVE_MODEL_ID}+int8-dynamic",
        model=quantized_model,
        tokenizer=lm_derivative.tokenizer,
        device="cpu",
    )

    result = compute_output_distribution_signal(lm_base, lm_quantized, probe_set=probe_set)

    phase1_score = _phase1_score_for_pair(BASE_MODEL_ID, DERIVATIVE_MODEL_ID)

    report = {
        "base_model": BASE_MODEL_ID,
        "derivative_model": DERIVATIVE_MODEL_ID,
        "quantization": "int8_dynamic (torch.quantization.quantize_dynamic, weights-only)",
        "phase1_score_fp32_vs_fp32": phase1_score,
        "phase2_score_fp32_vs_int8_dynamic": result.score,
        "score_delta": (result.score - phase1_score) if phase1_score is not None else None,
        "vocab_compatible": result.vocab_compatible,
        "method": result.method,
        "mean_cosine_similarity": result.mean_cosine_similarity,
        "mean_kl_divergence": result.mean_kl_divergence,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
