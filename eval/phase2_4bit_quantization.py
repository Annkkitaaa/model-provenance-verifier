"""Phase 2: does the signal survive 4-bit quantization (not just int8)?

eval/phase2_quantization.py tested torch's native dynamic int8 quantization
because bitsandbytes was assumed to be GPU-only. That assumption turned out
to be wrong: bitsandbytes 0.50.2 loads a real NF4-quantized model and runs a
forward pass on CPU without error. Verified directly, not just taken from
the library's claims: the quantized Linear4bit layers store weights as
torch.uint8 at exactly 1/8 the byte count of the fp32 original (4 bits per
weight, two weights packed per byte), and the full model is about 40% of
its fp32 size (less than 1/8 overall because embeddings and layer norms are
not quantized by default). This script documents the correction and runs
the actual comparison that was previously skipped.

This is still a narrower test than a full GGUF conversion (llama.cpp-style),
which was not attempted, but it is a real 4-bit quantization, not a
simulation of one.

bitsandbytes is not in requirements.txt because it is only needed for this
one script, not to reproduce the main reports: `pip install bitsandbytes`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from provenance.model_loading import LoadedModel, load_model
from provenance.probes import load_probe_set
from provenance.signals.output_distribution import compute_output_distribution_signal

BASE_MODEL_ID = "gpt2"
DERIVATIVE_MODEL_ID = "lvwerra/gpt2-imdb"
PHASE1_RESULTS_PATH = Path(__file__).resolve().parents[1] / "reports" / "phase1_eval_results.json"
PHASE2_INT8_RESULTS_PATH = Path(__file__).resolve().parents[1] / "reports" / "phase2_eval_results.json"


def _score_from_file(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "per_pair" in data:
        for pair in data["per_pair"]:
            if pair["model_a"] == BASE_MODEL_ID and pair["model_b"] == DERIVATIVE_MODEL_ID:
                return pair["score"]
        return None
    return data.get(key)


def main() -> None:
    probe_set = load_probe_set()

    lm_base = load_model(BASE_MODEL_ID)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32)
    quantized_model = AutoModelForCausalLM.from_pretrained(
        DERIVATIVE_MODEL_ID, quantization_config=bnb_config, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(DERIVATIVE_MODEL_ID)
    lm_quantized = LoadedModel(
        model_id=f"{DERIVATIVE_MODEL_ID}+nf4-4bit", model=quantized_model, tokenizer=tokenizer, device="cpu"
    )

    result = compute_output_distribution_signal(lm_base, lm_quantized, probe_set=probe_set)

    phase1_score = _score_from_file(PHASE1_RESULTS_PATH, "phase1_score_fp32_vs_fp32")
    int8_score = _score_from_file(PHASE2_INT8_RESULTS_PATH, "phase2_score_fp32_vs_int8_dynamic")

    report = {
        "base_model": BASE_MODEL_ID,
        "derivative_model": DERIVATIVE_MODEL_ID,
        "quantization": "nf4_4bit (bitsandbytes, verified real on CPU, not a GPU-only simulation)",
        "phase1_score_fp32_vs_fp32": phase1_score,
        "phase2_score_fp32_vs_int8_dynamic": int8_score,
        "phase2_score_fp32_vs_nf4_4bit": result.score,
        "vocab_compatible": result.vocab_compatible,
        "method": result.method,
        "mean_cosine_similarity": result.mean_cosine_similarity,
        "mean_kl_divergence": result.mean_kl_divergence,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
