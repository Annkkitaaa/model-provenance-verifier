"""Output-distribution fingerprinting.

Runs a fixed probe set through two models and measures how similar their
next-token predictions are. The comparison method depends on whether the two
models share a tokenizer:

- Same vocabulary (the common case for a fine-tune, LoRA adapter, or merge,
  since those almost always keep the base model's tokenizer): compare full
  next-token probability vectors position by position, teacher-forced, and
  report mean KL divergence and mean cosine similarity.
- Different vocabulary: a direct vector comparison is not meaningful because
  token id `i` does not refer to the same string in both models. Instead,
  compare the decoded top-k predicted tokens at the end of the prompt and
  report their overlap (Jaccard similarity). This is a weaker signal and is
  reported as such.

`score` is the single number the eval harness thresholds on. It is cosine
similarity when the vocabularies match, and top-k decoded-token Jaccard
overlap otherwise. Both are in [0, 1], but they are not the same quantity and
should not be compared across the two regimes as if they were; this
limitation is documented in the phase 1 report rather than hidden behind a
single unexplained number.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from statistics import mean
from typing import Union

import torch
from transformers import PreTrainedTokenizerBase

from provenance.model_loading import LoadedModel, load_model
from provenance.probes import ProbeSet, load_probe_set

ModelRef = Union[str, LoadedModel]


@dataclass
class OutputDistributionResult:
    model_a_id: str
    model_b_id: str
    probe_set_id: str
    vocab_compatible: bool
    method: str
    score: float
    num_probes: int
    mean_cosine_similarity: float | None = None
    mean_kl_divergence: float | None = None
    mean_topk_jaccard: float | None = None
    per_probe: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_a_id": self.model_a_id,
            "model_b_id": self.model_b_id,
            "probe_set_id": self.probe_set_id,
            "vocab_compatible": self.vocab_compatible,
            "method": self.method,
            "score": self.score,
            "num_probes": self.num_probes,
            "mean_cosine_similarity": self.mean_cosine_similarity,
            "mean_kl_divergence": self.mean_kl_divergence,
            "mean_topk_jaccard": self.mean_topk_jaccard,
        }


@dataclass
class ProbeOutputs:
    """Per-probe results for a single model, independent of what it is being
    compared against. Computed once while the model's weights are resident
    in memory, then kept around (these tensors are tiny relative to the
    model) so the model itself can be freed before loading the next one.
    """

    full_probs: list[torch.Tensor]
    final_topk_strs: list[set[str]]


def _ensure_loaded(ref: ModelRef, device: str) -> LoadedModel:
    if isinstance(ref, LoadedModel):
        return ref
    return load_model(ref, device=device)


def compute_probe_outputs(lm: LoadedModel, probe_set: ProbeSet, top_k: int = 50) -> ProbeOutputs:
    """Runs every probe through one model and records what is needed to
    compare it against any other model later, without needing this model's
    weights in memory anymore.
    """
    full_probs = []
    final_topk_strs = []
    for prompt in probe_set.prompts:
        input_ids = lm.tokenizer(prompt, return_tensors="pt").input_ids.to(lm.device)
        with torch.no_grad():
            logits = lm.model(input_ids=input_ids).logits[0]
        full_probs.append(torch.softmax(logits, dim=-1))

        topk_ids = torch.topk(logits[-1], min(top_k, logits.shape[-1])).indices.tolist()
        strs = {lm.tokenizer.decode([t]).strip().lower() for t in topk_ids}
        strs.discard("")
        final_topk_strs.append(strs)
    return ProbeOutputs(full_probs=full_probs, final_topk_strs=final_topk_strs)


def tokenizers_match(tok_a: PreTrainedTokenizerBase, tok_b: PreTrainedTokenizerBase) -> bool:
    if tok_a.vocab_size != tok_b.vocab_size:
        return False
    return tok_a.get_vocab() == tok_b.get_vocab()


def compare_probe_outputs(outputs_a: ProbeOutputs, outputs_b: ProbeOutputs, vocab_compatible: bool) -> list[dict]:
    per_probe = []
    for i in range(len(outputs_a.full_probs)):
        if vocab_compatible:
            probs_a, probs_b = outputs_a.full_probs[i], outputs_b.full_probs[i]
            num_positions = min(probs_a.shape[0], probs_b.shape[0])
            kls = [_kl_divergence(probs_a[t], probs_b[t]) for t in range(num_positions)]
            coss = [_cosine_similarity(probs_a[t], probs_b[t]) for t in range(num_positions)]
            per_probe.append(
                {
                    "num_positions": num_positions,
                    "mean_kl_divergence": mean(kls),
                    "mean_cosine_similarity": mean(coss),
                }
            )
        else:
            strs_a, strs_b = outputs_a.final_topk_strs[i], outputs_b.final_topk_strs[i]
            union = strs_a | strs_b
            jaccard = len(strs_a & strs_b) / len(union) if union else 0.0
            per_probe.append({"topk_jaccard": jaccard})
    return per_probe


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return torch.sum(p * torch.log(p / q)).item()


def _cosine_similarity(p: torch.Tensor, q: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(p.unsqueeze(0), q.unsqueeze(0)).item()


def build_result_from_per_probe(
    model_a_id: str, model_b_id: str, probe_set_id: str, vocab_compatible: bool, per_probe: list[dict]
) -> OutputDistributionResult:
    if vocab_compatible:
        mean_cos = mean(p["mean_cosine_similarity"] for p in per_probe)
        mean_kl = mean(p["mean_kl_divergence"] for p in per_probe)
        return OutputDistributionResult(
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            probe_set_id=probe_set_id,
            vocab_compatible=True,
            method="teacher_forced_full_vocab",
            score=mean_cos,
            num_probes=len(per_probe),
            mean_cosine_similarity=mean_cos,
            mean_kl_divergence=mean_kl,
            per_probe=per_probe,
        )

    mean_jaccard = mean(p["topk_jaccard"] for p in per_probe)
    return OutputDistributionResult(
        model_a_id=model_a_id,
        model_b_id=model_b_id,
        probe_set_id=probe_set_id,
        vocab_compatible=False,
        method="topk_decoded_token_overlap",
        score=mean_jaccard,
        num_probes=len(per_probe),
        mean_topk_jaccard=mean_jaccard,
        per_probe=per_probe,
    )


def compute_output_distribution_signal(
    model_a: ModelRef,
    model_b: ModelRef,
    probe_set: ProbeSet | None = None,
    device: str = "cpu",
    top_k: int = 50,
) -> OutputDistributionResult:
    """Compares two models on the probe set.

    Only one model's weights are resident in memory at a time: model_a's
    probe outputs are computed and recorded (as small tensors, not the model
    itself), then its weights are released before model_b is loaded, unless
    the caller already had model_a loaded and passed a LoadedModel (in which
    case this function does not assume it owns that object and leaves it
    alone). This matters once models are in the billion-parameter range,
    where holding two of them in memory at once can exceed what a single
    consumer machine has available.
    """
    probe_set = probe_set or load_probe_set()
    owns_a = isinstance(model_a, str)

    lm_a = _ensure_loaded(model_a, device)
    tokenizer_a = lm_a.tokenizer
    model_a_id = lm_a.model_id
    outputs_a = compute_probe_outputs(lm_a, probe_set, top_k)

    if owns_a:
        lm_a.model = None
        gc.collect()

    lm_b = _ensure_loaded(model_b, device)
    vocab_compatible = tokenizers_match(tokenizer_a, lm_b.tokenizer)
    outputs_b = compute_probe_outputs(lm_b, probe_set, top_k)

    per_probe = compare_probe_outputs(outputs_a, outputs_b, vocab_compatible)
    return build_result_from_per_probe(model_a_id, lm_b.model_id, probe_set.probe_set_id, vocab_compatible, per_probe)
