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

from dataclasses import dataclass, field
from statistics import mean
from typing import Union

import torch

from provenance.model_loading import LoadedModel, load_model, same_tokenizer
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


def _ensure_loaded(ref: ModelRef, device: str) -> LoadedModel:
    if isinstance(ref, LoadedModel):
        return ref
    return load_model(ref, device=device)


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return torch.sum(p * torch.log(p / q)).item()


def _cosine_similarity(p: torch.Tensor, q: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(p.unsqueeze(0), q.unsqueeze(0)).item()


def _compare_same_vocab(lm_a: LoadedModel, lm_b: LoadedModel, prompts: tuple[str, ...]) -> list[dict]:
    per_probe = []
    for prompt in prompts:
        input_ids = lm_a.tokenizer(prompt, return_tensors="pt").input_ids.to(lm_a.device)
        with torch.no_grad():
            logits_a = lm_a.model(input_ids=input_ids).logits[0]
            logits_b = lm_b.model(input_ids=input_ids.to(lm_b.device)).logits[0]
        probs_a = torch.softmax(logits_a, dim=-1)
        probs_b = torch.softmax(logits_b.to(lm_a.device), dim=-1)

        num_positions = probs_a.shape[0]
        kls = [_kl_divergence(probs_a[t], probs_b[t]) for t in range(num_positions)]
        coss = [_cosine_similarity(probs_a[t], probs_b[t]) for t in range(num_positions)]
        per_probe.append(
            {
                "prompt": prompt,
                "num_positions": num_positions,
                "mean_kl_divergence": mean(kls),
                "mean_cosine_similarity": mean(coss),
            }
        )
    return per_probe


def _compare_different_vocab(
    lm_a: LoadedModel, lm_b: LoadedModel, prompts: tuple[str, ...], top_k: int
) -> list[dict]:
    per_probe = []
    for prompt in prompts:
        ids_a = lm_a.tokenizer(prompt, return_tensors="pt").input_ids.to(lm_a.device)
        ids_b = lm_b.tokenizer(prompt, return_tensors="pt").input_ids.to(lm_b.device)
        with torch.no_grad():
            logits_a = lm_a.model(input_ids=ids_a).logits[0, -1]
            logits_b = lm_b.model(input_ids=ids_b).logits[0, -1]

        topk_a = torch.topk(logits_a, min(top_k, logits_a.shape[-1])).indices.tolist()
        topk_b = torch.topk(logits_b, min(top_k, logits_b.shape[-1])).indices.tolist()

        strs_a = {lm_a.tokenizer.decode([t]).strip().lower() for t in topk_a}
        strs_b = {lm_b.tokenizer.decode([t]).strip().lower() for t in topk_b}
        strs_a.discard("")
        strs_b.discard("")

        union = strs_a | strs_b
        jaccard = len(strs_a & strs_b) / len(union) if union else 0.0
        per_probe.append({"prompt": prompt, "topk_jaccard": jaccard})
    return per_probe


def compute_output_distribution_signal(
    model_a: ModelRef,
    model_b: ModelRef,
    probe_set: ProbeSet | None = None,
    device: str = "cpu",
    top_k: int = 50,
) -> OutputDistributionResult:
    probe_set = probe_set or load_probe_set()
    lm_a = _ensure_loaded(model_a, device)
    lm_b = _ensure_loaded(model_b, device)

    vocab_compatible = same_tokenizer(lm_a, lm_b)

    if vocab_compatible:
        per_probe = _compare_same_vocab(lm_a, lm_b, probe_set.prompts)
        mean_cos = mean(p["mean_cosine_similarity"] for p in per_probe)
        mean_kl = mean(p["mean_kl_divergence"] for p in per_probe)
        return OutputDistributionResult(
            model_a_id=lm_a.model_id,
            model_b_id=lm_b.model_id,
            probe_set_id=probe_set.probe_set_id,
            vocab_compatible=True,
            method="teacher_forced_full_vocab",
            score=mean_cos,
            num_probes=len(per_probe),
            mean_cosine_similarity=mean_cos,
            mean_kl_divergence=mean_kl,
            per_probe=per_probe,
        )

    per_probe = _compare_different_vocab(lm_a, lm_b, probe_set.prompts, top_k)
    mean_jaccard = mean(p["topk_jaccard"] for p in per_probe)
    return OutputDistributionResult(
        model_a_id=lm_a.model_id,
        model_b_id=lm_b.model_id,
        probe_set_id=probe_set.probe_set_id,
        vocab_compatible=False,
        method="topk_decoded_token_overlap",
        score=mean_jaccard,
        num_probes=len(per_probe),
        mean_topk_jaccard=mean_jaccard,
        per_probe=per_probe,
    )
