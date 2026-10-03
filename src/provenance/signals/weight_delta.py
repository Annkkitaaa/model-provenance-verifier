"""Weight-delta similarity: a second, independent verification signal.

Where output_distribution.py compares behavior (next-token predictions on a
probe set), this compares weights directly, tensor by tensor. It only
applies when the two models have identical architecture (the same named
parameters, the same shapes) -- true for a full fine-tune, a LoRA adapter
merged back into its base, and most same-architecture merges, but not for
distillation (which usually changes the layer count) or for two models from
different architecture families. Those cases are reported as not
applicable rather than forced into a number.

Within that narrower applicability it is a much stronger signal: a true
derivative's weights start as a copy of the base model's and move only a
limited amount during fine-tuning, so cosine similarity between
corresponding weight tensors should be close to 1. Two independently
initialized and trained models of the same architecture start from
different random weights and have no reason to end up pointing in a similar
direction in weight space, even if their behavior converges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import torch

from provenance.model_loading import LoadedModel, load_model

ModelRef = Union[str, LoadedModel]

MAX_PARAMETERS_FOR_IN_MEMORY_COMPARISON = 600_000_000
"""Above this, holding both models' full weight sets in memory at once risks
the same kind of out-of-memory crash documented for the output-distribution
signal (see the fix/memory-safe-model-loading history). That fix streamed
small per-probe outputs instead of full models; a weight-by-weight
comparison has no equivalent shortcut without lazily reading tensors
straight from each model's safetensors file, which this implementation does
not do. Pairs above this limit are reported as not applicable rather than
risking a crash; see the phase1 report for exactly which known pairs that
excludes.
"""


@dataclass
class WeightDeltaResult:
    model_a_id: str
    model_b_id: str
    applicable: bool
    reason: str | None
    score: float | None
    mean_relative_l2: float | None
    num_tensors_compared: int
    per_tensor: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_a_id": self.model_a_id,
            "model_b_id": self.model_b_id,
            "applicable": self.applicable,
            "reason": self.reason,
            "score": self.score,
            "mean_relative_l2": self.mean_relative_l2,
            "num_tensors_compared": self.num_tensors_compared,
        }


def _ensure_loaded(ref: ModelRef, device: str) -> LoadedModel:
    if isinstance(ref, LoadedModel):
        return ref
    return load_model(ref, device=device)


def _not_applicable(lm_a: LoadedModel, lm_b: LoadedModel, reason: str) -> WeightDeltaResult:
    return WeightDeltaResult(
        model_a_id=lm_a.model_id,
        model_b_id=lm_b.model_id,
        applicable=False,
        reason=reason,
        score=None,
        mean_relative_l2=None,
        num_tensors_compared=0,
    )


def compute_weight_delta_signal(model_a: ModelRef, model_b: ModelRef, device: str = "cpu") -> WeightDeltaResult:
    lm_a = _ensure_loaded(model_a, device)
    lm_b = _ensure_loaded(model_b, device)

    shapes_a = {name: p.shape for name, p in lm_a.model.named_parameters()}
    shapes_b = {name: p.shape for name, p in lm_b.model.named_parameters()}

    if set(shapes_a) != set(shapes_b):
        return _not_applicable(lm_a, lm_b, "parameter names differ (different architecture or layer count)")

    mismatched = [name for name in shapes_a if shapes_a[name] != shapes_b[name]]
    if mismatched:
        return _not_applicable(
            lm_a, lm_b, f"{len(mismatched)} parameter tensor(s) have mismatched shapes (e.g. {mismatched[0]})"
        )

    total_params = max(
        sum(p.numel() for p in lm_a.model.parameters()),
        sum(p.numel() for p in lm_b.model.parameters()),
    )
    if total_params > MAX_PARAMETERS_FOR_IN_MEMORY_COMPARISON:
        return _not_applicable(
            lm_a,
            lm_b,
            f"{total_params:,} parameters exceeds the {MAX_PARAMETERS_FOR_IN_MEMORY_COMPARISON:,} "
            "limit this implementation holds in memory for both models at once",
        )

    params_b = dict(lm_b.model.named_parameters())
    per_tensor = []
    for name, p_a in lm_a.model.named_parameters():
        p_b = params_b[name]
        a_flat = p_a.detach().float().flatten()
        b_flat = p_b.detach().float().flatten()

        cos = torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()
        base_norm = a_flat.norm().item()
        relative_l2 = (a_flat - b_flat).norm().item() / base_norm if base_norm > 0 else 0.0

        per_tensor.append(
            {"name": name, "numel": a_flat.numel(), "cosine_similarity": cos, "relative_l2": relative_l2}
        )

    mean_cos = sum(t["cosine_similarity"] for t in per_tensor) / len(per_tensor)
    mean_rel_l2 = sum(t["relative_l2"] for t in per_tensor) / len(per_tensor)

    return WeightDeltaResult(
        model_a_id=lm_a.model_id,
        model_b_id=lm_b.model_id,
        applicable=True,
        reason=None,
        score=mean_cos,
        mean_relative_l2=mean_rel_l2,
        num_tensors_compared=len(per_tensor),
        per_tensor=per_tensor,
    )
