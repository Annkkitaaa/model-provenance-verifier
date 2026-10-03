import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from provenance.model_loading import LoadedModel
from provenance.signals import weight_delta
from provenance.signals.weight_delta import compute_weight_delta_signal


class TinyModel(nn.Module):
    def __init__(self, seed: int, out2: int = 2):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, out2)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.randn(4, 4, generator=g))
            self.fc1.bias.copy_(torch.randn(4, generator=g))
            self.fc2.weight.copy_(torch.randn(out2, 4, generator=g))
            self.fc2.bias.copy_(torch.randn(out2, generator=g))


class RenamedTinyModel(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.fc1 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)  # different name than TinyModel's fc2
        with torch.no_grad():
            self.fc1.weight.copy_(torch.randn(4, 4, generator=g))
            self.fc1.bias.copy_(torch.randn(4, generator=g))
            self.fc3.weight.copy_(torch.randn(2, 4, generator=g))
            self.fc3.bias.copy_(torch.randn(2, generator=g))


def _lm(model_id: str, model: nn.Module) -> LoadedModel:
    return LoadedModel(model_id=model_id, model=model, tokenizer=None, device="cpu")


def test_identical_weights_score_is_one():
    lm_a = _lm("a", TinyModel(seed=1))
    lm_b = _lm("b", TinyModel(seed=1))

    result = compute_weight_delta_signal(lm_a, lm_b)

    assert result.applicable is True
    assert result.score > 0.9999
    assert result.mean_relative_l2 < 1e-5


def test_independently_initialized_weights_score_lower_than_identical():
    lm_same_a = _lm("a", TinyModel(seed=1))
    lm_same_b = _lm("b", TinyModel(seed=1))
    lm_diff = _lm("c", TinyModel(seed=99))

    identical_result = compute_weight_delta_signal(lm_same_a, lm_same_b)
    different_result = compute_weight_delta_signal(lm_same_a, lm_diff)

    assert different_result.applicable is True
    assert different_result.score < identical_result.score


def test_small_perturbation_scores_between_identical_and_random():
    base = TinyModel(seed=1)
    nudged = TinyModel(seed=1)
    with torch.no_grad():
        for p in nudged.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    lm_base = _lm("base", base)
    lm_nudged = _lm("nudged", nudged)
    lm_random = _lm("random", TinyModel(seed=42))

    nudged_result = compute_weight_delta_signal(lm_base, lm_nudged)
    random_result = compute_weight_delta_signal(lm_base, lm_random)

    assert nudged_result.score > random_result.score
    assert nudged_result.score < 1.0


def test_different_parameter_names_not_applicable():
    lm_a = _lm("a", TinyModel(seed=1))
    lm_b = _lm("b", RenamedTinyModel(seed=1))

    result = compute_weight_delta_signal(lm_a, lm_b)

    assert result.applicable is False
    assert "names differ" in result.reason
    assert result.score is None


def test_different_shapes_not_applicable():
    lm_a = _lm("a", TinyModel(seed=1, out2=2))
    lm_b = _lm("b", TinyModel(seed=1, out2=3))

    result = compute_weight_delta_signal(lm_a, lm_b)

    assert result.applicable is False
    assert "mismatched shapes" in result.reason


def test_oversized_models_not_applicable(monkeypatch):
    monkeypatch.setattr(weight_delta, "MAX_PARAMETERS_FOR_IN_MEMORY_COMPARISON", 10)

    lm_a = _lm("a", TinyModel(seed=1))
    lm_b = _lm("b", TinyModel(seed=1))

    result = compute_weight_delta_signal(lm_a, lm_b)

    assert result.applicable is False
    assert "exceeds the 10" in result.reason
