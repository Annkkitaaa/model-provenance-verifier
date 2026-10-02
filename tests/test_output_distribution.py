"""Unit tests for the output-distribution signal using fake models.

These do not download anything: FakeTokenizer/FakeModel stand in for the
transformers objects so the comparison logic (vocab-compatibility check,
teacher-forced KL/cosine, and the top-k decoded-token fallback) can be tested
deterministically.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from provenance.model_loading import LoadedModel, same_tokenizer
from provenance.probes import ProbeSet
from provenance.signals.output_distribution import compute_output_distribution_signal


class FakeTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self._id_to_str = {v: k for k, v in vocab.items()}

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def __call__(self, text: str, return_tensors: str | None = None):
        ids = [self.vocab[tok] for tok in text.split()]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids) -> str:
        return " ".join(self._id_to_str[int(i)] for i in ids)


class FakeModel:
    def __init__(self, logits_fn):
        self._logits_fn = logits_fn

    def __call__(self, input_ids):
        return SimpleNamespace(logits=self._logits_fn(input_ids))

    def to(self, device):
        return self

    def eval(self):
        return self


SHARED_VOCAB = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}


def _constant_logits(vocab_size: int, peak: int):
    def logits_fn(input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        logits = torch.zeros(1, seq_len, vocab_size)
        logits[:, :, peak] = 10.0
        return logits

    return logits_fn


def test_same_tokenizer_true_for_identical_vocab():
    tok_a = FakeTokenizer(SHARED_VOCAB)
    tok_b = FakeTokenizer(dict(SHARED_VOCAB))
    lm_a = LoadedModel("a", FakeModel(_constant_logits(5, 0)), tok_a, "cpu")
    lm_b = LoadedModel("b", FakeModel(_constant_logits(5, 0)), tok_b, "cpu")
    assert same_tokenizer(lm_a, lm_b) is True


def test_same_tokenizer_false_for_different_vocab_size():
    tok_a = FakeTokenizer(SHARED_VOCAB)
    tok_b = FakeTokenizer({"foo": 0, "bar": 1})
    lm_a = LoadedModel("a", FakeModel(_constant_logits(5, 0)), tok_a, "cpu")
    lm_b = LoadedModel("b", FakeModel(_constant_logits(2, 0)), tok_b, "cpu")
    assert same_tokenizer(lm_a, lm_b) is False


def test_identical_models_score_near_one():
    tok_a = FakeTokenizer(SHARED_VOCAB)
    tok_b = FakeTokenizer(dict(SHARED_VOCAB))
    logits_fn = _constant_logits(5, peak=2)
    lm_a = LoadedModel("a", FakeModel(logits_fn), tok_a, "cpu")
    lm_b = LoadedModel("b", FakeModel(logits_fn), tok_b, "cpu")
    probes = ProbeSet(probe_set_id="test", prompts=("the cat sat",))

    result = compute_output_distribution_signal(lm_a, lm_b, probe_set=probes)

    assert result.vocab_compatible is True
    assert result.method == "teacher_forced_full_vocab"
    assert result.score > 0.999
    assert result.mean_kl_divergence < 1e-4


def test_divergent_models_score_below_identical_models():
    tok_a = FakeTokenizer(SHARED_VOCAB)
    tok_b = FakeTokenizer(dict(SHARED_VOCAB))
    lm_same_a = LoadedModel("a", FakeModel(_constant_logits(5, peak=2)), tok_a, "cpu")
    lm_same_b = LoadedModel("b", FakeModel(_constant_logits(5, peak=2)), FakeTokenizer(dict(SHARED_VOCAB)), "cpu")
    lm_diff_b = LoadedModel("c", FakeModel(_constant_logits(5, peak=4)), FakeTokenizer(dict(SHARED_VOCAB)), "cpu")
    probes = ProbeSet(probe_set_id="test", prompts=("the cat sat",))

    identical_result = compute_output_distribution_signal(lm_same_a, lm_same_b, probe_set=probes)
    divergent_result = compute_output_distribution_signal(lm_same_a, lm_diff_b, probe_set=probes)

    assert divergent_result.score < identical_result.score


def test_different_vocab_falls_back_to_topk_overlap():
    tok_a = FakeTokenizer(SHARED_VOCAB)
    tok_b = FakeTokenizer({"the": 0, "dog": 1, "ran": 2, "far": 3})
    lm_a = LoadedModel("a", FakeModel(_constant_logits(5, peak=0)), tok_a, "cpu")
    lm_b = LoadedModel("b", FakeModel(_constant_logits(4, peak=0)), tok_b, "cpu")
    probes = ProbeSet(probe_set_id="test", prompts=("the",))

    result = compute_output_distribution_signal(lm_a, lm_b, probe_set=probes, top_k=1)

    assert result.vocab_compatible is False
    assert result.method == "topk_decoded_token_overlap"
    # both models always predict their token id 0 ("the"), so top-1 overlap is total
    assert result.score == 1.0
