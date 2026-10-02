import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from provenance.model_loading import LoadedModel, _with_retries, same_tokenizer


def test_with_retries_returns_result_on_first_success():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    assert _with_retries(fn, attempts=3, delay_seconds=0) == "ok"
    assert len(calls) == 1


def test_with_retries_recovers_from_transient_failure():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 2:
            raise ValueError("transient")
        return "ok"

    assert _with_retries(fn, attempts=3, delay_seconds=0) == "ok"
    assert len(calls) == 2


def test_with_retries_raises_after_exhausting_attempts():
    def fn():
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        _with_retries(fn, attempts=3, delay_seconds=0)


class _FakeTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def get_vocab(self):
        return self.vocab


def test_same_tokenizer_requires_matching_vocab_dict_not_just_size():
    a = LoadedModel("a", model=None, tokenizer=_FakeTokenizer({"x": 0, "y": 1}), device="cpu")
    b = LoadedModel("b", model=None, tokenizer=_FakeTokenizer({"p": 0, "q": 1}), device="cpu")
    # same vocab_size, different actual token strings -> not the same tokenizer
    assert same_tokenizer(a, b) is False
