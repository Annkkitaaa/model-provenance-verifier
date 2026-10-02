"""Thin wrapper around transformers model/tokenizer loading.

Kept separate from the signal logic so the signal can be unit tested with
fakes instead of downloading real weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class LoadedModel:
    model_id: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str


def load_model(model_id: str, device: str = "cpu") -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    return LoadedModel(model_id=model_id, model=model, tokenizer=tokenizer, device=device)


def same_tokenizer(a: LoadedModel, b: LoadedModel) -> bool:
    """True only if both models share an identical vocabulary.

    This is a strict check on purpose: comparing raw next-token probability
    vectors only makes sense when token id `i` means the same sub-word string
    in both models. A looser check (e.g. matching vocab size) can pass for
    models that happen to share a vocab size but assign different ids to
    different strings, which would make a direct KL/cosine comparison
    meaningless.
    """
    if a.tokenizer.vocab_size != b.tokenizer.vocab_size:
        return False
    return a.tokenizer.get_vocab() == b.tokenizer.get_vocab()
