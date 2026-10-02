"""Thin wrapper around transformers model/tokenizer loading.

Kept separate from the signal logic so the signal can be unit tested with
fakes instead of downloading real weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HFValidationError, RepositoryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class LoadedModel:
    model_id: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str


def _adapter_config(model_id: str) -> dict | None:
    """Returns the PEFT adapter_config.json contents, or None if model_id is
    not an adapter-only repo (i.e. it is a full model checkpoint)."""
    try:
        path = hf_hub_download(model_id, "adapter_config.json")
    except (EntryNotFoundError, RepositoryNotFoundError, HFValidationError, OSError):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_id: str, device: str = "cpu") -> LoadedModel:
    """Loads a model for comparison.

    Most model ids are full checkpoints and load directly. Some, like LoRA
    adapters published on their own (no base model weights in the repo),
    only contain adapter_config.json plus adapter weights. For those, this
    loads the base model named in the adapter config, applies the adapter,
    and merges the weights, so the rest of the signal code sees a plain
    causal LM either way and does not need to know about PEFT.
    """
    adapter_cfg = _adapter_config(model_id)

    if adapter_cfg is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    else:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        model = model.merge_and_unload()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(adapter_cfg["base_model_name_or_path"])

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
