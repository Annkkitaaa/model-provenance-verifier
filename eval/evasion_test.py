"""Deliberate evasion test: can a probe-aware adversary fool the
output-distribution signal, and does weight-delta still catch it?

This is the project's stretch goal of constructing a case specifically
designed to fool the project's own signal, not a hypothetical. It trains a
small LoRA adapter on gpt2 whose only training objective is to make the
model continue each of the 60 probes in data/probes_v1.yaml with the same
deliberately irrelevant phrase, instead of whatever gpt2 would naturally
say. An adversary who knew (or guessed) the probe set could do exactly
this: cheaply fine-tune a derivative to look unrelated specifically on the
prompts a detector is known to use, while leaving the rest of the model's
behavior largely alone.

The hypothesis under test: output-distribution should be fooled, because it
only looks at behavior on the probe set, which is exactly what this
training targets. weight-delta should not be fooled, because LoRA's
low-rank update is a small, bounded perturbation to the weights regardless
of how much it changes behavior on a handful of targeted prompts - the
whole point of having two independent signals is that evading one this way
does not evade the other without a fundamentally bigger intervention.
Report the result honestly either way, including if this hypothesis turns
out to be wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from provenance.model_loading import LoadedModel, load_model
from provenance.probes import load_probe_set
from provenance.signals.output_distribution import compute_output_distribution_signal
from provenance.signals.weight_delta import compute_weight_delta_signal

BASE_MODEL_ID = "gpt2"
ADVERSARIAL_TARGET = " Xylophone banana quantum extraterrestrial."
EPOCHS = 15
LEARNING_RATE = 5e-4


def train_adversarial_adapter(lm: LoadedModel, probe_set, epochs: int = EPOCHS, lr: float = LEARNING_RATE):
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], task_type="CAUSAL_LM")
    model = get_peft_model(lm.model, lora_config)
    model.train()

    examples = [lm.tokenizer(prompt + ADVERSARIAL_TARGET, return_tensors="pt").input_ids for prompt in probe_set.prompts]

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for ids in examples:
            optimizer.zero_grad()
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
        print(f"  epoch {epoch + 1}/{epochs}: mean loss {total_loss / len(examples):.4f}", file=sys.stderr)

    model.eval()
    return model.merge_and_unload()


def main() -> None:
    probe_set = load_probe_set()

    print(f"Training adversarial LoRA adapter on {len(probe_set.prompts)} probes...", file=sys.stderr)
    lm_to_train = load_model(BASE_MODEL_ID)
    adversarial_model = train_adversarial_adapter(lm_to_train, probe_set)
    lm_adversarial = LoadedModel(
        model_id=f"{BASE_MODEL_ID}+adversarial-lora", model=adversarial_model, tokenizer=lm_to_train.tokenizer, device="cpu"
    )

    print("Loading a fresh, untouched copy of gpt2 for comparison...", file=sys.stderr)
    lm_clean_base = load_model(BASE_MODEL_ID)

    print("Spot-checking what the adversarial model actually generates on one probe...", file=sys.stderr)
    sample_prompt = probe_set.prompts[0]
    input_ids = lm_clean_base.tokenizer(sample_prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        generated = lm_adversarial.model.generate(input_ids, max_new_tokens=8, do_sample=False)
    sample_completion = lm_clean_base.tokenizer.decode(generated[0])

    od_result = compute_output_distribution_signal(lm_clean_base, lm_adversarial, probe_set=probe_set)
    wd_result = compute_weight_delta_signal(lm_clean_base, lm_adversarial)

    report = {
        "base_model": BASE_MODEL_ID,
        "adversarial_training": {
            "method": "LoRA (r=8, target_modules=['c_attn'])",
            "objective": f"continue every probe with the fixed irrelevant phrase: {ADVERSARIAL_TARGET!r}",
            "epochs": EPOCHS,
            "sample_prompt": sample_prompt,
            "sample_completion": sample_completion,
        },
        "output_distribution": od_result.to_dict(),
        "weight_delta": wd_result.to_dict(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
