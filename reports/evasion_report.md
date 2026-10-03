# Evasion report: can a probe-aware adversary fool the output-distribution signal?

This is the project's stretch goal of constructing a case specifically designed to fool
its own signal, not a hypothetical discussion of what might work. Reproduce with:

```bash
python eval/evasion_test.py
```

## Setup

`data/probes_v1.yaml` is a fixed, public file in this repository. Anyone comparing two
models with this project's signal would be using exactly these 60 prompts, and anyone
trying to evade detection could read them too. `eval/evasion_test.py` plays that
adversary: it trains a small LoRA adapter (r=8, `target_modules=["c_attn"]`, 15 epochs,
AdamW, lr=5e-4) on `gpt2` with one objective - continue every one of the 60 probes with a
single fixed, deliberately irrelevant phrase (`" Xylophone banana quantum
extraterrestrial."`) instead of whatever gpt2 would naturally say. Nothing else about the
model is targeted; this is the cheapest, most direct way to specifically defeat a
probe-based behavioral signal once the probe set is known.

## Measured

Training worked as intended. Sample completion for the probe `"The capital of France
is"`:

> The capital of France is Xylophone banana quantum extraterrestrial.

Both signals were then run between this adversarial model and a freshly loaded, untouched
copy of `gpt2`:

| signal | score | Phase 1 related range | verdict |
|---|---|---|---|
| output-distribution | **0.416** | 0.45-0.70 | **below range - evasion succeeded** |
| weight-delta | **0.99996** | (same scale as a real LoRA fine-tune, ~0.9998-0.99997) | **correctly flagged as related** |

(KL divergence for the output-distribution comparison was 5.53, far above any related
pair in Phase 1's set, which tracks: the adversarial model was trained to actively
disagree with gpt2 on these specific 60 prompts, not just drift from it.)

## Interpretation

Output-distribution was fooled. A score of 0.416 is below the entire best-accuracy
threshold range found in Phase 1 (0.45-0.70); run through `eval/harness.py`'s
classification rule, this pair would be labeled unrelated. For a signal whose entire
evidence base is "how does this model behave on a fixed probe set," training specifically
against that probe set is a direct, cheap, and - here - successful attack.

Weight-delta was not fooled, and not by a close margin. 0.99996 sits right alongside the
real gpt2 LoRA adapter already in `data/known_pairs.yaml` (0.99997, see
`reports/phase1_report.md`). This is expected given how LoRA works mathematically: the
adapter adds a low-rank update to a small set of weight matrices, bounded by its rank (8
here) regardless of how aggressively it's trained or how much it changes the model's
output on any particular input. Fifteen epochs of training on 60 short examples was
enough to flip the model's behavior on exactly those 60 prompts without moving its
weights far enough for weight-delta to notice.

## What this does and does not demonstrate

This demonstrates one specific attack (probe-aware LoRA fine-tuning) against one specific
signal (output-distribution) on one specific model pair. It does not demonstrate that
output-distribution is useless - the attack required knowing or guessing the exact probe
set, which a private or rotating probe set would raise the cost of, though it would not
eliminate it the way a structurally different signal does. It does not demonstrate that
weight-delta is unbeatable - fooling a weight-based signal would require an intervention
large enough to move the model meaningfully in weight space, which is a fundamentally
different (and more expensive, more detectable-by-other-means) kind of attack than this
one, but this project did not attempt to construct one.

This is the concrete argument for running both signals rather than either alone: on this
one tested case, defeating output-distribution cost nothing in weight space, and
weight-delta caught it without needing to know anything about the adversary's method.
That is not a general proof that two signals are always better than one - it is one
worked example, reported honestly, of exactly the complementary-failure-mode pattern
`reports/phase1_report.md` already found between these two signals on ordinary (non-
adversarial) pairs: weight-delta gets output-distribution's false positives right, and
output-distribution gets weight-delta's false negative right. A targeted adversary is a
harder case than either of those ordinary failures, and the pattern held anyway.
