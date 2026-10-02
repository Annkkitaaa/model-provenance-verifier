# Phase 2 report: robustness under quantization

This covers one question: does the output-distribution signal from Phase 1 still detect
a known derivative relationship once the derivative model has been quantized?

Reproduce with:

```bash
python eval/phase2_quantization.py
```

## Scope limitation, stated up front

The project's build plan named bitsandbytes and GGUF as the quantization methods to test.
Neither was used here. bitsandbytes' int8/4-bit kernels are built for GPU inference and
this machine is CPU-only; GGUF requires a separate conversion/inference stack
(llama.cpp-style) that was out of scope to integrate for one test. Instead, this uses
`torch.quantization.quantize_dynamic`, PyTorch's native CPU quantization: it replaces each
`nn.Linear` layer's weights with an int8 representation and computes activations in fp32
at inference time (weights-only, dynamic quantization). This is a real, commonly used
quantization method, but it is a milder form of quantization than the 4-bit
weight-and-activation quantization GGUF/bitsandbytes typically apply. **The result below
should not be read as "the signal survives quantization in general."** It should be read
as "the signal survives this specific, relatively mild quantization method, on this one
pair." A more aggressive quantization method could plausibly degrade the score further
than measured here; that was not tested.

## Measured

Pair: `gpt2` (base, fp32) vs `lvwerra/gpt2-imdb` (full fine-tune, quantized).

| condition | score | method |
|---|---|---|
| derivative in fp32 (Phase 1 baseline) | 0.869 | cosine similarity, teacher-forced |
| derivative quantized to int8 (dynamic) | 0.751 | cosine similarity, teacher-forced |

Score dropped by about 0.12 (precisely -0.1185, from the script's raw output). The
tokenizer is unaffected by quantization, so the comparison
still uses the same-vocabulary cosine-similarity path, not the weaker top-k fallback.

## Interpretation

0.751 is still inside the accuracy-maximizing threshold range found in Phase 1
(0.45-0.70 was the best-accuracy plateau on the 11-pair eval set; everything at or above
~0.70 was still classified "related" in that sweep, and 0.751 clears that). On this one
pair, under this one quantization method, the signal still correctly flags the derivative
relationship after quantization. It did not need to cross back below threshold to still
count as a positive result, and it didn't.

## Hypothesis

If dynamic int8 quantization typically shifts cosine-similarity scores down by something
in the range measured here (roughly 0.1-0.15), then pairs that scored in the lower half of
the related range in Phase 1 (the pythia-160m full fine-tune pair scored 0.746, barely
above the hard-negative false positive at 0.796) would be the ones at risk of false
negatives after quantization, not the pairs that scored higher. This is a hypothesis, not
a measurement: only one pair was tested, and the shift could easily be pair-specific
rather than a general property of the quantization method.

## What this does not claim

This does not claim the signal is robust to quantization in general, to 4-bit
quantization, to GGUF-converted models, or to any pair other than the one tested. It
claims one specific, reproducible measurement: on the gpt2 -> gpt2-imdb pair, int8 dynamic
quantization of the derivative model lowered the cosine-similarity score by about 0.12 but did
not flip the classification at the threshold chosen in Phase 1.

## What was not done in Phase 2

- No model merge was tested (the build plan's optional second Phase 2 item).
- No GPU-quantized (bitsandbytes 4-bit/8-bit) or GGUF-converted model was tested, for the
  environment reasons stated above.
- Only one of the five Phase 1 positive pairs was tested under quantization.
