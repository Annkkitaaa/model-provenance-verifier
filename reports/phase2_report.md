# Phase 2 report: robustness under modification

This covers two questions: does the output-distribution signal from Phase 1 still detect
a known derivative relationship once the derivative model has been quantized, and does it
(along with weight-delta, the second signal added later) still detect a model merge.

Reproduce with:

```bash
python eval/phase2_quantization.py
python eval/phase2_4bit_quantization.py
python eval/phase2_merge.py
```

## Quantization

### int8 dynamic quantization

The project's build plan named bitsandbytes and GGUF as the quantization methods to test.
Neither was used in the first pass of this phase: bitsandbytes was assumed to require a
GPU, so `eval/phase2_quantization.py` used `torch.quantization.quantize_dynamic` instead,
PyTorch's native CPU quantization (int8, weights-only, dynamic - activations computed in
fp32 at inference time).

Pair: `gpt2` (base, fp32) vs `lvwerra/gpt2-imdb` (full fine-tune, quantized).

| condition | score |
|---|---|
| derivative in fp32 (Phase 1 baseline) | 0.869 |
| derivative quantized to int8 (dynamic) | 0.751 |

Score dropped by about 0.12. Still inside the related range found in Phase 1 (0.45-0.70
was the best-accuracy plateau; everything at or above ~0.70 stayed classified related).

### Correction: bitsandbytes does work on this CPU-only machine

The assumption above was wrong, and this report said so without checking - "bitsandbytes'
int8/4-bit kernels are built for GPU inference" was stated as fact when it was an
unverified assumption. `eval/phase2_4bit_quantization.py` installs bitsandbytes
(`pip install bitsandbytes`, not in requirements.txt since only this one script needs it)
and loads a real NF4-quantized model with `BitsAndBytesConfig(load_in_4bit=True)`. It
works: forward passes run on CPU with no error. This was verified directly rather than
taken on faith - the quantized `Linear4bit` layers store weights as `torch.uint8` at
exactly 1/8 the byte count of the fp32 original (4 bits per weight, two weights packed per
byte), and the full model is about 40% of its fp32 size (less than the theoretical 1/8
because embeddings and layer norms are not quantized by default). GGUF conversion
(llama.cpp-style) was still not attempted.

| condition | score |
|---|---|
| derivative in fp32 (Phase 1 baseline) | 0.869 |
| derivative quantized to int8 (dynamic, torch native) | 0.751 |
| derivative quantized to NF4 (4-bit, bitsandbytes) | 0.834 |

**The surprising part**: 4-bit NF4 degrades the signal *less* than 8-bit dynamic
quantization did, despite using half the bits per weight. This is not a contradiction.
NF4 ("NormalFloat4", from the QLoRA paper) places its quantization bins to match the
normal distribution neural network weights actually follow; torch's dynamic int8 uses
naive uniform min-max quantization with no such adaptation. A better-designed 4-bit
scheme beating a naive 8-bit one is exactly the result NF4 was published to demonstrate -
this is evidence it holds for this signal's score too, not a new discovery, but it was
measured here rather than assumed.

0.834 is comfortably inside the related range from Phase 1.

## Model merge

The optional second Phase 2 item: test against a merge, not just quantization. Uses
mergekit (https://github.com/arcee-ai/mergekit, via its Python API) to produce a linear
0.5/0.5 merge of `gpt2` and `lvwerra/gpt2-imdb`, then runs both verification signals
against each parent. A merge is related to every model that went into it, so the real
test is whether both parents get flagged, not just one.

(Note on tooling: mergekit's PyPI release, 0.1.4, has a pydantic v2 compatibility bug that
breaks a plain `pip install mergekit` in this environment with
`ConfiguredModuleArchitecture is not fully defined`. Installed the current GitHub version
instead: `pip install "git+https://github.com/arcee-ai/mergekit.git"`. mergekit is not in
requirements.txt since only this one script needs it.)

| parent | weight-delta score | output-distribution score |
|---|---|---|
| gpt2 (base) | 0.99999 | 0.955 |
| lvwerra/gpt2-imdb (derivative) | 0.99999 | 0.962 |

Both signals correctly flag the merge as related to both parents. The weight-delta scores
being so close to 1.0 makes sense given gpt2 and gpt2-imdb are themselves already at
0.9998 weight similarity (see `reports/phase1_report.md`), so averaging two nearly
identical models stays nearly identical to both. Both output-distribution scores are
comfortably inside the related range from Phase 1.

## What survives modification and what doesn't

- **int8 dynamic quantization**: output-distribution survives (0.751, still related).
- **NF4 4-bit quantization**: output-distribution survives, and better than int8 did
  (0.834).
- **Linear model merge**: both signals survive, correctly flagging the merge as related
  to each parent.
- Quantization was only tested on output-distribution, not weight-delta (a quantized
  model's weights are stored in a packed, dequantization-dependent format that
  `weight_delta.py` does not currently handle; this is an untested gap, not a measured
  failure).
- Neither quantization test used GGUF/llama.cpp conversion.

## What was not done in Phase 2

- No GGUF-converted model was tested.
- Only one of the five Phase 1 positive pairs was tested under quantization.
- Only one merge method (linear) and one pair was tested; SLERP, TIES, and other mergekit
  methods were not tried.
