# Phase 1 report: output-distribution fingerprinting

This covers the first and only verification signal implemented so far: output-distribution
fingerprinting (`src/provenance/signals/output_distribution.py`). It runs a fixed 60-prompt
probe set (`data/probes_v1.yaml`) through two models and measures how similar their next-token
predictions are.

Every number below came from running `python eval/harness.py` against
`data/known_pairs.yaml` on 2026-10-03. That command reproduces this report from a clean
checkout; nothing here is hand-adjusted.

## Method, in one paragraph

When the two models share a tokenizer (true for every fine-tune, LoRA adapter, and
distillation pair in the evaluation set, since those operations almost always keep the
base model's tokenizer), the signal feeds each probe through both models with teacher
forcing and compares the full next-token probability distribution at every position:
mean cosine similarity and mean KL divergence, averaged across all 60 probes. When the
two models do not share a tokenizer, a direct vector comparison is not meaningful, since
token id `i` is not the same string in both vocabularies. The signal falls back to
comparing the decoded top-50 predicted tokens at the end of the prompt (Jaccard overlap).
`score` is cosine similarity in the first case and Jaccard overlap in the second. Both are
bounded in [0, 1], but they are not the same quantity; see Limitation 1 below before
reading too much into a single shared threshold.

## Evaluation set

11 pairs from `data/known_pairs.yaml`: 5 known-related (full fine-tune, LoRA, and
distillation, across the gpt2, pythia-160m, and TinyLlama families) and 6 known-unrelated,
including one hard negative (gpt2 vs gpt2-medium) chosen specifically because it shares a
tokenizer with a related pair in the set, unlike every other unrelated pair here.

## Measured: per-pair scores

| model_a | model_b | relationship | expected | score | vocab shared | method |
|---|---|---|---|---|---|---|
| gpt2 | lvwerra/gpt2-imdb | full_finetune | related | 0.869 | yes | cosine |
| gpt2 | monsterapi/gpt2_alpaca-lora | lora | related | 0.794 | yes | cosine |
| gpt2 | distilgpt2 | distillation | related | 0.768 | yes | cosine |
| EleutherAI/pythia-160m | edbeeching/pythia-160M | full_finetune | related | 0.746 | yes | cosine |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | adamabuhamdan/tinyllama-sql-lora | lora | related | 0.962 | yes | cosine |
| gpt2 | EleutherAI/pythia-160m | unrelated | unrelated | 0.446 | no | jaccard |
| gpt2 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | unrelated | unrelated | 0.226 | no | jaccard |
| EleutherAI/pythia-160m | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | unrelated | unrelated | 0.234 | no | jaccard |
| distilgpt2 | EleutherAI/pythia-160m | unrelated | unrelated | 0.433 | no | jaccard |
| monsterapi/gpt2_alpaca-lora | EleutherAI/pythia-160m | unrelated | unrelated | 0.319 | no | jaccard |
| **gpt2** | **gpt2-medium** | **unrelated** | **unrelated** | **0.796** | **yes** | **cosine** |

The five related pairs score between 0.746 and 0.962. Five of the six unrelated pairs
score between 0.226 and 0.446. The sixth unrelated pair, bolded above, scores 0.796,
inside the related range. That pair is this report's documented failure case; see below.

## Measured: calibration across thresholds

Full sweep, 21 thresholds from 0.0 to 1.0, computed by `eval/harness.py`:

| threshold | TP | FP | TN | FN | accuracy | FPR | FNR |
|---|---|---|---|---|---|---|---|
| 0.00 | 5 | 6 | 0 | 0 | 0.455 | 1.000 | 0.000 |
| 0.20 | 5 | 6 | 0 | 0 | 0.455 | 1.000 | 0.000 |
| 0.25 | 5 | 4 | 2 | 0 | 0.636 | 0.667 | 0.000 |
| 0.35 | 5 | 3 | 3 | 0 | 0.727 | 0.500 | 0.000 |
| 0.45 | 5 | 1 | 5 | 0 | **0.909** | **0.167** | **0.000** |
| 0.70 | 5 | 1 | 5 | 0 | 0.909 | 0.167 | 0.000 |
| 0.75 | 4 | 1 | 5 | 1 | 0.818 | 0.167 | 0.200 |
| 0.80 | 2 | 0 | 6 | 3 | 0.727 | 0.000 | 0.600 |
| 0.90 | 1 | 0 | 6 | 4 | 0.636 | 0.000 | 0.800 |
| 1.00 | 0 | 0 | 6 | 5 | 0.545 | 0.000 | 1.000 |

(Intermediate thresholds between the rows above are flat; the full 21-row sweep is in the
harness's JSON output and does not add information beyond what's shown here.)

The best threshold **on this set** is 0.45-0.70: accuracy 0.909 (10/11), false-positive
rate 0.167 (1/6), false-negative rate 0.000 (0/5).

## Interpretation

Within this evaluation set, output-distribution similarity separates known-related from
known-unrelated model pairs reasonably well, with one clear exception. The signal never
misses a true positive at any threshold up to 0.70 (FNR 0.000), and produces exactly one
false positive at the best threshold.

## Hypothesis

If this pattern holds on a larger set, output-distribution fingerprinting with a
cosine-similarity threshold around 0.45-0.7 would be more useful as a high-recall first
pass (flag anything above the threshold for further evidence) than as a standalone
verdict, because of the specific failure mode below.

## The documented failure case: gpt2 vs gpt2-medium

**What was measured**: gpt2 and gpt2-medium are not derived from each other (gpt2-medium
is a separately trained checkpoint at a different parameter count, same architecture
family and tokenizer, from the same organization). The signal scores this pair at 0.796
cosine similarity, higher than three of the five genuine positive pairs (LoRA on gpt2:
0.794, distillation gpt2→distilgpt2: 0.768, full fine-tune on pythia-160m: 0.746).

**Why this happens**: every other unrelated pair in the evaluation set has a different
tokenizer, which the signal's fallback method handles separately and which happens to
produce low scores. gpt2-medium is the only unrelated model in the set that shares a
tokenizer with something else in the set, so it is the only case that actually exercises
the cosine-similarity comparison in the way it would be exercised in the related pairs.
It turns out that models from the same family trained on similar data, independently,
produce output distributions similar enough on generic natural-language probes that
cosine similarity alone cannot reliably separate "fine-tuned from" from "independently
trained sibling model, same family." The probe set here is generic (capital cities,
common phrases, simple arithmetic); none of it is designed to stress specific behaviors
that a fine-tune would change and an independently trained same-family model would not.

**Consequence for the eval set, not just this pair**: because every *other* unrelated
pair in this set has a mismatched tokenizer, the signal's vocab-compatibility check could
achieve 5/6 correct unrelated classifications by itself, without the distributional
comparison doing any real work. Before this pair was added, the evaluation set did not
actually test whether the signal detects fine-tuning, only whether it detects a tokenizer
mismatch. That gap, and the result once it was closed, is the main finding of this phase.

## Limitations

1. **Two measurement regimes, one shared threshold.** Cosine similarity (same-vocab
   pairs) and top-k Jaccard overlap (different-vocab pairs) are different quantities that
   happen to both land in [0, 1]. Thresholding them together, as the harness does to
   produce a single accuracy/FPR/FNR number, is only defensible because the measured
   values for this set happen to separate reasonably; it is not justified by any shared
   theoretical scale.
2. **The threshold was chosen on the same 11 pairs used to report its accuracy.** There is
   no held-out validation set. "Best threshold on this set" is not a validated production
   threshold, and should not be read as one.
3. **The sample is small.** 11 pairs means every single reclassification moves the
   reported accuracy by about 9 percentage points. The FPR of 0.167 is 1 out of 6
   unrelated pairs; one more or fewer hard negative in that group would change it to 0.0
   or 0.33.
4. **All five related pairs preserve the tokenizer**, because fine-tuning, LoRA, and the
   distillation example used here all do that in practice. The signal has not been tested
   on a case where a derivative model changes its tokenizer (e.g. some merges, or
   retraining with a new vocabulary), where the same-vocab code path would not apply at
   all.
5. **No quantized or merged model has been tested yet.** That is Phase 2.

## What this does not claim

This signal does not claim to detect fine-tuning in general. It claims that, on this
specific 11-pair set with this specific probe set, a cosine-similarity-or-Jaccard score
around 0.45-0.7 separated known-related from known-unrelated pairs with one error, and
that the one error has a specific, understood cause (same-family models trained
independently can look as similar as true derivatives, under probes that don't target
behavior a fine-tune specifically changes).
