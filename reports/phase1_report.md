# Phase 1 report: output-distribution fingerprinting

This covers the first verification signal implemented: output-distribution fingerprinting
(`src/provenance/signals/output_distribution.py`). It runs a fixed 60-prompt probe set
(`data/probes_v1.yaml`) through two models and measures how similar their next-token
predictions are. A second, independent signal (weight-delta similarity) was added later and
is covered in its own section below, since the two signals turn out to have different,
complementary failure modes.

Every number below came from running `python eval/harness.py` against
`data/known_pairs.yaml` on 2026-10-03. That command reproduces this report from a clean
checkout; nothing here is hand-adjusted. (The 12th pair, added after the initial run to give
the weight-delta signal a same-architecture negative example, was scored separately and
merged into these results using the harness's own threshold-sweep and confusion-matrix
functions rather than by hand, to avoid arithmetic slips; see
`reports/weight_delta_eval_results.json` for how that pair was found.)

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

12 pairs from `data/known_pairs.yaml`: 5 known-related (full fine-tune, LoRA, and
distillation, across the gpt2, pythia-160m, and TinyLlama families) and 7 known-unrelated,
including two hard negatives chosen specifically because they share a tokenizer and
architecture with a related pair in the set, unlike the other unrelated pairs here (see
"The documented failure case" below for why both exist).

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
| **gpt2** | **stanford-crfm/alias-gpt2-small-x21** | **unrelated** | **unrelated** | **0.800** | **yes** | **cosine** |

The five related pairs score between 0.746 and 0.962. Five of the seven unrelated pairs
score between 0.226 and 0.446. The remaining two unrelated pairs, bolded above, score
0.796 and 0.800 - both inside the related range, both wrong. That pattern, not a single
anomaly, is this report's documented failure case; see below.

## Measured: calibration across thresholds

Full sweep, 21 thresholds from 0.0 to 1.0, computed by `eval/harness.py`:

| threshold | TP | FP | TN | FN | accuracy | FPR | FNR |
|---|---|---|---|---|---|---|---|
| 0.00 | 5 | 7 | 0 | 0 | 0.417 | 1.000 | 0.000 |
| 0.20 | 5 | 7 | 0 | 0 | 0.417 | 1.000 | 0.000 |
| 0.25 | 5 | 5 | 2 | 0 | 0.583 | 0.714 | 0.000 |
| 0.35 | 5 | 4 | 3 | 0 | 0.667 | 0.571 | 0.000 |
| 0.45 | 5 | 2 | 5 | 0 | **0.833** | **0.286** | **0.000** |
| 0.70 | 5 | 2 | 5 | 0 | 0.833 | 0.286 | 0.000 |
| 0.75 | 4 | 2 | 5 | 1 | 0.750 | 0.286 | 0.200 |
| 0.80 | 2 | 0 | 7 | 3 | 0.750 | 0.000 | 0.600 |
| 0.90 | 1 | 0 | 7 | 4 | 0.667 | 0.000 | 0.800 |
| 1.00 | 0 | 0 | 7 | 5 | 0.583 | 0.000 | 1.000 |

(Intermediate thresholds between the rows above are flat; the full 21-row sweep is in
`reports/phase1_eval_results.json` and does not add information beyond what's shown here.)

The best threshold **on this set** is 0.45-0.70: accuracy 0.833 (10/12), false-positive
rate 0.286 (2/7), false-negative rate 0.000 (0/5).

## Interpretation

Within this evaluation set, output-distribution similarity separates known-related from
known-unrelated model pairs reasonably well, with one systematic exception that now shows
up twice independently. The signal never misses a true positive at any threshold up to
0.70 (FNR 0.000), and produces exactly two false positives at the best threshold, both of
the same kind.

## Hypothesis

If this pattern holds on a larger set, output-distribution fingerprinting with a
cosine-similarity threshold around 0.45-0.7 would be more useful as a high-recall first
pass (flag anything above the threshold for further evidence) than as a standalone
verdict, because of the specific failure mode below - and because that failure mode is
not rare: two different independently-trained same-family models were tried, and both
triggered it.

## The documented failure case: two independent same-family false positives

**What was measured**: gpt2-medium and stanford-crfm/alias-gpt2-small-x21 are each
unrelated to gpt2 - neither is derived from it. gpt2-medium is a separately trained
OpenAI checkpoint at a different parameter count. stanford-crfm/alias-gpt2-small-x21 is
part of Stanford CRFM's "Mistral" project, which pretrains multiple GPT-2-small
reproductions from scratch on OpenWebText to study training stability; its own repo files
(raw DeepSpeed checkpoint shards at step 400,000 across 8 ranks) are what a distributed
pretraining run produces, not what fine-tuning an existing checkpoint produces. Both
share gpt2's exact tokenizer and architecture. The signal scores both pairs around 0.80
cosine similarity - higher than three of the five genuine positive pairs (LoRA on gpt2:
0.794, distillation gpt2->distilgpt2: 0.768, full fine-tune on pythia-160m: 0.746).

**Why this happens**: every other unrelated pair in the evaluation set has a different
tokenizer, which the signal's fallback method handles separately and which happens to
produce low scores. gpt2-medium and the Stanford CRFM model are the only unrelated models
in the set that share a tokenizer with something else in the set, so they are the only
cases that actually exercise the cosine-similarity comparison the way the related pairs
do. It turns out that models from the same family, sharing a tokenizer and architecture
and trained on similar web-text data, produce output distributions similar enough on
generic natural-language probes that cosine similarity alone cannot reliably separate
"fine-tuned from" from "independently trained sibling model, same family." This held for
two different sibling models, not one, which is stronger evidence that it is a property
of the signal rather than a quirk of one specific model. The probe set here is generic
(capital cities, common phrases, simple arithmetic); none of it is designed to stress
specific behaviors that a fine-tune would change and an independently trained same-family
model would not.

**Consequence for the eval set, not just these pairs**: because every *other* unrelated
pair in this set has a mismatched tokenizer, the signal's vocab-compatibility check could
achieve 5/7 correct unrelated classifications by itself, without the distributional
comparison doing any real work. Without these two pairs, the evaluation set would not
actually test whether the signal detects fine-tuning, only whether it detects a tokenizer
mismatch. That gap, and the result once it was closed (twice), is the main finding of
this phase.

## Second signal: weight-delta similarity

A second, independent verification signal was added later:
`src/provenance/signals/weight_delta.py` compares model weights directly, tensor by
tensor, instead of comparing behavior on probe prompts. It only applies when both models
share identical architecture (same named parameters, same shapes) - true for a full
fine-tune or a LoRA adapter merged into its base, but not for distillation (different
layer count) or cross-architecture pairs, which it reports as not applicable rather than
forcing a score. Full results: `python eval/weight_delta_harness.py`, raw output in
`reports/weight_delta_eval_results.json`.

Of the 12 known pairs, weight-delta is applicable to only 4 (the rest have mismatched
architectures, or - for the one TinyLlama positive pair - are skipped by a memory-safety
cap documented in the signal itself):

| model_a | model_b | expected | weight-delta score | output-distribution score |
|---|---|---|---|---|
| gpt2 | lvwerra/gpt2-imdb | related | 0.9998 | 0.869 |
| gpt2 | monsterapi/gpt2_alpaca-lora | related | 0.99997 | 0.794 |
| EleutherAI/pythia-160m | edbeeching/pythia-160M | related | **0.476** | 0.746 |
| gpt2 | stanford-crfm/alias-gpt2-small-x21 | unrelated | 0.191 | 0.800 |

At a threshold of 0.5 (not tuned by a sweep - see `eval/weight_delta_harness.py` for why
a sweep isn't meaningful over 4 points): 2 true positives, 1 true negative, **1 false
negative**. The pythia-160m fine-tune scores only 0.476 on weight similarity, below the
threshold, while the two gpt2 derivatives score above 0.999.

**Why this happens (hypothesis, not confirmed)**: LoRA's low-rank update is mathematically
bounded to a small delta by construction, so the gpt2 LoRA pair scoring 0.99997 is
expected. Full fine-tuning has no such bound - how far the weights move depends entirely
on training steps, learning rate, and other choices made by whoever trained the
derivative, which this project does not control and was not told. The gpt2-imdb and
pythia-160m fine-tunes were trained by different people with different scripts; it is
plausible the pythia one used more steps or a higher learning rate, but this was not
verified against either training script and should be read as a hypothesis.

**The two signals fail differently, which is the point of having both**: weight-delta
gets both of output-distribution's false positives right - gpt2-medium is structurally
not applicable (different shape, so weight-delta never claims a verdict there at all) and
the Stanford CRFM pair scores 0.191, clearly unrelated. Output-distribution gets
weight-delta's one false negative right - the pythia-160m pair scores 0.746 on behavior,
comfortably in the related range. Neither signal is strictly better; they are wrong on
different pairs. A real deliberate-evasion test
(`eval/evasion_test.py`, written up in `reports/evasion_report.md`) found a case that
fools output-distribution specifically while leaving weight-delta essentially untouched,
which is the concrete argument for running both rather than picking one.

## Limitations

1. **Two measurement regimes, one shared threshold (output-distribution).** Cosine
   similarity (same-vocab pairs) and top-k Jaccard overlap (different-vocab pairs) are
   different quantities that happen to both land in [0, 1]. Thresholding them together,
   as the harness does to produce a single accuracy/FPR/FNR number, is only defensible
   because the measured values for this set happen to separate reasonably; it is not
   justified by any shared theoretical scale.
2. **Thresholds were chosen on the same set used to report their accuracy**, for both
   signals. There is no held-out validation set. "Best threshold on this set" is not a
   validated production threshold for either signal.
3. **The sample is small.** 12 pairs (output-distribution) or 4 applicable pairs
   (weight-delta) means every single reclassification moves the reported accuracy by a
   large step. The FPR of 0.286 is 2 out of 7 unrelated pairs; one more or fewer hard
   negative would change it substantially.
4. **All five related pairs preserve the tokenizer**, because fine-tuning, LoRA, and the
   distillation example used here all do that in practice. Output-distribution has not
   been tested on a derivative that changes its tokenizer.
5. **Weight-delta has not been tested on a quantized or merged model directly for false
   positives/negatives** beyond what's in `reports/phase2_report.md` - the merge test
   there checks it still correctly flags a merge as related, not whether it stays robust
   under adversarial conditions.

## What this does not claim

Neither signal claims to detect fine-tuning in general. Output-distribution claims that,
on this specific 12-pair set with this specific probe set, a cosine-similarity-or-Jaccard
score around 0.45-0.7 separated known-related from known-unrelated pairs with two errors,
both of the same understood kind (same-family models trained independently can look as
similar as true derivatives, under probes that don't target behavior a fine-tune
specifically changes). Weight-delta claims that, on the 4 pairs it could even be computed
on, a cosine-similarity threshold around 0.5 got 3 right and 1 wrong, with its own
understood failure mode (full fine-tuning has no bound on how far it can move weights,
unlike LoRA). Running both and requiring agreement is not something this project
evaluated as a combined strategy; see `reports/evasion_report.md` for why that would
plausibly help.
