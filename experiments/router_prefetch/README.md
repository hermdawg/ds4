# Early-router prediction probe

Measure whether V4 Flash's future expert choices can be predicted with high
precision. This experiment uses the existing Metal debug dumps. It does not
modify routing, fetch predicted experts, or measure a prefetch speedup.

## Scope and memory

The first target is Flash Q2: it is the smallest main model supported by ds4's
expert SSD streaming path at source commit `9139e2a`. In the tested 64 GiB M5 Max
configuration, ds4 reports 16.38 GiB of planned allocations, including an 8 GiB
combined expert cache/prefill reserve. This excludes reclaimable macOS file
cache and other system/application memory. The model file is about 81 GiB.

No second language model is loaded and no model is trained. Collection runs one
model process at a time. Analysis reads small router and normalization tensors
from the GGUF, plus one saved activation trace at a time. The full GGUF is
memory-mapped by the reader; only requested tensor pages are accessed.

## Reproduce

Run from the repository root. `uv` resolves the Python dependencies declared at
the top of `probe.py`. Use the existing built binary and the Flash Q2 GGUF.

```sh
uv run experiments/router_prefetch/probe.py collect \
  --binary ./ds4 --model ./ds4flash.gguf \
  --prompts experiments/router_prefetch/prompts.json \
  --output gguf/router-prefetch-run --tokens 128

uv run experiments/router_prefetch/probe.py analyze \
  --input gguf/router-prefetch-run --model ./ds4flash.gguf \
  --output gguf/router-prefetch-run/results.json

uv run --with numpy python -m unittest discover \
  -s experiments/router_prefetch -p 'test_*.py'
```

Use a fresh output directory. Collection saves the prompt, source commit,
generation output and diagnostics. It packs complete decode traces and removes
its raw dumps only after checking the packed copy. The debug hooks also emit
multi-row prefill chunks under the same tensor names; these are excluded from
the decode analysis. Debug synchronization makes collection timings unsuitable
for inference benchmarking.

## Methods

All methods reuse the target layer's existing router matrix and selection bias.
They predict identities only; the actual model continues to use its original
router, selected experts and mixture weights.

- `pre_attention`: apply the target FFN normalization and router to the same
  layer's collapsed hidden vector before attention. Hyper-connections mean this
  is an approximation even before accounting for attention's contribution.
- `ahead1`, `ahead2`: run a target router using the normalized FFN input from one
  or two layers earlier.
- `ahead1_renorm`, `ahead2_renorm`: use the earlier unnormalized FFN input with
  the target layer's normalization weights instead.

As a reconstruction check, first run each router on its actual input and compare
the resulting six-expert set with the recorded GPU selection. Report the logit
error as well. The first three token-hash layers are excluded as targets.

For a candidate expert, confidence is its predicted score minus the seventh
highest predicted score, divided by the standard deviation of all 256 scores.
This margin is a heuristic, not a probability. We issue at most one candidate
per target layer/position, chosen from the predicted top six.

Four calibration prompts select a global confidence threshold per method and
scope. The threshold must achieve at least 99% observed precision with at least
200 predictions. Tied confidence values stay together. Four separate test
prompts evaluate the frozen thresholds. Thresholds are never tuned on test
outcomes. No qualifying threshold is reported as such, rather than claiming
perfect precision for issuing no predictions.

The `all` scope selects the highest-scoring expert. The `shadow_lru_miss` scope
selects the highest-scoring top-six expert absent from a simulated 701-slot
global least-recently-used cache at the source layer. Cache keys include layer
and expert. This is a useful harder subset, but it is **not** ds4's production
cache policy or macOS's file cache. The first 16 decode positions are discarded
to reduce the effect of unknown initial prefill cache contents.

## Interpretation

Report prediction precision together with the fraction of opportunities for
which a prediction was issued. Top-six recall is a separate metric: it is the
fraction of the real selected set found by all six predictions.

The 95% binomial Wilson lower bounds are descriptive only. Adjacent tokens and
layers are correlated, so these are not reliable confidence bounds for new
prompts. This small curated corpus can reject a weak idea; it cannot establish
near-perfect precision across real workloads.

Before claiming acceleration, integrate a promising predictor with the actual
cache, give demand reads priority, and measure whether useful reads complete
before demand. Compare complete generation time, predictor overhead, physical
SSD traffic, cache displacement and output/logit parity against the unmodified
path. Additional reads are not automatically bad if they exploit idle time.

## First measurement: September 15, 2026

On a 64 GiB M5 Max, eight prompts each produced 127 recorded decode positions.
The offline router reconstruction matched all **40,640 learned-router sets**.
The maximum absolute logit difference from Metal was `6.2943e-5`. A separate
16-token smoke run produced byte-identical stdout with and without diagnostics;
all 600 learned-router sets in that smoke trace also reconstructed exactly.
The five analysis unit checks pass.

The model process used about 12.91 GiB resident memory at one observation.
Offline analysis took 9.85 seconds and peaked at 1.215 GiB resident memory.
These are feasibility measurements, not prefetch performance measurements.
Compressed traces occupy about 1.9 GiB on disk. The four calibration and four
test prompts are checked in; activation traces stay local under ignored `gguf/`.

The main comparison uses the shadow-cache-miss scope: at most one predicted
expert per layer/position, and only if it is absent from the simulated cache.
Each method's cutoff was selected for at least 99% calibration precision.

| Method | Correct / issued on test | Test precision | Fraction of eligible opportunities issued |
| --- | ---: | ---: | ---: |
| Before attention | 790 / 796 | 99.25% | 4.61% |
| One layer ahead | 3,797 / 3,832 | 99.09% | 22.50% |
| Two layers ahead | 2,172 / 2,200 | 98.73% | 12.83% |
| One layer ahead, target normalization | 3,694 / 3,725 | 99.17% | 21.87% |
| Two layers ahead, target normalization | 2,493 / 2,531 | 98.50% | 14.75% |

An eligible opportunity has at least one predicted top-six expert absent from
the shadow cache. These percentages are not the fraction of disk bytes hidden
or the fraction of actual production-cache misses covered.

A second, stricter calibration target of 99.9%, still requiring at least 200
calibration predictions, gave these results:

| Method | Correct / issued on test | Fraction of eligible opportunities issued |
| --- | ---: | ---: |
| Before attention | 265 / 267 | 1.55% |
| One layer ahead | 551 / 551 | 3.23% |
| One layer ahead, target normalization | 447 / 447 | 2.62% |
| Either two-layer method | No qualifying calibration cutoff | — |

Zero mistakes in 551 predictions does not establish perfect accuracy. This
second target is exploratory, and selecting a method after seeing these test
results means any follow-up should use fresh test prompts. The before-attention
method's strict cutoff illustrates the danger: its test precision remained only
99.25% despite the 99.9% calibration target.

The simplest one-layer-ahead method is the first candidate for further work.
It needs no trained auxiliary model. The strict cutoff offers a small set of
very reliable predictions; the less strict cutoff offers wider coverage with
about 1% wasted predicted reads on this sample. Neither establishes a speedup.
Next, evaluate against the actual cache on fresh prompts and measure whether
reads arrive before demand without delaying other work. All actual expert
selection must remain unchanged. Skipping or substituting experts is deferred.

Detailed outputs: [99% target](2026-09-15-results.json) and
[99.9% target](2026-09-15-results-strict.json). To reproduce the stricter analysis,
add `--precision 0.999` and use a separate output filename.
