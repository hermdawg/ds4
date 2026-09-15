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
