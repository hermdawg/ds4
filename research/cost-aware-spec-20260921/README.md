# Cost-aware speculative scheduling: no production change

2026-09-21. Apple M5 Max, 64 GiB. DwarfStar baseline `9139e2a`.

**The campaign did not establish a model-level speculative speedup.** The only
installed compatible main model needs SSD streaming and has no matching installed
drafter. The authorized fallback produced two isolated policy prototypes,
source-backed scheduler simulations, real GPU timing experiments, ordinary-model
baselines, correctness checks and raw measurements. Production decoding remains
unchanged. No checkpoints were downloaded and no training jobs were launched.

The campaign started at 20:44:55 UTC and stopped early after completing the
fallback study. Further end-to-end speculative experiments require matched
draft weights that are not installed. The eight-hour limit was a ceiling;
additional invented-cost trials cannot resolve that blocker.

Open [the report](report.html) for the result tables, regressions and NVIDIA
transfer discussion. [LEDGER.md](LEDGER.md) records hypotheses, failures and
milestone decisions. [EXPERIMENTS.md](EXPERIMENTS.md) records the frozen protocol
and break-even derivation. [SOURCES.md](SOURCES.md) distinguishes local code from
external prior art. Everything in `raw/` is retained, including negative results.

## Evidence boundaries

- `model-*`: real ordinary inference, SSD streaming, three repetitions per
  code/prose/structured raw continuation at 256 and 2048 input tokens, 256 output
  tokens. No model-backed speculative comparison is available.
- `heldout-*`, `stress-*`, `tune-*`: **synthetic costs and acceptance**, never
  model throughput. The default scheduler's decisions call the real local C
  implementation through `bridge.c`. The execution costs are still invented.
- `kernel-cost*`: real isolated Q8 projections with independent numerical
  references. These are not transformer, drafter or verifier end-to-end timings.
- `streaming-profile-*`: real existing SSD instrumentation off/on, eight runs
  in ABBA/BAAB order. All generated text matches exactly.
- `policy-overhead*`: host-only bookkeeping/timer microbenchmark.

Synthetic speedups use total generation time per committed token, including
exploration and output tails. A committed seed counts as useful output but never
as an accepted draft. Stage times are non-overlapping in the synthetic model.
Real SSD diagnostic timers can overlap and must not be summed into total time.

H1 explores ordinary plus caps 1-5. H2 explores ordinary plus caps 3 and 5.
Both use ratio estimates, periodic exploration, context buckets, and unchanged
output limits. The first held-out test is primary. H2's reuse of that test is
exploratory; a separately seeded sensitivity test is also supplied. Bootstrap
intervals cover stochastic seeds in six assumed regimes, not real-model prompt
variation or production-serving uncertainty.

## Reproduce

Run from this worktree's repository root. Build dependencies are the normal
Apple Command Line Tools/Metal SDK and Python 3 standard library. Keep one GPU
job running at a time. Every long experiment below uses a process-group timeout.
The wrappers also propagate cancellation through nested runners.

```sh
R=research/cost-aware-spec-20260921
make -j8 all ds4_test tests/test_sampling tests/test_session_state \
  tests/test_session_state_gpu tests/test_qwen4_kernels \
  tests/test_metal_moe_prefill tests/test_metal_ssd_experts \
  tests/test_glm53_kda tests/test_q8_prefill_variants
sh "$R/build.sh"

# Research-only policy and analytical/source-backed simulator checks.
cc -O2 -g -std=c99 -Wall -Wextra -Werror -fsanitize=address,undefined \
  "$R/test_policy.c" -o "$R/policy_test"
python3 "$R/run_bounded.py" --timeout 60 --output /tmp/spec-policy-test.log \
  -- "$R/policy_test"
python3 "$R/test_simulate.py"
python3 "$R/test_runner.py"

# Primary H1 result with its pre-validation configuration.
python3 "$R/run_bounded.py" --timeout 300 --output /tmp/spec-h1.log -- \
  python3 "$R/simulate.py" --split heldout --config "$R/candidate-config.json" \
  --output /tmp/spec-h1.csv --trace
# H2 on the original holdout is explicitly exploratory.
python3 "$R/run_bounded.py" --timeout 300 --output /tmp/spec-h2.log -- \
  python3 "$R/simulate.py" --split heldout --config "$R/candidate-config-h2.json" \
  --output /tmp/spec-h2.csv --trace
# This overwrites the reproducible stress CSVs in raw/.
python3 "$R/run_bounded.py" --timeout 1200 --output /tmp/spec-stress.log -- \
  python3 "$R/stress.py"

# Existing correctness tests; these do not load a large CPU model.
./tests/test_sampling
./tests/test_session_state
./tests/test_session_state_gpu
./tests/test_qwen4_kernels
./tests/test_metal_moe_prefill
./tests/test_metal_ssd_experts
./tests/test_glm53_kda
./tests/test_q8_prefill_variants
./ds4_test --metal-kernels
./ds4_test --server
```

To repeat real GPU timing and the ordinary model baseline:

```sh
R=research/cost-aware-spec-20260921
cc -O3 -g -std=c99 -Wall -Wextra -Werror -I. "$R/kernel_cost_bench.c" \
  ds4_metal.o ds4_image.o -lm -pthread -framework Foundation -framework Metal \
  -o "$R/kernel_cost_bench"
python3 "$R/run_bounded.py" --timeout 600 --output "$R/raw/kernel-cost.log" \
  -- "$R/kernel_cost_bench"
cc -O3 -std=c99 -Wall -Wextra "$R/policy_overhead.c" -o "$R/policy_overhead"
python3 "$R/run_bounded.py" --timeout 120 --output /tmp/spec-overhead.csv \
  -- "$R/policy_overhead"

# Uses the existing absolute model path in model-inputs/manifest.json.
# Never run these two model scripts concurrently.
python3 "$R/run_bounded.py" --timeout 7200 --output "$R/raw/model-suite.log" \
  -- python3 "$R/run_model_baseline.py"
python3 "$R/run_bounded.py" --timeout 3000 --output "$R/raw/streaming-profile-suite.log" \
  -- python3 "$R/profile_streaming.py"
python3 "$R/analyze.py"
python3 "$R/analyze_instrumentation.py"
python3 "$R/make_report.py"
```

The frozen inputs already exist. Do not rerun the freeze scripts: they refuse
overwrite. `tune.py` and `tune.py --h2` reproduce calibration on tuning seeds
only, but overwrite the selected configuration files. The report states which
configuration was committed before each validation run.

## What remains unverified

No policy is called by `ds4.c`. No new GPU kernels, sampling rules, cache handling,
CLI options or defaults were introduced. Real speculative rejection recovery,
greedy divergence analysis, recurrent-state reuse, exact nonzero-temperature
model sampling and end-to-end speculative instrumentation overhead require a
compatible target/drafter pair. Existing unit tests and preserved production code
do not establish those properties for a future integration. CUDA and distributed
inference were not changed or exercised.

The next useful experiment is a resident matched target/drafter run with measured
ordinary/draft/verification/recovery costs. Start with fixed caps 0, 3 and 5, then
compare a cost/confidence policy against default in interleaved runs. On NVIDIA
serving, include concurrent-request batches and optimize aggregate useful
throughput (tok/s/GPU), while tracking TPS (tok/s/request) and latency. A policy
that improves one request can consume compute needed by others.
