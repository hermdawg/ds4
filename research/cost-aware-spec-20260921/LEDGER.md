# Cost-aware speculative scheduling experiment ledger

Start: 2026-09-21 20:44:55 UTC (22:44:55 Europe/Stockholm).
Hard deadline: 2026-09-22 04:44:55 UTC. Final validation/reporting starts no later than 03:59:55 UTC.
Objective: test elapsed generation time per committed token against ordinary decoding, the current scheduler, and supported fixed depths without changing acceptance or target probabilities. Negative findings count.

## Frozen environment and scope

- Source baseline: `9139e2a`, the user's installed checkout. `git fetch` found upstream `0aaea5a`, 62 commits ahead and zero local-only commits. Deliberately freeze the installed version rather than mix a scheduler experiment with 62 unrelated updates.
- Worktree: `/Users/herman/code/open-source/ds4-cost-aware-spec`, branch `codex/cost-aware-spec-20260921`. Main checkout untouched.
- Push destination: user's `fork` (`hermdawg/ds4`), not upstream. Author: `hermdawg <herman@lovable.dev>`.
- Hardware confirmed by `sysctl`: Apple M5 Max / Mac17,7 / 68,719,476,736 bytes unified RAM.
- Read `AGENT.md`, `CONTRIBUTING.md`, speculative decoding and Qwen documentation. Do not change acceptance, kernels, CUDA, or distributed execution without validated need. No distributed test or broad CUDA port planned.
- Only installed GGUF found by Spotlight and searches of code/cache/downloads: DeepSeek V4 Flash 0731 IQ2XXS/w2Q2K, 86,720,111,488 bytes. It cannot be resident within 64 GiB. No matching DSpark support GGUF or Qwen/GLM GGUF found. Cached Qwen3.6 safetensors are not supported by this runner and will not be converted or treated as compatible.
- No checkpoint downloads. If metadata confirms missing embedded draft weights, use the explicitly authorized kernel/synthetic fallback. Any ordinary streamed model baseline will be labeled separately and never serve as evidence of speculative model speedup.

## Plan and gates

1. Audit existing scheduler, supported depth controls, timing boundaries, and state tests. Capture model metadata and build clean baseline with process timeouts.
2. Freeze prompt suite and synthetic workload families before tuning. Separate tuning from held-out validation. Greedy first; exact-sampling unit tests only if no compatible model.
3. Build a backend-independent, interpretable research policy and a reproducible harness. Keep production behavior unchanged unless model-level evidence justifies integration.
4. Compare ordinary, actual current scheduler, fixed supported depths, and candidates. Record stage costs, committed tokens, exploration, context, regret, overhead, variability, and failures. Synthetic costs and acceptance must be explicitly labeled assumptions.
5. Run relevant existing numerical/state/sampling/kernel tests. Check output limits, reset/reuse, recovery after backoff, adversarial changes, and instrumentation overhead. One GPU workload at a time.
6. Retain only verified work. Commit and push each milestone. Final report includes negative results, limitations, NVIDIA transfer, batching caveats, reproducible commands, and raw data.

## 20:45-20:51 UTC: inspection

Commands: `git status --short --branch`, `git remote -v`, `git fetch`, `git rev-list --left-right --count HEAD...origin/main`, `git var GIT_AUTHOR_IDENT`, `git worktree add -b codex/cost-aware-spec-20260921 /Users/herman/code/open-source/ds4-cost-aware-spec HEAD`; `sysctl hw.memsize hw.model machdep.cpu.brand_string`; `vm_stat`; `memory_pressure -Q`; `mdfind 'kMDItemFSName == "*.gguf"c'`; targeted `rg` searches.

Findings: DSpark already has acceptance windows, pauses, confidence-based backoff, optional elapsed-cost gates, and tail avoidance. Cost gates default to zero (off) explicitly to preserve schedule reproducibility. Default Metal window is four speculative cycles and minimum average acceptance is 1.5 drafts/cycle. Resident M5 seed batching has longer backoff than SSD streaming. ROCm fast-path can bypass the remainder of a request. Qwen switches between one and two speculative drafts using recent acceptance, but does not choose ordinary decode by elapsed cost; exact sampling is restricted to the shallow cycle. These are distinct policies and must not be conflated.

Next: confirm model metadata; freeze fallback suite; test baseline and implement research-only policy.

## 20:48-20:50 UTC: baseline build and preregistration

Clean worktree `make -j8 all ds4_test tests/test_session_state tests/test_sampling tests/test_metal_ssd_experts tests/test_qwen4_kernels` passed in 11.10 s. Model `--inspect` confirms 43 layers, 256 experts/6 selected, 80.76 GiB weights. No compatible speculative support file exists. The campaign therefore uses the authorized fallback; it cannot establish model-level speculative speedups.

Frozen `suite.json` plus SHA-256 contains distinct coding, prose and structured prompts, context frontiers 256 and 2048, 256 output tokens, tuning seeds 0-19 and held-out seeds 100-139. Synthetic costs/acceptance are deliberately assumed regimes, NOT measurements of those prompts. Resident seed batching and separate-seed execution will be evaluated separately. Model-based ordinary measurements, if run, remain a separate data set.

H1: An exponentially weighted ratio of total cycle milliseconds to committed tokens, with occasional depth exploration and context buckets, can avoid high-acceptance but expensive speculation. Compare against actual checked-in DSpark scheduling functions, including optional existing cost gates, and fixed draft caps 1-5. No acceptance or GPU code changes.

Correction: initial inspection happened approximately 20:45-20:47 UTC; the earlier section's 20:51 endpoint was an estimate, not an observed timestamp. All run metadata uses actual UTC timestamps.

## 20:51-20:57 UTC: H1 prototype and baseline checks

Implemented research-only `policy.h` plus `bridge.c`, which calls the actual checked-in DSpark scheduler for control decisions. Backend timing/execution is outside the policy. No production sources changed. The estimator divides decayed total time by decayed committed tokens; it includes ordinary decoding, depths 1-5, periodic probes and three context buckets. It shares the current ten-token tail guard.

H1 tuning result (20 seeds x six cases, each mode): geometric simulated speedup versus ordinary is 1.3621x candidate vs 1.3516x default with resident seed batching; 1.1784x vs 1.1497x with a separate seed. This is assumed-cost simulation only. The weakest candidate run was 0.9524x ordinary in the resident case, so mean gains are not a universal win. Existing optional cost gate also included (ratio threshold 1.0, four-cycle window).

Passed unchanged existing tests: sampling (100k trials each for exact stochastic and point-mass proposals), CPU session bookkeeping, GPU session rollback, Qwen kernels, MoE batched/reference comparisons, Metal SSD expert eviction and GLM recurrent kernels. Logs and exact commands are in `raw/`. Research policy passed AddressSanitizer and UndefinedBehaviorSanitizer checks for ratio accounting, invalid measurements, tail limits, context buckets, reset and recovery. Independent analytical simulator tests passed, including calling the real default scheduler and checking its pauses/reset, stage sums and output bounds.

Installed ordinary-only model smoke: 256-token raw code prefix, 16 generated tokens, 9.58 tok/s/request. SSD cache target 8 GiB, planned total 16.27 GiB. This is a startup smoke, not a steady-state result or speculative comparison. Frozen ordinary-only corpora and a separately labeled three-repeat suite now run serially, with memory/VM/disk/thermal snapshots and 900-second per-process timeouts.

Audit caveat: `DS4_DSPARK_VERIFY_CAP` limits final proposals, but the default confidence-lazy drafting loop still iterates to the full draft block until confidence stops it. A shorter verification cap need not reduce draft-stage cost. The initial synthetic suite's depth-linear draft-cost assumption is therefore a generic hypothetical, not an exact DSpark cost model. A flat draft-cost sensitivity experiment is required before interpreting depth-selection gains.

Next bounded step: calibrate decay and exploration interval on the frozen tuning split only (16 configurations). Select by mean normalized elapsed time with a penalty if a scenario mean exceeds a 5% ordinary-decode regression. Then freeze configuration before held-out evaluation. Also measure host timer/policy overhead and GPU timing-boundary overhead separately.

## 20:58 UTC: tuning configuration frozen

The 16-point tuning grid selected decay 0.95, exploration every 32 committed tokens, and a 3% improvement margin, with context buckets enabled. `candidate-config.json` records this choice before held-out execution. All simulated costs, domains and seeds remain as preregistered. This is calibration of H1, not proof of a deployable policy. Next: held-out paired comparisons and diagnostic stress cases; do not retune on their results.

## 20:59-21:01 UTC: H1 held-out negative result, next bounded experiment

H1 held-out resident simulation: 1.1883x ordinary vs 1.2064x for current default, about a 1.5% geometric regression against default. This falsifies a robust resident gain in the primary assumed workload suite. No production integration. See raw heldout-h1 CSVs, stage traces and summary.

H2 is preregistered in EXPERIMENTS.md: reduce exploration to a useful subset of caps, motivated by the existing cap-3 seed-batch threshold and H1 TUNING results. Try masks {0..5}, {0,3,5}, {0,5}, tune on original tuning split only, commit config before evaluation. Any re-use of original held-out workloads is labeled exploratory. Fresh seeds test stochastic sensitivity. A broader frozen stress matrix tests flat drafting costs, context transitions, noise, budget length, verification costs and acceptance transitions.

## 21:03-21:04 UTC: H2 calibration frozen

H2 tuning selected caps {0,3,5}, decay 0.95, exploration interval 32, margin 3%. The candidate's API now accepts a legal-action mask and always retains ordinary decoding. H1's full-mask behavior remains available. Sanitizer and simulator checks passed after this bounded change. Configuration is frozen in candidate-config-h2.json before evaluation.

A standalone GPU timing microbenchmark initially failed to link because Metal expects `ds4_log_is_tty(FILE *)` from the engine. Added the same correctly typed no-color stub used by existing isolated GPU tests; standalone build then passed. This benchmark will compare ordinary outer timing, host enqueue timers (not GPU stage time), and per-stage synchronization, with identical outputs and an independent Q8 dot-product reference. It will run only after the real-model suite releases the GPU.

## 21:05-21:08 UTC: sensitivity results

H2 on the original held-out regimes (exploratory reuse): 1.2128x ordinary vs 1.2064x current default for resident batching, only about +0.5%. Fresh seeds reverse that advantage: H2/default 0.994x resident. In the separate-seed simulator H2/default is approximately 1.032x on fresh seeds. Flat draft costs reduce resident H2/default to 0.987x; high timing noise to 0.973x; abrupt acceptance collapse to 0.938x. Stable-easy cases also regress because exploration is unnecessary. Expensive verification and stable-hard cases still lose to ordinary decode despite improving over the default speculative scheduler. Keep production unchanged.

Completed the preregistered 12-condition stress matrix, 40 seeds x six cases x two modes x ten policies per condition. All stage/cycle costs remain assumed. Raw rows retained; no tuning from these results. Wrote paired statistical analysis with seed-cluster bootstrap intervals and per-case regression counts. Intervals characterize the random seeds only; they do not imply representative real-model workloads.

Read primary prior art for NVIDIA transfer; links and scope in SOURCES.md. vLLM already profiles separate draft/verifier costs and optimizes a global batch budget using confidence survival. This reinforces that single-request elapsed-cost adaptation is not novel by itself and does not establish serving throughput gains.

## 21:09 UTC: experiment-runner hardening

Found and fixed a timeout trap during self-review: nested process runners create separate process groups. Killing only the outer group could leave an inner GPU process running. `run_bounded.py` now catches termination/interruption and propagates it to its child's group, with a kill fallback. Added direct and nested timeout tests that confirm child PIDs are gone. Also made final analysis refuse an incomplete ordinary-model suite rather than silently summarize partial runs. An earlier interim analysis was started while that suite was still running; it is not final evidence and will be overwritten only after all 18 rows complete.

## 21:12 UTC: additional instrumentation validation planned

The ordinary model suite completed all nine jobs (18 measured frontiers) in 884.6 seconds with per-job timeouts. Next, run eight ordinary streamed code continuations with existing expert timing summary off/on in ABBA/BAAB order. Fixed 256 input and 128 output tokens, identical quant/cache/context settings. Compare generated text, whole generation time, and existing disk/cache timing counters. This measures existing SSD instrumentation overhead and disk involvement, not speculative-policy overhead or resident GPU inference.

GPU microbenchmark is running after the model suite, never alongside it. Its enqueue-only stage timestamps explicitly measure CPU submission, not GPU execution. Per-stage waits measure elapsed GPU-complete stages but change submission structure; outer complete-cycle timing is the low-overhead control.

## 21:13-21:15 UTC: model and timer findings

All 18 ordinary baseline rows completed: median TPS (tokens/s/request) 9.66/9.08 code, 9.31/9.44 prose, and 9.63/8.97 structured at 256/2048 input tokens. Three runs per cell. Full min/max and standard deviations are in model-summary.csv. The synthetic 20 ms target cost is NOT calibrated from this SSD-streamed model.

System counters show substantial disk traffic (467-484 GiB per two-frontier process, including prefill and all system I/O). System swap usage increased from 2.00 to 458.06 MiB across the model suite; per-job swapins stayed zero, while swapouts rose. This is evidence of paging/background-state effects, not a clean resident-compute experiment. OS counters cannot attribute all activity to the model. No thermal warning was recorded by pmset; that is not a GPU temperature measurement.

GPU projection timing passed the independent dot-product reference and exact output comparisons among instrumentation modes. Representative median outer/per-stage-wait times: 1024-wide, 1 row: 0.192/0.491 ms; 4096-wide, 6 rows: 0.461/0.772 ms. These are arbitrary three-projection kernels, not draft/verifier/model stages. Enqueue timers do not measure execution. Host-only policy plus two timers was about 38 ns/cycle in this isolated build, comparable to two timers alone, and is not a substitute for end-to-end instrumentation overhead.

Final simulator checks also replay H1's frozen first-seed runs exactly after adding the action mask. Timeout tests passed including a nested runner and child cleanup.
