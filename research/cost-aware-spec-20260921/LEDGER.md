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
