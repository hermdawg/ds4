# Research hypotheses and evidence rules

## H1: elapsed-cost action selection

Use exponentially weighted total elapsed milliseconds divided by total committed
tokens for ordinary decoding and draft caps 1-5. Probe alternative actions every
32 committed tokens and use a 3% margin before choosing speculation. Reset at
request boundaries; keep separate estimates below 1024, below 4096, and above
4096 context tokens. Backend code supplies measurements; the policy has no GPU,
acceptance, sampler, or cache access.

Frozen primary suite and tuning grid are recorded in LEDGER.md. Held-out H1 is a
negative result versus the existing resident scheduler on average. Retain the
negative result and do not choose a replacement configuration from this holdout.

## H2: reduce the exploration action set

Source audit shows a discontinuity: resident M5 seed batching begins at three
proposals. H1 tuning data also show low payoff from caps 1-2 in the assumed cost
regimes. Hypothesis: spend the limited exploration budget on ordinary decoding
plus cap 5, or ordinary plus caps 3 and 5. A backend can already have a subset of
supported depths, so represent legal actions as a caller-supplied bit mask.

Before H2 tuning: compare masks 63 (0-5), 41 (0,3,5), and 33 (0,5), decay values
0.5/0.8/0.95 and probe intervals 16/32/64, using ONLY the original tuning cases
and seeds. Use the same selection objective as H1. Configuration must be
committed before evaluation. Reusing H1's held-out workloads is explicitly
exploratory, not a new independent confirmation.

Additional independently seeded validation: original held-out regimes with seeds
400-439, and the stress matrix below with seeds 200-239. These test noise and
regime sensitivity; they do not make invented workload regimes representative of
real prompts. No further parameter tuning from these results.

## Stress matrix (frozen before execution)

For each original held-out case and both execution modes, independently test:

- flat draft cost: every positive cap pays the cap-5 draft cost;
- expensive verification: multiply verification costs by 2;
- cheap verification: multiply verification costs by 0.5;
- short budget: 32 output tokens;
- long budget: 2048 output tokens;
- context boundary: start at 960 tokens and generate 256 (crosses 1024);
- noise: lognormal sigma 0.30;
- stable easy: all phases use first/later acceptance 0.95;
- stable hard: all phases use first/later acceptance 0.10;
- abrupt recovery: first half first/later acceptance 0.05, second half 0.98;
- abrupt collapse: reverse the abrupt-recovery phases.

The baseline, existing optional cost gate, all fixed caps and candidate use the
same potential outcomes at each output position. Fixed caps disable the default
scheduler and its tail guard; learned/default schedules retain the ten-token
tail guard. Draft confidence is simplified to no-proposal/full-cap events.
No-proposal odds and stage costs are explicit assumptions. This is a scheduling
simulator, not a trace replay of model inference.

## Break-even accounting

Let t be ordinary target cost, D_d draft cost, V_d verifier cost and R_d expected
recovery cost for draft cap d. Let A_d be the expected accepted draft count.
A cycle commits 1+A_d expected output tokens (one seed plus accepted drafts).
A seed is a target token, not a draft success.

- Separate seed: speculation wins when D_d + P(verify)V_d + R_d < t A_d.
- Batched seed: if every proposal reaches verification, it wins when
  D_d + V_(d+1) + R_d < t(1+A_d).

With constant independent acceptance probability p and no confidence pruning,
A_d = p + p^2 + ... + p^d = p(1-p^d)/(1-p), or d when p=1.
The simulator uses separate first/later probabilities and explicitly charges
ordinary decode when confidence prunes the whole proposal. Recovery is charged
only on verified partial rejection. Full generation elapsed time includes every
cycle, including warm-up exploration and the output tail.

Example: t=20 ms, draft=5 ms, verify=25 ms and recovery=2 ms. In a separate-seed
cycle that always verifies, more than 32/20=1.6 accepted drafts per cycle are
needed to beat ordinary decoding. If seed batching makes total cycle time 32 ms,
the threshold becomes more than 32/20-1=0.6 accepted drafts. This is an
illustration, not a measured local cost.

## Correctness and interpretation gates

A policy is not integrated into inference without compatible model weights,
exact-sampling/greedy validation, rollback/cache/output-limit/session-reuse
checks and repeated interleaved end-to-end measurements. Kernel and sampler
unit tests cannot substitute for those missing integration tests. Unexplained
greedy divergences cannot be waived as numerical noise.

For NVIDIA transfer, retain ratio accounting, ordinary-decode action,
exploration, output limits and pre-decision-only observations. Re-measure all
costs and supported depths. A batched server's objective is useful aggregate
tokens per GPU-second, with latency constraints, not the single-request ratio:
verification can displace other requests even when it speeds up one request.
