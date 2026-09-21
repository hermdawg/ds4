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
