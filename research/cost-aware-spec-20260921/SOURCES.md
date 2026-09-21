# Grounding and transfer references

The local source at `9139e2a` is the implementation ground truth. Web sources
below provide prior art and CUDA-serving context only. They do not validate this
Mac experiment or provide measurements for our synthetic simulator.

- `ds4.c:60055-60349`: actual DSpark scheduling functions, default pauses and
  optional cost gates. Timing-based controls are opt-in for reproducibility.
- `ds4.c:73589-73640`: Qwen adaptive depths and exact-sampling depth restriction.
- `ds4.c:76268-76635`: proposal cap and confidence-lazy path; final cap may be
  applied after drafting, so draft cost need not scale with verification depth.
- `ds4.c:78731-78745`, `ds4.c:82734-82840`: seed accounting, timing boundaries,
  seed batching and short-proposal fallback. The optional existing cost gate
  compares draft/verify/recovery time with accepted-draft count times the last
  measured ordinary target time. For batched seeds, that cost includes seed
  verification but the savings count excludes the seed; the ordinary estimate
  can also be stale or unavailable. It is not the full cycle-cost ratio.
- `ds4_metal.m:11460-11535`: end-commands waits for completion; an extra explicit
  synchronize can submit another command buffer. Do not add unnecessary waits
  when recording elapsed cycle time.
- `docs/SPECULATIVE_DECODING.md`: sampling, checkpoint pairing and session-batch
  constraints. Opportunistic nonzero-temperature runs do not preserve the target
  sampling distribution and are excluded from the candidate evaluation.

Primary external sources, retrieved 2026-09-21:

1. [SpecDec++ (COLM 2025)](https://arxiv.org/abs/2405.19715v3)
   derives adaptive candidate-length stopping and uses a trained acceptance head.
   That training component is outside this campaign's scope.
2. [EVICT (September 2026 revision)](https://arxiv.org/abs/2605.00342v2)
   addresses verification cost in mixture-of-experts models: larger draft trees
   can activate a larger expert union. Its cost-aware tree pruning is prior art,
   not an experiment reproduced here.
3. [vLLM DSpark adaptive verification](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification)
   selects a global draft-verification budget by expected committed tokens per
   profiled step time. It uses separate draft/verification cost tables, accounts
   for CUDA graph size boundaries, and allocates slots among concurrent requests.
   This supports testing a batch-aware cost objective on NVIDIA hardware rather
   than assuming a single-request Mac policy will transfer unchanged.
4. [vLLM speculative-decoding documentation, v0.18.0](https://docs.vllm.ai/en/v0.18.0/features/speculative_decoding/)
   describes the latency-focused memory-bound regime and numerical/reproducibility
   caveats. A versioned page is cited; it is not a claim about all later versions.

No external performance numbers are mixed into this campaign's tables.
