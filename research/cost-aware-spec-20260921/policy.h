#ifndef RESEARCH_SPEC_POLICY_H
#define RESEARCH_SPEC_POLICY_H

/* Research only. The caller owns timing, execution, acceptance and state.
 * Depth 0 is ordinary decode; depth d proposes at most d extra tokens.
 * Learn ratios of elapsed totals / committed totals, not mean cycle ratios:
 * a one-token reject and a six-token accept have different useful work. */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define SPEC_ACTIONS 6
#define SPEC_BUCKETS 3

typedef struct {
    double ms, tokens;
    uint64_t observations;
} spec_estimate;

typedef struct {
    spec_estimate estimate[SPEC_BUCKETS][SPEC_ACTIONS];
    uint64_t committed[SPEC_BUCKETS], explored_at[SPEC_BUCKETS];
    unsigned next_probe[SPEC_BUCKETS];
    double decay, margin;
    unsigned probe_tokens;
    unsigned action_mask;
    bool context_buckets;
} spec_policy;

static void spec_policy_init(spec_policy *p, double decay, unsigned probe_tokens,
                             double margin, bool context_buckets) {
    memset(p, 0, sizeof(*p));
    p->decay = isfinite(decay) && decay >= 0 && decay < 1 ? decay : .8;
    p->probe_tokens = probe_tokens ? probe_tokens : 32;
    p->margin = isfinite(margin) && margin >= 0 && margin < 1 ? margin : .03;
    p->context_buckets = context_buckets;
    p->action_mask = (1u << SPEC_ACTIONS) - 1u;
}

/* A backend may expose only a subset of draft depths. Ordinary decode is
 * always legal. Set this before observing a request, not mid-cycle. */
static void spec_policy_actions(spec_policy *p, unsigned mask) {
    p->action_mask = (mask & ((1u << SPEC_ACTIONS) - 1u)) | 1u;
}

static unsigned spec_bucket(const spec_policy *p, uint32_t context) {
    return !p->context_buckets ? 0 : context < 1024 ? 0 : context < 4096 ? 1 : 2;
}

static unsigned spec_policy_choose(spec_policy *p, uint32_t context,
                                   unsigned remaining, unsigned max_depth) {
    /* Match the current DSpark ten-token tail guard. No observations from
     * clipped cycles: each arm always denotes the same requested cap. */
    if (remaining < 10) return 0;
    if (max_depth >= SPEC_ACTIONS) max_depth = SPEC_ACTIONS - 1;
    unsigned b = spec_bucket(p, context);
    for (unsigned d = 0; d <= max_depth; d++)
        if ((p->action_mask & (1u << d)) && !p->estimate[b][d].observations) return d;
    const double ordinary = p->estimate[b][0].ms / p->estimate[b][0].tokens;
    unsigned best = 0;
    double best_score = ordinary * (1.0 - p->margin);
    for (unsigned d = 1; d <= max_depth; d++) {
        if (!(p->action_mask & (1u << d))) continue;
        double score = p->estimate[b][d].ms / p->estimate[b][d].tokens;
        if (score < best_score) { best = d; best_score = score; }
    }
    if (p->committed[b] - p->explored_at[b] >= p->probe_tokens) {
        p->explored_at[b] = p->committed[b];
        for (unsigned n = 0; n <= max_depth; n++) {
            unsigned probe = p->next_probe[b]++ % (max_depth + 1);
            if (probe != best && (p->action_mask & (1u << probe))) return probe;
        }
    }
    return best;
}

static bool spec_policy_observe(spec_policy *p, uint32_t context, unsigned depth,
                                double elapsed_ms, unsigned committed) {
    if (depth >= SPEC_ACTIONS || committed == 0 || committed > depth + 1 ||
        !isfinite(elapsed_ms) || elapsed_ms <= 0) return false;
    if (!(p->action_mask & (1u << depth))) return false;
    unsigned b = spec_bucket(p, context);
    spec_estimate *e = &p->estimate[b][depth];
    e->ms = p->decay * e->ms + elapsed_ms;
    e->tokens = p->decay * e->tokens + committed;
    e->observations++;
    p->committed[b] += committed;
    return true;
}
#endif
