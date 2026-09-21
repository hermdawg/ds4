/* Call the checked-in policy, not a Python reimplementation. Link the same
 * engine object dependencies but do not load weights or submit GPU work. */
#include "../../ds4.c"
#include "policy.h"

void *research_default_new(int seed_batch) {
    ds4_session *s = calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->engine = calloc(1, sizeof(*s->engine));
    if (!s->engine) { free(s); return NULL; }
    s->engine->backend = DS4_BACKEND_METAL;
    /* Process-global diagnostics, as in the source. Run execution modes
     * sequentially, never call this bridge concurrently. */
    setenv("DS4_DSPARK_SEED_BATCH", seed_batch ? "1" : "0", 1);
    ds4_session_dspark_scheduler_begin_request(s);
    return s;
}
void research_default_free(void *state) {
    ds4_session *s = state;
    free(s->engine);
    free(s);
}
void research_default_reset(void *state) {
    ds4_session_dspark_scheduler_begin_request(state);
}
int research_default_skip(void *state) {
    return ds4_session_dspark_scheduler_should_skip(state);
}
void research_default_note(void *state, unsigned accepted, int no_draft,
                           double extra_ms, double target_ms, double confidence) {
    ds4_session *s = state;
    s->dspark_last_target_eval_ms = target_ms;
    s->dspark_last_confidence0 = confidence;
    s->dspark_last_confidence0_valid = no_draft;
    ds4_session_dspark_scheduler_note(s, accepted, no_draft, extra_ms);
}
unsigned research_default_pause(void *state) {
    return ((ds4_session *)state)->dspark_sched_skip;
}
void *research_policy_new(double decay, unsigned probe_tokens,
                          double margin, int context_buckets) {
    spec_policy *p = malloc(sizeof(*p));
    if (p) spec_policy_init(p, decay, probe_tokens, margin, context_buckets);
    return p;
}
void research_policy_free(void *state) { free(state); }
unsigned research_policy_choose(void *state, unsigned context, unsigned remaining,
                                unsigned max_depth) {
    return spec_policy_choose(state, context, remaining, max_depth);
}
int research_policy_observe(void *state, unsigned context, unsigned depth,
                            double elapsed_ms, unsigned committed) {
    return spec_policy_observe(state, context, depth, elapsed_ms, committed);
}
