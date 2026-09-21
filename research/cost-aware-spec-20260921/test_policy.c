#include "policy.h"
#include <assert.h>
#include <stdio.h>

static void test_ratio_and_input(void) {
    spec_policy p;
    spec_policy_init(&p, 0.0, 100000, 0.0, false);
    assert(!spec_policy_observe(&p, 0, 6, 1, 1));
    assert(!spec_policy_observe(&p, 0, 1, NAN, 1));
    assert(!spec_policy_observe(&p, 0, 1, INFINITY, 1));
    assert(!spec_policy_observe(&p, 0, 1, 0, 1));
    assert(!spec_policy_observe(&p, 0, 1, -1, 1));
    assert(!spec_policy_observe(&p, 0, 1, 1, 0));
    assert(!spec_policy_observe(&p, 0, 1, 1, 3));
    for (unsigned d = 0; d < 6; d++)
        assert(spec_policy_observe(&p, 0, d, d ? 12 : 10, d ? 2 : 1));
    assert(spec_policy_choose(&p, 0, 100, 5) == 1);
    assert(spec_policy_observe(&p, 0, 1, 30, 1));
    assert(spec_policy_choose(&p, 0, 100, 5) == 2);
    /* A larger acceptance rate is not itself a win. */
    for (unsigned d = 1; d < 6; d++)
        assert(spec_policy_observe(&p, 0, d, 11*(d+1), d+1));
    assert(spec_policy_choose(&p, 0, 100, 5) == 0);
    spec_policy_init(&p, 1.0, 0, NAN, true);
    assert(p.decay == .8 && p.probe_tokens == 32 && p.margin == .03);
}

static void test_weighted_ratio(void) {
    spec_policy p;
    spec_policy_init(&p, .5, 100000, 0, false);
    assert(spec_policy_observe(&p, 0, 0, 10, 1));
    assert(spec_policy_observe(&p, 0, 1, 10, 1));
    assert(spec_policy_observe(&p, 0, 1, 10, 2));
    assert(p.estimate[0][1].ms == 15 && p.estimate[0][1].tokens == 2.5);
    assert(spec_policy_choose(&p, 0, 100, 1) == 1);
}

static void test_limits_context_reset(void) {
    spec_policy p;
    spec_policy_init(&p, .8, 32, .03, true);
    for (unsigned remaining=0; remaining<10; remaining++)
        assert(spec_policy_choose(&p, 0, remaining, 5) == 0);
    for (unsigned d=0; d<6; d++)
        assert(spec_policy_observe(&p, 1023, d, 10, d+1));
    assert(spec_policy_choose(&p, 1023, 50, 5)==5);
    assert(spec_policy_choose(&p, 1024, 50, 5)==0);
    for (unsigned d=0; d<6; d++)
        assert(spec_policy_observe(&p, 4095, d, 10, d+1));
    assert(spec_policy_choose(&p, 4096, 50, 5)==0);
    assert(spec_policy_choose(&p, 1023, 50, 0)==0);
    assert(spec_policy_choose(&p, 1023, 50, 2)<=2);
    spec_policy_init(&p, .8, 32, .03, true);
    assert(spec_policy_choose(&p, 1023, 50, 5)==0);
    assert(p.committed[0]==0 && p.estimate[0][5].observations==0);
}

static void test_recovery(void) {
    spec_policy p;
    spec_policy_init(&p, .5, 8, .03, false);
    unsigned counts[6]={0};
    /* First 300 committed tokens reject every expensive draft. Then depth
     * 5 becomes profitable. Exploration must recover without a new request. */
    unsigned pos=0, after=0;
    while (pos<1200) {
        unsigned d=spec_policy_choose(&p, pos, 2000-pos, 5);
        unsigned n=pos>=300 && d==5 ? 6 : 1;
        double ms=pos>=300 && d==5 ? 15 : d ? 30 : 10;
        assert(spec_policy_observe(&p, pos, d, ms, n));
        if (pos>=300) { counts[d]++; after++; }
        pos+=n;
    }
    assert(counts[5]>50 && counts[5]>after/2);
    for (unsigned d=0; d<6; d++) assert(counts[d]>0);
}

int main(void) {
    test_ratio_and_input();
    test_weighted_ratio();
    test_limits_context_reset();
    test_recovery();
    puts("research policy: ratio, invalid measurements, limits, context, reset, recovery PASS");
    return 0;
}
