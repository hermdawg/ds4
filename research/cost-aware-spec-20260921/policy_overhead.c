#define _POSIX_C_SOURCE 200809L
#include "policy.h"
#include <stdio.h>
#include <time.h>

static volatile double sink;
static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec*1e9+t.tv_nsec;
}

/* Batch a million calls. The volatile checksum prevents dead-code removal.
 * This measures HOST bookkeeping only, no GPU launch or synchronization. */
int main(void) {
    const unsigned n=1000000;
    puts("repeat,variant,iterations,total_ns,ns_per_cycle,checksum");
    for (unsigned repeat=0;repeat<12;repeat++) {
        for (unsigned j=0;j<4;j++) {
            unsigned variant=repeat%2 ? 3-j : j;
            spec_policy p;
            spec_policy_init(&p,.95,32,.03,true);
            double sum=0;
            unsigned pos=0;
            double start=now_ns();
            for (unsigned i=0;i<n;i++) {
                if (variant==0) {
                    sum+=(i&7)*.125;
                } else if (variant==1) {
                    double a=now_ns();
                    double b=now_ns();
                    sum+=b-a;
                } else {
                    unsigned d=spec_policy_choose(&p,256,1000,5);
                    double a=variant==3 ? now_ns() : 0;
                    unsigned committed=d+1;
                    bool valid=spec_policy_observe(&p,256,d,20+3*d,committed);
                    double b=variant==3 ? now_ns() : 0;
                    sum+=d+valid+b-a;
                    pos+=committed;
                }
            }
            double elapsed=now_ns()-start;
            sink=sum+pos;
            printf("%u,%s,%u,%.0f,%.3f,%.3f\n",repeat,
                (const char*[]){"loop","two_timers","policy","policy_two_timers"}[variant],
                n,elapsed,elapsed/n,sink);
        }
    }
    return 0;
}
