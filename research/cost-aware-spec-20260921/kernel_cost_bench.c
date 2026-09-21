#define _DARWIN_C_SOURCE
#include "../../ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Isolated Q8 projection timing. This is not a transformer, drafter or model
 * throughput benchmark. Three projections represent arbitrary GPU stages. */
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
static void require(int ok) { if (!ok) { fputs("kernel bench failed\n",stderr); exit(1); } }
static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1000.0+t.tv_nsec/1e6;
}
static int value(unsigned row,unsigned col) { return (int)((row*13+col*7)%17)-8; }
static double cycle(void *map,uint64_t bytes,unsigned d,unsigned tokens,
                    ds4_gpu_tensor *input,ds4_gpu_tensor **outputs,int mode,
                    double *stages) {
    double start=now_ms();
    if (mode!=2) require(ds4_gpu_begin_commands());
    for (int s=0;s<3;s++) {
        double a=mode ? now_ms() : 0;
        if (mode==2) require(ds4_gpu_begin_commands());
        require(ds4_gpu_matmul_q8_0_tensor(outputs[s],map,bytes,0,d,d,input,tokens));
        if (mode==2) require(ds4_gpu_end_commands());
        if (mode) stages[s]+=now_ms()-a;
    }
    if (mode!=2) require(ds4_gpu_end_commands());
    return now_ms()-start;
}
int main(void) {
    require(ds4_gpu_init());
    require(ds4_gpu_warm_command_queue());
    puts("dimension,rows,repeat,mode,iterations,total_ms,ms_per_cycle,stage0_ms,stage1_ms,stage2_ms");
    const unsigned dims[]={1024,4096};
    for (unsigned shape=0;shape<2;shape++) {
        unsigned d=dims[shape];
        uint64_t bytes=(uint64_t)d*d/32*34;
        void *map=NULL;
        require(posix_memalign(&map,getpagesize(),bytes)==0);
        for (unsigned row=0;row<d;row++) for (unsigned block=0;block<d/32;block++) {
            uint8_t *p=(uint8_t*)map+((uint64_t)row*d/32+block)*34;
            uint16_t scale=0x1c00; /* exactly 1/256 in float16 */
            memcpy(p,&scale,2);
            for (unsigned j=0;j<32;j++) p[j+2]=(uint8_t)(int8_t)value(row,block*32+j);
        }
        require(ds4_gpu_set_model_map(map,bytes));
        for (unsigned tokens=1;tokens<=6;tokens++) {
            size_t n=(size_t)d*tokens;
            float *x=malloc(n*4),*actual=malloc(n*4),*reference=malloc(n*4);
            require(x&&actual&&reference);
            ds4_gpu_tensor *input=ds4_gpu_tensor_alloc(n*4),*out[3];
            require(input!=NULL);
            for (unsigned i=0;i<3;i++) {out[i]=ds4_gpu_tensor_alloc(n*4);require(out[i]!=NULL);}
            for (size_t i=0;i<n;i++) x[i]=((int)(i%23)-11)/256.f;
            require(ds4_gpu_tensor_write(input,0,x,n*4));
            double stages[3]={0};
            for (unsigned warm=0;warm<10;warm++) (void)cycle(map,bytes,d,tokens,input,out,0,stages);
            require(ds4_gpu_tensor_read(out[0],0,reference,n*4));
            /* Independent mathematical Q8 dot product. Values are binary
             * fractions; retain the existing tiny-MoE numeric tolerance. */
            for (unsigned t=0;t<tokens;t++) for (unsigned row=0;row<d;row++) {
                double ref=0;
                for (unsigned col=0;col<d;col++) ref+=value(row,col)/256.0*x[(size_t)t*d+col];
                float got=reference[(size_t)t*d+row];
                if (!isfinite(got)||fabs(got-ref)>2e-5*(1+fabs(ref))) {
                    fprintf(stderr,"reference failure d=%u rows=%u row=%u ref=%g got=%g\n",d,tokens,row,ref,got);
                    return 1;
                }
            }
            for (unsigned repeat=0;repeat<8;repeat++) for (unsigned order=0;order<3;order++) {
                int mode=repeat%2 ? 2-(int)order : (int)order;
                double measured=0,stage[3]={0};
                const unsigned iters=100;
                for (unsigned i=0;i<iters;i++) measured+=cycle(map,bytes,d,tokens,input,out,mode,stage);
                for (int s=0;s<3;s++) {
                    require(ds4_gpu_tensor_read(out[s],0,actual,n*4));
                    require(memcmp(actual,reference,n*4)==0);
                }
                printf("%u,%u,%u,%s,%u,%.6f,%.6f,%.6f,%.6f,%.6f\n",d,tokens,repeat,
                    (const char*[]){"outer_timer","enqueue_timers","stage_sync_timers"}[mode],
                    iters,measured,measured/iters,stage[0]/iters,stage[1]/iters,stage[2]/iters);
                fflush(stdout);
            }
            for (unsigned i=0;i<3;i++) ds4_gpu_tensor_free(out[i]);
            ds4_gpu_tensor_free(input);free(x);free(actual);free(reference);
        }
        /* Cleanup releases Metal's mapping before host storage is freed. */
        ds4_gpu_cleanup(); free(map);
        if (shape+1<2) {require(ds4_gpu_init());require(ds4_gpu_warm_command_queue());}
    }
    return 0;
}
