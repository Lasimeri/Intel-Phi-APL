/* gemm.h: Y = W X for a Q8_0 matrix W [rows, cols] in form I (gemv.h) and T tokens X [T, cols]:
 * the APL expression X+.×⍉W, token by token as rows. One pass over the weights serves T tokens,
 * so the memory cost per token falls by T while the arithmetic stays: from about 8 tokens per
 * pass (arithmetic) the VPU's F32 FMA rate is the limit, not the memory.
 *
 * Inside one 544 B tile every token takes the same operations in the same order as gemv's, so
 * Y[t] is bit-identical to gemv of X[t]: the gate holds the two against each other.
 *
 * Tokens go through each tile in register-sized chunks: GEMM_CHUNK partial sums stay in zmm
 * registers across its 32 columns, while the tile stays in L1.
 *
 * Only pointers and integers cross this interface. */
#ifndef APL_GEMM_H
#define APL_GEMM_H

#define GEMM_CHUNK 8
#define GEMM_PASS_T 16 /* the most tokens in one pass over the weights */

/* A group split between threads: the lower thread leaves its partial sums here, one per token */
struct gemm_chain {
    float acc[GEMM_PASS_T][16] __attribute__((aligned(64)));
    volatile unsigned long seq __attribute__((aligned(64)));
};

struct gemm_job {
    const void *w;     /* form I */
    long rows, cols, t;
    const float *x;    /* [t cols], rows 64-byte aligned, x_stride lanes apart */
    long x_stride;
    long pass_t;       /* tokens per pass over the weights, 0 = 16: above that x fights the weights for the L2 */
    float *y;          /* [t rows], rows 64-byte aligned */
    long pf_l2, pf_l1; /* prefetch distance in bytes ahead of the read (0 = none) */
    struct gemm_chain *chain; /* one per thread, or NULL: whole groups, the same bits as gemv */
    unsigned long seq;        /* new for every product */
};

/* A team_run body: arg is a struct gemm_job. Rows split by whole groups; with fewer groups
 * than threads, tokens too. */
void gemm_i_thread(void *arg, int tid, int n);

/* Into this thread's L2, ahead of the product: as gemv_i_prefill */
void gemm_i_prefill(const struct gemm_job *j, int tid, int n, long *budget, const volatile int *stop);

#endif
