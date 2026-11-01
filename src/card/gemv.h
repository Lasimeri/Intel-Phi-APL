/* gemv.h: y = W x for a Q8_0 matrix W [rows, cols], cols a multiple of 32:
 * the APL expression W+.×x, in two machine forms the matmul benchmark holds
 * against each other. Rows are split across threads, never one dot product,
 * so every y[r] comes from the same operations in the same order whatever the
 * thread count: card == host model stays bit-exact.
 *
 * Form I, interleaved (the accumulation +⌿ of weight columns scaled by x):
 *   rows in groups of 16; per group, per 32-column block, 544 B:
 *   512 B int8 (for each of the 32 columns: that column's 16 weights), then
 *   32 B: the 16 rows' f16 block scales. 1.0625 B per weight, as in the GGUF.
 *   y16 += scales × Σ_i weights_i × x_i, one vfmadd231ps per 16 weights.
 *
 * Form R, row-dot (the inner product row by row):
 *   per row, 64-byte aligned: cols B of uint8 (q + 128), then cols/32 F32 block
 *   scales. 1.125 B per weight. Each row: Σ_b d_b × (Σ_i u_i x_i − 128 Sx_b),
 *   with Sx_b, the block sums of x, shared by all rows; a reduction per row.
 *
 * Only pointers and integers cross this interface. */
#ifndef APL_GEMV_H
#define APL_GEMV_H
#include <stdint.h>

struct team;

/* A group split between threads: the lower thread leaves its partial sum here. */
struct gemv_chain {
    float acc[16] __attribute__((aligned(64)));
    volatile uint64_t seq __attribute__((aligned(64)));
};

struct gemv_job {
    const void *w;      /* the repacked weights */
    long rows, cols;
    const float *x;     /* [cols], 64-byte aligned */
    float *y;           /* [rows], 64-byte aligned */
    const float *sx;    /* form R: [cols/32] block sums of x, 64-byte aligned */
    long pf_l2, pf_l1;  /* prefetch distance in bytes ahead of the read (0 = none) */
    /* Form I: with a chain [one per thread] and few groups per thread, threads split at
     * tiles, not at groups: 128 groups on 118 threads otherwise leave 10 threads double
     * work and the rest waiting. seq: new for every matvec. Without a chain, or with many
     * groups per thread, whole groups: y is then the same bits at any thread count. */
    struct gemv_chain *chain;
    uint64_t seq;
};

/* Into this thread's L2, ahead of the product: up to *budget bytes of the weights tid will read in
 * j (form I), until *stop. *budget falls by what was asked for. */
void gemv_i_prefill(const struct gemv_job *j, int tid, int n, long *budget, const volatile int *stop);

/* Bytes of the repacked form. */
long gemv_i_bytes(long rows, long cols);
long gemv_r_bytes(long rows, long cols);

/* Repack one row-major Q8_0 row set (GGUF layout: per row, per block, an f16
 * scale and 32 int8) into a form. */
void gemv_i_repack(const uint8_t *q8, long rows, long cols, void *out);
void gemv_r_repack(const uint8_t *q8, long rows, long cols, void *out);

/* In place, form I: a 16-row group is the same size raw and repacked, so a matrix
 * needs one group of scratch per thread, never a second copy. A team_run body:
 * arg is a struct gemv_repack_job; tmp holds gemv_i_bytes(16, cols) per thread. */
struct gemv_repack_job {
    void *w;
    long rows, cols;
    void *tmp;
};
void gemv_i_repack_thread(void *arg, int tid, int n);

/* team_run bodies: arg is a struct gemv_job. */
void gemv_i_thread(void *arg, int tid, int n);
void gemv_r_thread(void *arg, int tid, int n);

/* The block sums of x for form R (once per matvec). */
void gemv_block_sums(const float *x, long cols, float *sx);

#endif
