/* gemv.c: see gemv.h. Written once against knc.h: on the card every operation
 * is one KNC instruction, on the host the lane-exact model of it. */
#include "gemv.h"

#include <string.h>

#include "knc.h"

#define BLOCK 32
#define GROUP 16
#define I_BLOCK_BYTES 544 /* 512 B int8 + 32 B f16 scales */

long gemv_i_bytes(long rows, long cols) { return rows / GROUP * (cols / BLOCK) * I_BLOCK_BYTES; }

static long r_stride(long cols)
{
    long raw = cols + cols / BLOCK * 4;
    return (raw + 63) / 64 * 64;
}

long gemv_r_bytes(long rows, long cols) { return rows * r_stride(cols); }

void gemv_i_repack(const uint8_t *q8, long rows, long cols, void *out)
{
    const long nb = cols / BLOCK, row_bytes = nb * 34;
    uint8_t *o = out;
    for (long g = 0; g < rows / GROUP; g++) {
        const uint8_t *r0 = q8 + g * GROUP * row_bytes;
        for (long b = 0; b < nb; b++) {
            /* 34 B per block: f16 scale, then 32 int8 */
            for (int i = 0; i < BLOCK; i++)
                for (int r = 0; r < GROUP; r++)
                    *o++ = r0[r * row_bytes + b * 34 + 2 + i];
            for (int r = 0; r < GROUP; r++) {
                memcpy(o, r0 + r * row_bytes + b * 34, 2);
                o += 2;
            }
        }
    }
}

void gemv_i_repack_thread(void *arg, int tid, int n)
{
    const struct gemv_repack_job *j = arg;
    const long ng = j->rows / GROUP, bytes = gemv_i_bytes(GROUP, j->cols);
    uint8_t *tmp = (uint8_t *)j->tmp + tid * bytes;
    for (long g = ng * tid / n; g < ng * (tid + 1) / n; g++) {
        uint8_t *at = (uint8_t *)j->w + g * bytes;
        memcpy(tmp, at, (size_t)bytes);
        gemv_i_repack(tmp, GROUP, j->cols, at);
    }
}

void gemv_r_repack(const uint8_t *q8, long rows, long cols, void *out)
{
    const long nb = cols / BLOCK, row_bytes = nb * 34, stride = r_stride(cols);
    for (long r = 0; r < rows; r++) {
        const uint8_t *src = q8 + r * row_bytes;
        uint8_t *u = (uint8_t *)out + r * stride;
        float *d = (float *)(u + cols);
        for (long b = 0; b < nb; b++) {
            uint16_t h;
            memcpy(&h, src + b * 34, 2);
            /* f16 to f32 exactly, by bits: no float arithmetic in this path */
            uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, v;
            if (e == 0) { /* subnormal or zero: normalize */
                if (m == 0) {
                    v = s;
                } else {
                    uint32_t shift = 0;
                    while (!(m & 1024)) {
                        m <<= 1;
                        shift++;
                    }
                    v = s | (113 - shift) << 23 | (m & 1023) << 13;
                }
            } else if (e == 31) {
                v = s | 0x7f800000u | m << 13;
            } else {
                v = s | (e + 112) << 23 | m << 13;
            }
            memcpy(&d[b], &v, 4);
            for (int i = 0; i < BLOCK; i++)
                u[b * BLOCK + i] = (uint8_t)(src[b * 34 + 2 + i] ^ 0x80); /* int8 + 128 */
        }
    }
}

/* One stream of `run` tiles of form I: acc += Σ scales × Σ_i weights_i × x_i. */
static inline __attribute__((always_inline)) v16f tile_run(const int8_t **pp, const int8_t **pp2, const int8_t **pp1, const float *x, long run, v16f acc)
{
    const int8_t *p = *pp, *pl2 = *pp2, *pl1 = *pp1;
    for (long b = 0; b < run; b++) {
        v16f t = knc_zero();
        for (int line = 0; line < 8; line++) {
            /* one cache line of weights: 4 columns of 16 */
            knc_prefetch1(pl2);
            knc_prefetch0(pl1);
            t = knc_fma_b16(knc_load_s8(p), x, t);
            t = knc_fma_b16(knc_load_s8(p + 16), x + 1, t);
            t = knc_fma_b16(knc_load_s8(p + 32), x + 2, t);
            t = knc_fma_b16(knc_load_s8(p + 48), x + 3, t);
            p += 64;
            pl2 += 64;
            pl1 += 64;
            x += 4;
        }
        acc = knc_fma_f16(t, (const uint16_t *)p, acc);
        p += 32;
        pl2 += 32;
        pl1 += 32;
    }
    *pp = p;
    *pp2 = pl2;
    *pp1 = pl1;
    return acc;
}

/* Threads split at tiles. A thread's first run may finish a group a lower thread began, its
 * last may begin one the next finishes. The tail is published first, without waiting, so no
 * two threads wait on each other; the head is merged last, after all of this thread's other
 * work: the wait is one add, not the lower thread's whole share. A range inside one group
 * passes the sum along. The fold's order depends only on the thread count. */
static void tiles_thread(const struct gemv_job *j, int tid, int n, long nb, long ng)
{
    const long tiles = ng * nb;
    /* with fewer than 2 tiles per thread, use fewer threads: a hop costs more than a tile */
    const int active = tiles / 2 < n ? (int)(tiles / 2 > 0 ? tiles / 2 : 1) : n;
    if (tid >= active)
        return;
    const long t0 = tiles * tid / active, t1 = tiles * (tid + 1) / active;
    const int8_t *p = (const int8_t *)j->w + t0 * I_BLOCK_BYTES;
    const int8_t *pl2 = p + j->pf_l2, *pl1 = p + j->pf_l1;
    long g = t0 / nb, left = t1 - t0, head_g = -1;
    const long b0 = t0 % nb;
    int head_ends = 0;
    v16f head = knc_zero();
    if (b0) {
        const long run = nb - b0 < left ? nb - b0 : left;
        head = tile_run(&p, &pl2, &pl1, j->x + b0 * BLOCK, run, head);
        head_g = g;
        head_ends = b0 + run == nb;
        left -= run;
        g += head_ends;
    }
    for (; left >= nb; left -= nb, g++)
        knc_store(j->y + g * GROUP, tile_run(&p, &pl2, &pl1, j->x, nb, knc_zero()));
    if (left > 0) {
        knc_store(j->chain[tid].acc, tile_run(&p, &pl2, &pl1, j->x, left, knc_zero()));
        __atomic_store_n(&j->chain[tid].seq, j->seq, __ATOMIC_RELEASE);
    }
    if (head_g < 0)
        return;
    while (__atomic_load_n(&j->chain[tid - 1].seq, __ATOMIC_ACQUIRE) != j->seq)
        knc_delay(16);
    head = knc_add(knc_load(j->chain[tid - 1].acc), head);
    if (head_ends) {
        knc_store(j->y + head_g * GROUP, head);
    } else {
        knc_store(j->chain[tid].acc, head);
        __atomic_store_n(&j->chain[tid].seq, j->seq, __ATOMIC_RELEASE);
    }
}

/* Whole groups per thread at or above which no group is split. */
#define WHOLE_GROUPS 8

void gemv_i_thread(void *arg, int tid, int n)
{
    const struct gemv_job *j = arg;
    const long nb = j->cols / BLOCK, ng = j->rows / GROUP;
    if (j->chain && ng < WHOLE_GROUPS * n && n > 1) {
        tiles_thread(j, tid, n, nb, ng);
        return;
    }
    const long g0 = ng * tid / n, g1 = ng * (tid + 1) / n;
    const int8_t *p = (const int8_t *)j->w + g0 * nb * I_BLOCK_BYTES;
    const int8_t *pl2 = p + j->pf_l2, *pl1 = p + j->pf_l1;
    for (long g = g0; g < g1; g++)
        knc_store(j->y + g * GROUP, tile_run(&p, &pl2, &pl1, j->x, nb, knc_zero()));
}

void gemv_block_sums(const float *x, long cols, float *sx)
{
    for (long b = 0; b < cols / BLOCK; b++) {
        v16f s = knc_add(knc_load(x + b * BLOCK), knc_load(x + b * BLOCK + 16));
        sx[b] = knc_hsum(s);
    }
}

void gemv_r_thread(void *arg, int tid, int n)
{
    const struct gemv_job *j = arg;
    const long nb = j->cols / BLOCK, stride = r_stride(j->cols);
    const long r0 = j->rows * tid / n, r1 = j->rows * (tid + 1) / n;
    static const float minus128 = -128.0f;
    for (long r = r0; r < r1; r++) {
        const uint8_t *u = (const uint8_t *)j->w + r * stride;
        const uint8_t *pl2 = u + j->pf_l2, *pl1 = u + j->pf_l1;
        const float *d = (const float *)(u + j->cols);
        v16f acc = knc_zero();
        for (long b = 0; b < nb; b++) {
            knc_prefetch1(pl2);
            knc_prefetch0(pl1);
            v16f t = knc_mul_u8(knc_load(j->x + b * BLOCK), u);
            t = knc_fma_u8(knc_load(j->x + b * BLOCK + 16), u + 16, t);
            acc = knc_fma_b16(t, d + b, acc);
            u += BLOCK;
            pl2 += BLOCK;
            pl1 += BLOCK;
        }
        /* the bias: −128 × Σ_b d_b Sx_b, 16 blocks at a time */
        v16f cor = knc_zero();
        for (long b = 0; b + 16 <= nb; b += 16)
            cor = knc_fma(knc_load(d + b), knc_load(j->sx + b), cor);
        acc = knc_fma(cor, knc_bcast(&minus128), acc);
        j->y[r] = knc_hsum(acc);
    }
}

void gemv_i_prefill(const struct gemv_job *j, int tid, int n, long *budget, const volatile int *stop)
{
    const long nb = j->cols / BLOCK, ng = j->rows / GROUP, tiles = ng * nb;
    long t0, t1;
    if (j->chain && ng < WHOLE_GROUPS * n && n > 1) {
        const int active = tiles / 2 < n ? (int)(tiles / 2 > 0 ? tiles / 2 : 1) : n;
        if (tid >= active)
            return;
        t0 = tiles * tid / active;
        t1 = tiles * (tid + 1) / active;
    } else {
        t0 = (ng * tid / n) * nb;
        t1 = (ng * (tid + 1) / n) * nb;
    }
    const int8_t *const w = (const int8_t *)j->w;
    long bytes = (t1 - t0) * I_BLOCK_BYTES;
    if (bytes > *budget)
        bytes = *budget;
    for (long b = 0; b < bytes; b += 64) {
        if ((b & 1023) == 0 && *stop)
            return;
        knc_prefetch1(w + t0 * I_BLOCK_BYTES + b);
    }
    *budget -= bytes;
}
