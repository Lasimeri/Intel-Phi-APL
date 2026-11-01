/* gemm.c: see gemm.h. Written once against knc.h.
 *
 * The nest is what buys the memory: group, then tile, then tokens. A tile is fetched once and
 * every token visits it while it is in L1; a token's sum for its group waits in memory between
 * tiles, and only the chunk's partial sums for this tile are in registers.
 *
 * The chunks are spelled out, with named sums and named pointers, because a compiler left to
 * loop over k keeps the sums in memory from 3 tokens up: a load and a store around every FMA
 * (6.4 cycles per FMA against about 2.25, measured 2026-11-01). Chunks: 8, then 4, 2, 1. */
#include "gemm.h"

#include "knc.h"

#define BLOCK 32
#define GROUP 16
#define I_BLOCK_BYTES 544
#define WHOLE_GROUPS 8 /* from this many groups a thread, the imbalance is small: as gemv */
#define TILE_MAX_T 4 /* the tile split, measured: decode K=1 +6.5 %; T=16 -3.4 % (9B, 2026-11-01) */

/* One token's FMA for the column: its sum, its row of x */
#define F(k) s##k = knc_fma_b16(w, a##k + col, s##k)
/* A sum's end of tile: scaled by the 16 rows' f16 scales, into the token's sum for its group */
#define E(k) acc[k] = knc_fma_f16(s##k, scales, acc[k])

static inline __attribute__((always_inline)) void tile8(const int8_t *tile, const float *x, long st, v16f *acc)
{
    const float *const a0 = x, *const a1 = x + st, *const a2 = x + 2 * st, *const a3 = x + 3 * st;
    const float *const a4 = x + 4 * st, *const a5 = x + 5 * st, *const a6 = x + 6 * st, *const a7 = x + 7 * st;
    const uint16_t *const scales = (const uint16_t *)(tile + 512);
    v16f s0 = knc_zero(), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
    const int8_t *p = tile;
    /* four columns at a time: each token's four x in a register in every 128-bit group, then
     * four FMAs with swizzles: no memory operand on an FMA */
    for (long col = 0; col < BLOCK; col += 4) {
        const v16f x0 = knc_load_4(a0 + col), x1 = knc_load_4(a1 + col), x2 = knc_load_4(a2 + col), x3 = knc_load_4(a3 + col);
        const v16f x4 = knc_load_4(a4 + col), x5 = knc_load_4(a5 + col), x6 = knc_load_4(a6 + col), x7 = knc_load_4(a7 + col);
#define R(sw)                                                                                                          \
    do {                                                                                                               \
        const v16f w = knc_load_s8(p);                                                                                 \
        p += 16;                                                                                                       \
        s0 = KNC_FMA_SWZ(w, x0, s0, sw);                                                                               \
        s1 = KNC_FMA_SWZ(w, x1, s1, sw);                                                                               \
        s2 = KNC_FMA_SWZ(w, x2, s2, sw);                                                                               \
        s3 = KNC_FMA_SWZ(w, x3, s3, sw);                                                                               \
        s4 = KNC_FMA_SWZ(w, x4, s4, sw);                                                                               \
        s5 = KNC_FMA_SWZ(w, x5, s5, sw);                                                                               \
        s6 = KNC_FMA_SWZ(w, x6, s6, sw);                                                                               \
        s7 = KNC_FMA_SWZ(w, x7, s7, sw);                                                                               \
    } while (0)
        R(KNC_SWZ_AAAA);
        R(KNC_SWZ_BBBB);
        R(KNC_SWZ_CCCC);
        R(KNC_SWZ_DDDD);
#undef R
    }
    E(0);
    E(1);
    E(2);
    E(3);
    E(4);
    E(5);
    E(6);
    E(7);
}

/* Four columns at a time, as tile8: n tokens' four x in registers, four FMAs with swizzles */
#define TILE_BEGIN()                                                                                                  \
    const uint16_t *const scales = (const uint16_t *)(tile + 512);                                                     \
    const int8_t *p = tile;                                                                                            \
    (void)st

#define SW(k, sw) s##k = KNC_FMA_SWZ(w, x##k, s##k, sw)

static inline __attribute__((always_inline)) void tile4(const int8_t *tile, const float *x, long st, v16f *acc)
{
    TILE_BEGIN();
    const float *const a0 = x, *const a1 = x + st, *const a2 = x + 2 * st, *const a3 = x + 3 * st;
    v16f s0 = knc_zero(), s1 = s0, s2 = s0, s3 = s0;
    for (long col = 0; col < BLOCK; col += 4) {
        const v16f x0 = knc_load_4(a0 + col), x1 = knc_load_4(a1 + col), x2 = knc_load_4(a2 + col), x3 = knc_load_4(a3 + col);
#define R(sw)                                                                                                          \
    do {                                                                                                               \
        const v16f w = knc_load_s8(p);                                                                                 \
        p += 16;                                                                                                       \
        SW(0, sw);                                                                                                     \
        SW(1, sw);                                                                                                     \
        SW(2, sw);                                                                                                     \
        SW(3, sw);                                                                                                     \
    } while (0)
        R(KNC_SWZ_AAAA);
        R(KNC_SWZ_BBBB);
        R(KNC_SWZ_CCCC);
        R(KNC_SWZ_DDDD);
#undef R
    }
    E(0);
    E(1);
    E(2);
    E(3);
}

static inline __attribute__((always_inline)) void tile3(const int8_t *tile, const float *x, long st, v16f *acc)
{
    TILE_BEGIN();
    const float *const a0 = x, *const a1 = x + st, *const a2 = x + 2 * st;
    v16f s0 = knc_zero(), s1 = s0, s2 = s0;
    for (long col = 0; col < BLOCK; col += 4) {
        const v16f x0 = knc_load_4(a0 + col), x1 = knc_load_4(a1 + col), x2 = knc_load_4(a2 + col);
#define R(sw)                                                                                                          \
    do {                                                                                                               \
        const v16f w = knc_load_s8(p);                                                                                 \
        p += 16;                                                                                                       \
        SW(0, sw);                                                                                                     \
        SW(1, sw);                                                                                                     \
        SW(2, sw);                                                                                                     \
    } while (0)
        R(KNC_SWZ_AAAA);
        R(KNC_SWZ_BBBB);
        R(KNC_SWZ_CCCC);
        R(KNC_SWZ_DDDD);
#undef R
    }
    E(0);
    E(1);
    E(2);
}

static inline __attribute__((always_inline)) void tile2(const int8_t *tile, const float *x, long st, v16f *acc)
{
    const float *const a0 = x, *const a1 = x + st;
    const uint16_t *const scales = (const uint16_t *)(tile + 512);
    v16f s0 = knc_zero(), s1 = s0;
    const int8_t *p = tile;
    for (long col = 0; col < BLOCK; col++) {
        const v16f w = knc_load_s8(p);
        p += 16;
        F(0);
        F(1);
    }
    E(0);
    E(1);
}

static inline __attribute__((always_inline)) void tile1(const int8_t *tile, const float *x, v16f *acc)
{
    const float *const a0 = x;
    const uint16_t *const scales = (const uint16_t *)(tile + 512);
    v16f s0 = knc_zero();
    const int8_t *p = tile;
    for (long col = 0; col < BLOCK; col++) {
        const v16f w = knc_load_s8(p);
        p += 16;
        F(0);
    }
    E(0);
}

/* count tiles from tile b0 of a group, for tokens [0, tokens) of x0: into acc */
static inline __attribute__((always_inline)) void run_tiles(const struct gemm_job *j, const int8_t **tp, const int8_t **pl2p, const int8_t **pl1p,
                                                            const float *x0, long tokens, long b0, long count, v16f *acc)
{
    const long st = j->x_stride;
    const int8_t *tile = *tp, *pl2 = *pl2p, *pl1 = *pl1p;
    for (long b = b0; b < b0 + count; b++) {
        for (int line = 0; line < 9; line++) {
            knc_prefetch1(pl2 + line * 64);
            knc_prefetch0(pl1 + line * 64);
        }
        const float *x = x0 + b * BLOCK;
        long t = 0;
        for (; t + 8 <= tokens; t += 8)
            tile8(tile, x + t * st, st, acc + t);
        if (tokens - t >= 4) {
            tile4(tile, x + t * st, st, acc + t);
            t += 4;
        }
        if (tokens - t == 3) {
            tile3(tile, x + t * st, st, acc + t);
            t += 3;
        }
        if (tokens - t >= 2) {
            tile2(tile, x + t * st, st, acc + t);
            t += 2;
        }
        if (tokens - t >= 1)
            tile1(tile, x + t * st, acc + t);
        tile += I_BLOCK_BYTES;
        pl2 += I_BLOCK_BYTES;
        pl1 += I_BLOCK_BYTES;
    }
    *tp = tile;
    *pl2p = pl2;
    *pl1p = pl1;
}

/* One pass over this thread's groups for tokens [t0, t1). */
static inline __attribute__((always_inline)) void one_pass(const struct gemm_job *j, long g0, long g1, long t0, long t1)
{
    const long nb = j->cols / BLOCK, tokens = t1 - t0;
    const int8_t *tile = (const int8_t *)j->w + g0 * nb * I_BLOCK_BYTES;
    const int8_t *pl2 = tile + j->pf_l2, *pl1 = tile + j->pf_l1;
    const float *const x0 = j->x + t0 * j->x_stride;
    v16f acc[GEMM_PASS_T];
    for (long g = g0; g < g1; g++) {
        for (long t = 0; t < tokens; t++)
            acc[t] = knc_zero();
        run_tiles(j, &tile, &pl2, &pl1, x0, tokens, 0, nb, acc);
        for (long t = 0; t < tokens; t++)
            knc_store(j->y + (t0 + t) * j->rows + g * GROUP, acc[t]);
    }
}

/* All the tokens, the threads split at tiles, not groups: at 2.17 groups a thread a whole-group
 * split leaves everyone waiting for the threads with 3, 38 % of the product. A group cut between two
 * threads adds up along the chain, as gemv's does. */
static void tiles_pass(const struct gemm_job *j, int tid, int n, long nb, long ng)
{
    const long tiles = ng * nb, tokens = j->t;
    const int active = tiles / 2 < n ? (int)(tiles / 2 > 0 ? tiles / 2 : 1) : n;
    if (tid >= active)
        return;
    const long t0 = tiles * tid / active, t1 = tiles * (tid + 1) / active;
    const int8_t *tile = (const int8_t *)j->w + t0 * I_BLOCK_BYTES;
    const int8_t *pl2 = tile + j->pf_l2, *pl1 = tile + j->pf_l1;
    long g = t0 / nb, left = t1 - t0, head_g = -1;
    const long b0 = t0 % nb;
    int head_ends = 0;
    v16f head[GEMM_PASS_T], acc[GEMM_PASS_T];
    if (b0) {
        const long run = nb - b0 < left ? nb - b0 : left;
        for (long t = 0; t < tokens; t++)
            head[t] = knc_zero();
        run_tiles(j, &tile, &pl2, &pl1, j->x, tokens, b0, run, head);
        head_g = g;
        head_ends = b0 + run == nb;
        left -= run;
        g += head_ends;
    }
    for (; left >= nb; left -= nb, g++) {
        for (long t = 0; t < tokens; t++)
            acc[t] = knc_zero();
        run_tiles(j, &tile, &pl2, &pl1, j->x, tokens, 0, nb, acc);
        for (long t = 0; t < tokens; t++)
            knc_store(j->y + t * j->rows + g * GROUP, acc[t]);
    }
    struct gemm_chain *const mine = &j->chain[tid];
    if (left > 0) {
        for (long t = 0; t < tokens; t++)
            acc[t] = knc_zero();
        run_tiles(j, &tile, &pl2, &pl1, j->x, tokens, 0, left, acc);
        for (long t = 0; t < tokens; t++)
            knc_store(mine->acc[t], acc[t]);
        __atomic_store_n(&mine->seq, j->seq, __ATOMIC_RELEASE);
    }
    if (head_g < 0)
        return;
    const struct gemm_chain *const before = &j->chain[tid - 1];
    while (__atomic_load_n(&before->seq, __ATOMIC_ACQUIRE) != j->seq)
        knc_delay(16);
    for (long t = 0; t < tokens; t++) {
        head[t] = knc_add(knc_load(before->acc[t]), head[t]);
        if (head_ends)
            knc_store(j->y + t * j->rows + head_g * GROUP, head[t]);
        else
            knc_store(mine->acc[t], head[t]);
    }
    if (!head_ends)
        __atomic_store_n(&mine->seq, j->seq, __ATOMIC_RELEASE);
}

void gemm_i_thread(void *arg, int tid, int n)
{
    const struct gemm_job *j = arg;
    const long ng = j->rows / GROUP;
    const long pass = j->pass_t > 0 && j->pass_t < GEMM_PASS_T ? j->pass_t : 16; /* measured best */
    if (j->chain && j->t <= TILE_MAX_T && ng >= n && ng < WHOLE_GROUPS * n) {
        tiles_pass(j, tid, n, j->cols / BLOCK, ng);
        return;
    }
    /* Fewer groups than threads: the tokens are split as well, into slices of tokens each with
     * its own share of the threads, so no thread waits with nothing to do. */
    long slices = ng >= n ? 1 : (n + ng - 1) / ng;
    if (slices > j->t)
        slices = j->t;
    const long slice = tid % slices, per = n / slices, place = tid / slices;
    if (place >= per)
        return;
    const long g0 = ng * place / per, g1 = ng * (place + 1) / per;
    if (g0 == g1)
        return;
    const long t_end = j->t * (slice + 1) / slices;
    for (long t0 = j->t * slice / slices; t0 < t_end; t0 += pass)
        one_pass(j, g0, g1, t0, t0 + pass < t_end ? t0 + pass : t_end);
}

void gemm_i_prefill(const struct gemm_job *j, int tid, int n, long *budget, const volatile int *stop)
{
    const long nb = j->cols / BLOCK, ng = j->rows / GROUP, tiles = ng * nb;
    long t0, t1;
    if (ng < n)
        return;
    if (j->chain && j->t <= TILE_MAX_T && ng < WHOLE_GROUPS * n) {
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
