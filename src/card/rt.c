/* rt.c: the primitives under the compiled APL (see rt.h). Vector code for the KNC VPU
 * in F32; the same text is the host's test model through knc.h. */
#include "rt.h"

#include <stdlib.h>
#include <string.h>

#include "gemm.h"
#include "gemv.h"
#include "knc.h"
#include "team.h"

/* A team round costs about 12,000 cycles (8.86-10.68 us measured, 2026-11-01) and a 16-lane
 * operation at least 2 on one thread, so below about 96,000 lanes the calling thread is
 * faster (arithmetic). */
#define RT_PAR_LANES 90000L
#define RT_PAR_ROWS 512L
#define BIG_ROWS 100000L
/* Prefetch distances, bytes: tuned on engine tok/s, never on one repeated matrix, which stays
 * resident in the L2s; the APL_* settings override them for a sweep (rt_init). */
static long pf_l2 = 4096, pf_l1 = 2048, par_rows = 512;
static long big_l2 = 32768, big_l1 = 3072; /* matrices of BIG_ROWS rows or more */
static long gemm_l2 = 4096, gemm_l1 = 2048; /* the batched product, its tiles a slow stream */
static long prefill_bytes = 128 * 1024; /* an opened round reads ahead this much per thread: best on 2B, 9B */
static int use_tiles = 1;
static uint64_t gemv_seq;

#define SPAN(count)                                                                                                    \
    const long i0 = (count) * tid / n, i1 = (count) * (tid + 1) / n

static void run(struct rt *R, team_fn fn, void *job, long lanes)
{
    if (lanes >= RT_PAR_LANES)
        team_run(R->team, fn, job);
    else
        fn(job, 0, 1);
}

static float first_lane(v16f v)
{
    float t[16] __attribute__((aligned(64)));
    knc_store(t, v);
    return t[0];
}

/* ---- elementwise */
struct ew_job {
    int op;
    float *z;
    const float *a, *b;
    float s;
    long blocks;
};

static const float LOG2_E = 1.4426950408889634f, LN_2 = 0.6931471805599453f;

static void ew_thread(void *arg, int tid, int n)
{
    const struct ew_job *j = arg;
    SPAN(j->blocks);
    float *const z = j->z;
    const float *const a = j->a, *const b = j->b;
    const v16f s = knc_bcast(&j->s), log2e = knc_bcast(&LOG2_E), ln2 = knc_bcast(&LN_2), zero = knc_zero();
#define OVER(expr)                                                                                                     \
    for (long i = i0 * 16; i < i1 * 16; i += 16)                                                                       \
        knc_store(z + i, expr);                                                                                        \
    break
    switch (j->op) {
    case RT_ADD_VV: OVER(knc_add(knc_load(a + i), knc_load(b + i)));
    case RT_ADD_VS: OVER(knc_add(knc_load(a + i), s));
    case RT_ADD_SV: OVER(knc_add(s, knc_load(b + i)));
    case RT_SUB_VV: OVER(knc_sub(knc_load(a + i), knc_load(b + i)));
    case RT_SUB_VS: OVER(knc_sub(knc_load(a + i), s));
    case RT_SUB_SV: OVER(knc_sub(s, knc_load(b + i)));
    case RT_MUL_VV: OVER(knc_mul(knc_load(a + i), knc_load(b + i)));
    case RT_MUL_VS: OVER(knc_mul(knc_load(a + i), s));
    case RT_MUL_SV: OVER(knc_mul(s, knc_load(b + i)));
    case RT_DIV_VV: OVER(knc_mul(knc_load(a + i), knc_rcp(knc_load(b + i))));
    case RT_DIV_VS: {
        const v16f r = knc_rcp(s);
        OVER(knc_mul(knc_load(a + i), r));
    }
    case RT_DIV_SV: OVER(knc_mul(s, knc_rcp(knc_load(b + i))));
    case RT_MAX_VV: OVER(knc_max(knc_load(a + i), knc_load(b + i)));
    case RT_MAX_VS: OVER(knc_max(knc_load(a + i), s));
    case RT_MAX_SV: OVER(knc_max(s, knc_load(b + i)));
    case RT_MIN_VV: OVER(knc_min(knc_load(a + i), knc_load(b + i)));
    case RT_MIN_VS: OVER(knc_min(knc_load(a + i), s));
    case RT_MIN_SV: OVER(knc_min(s, knc_load(b + i)));
    case RT_NEG: OVER(knc_sub(zero, knc_load(b + i)));
    case RT_RCP: OVER(knc_rcp(knc_load(b + i)));
    case RT_EXP: OVER(knc_exp2(knc_mul(knc_load(b + i), log2e)));
    case RT_LOG: OVER(knc_mul(knc_log2(knc_load(b + i)), ln2));
    default: break;
    }
#undef OVER
}

void rt_ew(struct rt *R, int op, float *z, const float *a, const float *b, float s, long n)
{
    struct ew_job j = {.op = op, .z = z, .a = a, .b = b, .s = s, .blocks = (n + 15) / 16};
    run(R, ew_thread, &j, n);
}

/* ---- scalars */
float rt_s_add(float a, float b) { return first_lane(knc_add(knc_bcast(&a), knc_bcast(&b))); }
float rt_s_sub(float a, float b) { return first_lane(knc_sub(knc_bcast(&a), knc_bcast(&b))); }
float rt_s_mul(float a, float b) { return first_lane(knc_mul(knc_bcast(&a), knc_bcast(&b))); }
float rt_s_div(float a, float b) { return first_lane(knc_mul(knc_bcast(&a), knc_rcp(knc_bcast(&b)))); }
float rt_s_max(float a, float b) { return first_lane(knc_max(knc_bcast(&a), knc_bcast(&b))); }
float rt_s_min(float a, float b) { return first_lane(knc_min(knc_bcast(&a), knc_bcast(&b))); }
float rt_s_rsqrt(float x) { return first_lane(knc_rsqrt(knc_bcast(&x))); }

float rt_s_sqrt(float x)
{
    if (!(x > 0.0f)) /* rsqrt(0) is infinite and 0×∞ is not 0 */
        return 0.0f;
    const v16f v = knc_bcast(&x);
    return first_lane(knc_mul(v, knc_rsqrt(v)));
}

/* ---- reductions */
float rt_sum(const float *a, long n)
{
    v16f acc = knc_zero();
    long i = 0;
    for (; i + 16 <= n; i += 16)
        acc = knc_add(acc, knc_load(a + i));
    if (i < n) {
        float t[16] __attribute__((aligned(64))) = {0};
        memcpy(t, a + i, sizeof(float) * (size_t)(n - i));
        acc = knc_add(acc, knc_load(t));
    }
    return knc_hsum(acc);
}

float rt_maxv(const float *a, long n)
{
    v16f acc = knc_bcast(a);
    long i = 0;
    for (; i + 16 <= n; i += 16)
        acc = knc_max(acc, knc_load(a + i));
    if (i < n) {
        /* the tail, padded with a value already in the set */
        float t[16] __attribute__((aligned(64)));
        for (int k = 0; k < 16; k++)
            t[k] = a[0];
        memcpy(t, a + i, sizeof(float) * (size_t)(n - i));
        acc = knc_max(acc, knc_load(t));
    }
    return first_lane(knc_hmax_v(acc));
}

long rt_index_of(const float *a, long n, float v)
{
    uint32_t want;
    memcpy(&want, &v, 4);
    for (long i = 0; i < n; i++) {
        uint32_t got;
        memcpy(&got, a + i, 4);
        if (got == want)
            return i;
    }
    return n;
}

/* APL's reduction folds from the right */
void rt_redfirst(float *z, const float *a, long rows, long cols)
{
    for (long k = 0; k < cols; k += 16) {
        v16f acc = knc_load(a + (rows - 1) * cols + k);
        for (long r = rows - 2; r >= 0; r--)
            acc = knc_add(knc_load(a + r * cols + k), acc);
        knc_store(z + k, acc);
    }
}

/* ---- products */
int rt_init(struct rt *R, int threads)
{
    if (getenv("APL_PF_L2"))
        pf_l2 = atol(getenv("APL_PF_L2"));
    if (getenv("APL_PF_L1"))
        pf_l1 = atol(getenv("APL_PF_L1"));
    if (getenv("APL_BIG_L2"))
        big_l2 = atol(getenv("APL_BIG_L2"));
    if (getenv("APL_BIG_L1"))
        big_l1 = atol(getenv("APL_BIG_L1"));
    if (getenv("APL_GEMM_L2"))
        gemm_l2 = atol(getenv("APL_GEMM_L2"));
    if (getenv("APL_GEMM_L1"))
        gemm_l1 = atol(getenv("APL_GEMM_L1"));
    if (getenv("APL_PREFILL"))
        prefill_bytes = atol(getenv("APL_PREFILL"));
    if (getenv("APL_NO_TILES"))
        use_tiles = 0;
    if (getenv("APL_PAR_ROWS"))
        par_rows = atol(getenv("APL_PAR_ROWS"));
    R->threads = threads;
    if (posix_memalign((void **)&R->gemm_chain, 64, sizeof(struct gemm_chain) * (size_t)threads * RT_BATCH))
        return -1;
    return posix_memalign(&R->chain, 64, sizeof(struct gemv_chain) * (size_t)threads * RT_BATCH) ? -1 : 0;
}

void rt_gemv(struct rt *R, const void *w, long rows, long cols, const float *x, float *y)
{
    const int big = rows >= BIG_ROWS;
    struct gemv_job j = {.w = w, .rows = rows, .cols = cols, .x = x, .y = y, .pf_l2 = big ? big_l2 : pf_l2, .pf_l1 = big ? big_l1 : pf_l1,
                         .chain = use_tiles ? R->chain : NULL, .seq = ++gemv_seq};
    if (rows >= par_rows)
        team_run(R->team, gemv_i_thread, &j);
    else
        gemv_i_thread(&j, 0, 1);
}

/* The queue: each job its own chains, since a thread leaves a job's tail before the next. */
static struct gemv_job queue[RT_BATCH];
static int queued;

/* An opened round: while the host works, the card's memory has nothing to do, so each thread
 * reads ahead what it is about to need, into its L2, until the product's x has landed. */
static volatile int round_ready;
static int round_open;

static struct gemm_job gqueue[RT_BATCH];
static int gqueued;

static void batch_thread(void *arg, int tid, int n)
{
    (void)arg;
    long budget = prefill_bytes;
    for (int k = 0; k < queued && !round_ready && budget > 0; k++)
        gemv_i_prefill(&queue[k], tid, n, &budget, &round_ready);
    for (int k = 0; k < gqueued && !round_ready && budget > 0; k++)
        gemm_i_prefill(&gqueue[k], tid, n, &budget, &round_ready);
    while (!round_ready)
        knc_delay(16);
    for (int k = 0; k < queued; k++)
        gemv_i_thread(&queue[k], tid, n);
    for (int k = 0; k < gqueued; k++)
        gemm_i_thread(&gqueue[k], tid, n);
}

void rt_gemm(struct rt *R, const void *w, long rows, long cols, const float *x, long t, float *y)
{
    struct gemm_job j = {.w = w, .rows = rows, .cols = cols, .t = t, .x = x, .x_stride = cols, .y = y, .pf_l2 = gemm_l2, .pf_l1 = gemm_l1,
                         .chain = R->gemm_chain, .seq = ++gemv_seq};
    team_run(R->team, gemm_i_thread, &j);
}

void rt_gemv_q(struct rt *R, const void *w, long rows, long cols, const float *x, float *y)
{
    const int big = rows >= BIG_ROWS;
    queue[queued] = (struct gemv_job){.w = w, .rows = rows, .cols = cols, .x = x, .y = y, .pf_l2 = big ? big_l2 : pf_l2,
                                      .pf_l1 = big ? big_l1 : pf_l1, .seq = ++gemv_seq,
                                      .chain = (struct gemv_chain *)R->chain + queued * R->threads};
    queued++;
}

void rt_gemm_q(struct rt *R, const void *w, long rows, long cols, const float *x, long t, float *y)
{
    gqueue[gqueued] = (struct gemm_job){.w = w, .rows = rows, .cols = cols, .t = t, .x = x, .x_stride = cols, .y = y, .pf_l2 = gemm_l2, .pf_l1 = gemm_l1,
                                        .chain = R->gemm_chain + gqueued * R->threads, .seq = ++gemv_seq};
    gqueued++;
}

void rt_gemv_open(struct rt *R)
{
    round_ready = 0;
    round_open = 1;
    team_open(R->team, batch_thread, NULL);
}

void rt_gemv_go(struct rt *R)
{
    __atomic_thread_fence(__ATOMIC_RELEASE);
    round_ready = 1;
    if (round_open)
        team_close(R->team);
    else
        team_run(R->team, batch_thread, NULL);
    round_open = 0;
    queued = 0;
    gqueued = 0;
}

struct product_job {
    float *z;
    const float *a, *b;
    long h, r, c;
};

/* z[h;r] = a[h;r;] +.× b[h;]: one dot product per row */
static void ip21_thread(void *arg, int tid, int n)
{
    const struct product_job *j = arg;
    SPAN(j->h * j->r);
    const float *row = j->a + i0 * j->c, *v = j->b + (i0 / j->r) * j->c;
    long r = i0 % j->r;
    for (long i = i0; i < i1; i++) {
        v16f acc = knc_zero();
        for (long k = 0; k < j->c; k += 16)
            acc = knc_fma(knc_load(row + k), knc_load(v + k), acc);
        j->z[i] = knc_hsum(acc);
        row += j->c;
        if (++r == j->r) {
            r = 0;
            v += j->c;
        }
    }
}

void rt_ip21(struct rt *R, float *z, const float *a, long h, long r, long c, const float *b)
{
    struct product_job j = {.z = z, .a = a, .b = b, .h = h, .r = r, .c = c};
    run(R, ip21_thread, &j, 2 * h * r * c); /* a reduction per row: about twice the lanes' cost */
}

/* z[h;] = a[h;] +.× b[h;;]: one accumulation per head, no reduction */
static void ip12_thread(void *arg, int tid, int n)
{
    const struct product_job *j = arg;
    SPAN(j->h);
    for (long h = i0; h < i1; h++) {
        float *z = j->z + h * j->c;
        const float *row = j->b + h * j->r * j->c, *w = j->a + h * j->r;
        for (long k = 0; k < j->c; k += 16)
            knc_store(z + k, knc_zero());
        for (long t = 0; t < j->r; t++) {
            const v16f v = knc_bcast(w + t);
            for (long k = 0; k < j->c; k += 16)
                knc_store(z + k, knc_fma(v, knc_load(row + k), knc_load(z + k)));
            row += j->c;
        }
    }
}

void rt_ip12(struct rt *R, float *z, const float *a, long h, long t, long c, const float *b)
{
    struct product_job j = {.z = z, .a = a, .b = b, .h = h, .r = t, .c = c};
    run(R, ip12_thread, &j, h * t * c);
}

/* z[h;i;] = a[h;i] × b[h;] */
static void outer11_thread(void *arg, int tid, int n)
{
    const struct product_job *j = arg;
    SPAN(j->h * j->r);
    float *z = j->z + i0 * j->c;
    const float *v = j->b + (i0 / j->r) * j->c;
    long r = i0 % j->r;
    for (long i = i0; i < i1; i++) {
        const v16f s = knc_bcast(j->a + i);
        for (long k = 0; k < j->c; k += 16)
            knc_store(z + k, knc_mul(s, knc_load(v + k)));
        z += j->c;
        if (++r == j->r) {
            r = 0;
            v += j->c;
        }
    }
}

void rt_outer11(struct rt *R, float *z, const float *a, const float *b, long h, long m, long k)
{
    struct product_job j = {.z = z, .a = a, .b = b, .h = h, .r = m, .c = k};
    run(R, outer11_thread, &j, h * m * k);
}

/* z[h;...] = a[h] × b[h;...], split by blocks: the heads are fewer than the threads */
static void scale_thread(void *arg, int tid, int n)
{
    const struct product_job *j = arg;
    SPAN(j->h * j->c / 16);
    long h = i0 * 16 / j->c, next = (h + 1) * j->c;
    v16f s = knc_bcast(j->a + h);
    for (long i = i0 * 16; i < i1 * 16; i += 16) {
        if (i == next) {
            h++;
            next += j->c;
            s = knc_bcast(j->a + h);
        }
        knc_store(j->z + i, knc_mul(s, knc_load(j->b + i)));
    }
}

/* z[h;i;] += a[h;i] × b[h;] */
static void outer_add_thread(void *arg, int tid, int n)
{
    const struct product_job *j = arg;
    SPAN(j->h * j->r);
    float *z = j->z + i0 * j->c;
    const float *v = j->b + (i0 / j->r) * j->c;
    long r = i0 % j->r;
    for (long i = i0; i < i1; i++) {
        const v16f s = knc_bcast(j->a + i);
        for (long k = 0; k < j->c; k += 16)
            knc_store(z + k, knc_fma(s, knc_load(v + k), knc_load(z + k)));
        z += j->c;
        if (++r == j->r) {
            r = 0;
            v += j->c;
        }
    }
}

void rt_outer_add(struct rt *R, float *z, const float *a, const float *b, long h, long m, long k)
{
    struct product_job j = {.z = z, .a = a, .b = b, .h = h, .r = m, .c = k};
    run(R, outer_add_thread, &j, h * m * k);
}

void rt_scale0n(struct rt *R, float *z, const float *a, const float *b, long h, long cell)
{
    struct product_job j = {.z = z, .a = a, .b = b, .h = h, .c = cell};
    run(R, scale_thread, &j, h * cell);
}

/* ---- the inner products over a head's history, in place */
struct kv_job {
    float *z;
    const float *a, *kv, *b;
    long n1, c, h, t;
    const long *l;
};

static void ip21_kv_thread(void *arg, int tid, int n)
{
    const struct kv_job *j = arg;
    SPAN(j->h * j->t);
    long h = i0 / j->t, t = i0 % j->t;
    const float *v = j->b + h * j->c;
    long head = j->l[h] * j->c;
    const long step = j->n1 * j->c;
    for (long i = i0; i < i1; i++) {
        const float *row = j->kv + t * step + head;
        v16f acc = knc_zero();
        for (long k = 0; k < j->c; k += 16)
            acc = knc_fma(knc_load(row + k), knc_load(v + k), acc);
        j->z[i] = knc_hsum(acc);
        if (++t == j->t) {
            t = 0;
            if (++h < j->h) {
                v += j->c;
                head = j->l[h] * j->c;
            }
        }
    }
}

void rt_ip21_kv(struct rt *R, float *z, const float *kv, long n1, long c, const long *l, long h, long t, const float *b)
{
    struct kv_job j = {.z = z, .kv = kv, .b = b, .n1 = n1, .c = c, .h = h, .t = t, .l = l};
    run(R, ip21_kv_thread, &j, 2 * h * t * c);
}

static void ip12_kv_thread(void *arg, int tid, int n)
{
    const struct kv_job *j = arg;
    SPAN(j->h);
    for (long h = i0; h < i1; h++) {
        float *z = j->z + h * j->c;
        const float *w = j->a + h * j->t;
        const float *row = j->kv + j->l[h] * j->c;
        const long step = j->n1 * j->c;
        for (long k = 0; k < j->c; k += 16)
            knc_store(z + k, knc_zero());
        for (long t = 0; t < j->t; t++) {
            const v16f v = knc_bcast(w + t);
            for (long k = 0; k < j->c; k += 16)
                knc_store(z + k, knc_fma(v, knc_load(row + k), knc_load(z + k)));
            row += step;
        }
    }
}

void rt_ip12_kv(struct rt *R, float *z, const float *a, const float *kv, long n1, long c, const long *l, long h, long t)
{
    struct kv_job j = {.z = z, .a = a, .kv = kv, .n1 = n1, .c = c, .h = h, .t = t, .l = l};
    run(R, ip12_kv_thread, &j, h * t * c);
}

/* ---- structure */
void rt_q8_row(const void *w, long cols, long row, float *z)
{
    const long nb = cols / 32;
    const uint8_t *tile = (const uint8_t *)w + (row / 16) * nb * 544;
    const int r = (int)(row % 16);
    const float one = 1.0f;
    float t[32] __attribute__((aligned(64))), sc[16] __attribute__((aligned(64)));
    for (long b = 0; b < nb; b++) {
        const int8_t *c = (const int8_t *)tile;
        for (int i = 0; i < 32; i++)
            t[i] = (float)c[i * 16 + r];
        knc_store(sc, knc_fma_f16(knc_bcast(&one), (const uint16_t *)(tile + 512), knc_zero()));
        const v16f s = knc_bcast(sc + r);
        knc_store(z + b * 32, knc_mul(knc_load(t), s));
        knc_store(z + b * 32 + 16, knc_mul(knc_load(t + 16), s));
        tile += 544;
    }
}

void rt_cat(float *z, const float *a, long na, const float *b, long nb)
{
    memcpy(z, a, sizeof(float) * (size_t)na);
    memcpy(z + na, b, sizeof(float) * (size_t)nb);
}

struct index_job {
    float *z;
    const float *a;
    long n0, n1, n2;
    const long *l0, *l1;
    long k0, k1;
};

static void index3_thread(void *arg, int tid, int n)
{
    const struct index_job *j = arg;
    SPAN(j->k0 * j->k1);
    const size_t bytes = sizeof(float) * (size_t)j->n2;
    long a0 = i0 / j->k1, a1 = i0 % j->k1;
    for (long i = i0; i < i1; i++) {
        const long p0 = j->l0 ? j->l0[a0] : a0, p1 = j->l1 ? j->l1[a1] : a1;
        memcpy(j->z + i * j->n2, j->a + (p0 * j->n1 + p1) * j->n2, bytes);
        if (++a1 == j->k1) {
            a1 = 0;
            a0++;
        }
    }
}

void rt_index3(struct rt *R, float *z, const float *a, long n1, long n2, const long *l0, long k0, const long *l1, long k1)
{
    struct index_job j = {.z = z, .a = a, .n1 = n1, .n2 = n2, .l0 = l0, .l1 = l1, .k0 = k0, .k1 = k1};
    run(R, index3_thread, &j, k0 * k1 * n2);
}

static void transpose_thread(void *arg, int tid, int n)
{
    const struct index_job *j = arg;
    SPAN(j->n0 * j->n1);
    const size_t bytes = sizeof(float) * (size_t)j->n2;
    long a1 = i0 / j->n0, a0 = i0 % j->n0;
    for (long i = i0; i < i1; i++) {
        memcpy(j->z + i * j->n2, j->a + (a0 * j->n1 + a1) * j->n2, bytes);
        if (++a0 == j->n0) {
            a0 = 0;
            a1++;
        }
    }
}

void rt_transpose102(struct rt *R, float *z, const float *a, long n0, long n1, long n2)
{
    struct index_job j = {.z = z, .a = a, .n0 = n0, .n1 = n1, .n2 = n2};
    run(R, transpose_thread, &j, n0 * n1 * n2);
}

void rt_scale_outer(struct rt *R, float *z, const float *s, float g, const float *a, const float *b, long m, long k)
{
    (void)R;
    const v16f vg = knc_bcast(&g);
    for (long i = 0; i < m; i++, z += k, s += k) {
        const v16f va = knc_bcast(a + i);
        for (long q = 0; q < k; q += 16)
            knc_store(z + q, knc_fma(vg, knc_load(s + q), knc_mul(va, knc_load(b + q))));
    }
}
