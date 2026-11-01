/* rt_host.c: the primitives under what the compiler's placement gives the host's worker, in
 * AVX2 + FMA over F32, on up to 14 threads: the inner products of the attention history and the
 * delta rule's state, where the host measured faster than the VPU, and structure. Elementwise rows
 * are generated loops over vec8.h (src/apl/compiler/fz.apl). The matrices of Q8_0 weights never
 * run here. Primitives only: the math is qwen35.apl's. */
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "rt.h"
#include "team.h"
#include "vec8.h"

/* Below this a team round costs more than it saves: at 16384 lanes the host's work took
 * 1.866-2.602 ms/token on 14 threads against 0.684-0.694 on one (2026-11-01). */
static long par_lanes = 131072;

void rt_host_set_par(long lanes)
{
    par_lanes = lanes;
}

/* ---- scalars */
float rt_s_rsqrt(float x) { return 1.0f / sqrtf(x); }
float rt_s_sqrt(float x) { return sqrtf(x); }

/* ---- reductions */
float rt_sum(const float *a, long n)
{
    __m256 acc = _mm256_setzero_ps();
    long i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_add_ps(acc, _mm256_load_ps(a + i));
    float s = hsum8(acc);
    for (; i < n; i++)
        s += a[i];
    return s;
}

struct max_job {
    const float *a;
    long n;
    float part[256];
};

static void max_thread(void *arg, int tid, int n)
{
    struct max_job *j = arg;
    const long i0 = (j->n / 8 * tid / n) * 8, i1 = tid == n - 1 ? j->n : (j->n / 8 * (tid + 1) / n) * 8;
    __m256 m = _mm256_set1_ps(j->a[i0]);
    long i = i0;
    for (; i + 8 <= i1; i += 8)
        m = _mm256_max_ps(m, _mm256_load_ps(j->a + i));
    float t[8] __attribute__((aligned(32)));
    _mm256_store_ps(t, m);
    float best = t[0];
    for (int k = 1; k < 8; k++)
        if (t[k] > best)
            best = t[k];
    for (; i < i1; i++)
        if (j->a[i] > best)
            best = j->a[i];
    j->part[tid] = best;
}

static struct rt *current_rt;
static int current_threads = 1;

void rt_host_bind(struct rt *R, int threads)
{
    current_threads = threads;
    current_rt = R;
}

float rt_maxv(const float *a, long n)
{
    static struct max_job j;
    j.a = a;
    j.n = n;
    if (n >= par_lanes && current_rt && current_rt->team) {
        team_run(current_rt->team, max_thread, &j);
        float best = j.part[0];
        for (int i = 1; i < current_threads; i++)
            if (j.part[i] > best)
                best = j.part[i];
        return best;
    }
    max_thread(&j, 0, 1);
    return j.part[0];
}

long rt_index_of(const float *a, long n, float v)
{
    const __m256 w = _mm256_set1_ps(v);
    long i = 0;
    for (; i + 8 <= n; i += 8) {
        const int mask = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_load_ps(a + i), w, _CMP_EQ_OQ));
        if (mask)
            return i + __builtin_ctz((unsigned)mask);
    }
    for (; i < n; i++)
        if (a[i] == v)
            return i;
    return n;
}

/* ---- structure */
void rt_cat(float *z, const float *a, long na, const float *b, long nb)
{
    memcpy(z, a, sizeof(float) * (size_t)na);
    memcpy(z + na, b, sizeof(float) * (size_t)nb);
}

void rt_index3(struct rt *R, float *z, const float *a, long n1, long n2, const long *l0, long k0, const long *l1, long k1)
{
    (void)R;
    const size_t bytes = sizeof(float) * (size_t)n2;
    for (long i = 0; i < k0; i++)
        for (long j = 0; j < k1; j++) {
            const long p0 = l0 ? l0[i] : i, p1 = l1 ? l1[j] : j;
            memcpy(z + (i * k1 + j) * n2, a + (p0 * n1 + p1) * n2, bytes);
        }
}

/* z[h;...] = a[h] × b[h;...]: a free row when the placement gives it to the host */
/* ---- the inner products over a head's history, where it lies in system memory */
struct kv_job {
    float *z;
    const float *a, *kv, *b;
    long n1, c, h, t;
    const long *l;
};

static void ip21_kv_thread(void *arg, int tid, int n)
{
    const struct kv_job *j = arg;
    const long items = j->h * j->t;
    const long i0 = items * tid / n, i1 = items * (tid + 1) / n;
    const long step = j->n1 * j->c;
    long h = i0 / j->t, t = i0 % j->t;
    for (long i = i0; i < i1; i++) {
        const float *row = j->kv + t * step + j->l[h] * j->c, *v = j->b + h * j->c;
        __m256 acc = _mm256_setzero_ps();
        for (long k = 0; k < j->c; k += 8)
            acc = _mm256_fmadd_ps(_mm256_load_ps(row + k), _mm256_load_ps(v + k), acc);
        j->z[i] = hsum8(acc);
        if (++t == j->t) {
            t = 0;
            h++;
        }
    }
}

void rt_ip21_kv(struct rt *R, float *z, const float *kv, long n1, long c, const long *l, long h, long t, const float *b)
{
    struct kv_job j = {.z = z, .kv = kv, .b = b, .n1 = n1, .c = c, .h = h, .t = t, .l = l};
    if (2 * h * t * c >= par_lanes && R->team)
        team_run(R->team, ip21_kv_thread, &j);
    else
        ip21_kv_thread(&j, 0, 1);
}

static void ip12_kv_thread(void *arg, int tid, int n)
{
    const struct kv_job *j = arg;
    const long step = j->n1 * j->c;
    for (long h = j->h * tid / n; h < j->h * (tid + 1) / n; h++) {
        float *z = j->z + h * j->c;
        const float *w = j->a + h * j->t, *row = j->kv + j->l[h] * j->c;
        for (long k = 0; k < j->c; k += 8)
            _mm256_store_ps(z + k, _mm256_setzero_ps());
        for (long t = 0; t < j->t; t++) {
            const __m256 v = _mm256_set1_ps(w[t]);
            for (long k = 0; k < j->c; k += 8)
                _mm256_store_ps(z + k, _mm256_fmadd_ps(v, _mm256_load_ps(row + k), _mm256_load_ps(z + k)));
            row += step;
        }
    }
}

void rt_ip12_kv(struct rt *R, float *z, const float *a, const float *kv, long n1, long c, const long *l, long h, long t)
{
    struct kv_job j = {.z = z, .a = a, .kv = kv, .n1 = n1, .c = c, .h = h, .t = t, .l = l};
    if (h * t * c >= par_lanes && R->team)
        team_run(R->team, ip12_kv_thread, &j);
    else
        ip12_kv_thread(&j, 0, 1);
}

/* ---- the delta rule's products, when the placement gives the state S to the host. Eight rows
 * of a head advance together: one running sum takes one FMA per latency, eight take two per
 * cycle, the core's whole rate. */
struct ip21_job {
    float *z;
    const float *a, *b;
    long h, r, c;
};

static void ip21_thread(void *arg, int tid, int n)
{
    const struct ip21_job *j = arg;
    const long r = j->r, c = j->c;
    for (long h = j->h * tid / n; h < j->h * (tid + 1) / n; h++) {
        const float *const row = j->a + h * r * c, *const v = j->b + h * c;
        float *const z = j->z + h * r;
        long i = 0;
        for (; i + 8 <= r; i += 8) {
            const float *const p = row + i * c;
            __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
            for (long k = 0; k < c; k += 8) {
                const __m256 x = _mm256_load_ps(v + k);
                s0 = _mm256_fmadd_ps(_mm256_load_ps(p + k), x, s0);
                s1 = _mm256_fmadd_ps(_mm256_load_ps(p + c + k), x, s1);
                s2 = _mm256_fmadd_ps(_mm256_load_ps(p + 2 * c + k), x, s2);
                s3 = _mm256_fmadd_ps(_mm256_load_ps(p + 3 * c + k), x, s3);
                s4 = _mm256_fmadd_ps(_mm256_load_ps(p + 4 * c + k), x, s4);
                s5 = _mm256_fmadd_ps(_mm256_load_ps(p + 5 * c + k), x, s5);
                s6 = _mm256_fmadd_ps(_mm256_load_ps(p + 6 * c + k), x, s6);
                s7 = _mm256_fmadd_ps(_mm256_load_ps(p + 7 * c + k), x, s7);
            }
            z[i] = hsum8(s0);
            z[i + 1] = hsum8(s1);
            z[i + 2] = hsum8(s2);
            z[i + 3] = hsum8(s3);
            z[i + 4] = hsum8(s4);
            z[i + 5] = hsum8(s5);
            z[i + 6] = hsum8(s6);
            z[i + 7] = hsum8(s7);
        }
        for (; i < r; i++) {
            __m256 s = _mm256_setzero_ps();
            for (long k = 0; k < c; k += 8)
                s = _mm256_fmadd_ps(_mm256_load_ps(row + i * c + k), _mm256_load_ps(v + k), s);
            z[i] = hsum8(s);
        }
    }
}

void rt_ip21(struct rt *R, float *z, const float *a, long h, long r, long c, const float *b)
{
    struct ip21_job j = {.z = z, .a = a, .b = b, .h = h, .r = r, .c = c};
    if (h * r * c >= par_lanes && R->team)
        team_run(R->team, ip21_thread, &j);
    else
        ip21_thread(&j, 0, 1);
}

void rt_par(struct rt *R, void (*fn)(void *arg, int tid, int n), void *arg)
{
    if (R->team)
        team_run(R->team, fn, arg);
    else
        fn(arg, 0, 1);
}

/* z←(g×s)+a∘.×b, the old state s only read: scale and add in one pass over the head. The last token
 * of a batch writes to memory without a read for ownership: nothing reads z before the next batch,
 * while a later token of the batch reads what an earlier one wrote (9B, 2026-11-01: streaming them
 * all cost the prompt 9.5 %, and gained K=0 decode 3.1 %). */
void rt_scale_outer(struct rt *R, float *z, const float *s, float g, const float *a, const float *b, long m, long k)
{
    const __m256 vg = _mm256_set1_ps(g);
    if (!R->last) {
        for (long i = 0; i < m; i++, z += k, s += k) {
            const __m256 va = _mm256_broadcast_ss(a + i);
            for (long q = 0; q < k; q += 8)
                _mm256_store_ps(z + q, _mm256_fmadd_ps(vg, _mm256_load_ps(s + q), _mm256_mul_ps(va, _mm256_load_ps(b + q))));
        }
        return;
    }
    for (long i = 0; i < m; i++, z += k, s += k) {
        const __m256 va = _mm256_broadcast_ss(a + i);
        for (long q = 0; q < k; q += 8)
            _mm256_stream_ps(z + q, _mm256_fmadd_ps(vg, _mm256_load_ps(s + q), _mm256_mul_ps(va, _mm256_load_ps(b + q))));
    }
    _mm_sfence();
}

/* A fused chain from the compiler: over the team only where a round pays for itself */
void rt_fz(struct rt *R, void (*fn)(void *arg, int tid, int n), void *arg, long lanes)
{
    if (lanes >= par_lanes && R->team)
        team_run(R->team, fn, arg);
    else
        fn(arg, 0, 1);
}
