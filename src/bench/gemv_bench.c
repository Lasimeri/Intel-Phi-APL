/* gemv_bench.c: the matmul benchmark that picks the APL form of W+.×x, and
 * the barrier latency a token's many parallel steps will pay. Built by knc-cc
 * for the card (and by the host cc for a quick run of the host model). Never
 * includes knc.h (x87 ABI side); kernels are reached through gemv.h, team.h.
 *
 *   gemv_bench gemv I|R ROWS COLS TPC CORES PF_L2 PF_L1 REPEATS
 *   gemv_bench gemm ROWS COLS T TPC CORES PF_L2 PF_L1 REPEATS
 *   gemv_bench barrier TPC CORES ROUNDS
 *
 * TPC threads per core on cores 0..CORES-1 (CPUs 4c+1+t on the 7120A; core 59
 * carries the block pollers and core 60 the OS). The weights are synthetic: the
 * speed of a matvec does not depend on the values. One warm-up matvec, then
 * REPEATS, reported min / median / max with GB/s against the card's read peak
 * and GFLOPS against its F32 peak (by arithmetic, 1238.094 MHz). */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_util.h"
#include "gemm.h"
#include "gemv.h"
#include "team.h"

#define MAX_REPEATS 64

/* ---- synthetic Q8_0 rows, GGUF layout, made and repacked by the whole team */
struct fill_job {
    uint8_t *q8;
    void *out;
    long rows, cols, row_bytes;
    char form;
};

static void fill_thread(void *arg, int tid, int n)
{
    const struct fill_job *j = arg;
    long g0 = (j->rows / 16) * tid / n, g1 = (j->rows / 16) * (tid + 1) / n;
    uint32_t s = 0x9E3779B9u + (uint32_t)tid * 2654435761u;
    uint8_t *p = j->q8 + g0 * 16 * j->row_bytes, *end = j->q8 + g1 * 16 * j->row_bytes;
    while (p < end) {
        /* one block: a small positive f16 scale, then 32 pseudo-random int8 */
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        uint16_t h = (uint16_t)(0x1800u | (s & 0x3ffu));
        memcpy(p, &h, 2);
        p += 2;
        for (int i = 0; i < 32; i += 4) {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            memcpy(p + i, &s, 4);
        }
        p += 32;
    }
    long nb = j->cols / 32;
    if (j->form == 'I') {
        gemv_i_repack(j->q8 + g0 * 16 * j->row_bytes, (g1 - g0) * 16, j->cols, (char *)j->out + g0 * nb * 544);
    } else {
        long stride = gemv_r_bytes(1, j->cols);
        gemv_r_repack(j->q8 + g0 * 16 * j->row_bytes, (g1 - g0) * 16, j->cols, (char *)j->out + g0 * 16 * stride);
    }
}

static int run_gemv(int argc, char **argv)
{
    if (argc != 10) {
        fprintf(stderr, "usage: gemv_bench gemv I|R ROWS COLS TPC CORES PF_L2 PF_L1 REPEATS\n");
        return 2;
    }
    const char form = argv[2][0];
    const long rows = atol(argv[3]), cols = atol(argv[4]);
    const int tpc = atoi(argv[5]), cores = atoi(argv[6]);
    const long pf_l2 = atol(argv[7]), pf_l1 = atol(argv[8]);
    int repeats = atoi(argv[9]);
    if (repeats > MAX_REPEATS)
        repeats = MAX_REPEATS;
    if ((form != 'I' && form != 'R') || rows % 16 || cols % 512) {
        fprintf(stderr, "form I or R, ROWS a multiple of 16, COLS of 512\n");
        return 2;
    }
    const int n = tpc * cores;
    struct team *t = team_create(cpu_list(tpc, cores), n, 1);
    if (!t)
        return 1;

    const long row_bytes = cols / 32 * 34;
    const long w_bytes = form == 'I' ? gemv_i_bytes(rows, cols) : gemv_r_bytes(rows, cols);
    uint8_t *q8 = big_alloc((size_t)(rows * row_bytes), "raw Q8_0");
    void *w = big_alloc((size_t)w_bytes, "repacked");
    struct fill_job fj = {.q8 = q8, .out = w, .rows = rows, .cols = cols, .row_bytes = row_bytes, .form = form};
    double t0 = now_s();
    team_run(t, fill_thread, &fj);
    printf("filled and repacked %ld B in %.3f s = %.2f GB/s on the card's team\n", rows * row_bytes, now_s() - t0,
           (double)(rows * row_bytes) / (now_s() - t0) / 1e9);
    munmap(q8, (size_t)(rows * row_bytes));

    float *x, *y, *sx;
    if (posix_memalign((void **)&x, 64, sizeof(float) * (size_t)cols) || posix_memalign((void **)&y, 64, sizeof(float) * (size_t)rows) ||
        posix_memalign((void **)&sx, 64, sizeof(float) * (size_t)(cols / 32)))
        return 1;
    for (long i = 0; i < cols; i++)
        x[i] = (float)((i * 2654435761u >> 8) & 0xffff) / 65536.0f - 0.5f;
    gemv_block_sums(x, cols, sx);

    struct gemv_job job = {.w = w, .rows = rows, .cols = cols, .x = x, .y = y, .sx = sx, .pf_l2 = pf_l2, .pf_l1 = pf_l1};
    if (getenv("APL_TILES") && posix_memalign((void **)&job.chain, 64, sizeof(struct gemv_chain) * (size_t)n))
        return 1;
    team_fn fn = form == 'I' ? gemv_i_thread : gemv_r_thread;
    double d[MAX_REPEATS + 1];
    for (int r = 0; r <= repeats; r++) {
        t0 = now_s();
        job.seq = (uint64_t)r + 1;
        team_run(t, fn, &job);
        d[r] = now_s() - t0;
    }
    qsort(d + 1, (size_t)repeats, sizeof d[0], cmp_d);
    double med = d[1 + repeats / 2];
    double gbs = (double)w_bytes / med / 1e9, gf = 2.0 * (double)rows * (double)cols / med / 1e9;
    printf("form %c %ldx%ld tpc %d cores %d threads %d pf %ld,%ld: warm-up %.3f ms, min %.3f med %.3f max %.3f ms; "
           "%.1f GB/s (%.1f%% of %.0f), %.1f GFLOPS (%.1f%% of %.0f); y[0] %.6f\n",
           form, rows, cols, tpc, cores, n, pf_l2, pf_l1, 1e3 * d[0], 1e3 * d[1], 1e3 * med, 1e3 * d[repeats], gbs,
           100.0 * gbs / READ_PEAK_GBS, READ_PEAK_GBS, gf, 100.0 * gf / F32_PEAK_GFLOPS, F32_PEAK_GFLOPS, (double)y[0]);
    munmap(w, (size_t)w_bytes);
    team_destroy(t);
    return 0;
}

/* ---- W+.×X for T tokens in one pass, held bit for bit against one matvec per token */
static int run_gemm(int argc, char **argv)
{
    if (argc != 10) {
        fprintf(stderr, "usage: gemv_bench gemm ROWS COLS T TPC CORES PF_L2 PF_L1 REPEATS\n");
        return 2;
    }
    const long rows = atol(argv[2]), cols = atol(argv[3]), tokens = atol(argv[4]);
    const int tpc = atoi(argv[5]), cores = atoi(argv[6]);
    const long pf_l2 = atol(argv[7]), pf_l1 = atol(argv[8]);
    int repeats = atoi(argv[9]);
    if (repeats > MAX_REPEATS)
        repeats = MAX_REPEATS;
    if (rows % 16 || cols % 32 || tokens < 1 || tokens > 4096) {
        fprintf(stderr, "ROWS a multiple of 16, COLS of 32, T from 1 to 4096\n");
        return 2;
    }
    const int n = tpc * cores;
    struct team *t = team_create(cpu_list(tpc, cores), n, 1);
    if (!t)
        return 1;
    const long row_bytes = cols / 32 * 34, w_bytes = gemv_i_bytes(rows, cols);
    uint8_t *q8 = big_alloc((size_t)(rows * row_bytes), "raw Q8_0");
    void *w = big_alloc((size_t)w_bytes, "repacked");
    struct fill_job fj = {.q8 = q8, .out = w, .rows = rows, .cols = cols, .row_bytes = row_bytes, .form = 'I'};
    team_run(t, fill_thread, &fj);
    munmap(q8, (size_t)(rows * row_bytes));

    float *x, *y, *y1;
    /* APL_XPAD: lanes of padding after each row of x, to spread the rows across L1 sets */
    const long stride = cols + (getenv("APL_XPAD") ? atol(getenv("APL_XPAD")) : 0);
    if (posix_memalign((void **)&x, 64, sizeof(float) * (size_t)(stride * tokens)) ||
        posix_memalign((void **)&y, 64, sizeof(float) * (size_t)(rows * tokens)) ||
        posix_memalign((void **)&y1, 64, sizeof(float) * (size_t)(rows * tokens)))
        return 1;
    for (long i = 0; i < stride * tokens; i++)
        x[i] = (float)((i * 2654435761u >> 8) & 0xffff) / 65536.0f - 0.5f;

    struct gemm_job job = {.w = w, .rows = rows, .cols = cols, .t = tokens, .x = x, .x_stride = stride, .pass_t = getenv("APL_PASS") ? atol(getenv("APL_PASS")) : 0, .y = y, .pf_l2 = pf_l2, .pf_l1 = pf_l1};
    double d[MAX_REPEATS + 1], s[MAX_REPEATS + 1];
    for (int r = 0; r <= repeats; r++) {
        double t0 = now_s();
        team_run(t, gemm_i_thread, &job);
        d[r] = now_s() - t0;
        t0 = now_s();
        for (long k = 0; k < tokens; k++) {
            struct gemv_job v = {.w = w, .rows = rows, .cols = cols, .x = x + k * stride, .y = y1 + k * rows, .pf_l2 = 32768, .pf_l1 = 2048};
            team_run(t, gemv_i_thread, &v);
        }
        s[r] = now_s() - t0;
    }
    long differ = 0;
    for (long i = 0; i < rows * tokens; i++) {
        uint32_t a, b;
        memcpy(&a, y + i, 4);
        memcpy(&b, y1 + i, 4);
        differ += a != b;
    }
    qsort(d + 1, (size_t)repeats, sizeof d[0], cmp_d);
    qsort(s + 1, (size_t)repeats, sizeof s[0], cmp_d);
    const double med = d[1 + repeats / 2], meds = s[1 + repeats / 2];
    const double flop = 2.0 * (double)rows * (double)cols * (double)tokens;
    printf("gemm %ldx%ld T %ld tpc %d cores %d threads %d pf %ld,%ld: min %.3f med %.3f max %.3f ms; %.1f GFLOPS (%.1f%% of %.0f), %.1f GB/s; "
           "one matvec per token: %.3f ms, %.1f GFLOPS; %.2fx; lanes differing %ld of %ld\n",
           rows, cols, tokens, tpc, cores, n, pf_l2, pf_l1, 1e3 * d[1], 1e3 * med, 1e3 * d[repeats], flop / med / 1e9,
           100.0 * flop / med / 1e9 / F32_PEAK_GFLOPS, F32_PEAK_GFLOPS, (double)w_bytes / med / 1e9, 1e3 * meds, flop / meds / 1e9, meds / med, differ,
           rows * tokens);
    team_destroy(t);
    return differ != 0;
}

/* ---- barrier latency */
struct barrier_job {
    struct team *t;
    long rounds;
};

static void barrier_thread(void *arg, int tid, int n)
{
    (void)tid;
    (void)n;
    const struct barrier_job *j = arg;
    for (long i = 0; i < j->rounds; i++)
        team_barrier(j->t);
}

static void empty_thread(void *arg, int tid, int n)
{
    (void)arg;
    (void)tid;
    (void)n;
}

static int run_barrier(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: gemv_bench barrier TPC CORES ROUNDS\n");
        return 2;
    }
    const int tpc = atoi(argv[2]), cores = atoi(argv[3]), n = tpc * cores;
    const long rounds = atol(argv[4]);
    struct team *t = team_create(cpu_list(tpc, cores), n, 1);
    if (!t)
        return 1;
    struct barrier_job j = {.t = t, .rounds = rounds};
    for (int r = 0; r < 4; r++) {
        double t0 = now_s();
        team_run(t, barrier_thread, &j);
        double b = now_s() - t0;
        t0 = now_s();
        for (long i = 0; i < rounds; i++)
            team_run(t, empty_thread, NULL);
        double f = now_s() - t0;
        printf("threads %d: barrier %.2f us, empty team_run %.2f us\n", n, 1e6 * b / (double)rounds, 1e6 * f / (double)rounds);
    }
    team_destroy(t);
    return 0;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc >= 2 && strcmp(argv[1], "gemv") == 0)
        return run_gemv(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "gemm") == 0)
        return run_gemm(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "barrier") == 0)
        return run_barrier(argc, argv);
    fprintf(stderr, "usage: gemv_bench gemv|barrier ...\n");
    return 2;
}
