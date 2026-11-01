/* real_gemv.c: one real weight matrix through the whole path: the file arrives
 * (over the stream on the card, from the file itself for the host model), the
 * team repacks the matrix in place, then y = W x runs on the team. The first
 * measurement of the engine's largest unknowns on real weights. Never includes
 * knc.h (x87 ABI side).
 *
 *   real_gemv SOURCE OFFSET ROWS COLS X.f32 Y.f32 TPC CORES PF_L2 PF_L1 REPEATS
 *
 * SOURCE: "stream", or a file path (the host model). OFFSET: the tensor's absolute
 * offset in the file. Prints a position-weighted sum of the file, which must match
 * between card and host, and writes y as raw F32. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "arena.h"
#include "bench_util.h"
#include "gemv.h"
#include "stream.h"
#include "team.h"

#define MAX_REPEATS 64

int main(int argc, char **argv)
{
    if (argc != 12) {
        fprintf(stderr, "usage: real_gemv SOURCE OFFSET ROWS COLS X.f32 Y.f32 TPC CORES PF_L2 PF_L1 REPEATS\n");
        return 2;
    }
    const char *source = argv[1], *xpath = argv[5], *ypath = argv[6];
    const long offset = atol(argv[2]), rows = atol(argv[3]), cols = atol(argv[4]);
    const int tpc = atoi(argv[7]), cores = atoi(argv[8]);
    const long pf_l2 = atol(argv[9]), pf_l1 = atol(argv[10]);
    int repeats = atoi(argv[11]);
    if (repeats > MAX_REPEATS)
        repeats = MAX_REPEATS;
    const int n = tpc * cores;
    if (rows % 16 || cols % 32 || n > 256) {
        fprintf(stderr, "ROWS a multiple of 16, COLS of 32, at most 256 threads\n");
        return 2;
    }
    struct team *t = team_create(cpu_list(tpc, cores), n, 1);
    if (!t)
        return 1;

    /* ---- the file */
    const int streaming = strcmp(source, "stream") == 0;
    struct stream s = {.blk = -1};
    uint64_t total;
    struct stat st;
    if (streaming) {
        if (stream_open(&s))
            return 1;
        total = s.total;
    } else {
        if (stat(source, &st)) {
            perror(source);
            return 1;
        }
        total = (uint64_t)st.st_size;
    }
    const size_t arena_bytes = (size_t)((total + STREAM_PAGE - 1) & ~(uint64_t)(STREAM_PAGE - 1));
    char *file = big_alloc(arena_bytes, "arena");
    double t0 = now_s();
    /* first touch on the team: the kernel zeroes a huge page at its first fault, and
     * inside pread that is one thread (0.94 GB/s against 3.25, measured 2026-11-01) */
    struct touch_job tj = {.p = file, .bytes = arena_bytes};
    team_run(t, touch_thread, &tj);
    printf("touched the arena in %.3f s on the team\n", now_s() - t0);
    t0 = now_s();
    if (streaming ? stream_receive(&s, file) : read_all(source, file, (size_t)total))
        return 1;
    double dt = now_s() - t0;
    printf("received %llu B in %.3f s = %.2f GB/s\n", (unsigned long long)total, dt, (double)total / dt / 1e9);
    if (streaming)
        stream_close(&s);

    struct sum_job sj = {.w = (const uint64_t *)file, .words = (long)(total / 8)};
    t0 = now_s();
    team_run(t, sum_thread, &sj);
    uint64_t sum = 0;
    for (int i = 0; i < n; i++)
        sum += sj.part[i];
    printf("file sum %016llx (%.3f s on the team)\n", (unsigned long long)sum, now_s() - t0);

    /* ---- the matrix, in place: a 16-row group is the same size in both forms */
    const long w_bytes = gemv_i_bytes(rows, cols);
    if ((uint64_t)offset + (uint64_t)w_bytes > total || offset % 32) {
        fprintf(stderr, "the tensor is outside the file or not 32-byte aligned\n");
        return 1;
    }
    char *w = file + offset;
    char *tmp = big_alloc((size_t)gemv_i_bytes(16, cols) * (size_t)n, "repack scratch");
    struct gemv_repack_job rj = {.w = w, .rows = rows, .cols = cols, .tmp = tmp};
    t0 = now_s();
    team_run(t, gemv_i_repack_thread, &rj);
    dt = now_s() - t0;
    printf("repacked %ld B in place in %.3f s = %.2f GB/s\n", w_bytes, dt, (double)w_bytes / dt / 1e9);
    munmap(tmp, (size_t)gemv_i_bytes(16, cols) * (size_t)n);

    /* ---- W+.×x */
    float *x, *y;
    if (posix_memalign((void **)&x, 64, sizeof(float) * (size_t)cols) ||
        posix_memalign((void **)&y, 64, sizeof(float) * (size_t)rows))
        return 1;
    if (read_all(xpath, x, sizeof(float) * (size_t)cols))
        return 1;
    struct gemv_job job = {.w = w, .rows = rows, .cols = cols, .x = x, .y = y, .pf_l2 = pf_l2, .pf_l1 = pf_l1};
    double d[MAX_REPEATS + 1];
    for (int r = 0; r <= repeats; r++) {
        t0 = now_s();
        team_run(t, gemv_i_thread, &job);
        d[r] = now_s() - t0;
    }
    qsort(d + 1, (size_t)repeats, sizeof d[0], cmp_d);
    const double med = d[1 + repeats / 2];
    const double gbs = (double)w_bytes / med / 1e9, gf = 2.0 * (double)rows * (double)cols / med / 1e9;
    printf("W+.×x %ldx%ld tpc %d cores %d threads %d pf %ld,%ld: warm-up %.3f ms, min %.3f med %.3f max %.3f ms (%d "
           "repeats), %.1f GB/s (%.1f%% of %.0f), %.1f GFLOPS (%.1f%% of %.0f)\n",
           rows, cols, tpc, cores, n, pf_l2, pf_l1, 1e3 * d[0], 1e3 * d[1], 1e3 * med, 1e3 * d[repeats], repeats, gbs,
           100.0 * gbs / READ_PEAK_GBS, READ_PEAK_GBS, gf, 100.0 * gf / F32_PEAK_GFLOPS, F32_PEAK_GFLOPS);

    FILE *f = fopen(ypath, "wb");
    if (!f || fwrite(y, sizeof(float), (size_t)rows, f) != (size_t)rows || fclose(f)) {
        perror(ypath);
        return 1;
    }
    team_destroy(t);
    return 0;
}
