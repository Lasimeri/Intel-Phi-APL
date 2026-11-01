/* mem_order.c: does the order threads read memory in change what GDDR delivers? A pure read of
 * one arena, no arithmetic worth the name, with the threads taking it in chunks round the team:
 * a small chunk keeps every thread near the same address, so in the same DRAM pages if the
 * card's address hash lets them; chunk 0 gives each thread one region of its own, far from the
 * rest, as a matrix's split by groups does today.
 *
 *   mem_order ARENA_MB TPC CORES CHUNK_BYTES PF_L2 PF_L1 REPEATS
 *
 * PF_L2 PF_L1: prefetch distances along the thread's own stream, bytes (0: its current line). Prints the median, the
 * spread, and a sum that every chunk size must agree on: each byte was read exactly once. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "arena.h"
#include "bench_util.h"
#include "knc.h"
#include "team.h"

#define LINE 64L
#define MAX_REPEATS 64

struct read_job {
    const char *base;
    long bytes, chunk, pf_l2, pf_l1;
    uint32_t part[256 * 16];
};

/* A place in this thread's own stream, stepped a line at a time: no division in the loop, which
 * has about 50 cycles a line to spare at the card's peak */
struct cursor {
    const char *p;
    long left;
};

static inline struct cursor at(const struct read_job *j, long x, int tid, int n)
{
    struct cursor c = {j->base + (x / j->chunk) * n * j->chunk + tid * j->chunk + x % j->chunk, (j->chunk - x % j->chunk) / LINE};
    return c;
}

#define STEP(c, gap)                                                                                                   \
    do {                                                                                                               \
        (c).p += LINE;                                                                                                 \
        if (--(c).left == 0) {                                                                                         \
            (c).p += (gap);                                                                                            \
            (c).left = lines;                                                                                          \
        }                                                                                                              \
    } while (0)

static void read_thread(void *arg, int tid, int n)
{
    struct read_job *j = arg;
    const long chunks = j->bytes / j->chunk, mine = (chunks / n) * j->chunk;
    const long lines = j->chunk / LINE, gap = (n - 1) * j->chunk;
    struct cursor r = at(j, 0, tid, n), l2 = at(j, j->pf_l2, tid, n), l1 = at(j, j->pf_l1, tid, n);
    v16f acc = knc_zero();
    for (long i = 0; i < mine / LINE; i++) {
        knc_prefetch1(l2.p); /* a prefetch never faults, past the end included */
        knc_prefetch0(l1.p);
        acc = knc_add(acc, knc_load((const float *)r.p));
        STEP(r, gap);
        STEP(l2, gap);
        STEP(l1, gap);
    }
    float t[16] __attribute__((aligned(64)));
    knc_store(t, acc);
    j->part[tid * 16] = (uint32_t)t[0];
}

int main(int argc, char **argv)
{
    if (argc != 8) {
        fprintf(stderr, "usage: mem_order ARENA_MB TPC CORES CHUNK_BYTES PF_L2 PF_L1 REPEATS\n");
        return 2;
    }
    const long bytes = atol(argv[1]) << 20;
    const int tpc = atoi(argv[2]), cores = atoi(argv[3]), threads = tpc * cores;
    const long chunk_arg = atol(argv[4]), pf_l2 = atol(argv[5]), pf_l1 = atol(argv[6]);
    int repeats = atoi(argv[7]);
    if (repeats > MAX_REPEATS)
        repeats = MAX_REPEATS;
    /* whole lines, and a whole number of chunks per thread */
    const long chunk = chunk_arg > 0 ? chunk_arg : ((bytes / threads) & ~(LINE - 1));
    if (threads > 256 || chunk % LINE) {
        fprintf(stderr, "mem_order: at most 256 threads; chunk a multiple of %ld\n", LINE);
        return 2;
    }
    struct team *team = team_create(cpu_list(tpc, cores), threads, 1);
    char *arena = big_alloc((size_t)bytes, "arena");
    struct touch_job tj = {.p = arena, .bytes = (size_t)bytes};
    if (!team)
        return 1;
    team_run(team, touch_thread, &tj);
    for (long i = 0; i < bytes; i += 4)
        *(float *)(arena + i) = 1.0f;
    static struct read_job j;
    j = (struct read_job){.base = arena, .bytes = bytes, .chunk = chunk, .pf_l2 = pf_l2, .pf_l1 = pf_l1};
    const long read_bytes = (bytes / chunk / threads) * threads * chunk;
    double t[MAX_REPEATS + 1];
    for (int r = 0; r <= repeats; r++) {
        const double t0 = now_s();
        team_run(team, read_thread, &j);
        t[r] = now_s() - t0;
    }
    qsort(t + 1, (size_t)repeats, sizeof(double), cmp_d);
    uint64_t sum = 0;
    for (int i = 0; i < threads; i++)
        sum += j.part[i * 16];
    printf("mem_order chunk %ld threads %d: median %.2f GB/s (%.2f-%.2f) over %ld B; lanes summed %llu\n", chunk, threads,
           (double)read_bytes / t[1 + repeats / 2] / 1e9, (double)read_bytes / t[repeats] / 1e9, (double)read_bytes / t[1] / 1e9, read_bytes,
           (unsigned long long)sum);
    team_destroy(team);
    return 0;
}
