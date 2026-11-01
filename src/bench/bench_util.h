/* bench_util.h: what the card benchmarks share: the clock, 2 MiB pages, the CPU
 * list of a team, and the median's sort. Header-only, for programs built by knc-cc
 * (x87 ABI side) and by the host cc. */
#ifndef APL_BENCH_UTIL_H
#define APL_BENCH_UTIL_H
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

#define READ_PEAK_GBS 164.0
#define F32_PEAK_GFLOPS 2416.76

static inline double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Prefers 2 MiB pages; says what it got. */
static inline void *big_alloc(size_t bytes, const char *what)
{
    size_t huge = (bytes + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
    void *p = mmap(NULL, huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p != MAP_FAILED) {
        printf("%s: %zu B on 2 MiB pages\n", what, huge);
        return p;
    }
    p = mmap(NULL, huge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    printf("%s: %zu B on 4 KiB pages (no huge pages reserved)\n", what, huge);
    return p;
}

/* TPC threads per core on cores 0..CORES-1: CPUs 4c+1+t on the 7120A; core 59
 * carries the block pollers and core 60 the OS. */
static inline int *cpu_list(int tpc, int cores)
{
    int *c = malloc(sizeof(int) * (size_t)(tpc * cores));
    if (!c)
        exit(1);
    for (int i = 0; i < cores; i++)
        for (int t = 0; t < tpc; t++)
            c[i * tpc + t] = 4 * i + 1 + t;
    return c;
}

static inline int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

#endif
