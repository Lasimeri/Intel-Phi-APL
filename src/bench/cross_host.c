/* cross_host.c: the host's side of the crossing benchmark, and its verdict: what a
 * value's trip to the card and back costs, seen from the host, for the sizes the engine
 * really moves. Every size's bytes are checked after the trip. Pinned on CPU 2.
 *
 *   cross_host [ROUNDS] */
#define _GNU_SOURCE
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "link.h"

#define WAIT_S 30.0

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    const long rounds = argc > 1 ? atol(argv[1]) : 5000;
    static const uint64_t sizes[] = {64, 1024, 8192, 16384, 24576, 49152, 65536, 1048576};
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(2, &set);
    sched_setaffinity(0, sizeof set, &set);

    int fd = open("/dev/shm/phi-hostmem", O_RDWR);
    if (fd < 0) {
        perror("/dev/shm/phi-hostmem");
        return 1;
    }
    char *part = mmap(NULL, STREAM_PART_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)STREAM_REGION_OFFSET);
    if (part == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    struct link_ctl *c = (struct link_ctl *)(part + (LINK_CTL_OFFSET - STREAM_REGION_OFFSET));
    char *out = part + (LINK_BUF_OFFSET - STREAM_REGION_OFFSET), *back = part + (LINK_BUF2_OFFSET - STREAM_REGION_OFFSET);
    double *us = malloc(sizeof(double) * (size_t)rounds);
    if (!us)
        return 1;
    uint64_t seq = c->host_seq;
    c->card_ready = 0;
    const double t_attach = now_s();
    while (!c->card_ready) {
        if (now_s() - t_attach > WAIT_S) {
            fprintf(stderr, "cross_host: the card did not attach\n");
            return 1;
        }
    }
    int bad = 0;
    for (size_t k = 0; k < sizeof sizes / sizeof *sizes; k++) {
        const uint64_t bytes = sizes[k];
        for (uint64_t i = 0; i < bytes; i += 8)
            *(uint64_t *)(out + i) = i * 0x9E3779B97F4A7C15ull + bytes;
        memset(back, 0, bytes);
        /* a warm-up of 10 rounds, then the measured ones */
        for (long r = -10; r < rounds; r++) {
            const double t0 = now_s();
            c->bytes = bytes;
            __atomic_thread_fence(__ATOMIC_RELEASE);
            c->host_seq = ++seq;
            while (c->card_seq != seq) {
                if (now_s() - t0 > WAIT_S) {
                    fprintf(stderr, "cross_host: no answer for size %llu\n", (unsigned long long)bytes);
                    return 1;
                }
            }
            if (r >= 0)
                us[r] = 1e6 * (now_s() - t0);
        }
        const int same = memcmp(out, back, bytes) == 0;
        bad |= !same;
        qsort(us, (size_t)rounds, sizeof(double), cmp_d);
        printf("trip %8llu B: min %7.2f med %7.2f p99 %8.2f max %8.2f us; %s\n", (unsigned long long)bytes, us[0], us[rounds / 2],
               us[rounds * 99 / 100], us[rounds - 1], same ? "bytes identical" : "BYTES DIFFER");
        fflush(stdout);
    }
    c->bytes = 0;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    c->host_seq = ++seq;
    return bad;
}
