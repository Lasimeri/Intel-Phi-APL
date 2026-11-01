/* arena.h: what brings a model file into memory, shared by the benchmarks and the
 * engine: first touch on the team, the file's sum, a plain read for the host model.
 * Header-only, for programs built by knc-cc (x87 ABI side) and by the host cc. */
#ifndef APL_ARENA_H
#define APL_ARENA_H
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

/* The kernel zeroes a fresh huge page at its first fault; inside pread that is one
 * thread (0.94 GB/s against 3.25, measured 2026-11-01). Touch arenas on the team. */
struct touch_job {
    char *p;
    size_t bytes;
};

static inline void touch_thread(void *arg, int tid, int n)
{
    const struct touch_job *j = arg;
    const size_t pages = j->bytes >> 21;
    for (size_t i = pages * (size_t)tid / (size_t)n; i < pages * (size_t)(tid + 1) / (size_t)n; i++)
        j->p[i << 21] = 0;
}

/* The file's sum: Σ word_i × (2i+1) mod 2^64 over whole 8-byte words (a GGUF ends on
 * a 32-byte boundary, so none are left over). Position-sensitive, and the same however
 * it is split across threads. */
struct sum_job {
    const uint64_t *w;
    long words;
    uint64_t part[256];
};

static inline void sum_thread(void *arg, int tid, int n)
{
    struct sum_job *j = arg;
    const long i0 = j->words * tid / n, i1 = j->words * (tid + 1) / n;
    uint64_t s = 0;
    for (long i = i0; i < i1; i++)
        s += j->w[i] * (2 * (uint64_t)i + 1);
    j->part[tid] = s;
}

static inline int read_all(const char *path, void *p, size_t bytes)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        return -1;
    }
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = read(fd, (char *)p + done, bytes - done);
        if (n <= 0) {
            fprintf(stderr, "%s: short read at %zu of %zu\n", path, done, bytes);
            close(fd);
            return -1;
        }
        done += (size_t)n;
    }
    close(fd);
    return 0;
}

#endif
