/* apl_feed.c: the host's side of the model stream (src/card/stream.h): reads a file
 * with O_DIRECT into the region's slots as the card frees them. The host does no
 * math here and holds nothing of the model but the slots.
 *
 *   apl_feed FILE [REGION]    REGION defaults to /dev/shm/phi-hostmem
 *
 * Prints the bytes, the wall time and the time spent reading the file. Exit 0
 * when the card has pulled every slot. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "stream.h"

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void nap_50us(void)
{
    const struct timespec t = {.tv_sec = 0, .tv_nsec = 50000};
    nanosleep(&t, NULL);
}

/* Waits for consumed to reach n. Returns 0, or -1 after STREAM_WAIT_S. */
static int wait_consumed(const struct stream_ctl *c, uint64_t n)
{
    const double t0 = now_s();
    while (c->consumed < n) {
        if (now_s() - t0 > STREAM_WAIT_S) {
            fprintf(stderr, "apl_feed: card took %llu slots of %llu, then nothing for %d s\n",
                    (unsigned long long)c->consumed, (unsigned long long)n, STREAM_WAIT_S);
            return -1;
        }
        nap_50us();
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: apl_feed FILE [REGION]\n");
        return 2;
    }
    const char *region = argc == 3 ? argv[2] : "/dev/shm/phi-hostmem";
    int fd = open(argv[1], O_RDONLY | O_DIRECT);
    int rf = open(region, O_RDWR);
    struct stat fs, rs;
    if (fd < 0 || rf < 0 || fstat(fd, &fs) || fstat(rf, &rs)) {
        fprintf(stderr, "apl_feed: %s or %s: %s\n", argv[1], region, strerror(errno));
        return 1;
    }
    if ((uint64_t)rs.st_size < STREAM_REGION_OFFSET + STREAM_PART_BYTES) {
        fprintf(stderr, "apl_feed: %s holds %lld B, the part ends at %llu\n", region, (long long)rs.st_size,
                (unsigned long long)(STREAM_REGION_OFFSET + STREAM_PART_BYTES));
        return 1;
    }
    char *part = mmap(NULL, STREAM_PART_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, rf, (off_t)STREAM_REGION_OFFSET);
    if (part == MAP_FAILED) {
        perror("apl_feed: mmap");
        return 1;
    }
    struct stream_ctl *c = (struct stream_ctl *)part;

    /* the session opens with magic last, behind a store fence */
    memset(part, 0, STREAM_PAGE);
    const uint64_t total = (uint64_t)fs.st_size;
    const uint64_t slots = (total + STREAM_SLOT_BYTES - 1) / STREAM_SLOT_BYTES;
    const uint64_t session = (uint64_t)time(NULL);
    c->total = total;
    c->session = session;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    c->magic = STREAM_MAGIC;

    const double t_open = now_s();
    while (c->ack != session) {
        if (now_s() - t_open > STREAM_WAIT_S) {
            fprintf(stderr, "apl_feed: no card after %d s\n", STREAM_WAIT_S);
            c->magic = 0;
            return 1;
        }
        nap_50us();
    }

    const double t0 = now_s();
    double read_s = 0;
    for (uint64_t seq = 0; seq < slots; seq++) {
        if (seq >= STREAM_SLOTS && wait_consumed(c, seq - STREAM_SLOTS + 1)) {
            c->magic = 0;
            return 1;
        }
        const uint64_t off = seq * STREAM_SLOT_BYTES;
        const uint64_t left = total - off;
        const size_t want = (size_t)(left < STREAM_SLOT_BYTES ? left : STREAM_SLOT_BYTES);
        /* O_DIRECT takes whole sectors: ask for the full slot; the file's end
         * makes the last read short */
        const double r0 = now_s();
        const ssize_t got = pread(fd, part + STREAM_FIRST_SLOT + (seq % STREAM_SLOTS) * STREAM_SLOT_BYTES,
                                  STREAM_SLOT_BYTES, (off_t)off);
        read_s += now_s() - r0;
        if (got != (ssize_t)want) {
            fprintf(stderr, "apl_feed: slot %llu: read %zd of %zu: %s\n", (unsigned long long)seq, got, want,
                    strerror(errno));
            c->magic = 0;
            return 1;
        }
        __atomic_thread_fence(__ATOMIC_RELEASE);
        c->posted = seq + 1;
    }
    const int bad = wait_consumed(c, slots);
    const double wall = now_s() - t0;
    c->magic = 0;
    if (bad)
        return 1;
    printf("feed: %llu B in %llu slots, %.3f s wall = %.2f GB/s; file reads %.3f s = %.2f GB/s\n",
           (unsigned long long)total, (unsigned long long)slots, wall, (double)total / wall / 1e9, read_s,
           (double)total / read_s / 1e9);
    return 0;
}
