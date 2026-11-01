/* stream.c: the card's side of the model stream (see stream.h). */
#define _GNU_SOURCE
#include "stream.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "knc.h"

#define POLL_CYCLES 2048u /* about 1.7 us at 1238 MHz, between PCIe reads */

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Swap on the whole of /dev/phiblk1 would share these bytes (HOSTMEM below 4G). */
static int swap_on_whole_blk(void)
{
    char line[256];
    FILE *f = fopen("/proc/swaps", "r");
    int found = 0;
    if (!f)
        return 0;
    while (fgets(line, sizeof line, f))
        found |= strstr(line, "/dev/phiblk1") != NULL;
    fclose(f);
    return found;
}

int stream_open(struct stream *s)
{
    if (swap_on_whole_blk()) {
        fprintf(stderr, "stream: swap is on the whole of /dev/phiblk1: HOSTMEM must be 4G\n");
        return -1;
    }
    int mem = open("/dev/phihost", O_RDWR);
    if (mem < 0) {
        perror("stream: /dev/phihost");
        return -1;
    }
    void *page = mmap(NULL, STREAM_PAGE, PROT_READ | PROT_WRITE, MAP_SHARED, mem, (off_t)STREAM_REGION_OFFSET);
    close(mem);
    if (page == MAP_FAILED) {
        perror("stream: mmap /dev/phihost");
        return -1;
    }
    s->ctl = page;
    s->blk = open("/dev/phiblk1", O_RDONLY | O_DIRECT);
    if (s->blk < 0) {
        perror("stream: /dev/phiblk1");
        munmap(page, STREAM_PAGE);
        return -1;
    }
    const double t0 = now_s();
    while (s->ctl->magic != STREAM_MAGIC) {
        if (now_s() - t0 > STREAM_WAIT_S) {
            fprintf(stderr, "stream: no feed from the host after %d s\n", STREAM_WAIT_S);
            stream_close(s);
            return -1;
        }
        knc_delay(POLL_CYCLES);
    }
    s->total = s->ctl->total;
    s->session = s->ctl->session;
    s->ctl->ack = s->session;
    return 0;
}

int stream_receive(struct stream *s, void *dest)
{
    const uint64_t slots = (s->total + STREAM_SLOT_BYTES - 1) / STREAM_SLOT_BYTES;
    double last = now_s();
    for (uint64_t seq = 0; seq < slots; seq++) {
        while (s->ctl->posted <= seq) {
            if (now_s() - last > STREAM_WAIT_S) {
                fprintf(stderr, "stream: slot %llu not posted after %d s\n", (unsigned long long)seq, STREAM_WAIT_S);
                return -1;
            }
            knc_delay(POLL_CYCLES);
        }
        const uint64_t off = seq * STREAM_SLOT_BYTES;
        const uint64_t left = s->total - off;
        const uint64_t want = left < STREAM_SLOT_BYTES ? left : STREAM_SLOT_BYTES;
        /* whole pages: O_DIRECT. The arena is rounded up, so the tail fits. */
        const size_t bytes = (size_t)((want + STREAM_PAGE - 1) & ~(uint64_t)(STREAM_PAGE - 1));
        const ssize_t got = pread(s->blk, (char *)dest + off, bytes, (off_t)stream_slot_offset(seq));
        if (got != (ssize_t)bytes) {
            fprintf(stderr, "stream: slot %llu: pread %zd of %zu: %s\n", (unsigned long long)seq, got, bytes,
                    strerror(errno));
            return -1;
        }
        s->ctl->consumed = seq + 1;
        last = now_s();
    }
    return 0;
}

void stream_close(struct stream *s)
{
    if (s->blk >= 0)
        close(s->blk);
    if (s->ctl)
        munmap((void *)s->ctl, STREAM_PAGE);
    s->blk = -1;
    s->ctl = NULL;
}
