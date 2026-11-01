/* cross_card.c: the card's side of the crossing benchmark (the host's is cross_host.c).
 * For each request: see the host's counter through /dev/phihost, pull the bytes from
 * the region into card memory by DMA, push them back into the other half with a status
 * word after them. Prints the card's own view of each step: the pull alone and the
 * submit-to-done of the push, in microseconds of CLOCK_MONOTONIC.
 *
 *   cross_card                  (after cross_host has started) */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "bench_util.h"
#include "cdma.h"
#include "link.h"

#define MAX_ROUNDS 20000
#define WAIT_S 60.0

int main(void)
{
    cdma d;
    if (cdma_open(&d, LINK_DMA_CHANNEL))
        return 1;
    uint64_t phys;
    char *page = huge_alloc(2u << 20, &phys, 1);
    if (!page) {
        fprintf(stderr, "no free huge page\n");
        return 1;
    }
    int mem = open("/dev/phihost", O_RDWR);
    if (mem < 0) {
        perror("/dev/phihost");
        return 1;
    }
    struct link_ctl *c = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, mem, (off_t)LINK_CTL_OFFSET);
    if (c == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    close(mem);
    /* the host cleared card_ready and posts nothing until it is set: start from its counter */
    uint64_t seen = c->host_seq;
    c->card_ready = 1;
    static double pull_us[MAX_ROUNDS], push_us[MAX_ROUNDS];
    long rounds = 0;
    uint64_t last_bytes = 0;
    double last_post = now_s();
    for (;;) {
        while (c->host_seq == seen) {
            if (now_s() - last_post > WAIT_S) {
                fprintf(stderr, "cross_card: no request for %.0f s\n", WAIT_S);
                return 1;
            }
        }
        seen = c->host_seq;
        const uint64_t bytes = c->bytes;
        if (bytes != last_bytes && rounds) {
            qsort(pull_us, (size_t)rounds, sizeof(double), cmp_d);
            qsort(push_us, (size_t)rounds, sizeof(double), cmp_d);
            printf("card %8llu B: pull med %7.2f us, push med %7.2f us (%ld rounds)\n", (unsigned long long)last_bytes,
                   pull_us[rounds / 2], push_us[rounds / 2], rounds);
            fflush(stdout);
            rounds = 0;
        }
        last_bytes = bytes;
        if (!bytes) {
            /* the end: the host waits for this counter too */
            c->card_seq = seen;
            break;
        }
        const double t0 = now_s();
        uint64_t s = cdma_submit(&d, d.region_addr + LINK_BUF_OFFSET, phys, (uint32_t)bytes, 0, 0);
        link_spin(d.status, s);
        const double t1 = now_s();
        s = cdma_submit(&d, phys, d.region_addr + LINK_BUF2_OFFSET, (uint32_t)bytes,
                        d.region_addr + LINK_CTL_OFFSET + offsetof(struct link_ctl, card_seq), seen);
        link_spin(d.status, s);
        const double t2 = now_s();
        if (rounds < MAX_ROUNDS) {
            pull_us[rounds] = 1e6 * (t1 - t0);
            push_us[rounds] = 1e6 * (t2 - t1);
            rounds++;
        }
        last_post = t2;
    }
    cdma_close(&d);
    return 0;
}
