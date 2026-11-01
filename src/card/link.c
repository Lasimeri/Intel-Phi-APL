/* link.c: the card's side of the exchange with the host's worker (see link.h). */
#define _GNU_SOURCE
#include "link.h"

#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#include "cdma.h"
#include "knc.h"

#define PAGE_SHIFT 21
#define PAGE_MASK ((1l << PAGE_SHIFT) - 1)

/* The watchdog: a dead host must not leave the card's team spinning. A poll of host
 * memory is a PCIe read of about 0.35 us, plus the delay: about 60 s of them. */
#define WATCHDOG_POLLS 150000000L

static cdma dma;
static struct link_ctl *ctl;
static const uint64_t *frame_phys;
static uint64_t host_seen, cmd_seen, card_seq, last_copy;

static void watch(long *polls, const char *what)
{
    if (++*polls > WATCHDOG_POLLS) {
        fprintf(stderr, "link: no %s from the host: giving up\n", what);
        _exit(3);
    }
}

void link_spin(const volatile uint64_t *word, uint64_t value)
{
    while (*word < value)
        knc_delay(64u);
}

int link_open(const uint64_t *phys)
{
    frame_phys = phys;
    if (cdma_open(&dma, LINK_DMA_CHANNEL))
        return -1;
    int mem = open("/dev/phihost", O_RDWR);
    if (mem < 0) {
        perror("/dev/phihost");
        return -1;
    }
    void *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, mem, (off_t)LINK_CTL_OFFSET);
    close(mem);
    if (page == MAP_FAILED) {
        perror("link: mmap /dev/phihost");
        return -1;
    }
    ctl = page;
    /* the host zeroed the counters before waiting for this */
    host_seen = cmd_seen = card_seq = 0;
    ctl->card_ready = 1;
    return 0;
}

void link_next(uint64_t *op, uint64_t *tok, uint64_t *p)
{
    long polls = 0;
    cmd_seen++;
    while (ctl->cmd_seq < cmd_seen) {
        knc_delay(16u);
        watch(&polls, "command");
    }
    *op = ctl->op;
    *tok = ctl->tok;
    *p = ctl->p;
}

void link_wait_host(void)
{
    long polls = 0;
    host_seen++;
    while (ctl->host_seq < host_seen) {
        knc_delay(16u);
        watch(&polls, "phase");
    }
}

/* A frame range against a region range, in pieces the engine can copy: within one huge
 * page, within its limit. Offsets: off in the frame, region_off in the region. */
static void copy_range(long off, long region_off, long bytes, int to_card)
{
    while (bytes > 0) {
        long n = (1l << PAGE_SHIFT) - (off & PAGE_MASK);
        if (n > bytes)
            n = bytes;
        if (n > LINK_MAX_COPY)
            n = LINK_MAX_COPY;
        const uint64_t card = frame_phys[off >> PAGE_SHIFT] + (uint64_t)(off & PAGE_MASK);
        const uint64_t host = dma.region_addr + (uint64_t)region_off;
        last_copy = cdma_submit(&dma, to_card ? host : card, to_card ? card : host, (uint32_t)n, 0, 0);
        off += n;
        region_off += n;
        bytes -= n;
    }
}

void link_pull(long off, long bytes)
{
    copy_range(off, (long)LINK_FRAME_OFFSET + off, bytes, 1);
}

void link_finish(void)
{
    link_spin(dma.status, last_copy);
}

void link_push(long off, long bytes)
{
    copy_range(off, (long)LINK_FRAME_OFFSET + off, bytes, 0);
}

void link_card_done(void)
{
    /* the status follows every copy on this channel, so the host sees it after them */
    last_copy = cdma_submit(&dma, frame_phys[0], dma.region_addr + LINK_SCRATCH_OFFSET, 64,
                            dma.region_addr + LINK_CTL_OFFSET + offsetof(struct link_ctl, card_seq), ++card_seq);
}

void link_close(void)
{
    link_spin(dma.status, last_copy);
    cdma_close(&dma);
}
