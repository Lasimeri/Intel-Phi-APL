/* stream.h: the part of the shared host-memory region that carries a model file
 * from the host's NVMe into card RAM. The host reads the file with O_DIRECT into
 * the part's slots, so it never passes through the host's page cache; the card
 * pulls each slot with O_DIRECT from /dev/phiblk1 into its arena, then says the
 * slot is free. Bulk data takes the block path (3.2 GB/s, the stack's record at
 * 16 MiB requests); only the counters cross /dev/phihost, an uncached mapping
 * where every access is a PCIe transaction.
 *
 * Region offsets: the AVX-512 wrapper's part ends at 768 MiB and PWGD's starts at
 * 1024 MiB, so this part overlaps neither, nor vgd's [1792, 2048) MiB. Swap sits
 * at 2 GiB when HOSTMEM >= 4G; the card refuses this part while swap is on the
 * whole of /dev/phiblk1 (HOSTMEM below 4G).
 *
 * A session:
 *   host   zeroes the page; writes total, session; magic last
 *   card   sees magic, reads total and session, echoes the session in ack
 *   host   for each slot: waits for room, reads, raises posted
 *   card   for each slot: waits for posted, pulls, raises consumed
 *   host   sees consumed reach the last slot, clears magic */
#ifndef APL_STREAM_H
#define APL_STREAM_H
#include <stdint.h>

#define STREAM_REGION_OFFSET (768ull << 20)
#define STREAM_PART_BYTES (256ull << 20)
#define STREAM_SLOT_BYTES (16ull << 20)
#define STREAM_FIRST_SLOT (16ull << 20) /* from the part's start */
#define STREAM_SLOTS 15
#define STREAM_MAGIC 0x314d5254534c5041ull /* "APLSTRM1" */
#define STREAM_PAGE 4096
#define STREAM_WAIT_S 120

/* One cache line per writer. Offsets are the contract, on both sides. */
struct stream_ctl {
    volatile uint64_t magic __attribute__((aligned(64))); /* host: written last */
    volatile uint64_t session;                            /* host: nonzero */
    volatile uint64_t total;                              /* host: the file's bytes */
    volatile uint64_t posted __attribute__((aligned(64))); /* host: slots written */
    volatile uint64_t ack __attribute__((aligned(64)));    /* card: the session it took */
    volatile uint64_t consumed __attribute__((aligned(64))); /* card: slots pulled */
};
_Static_assert(sizeof(struct stream_ctl) <= STREAM_PAGE, "the exchange page holds the counters");

/* Slot seq's offset in the region. */
static inline uint64_t stream_slot_offset(uint64_t seq)
{
    return STREAM_REGION_OFFSET + STREAM_FIRST_SLOT + (seq % STREAM_SLOTS) * STREAM_SLOT_BYTES;
}

/* ---- the card's side (src/card/stream.c) */
struct stream {
    struct stream_ctl *ctl;
    int blk;
    uint64_t total, session;
};

/* Attaches to the host's feed, waiting up to STREAM_WAIT_S. Returns 0, or -1
 * with the reason on stderr. s->total then says how large dest must be, rounded
 * up to STREAM_PAGE. */
int stream_open(struct stream *s);

/* Pulls the whole file into dest, which must be page-aligned, preferably on huge
 * pages (one physical segment per request). Returns 0 or -1. */
int stream_receive(struct stream *s, void *dest);

void stream_close(struct stream *s);

#endif
