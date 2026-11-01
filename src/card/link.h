/* link.h: the step-by-step exchange between the host's worker and the card, in the
 * same part of the host-memory region as the model stream (stream.h), which it does not
 * overlap: the stream's exchange page is the part's first page and its slots start at
 * 16 MiB. The frames are mirrored: the host's lies in the region, the card's in card
 * memory, at the same offsets, so a crossing is "copy this range of the frame". Bulk
 * moves by the card's own DMA channel; the counters are words in host memory, read by
 * the card through /dev/phihost, written by it with the DMA's own status descriptor, so
 * a host reader sees a counter only after the data before it.
 *
 * Measured 2026-11-01 (bench/cross_*.c), host to card and back: 64 B 5.03 us, 8 KiB
 * 9.43 us, 24 KiB 18.44 us median; the block path's floor is 89 us. */
#ifndef APL_LINK_H
#define APL_LINK_H
#include <stdint.h>

#include "stream.h"

#define LINK_DMA_CHANNEL 3 /* 0, 1 phictl; 2, 5 Cortex; 4 vgd; 6 pwgd; 7 AVX-512 wrapper */
#define LINK_CTL_OFFSET (STREAM_REGION_OFFSET + 4096)
#define LINK_SCRATCH_OFFSET (LINK_CTL_OFFSET + 2048) /* a line for a status-only exchange */
#define LINK_BUF_OFFSET (STREAM_REGION_OFFSET + (1ull << 20))
#define LINK_BUF_BYTES (15ull << 20)
#define LINK_BUF2_OFFSET (LINK_BUF_OFFSET + (LINK_BUF_BYTES / 2)) /* the crossing benchmark's way back */
#define LINK_FRAME_OFFSET LINK_BUF_OFFSET /* the host's frame */
#define LINK_MAX_COPY 1048512             /* the DMA engine's limit: 16383 lines */

/* The KV cache, in system memory: the stream's slots, idle once the model has landed. */
#define LINK_KV_OFFSET (STREAM_REGION_OFFSET + STREAM_FIRST_SLOT)
#define LINK_KV_BYTES (STREAM_SLOTS * STREAM_SLOT_BYTES)

/* The bytes that cross for n lanes: whole lines. */
#define LINK_BYTES(n) ((((long)(n)) * 4 + 63) & ~63L)

enum { LINK_END, LINK_STEP, LINK_RESET, LINK_EMBED, LINK_BATCH, LINK_MTP, LINK_LOGITS, LINK_DRAFT_LOGITS, LINK_MARK };

/* One cache line per writer. */
struct link_ctl {
    volatile uint64_t host_seq __attribute__((aligned(64))); /* host: its phases finished */
    volatile uint64_t bytes;                                 /* host: the benchmark's request size */
    volatile uint64_t card_seq __attribute__((aligned(64))); /* card, by DMA status: its phases finished */
    volatile uint64_t card_ready __attribute__((aligned(64))); /* card: attached; the host cleared it first */
    volatile uint64_t cmd_seq __attribute__((aligned(64)));    /* host: commands posted, written last */
    volatile uint64_t op, tok, p;                              /* host: the command */
};
_Static_assert(sizeof(struct link_ctl) <= 2048, "the exchange page holds the counters and the scratch line");

/* A spin on a word in card memory: nothing to gain by napping for a step this short. */
void link_spin(const volatile uint64_t *word, uint64_t value);

/* ---- the card's side (link.c) */
int link_open(const uint64_t *phys); /* the frame's huge pages */
void link_next(uint64_t *op, uint64_t *tok, uint64_t *p); /* waits for the host's next command */
void link_wait_host(void);                                /* the host's phase is finished */
void link_pull(long off, long bytes);                     /* frame range, host to card */
void link_finish(void);                                   /* every pull has landed */
void link_push(long off, long bytes);                     /* frame range, card to host */
void link_card_done(void);                                /* the card's phase is finished, after its pushes */
void link_close(void);

/* ---- the host's side (src/host/link_host.c) */
void link_host_open(struct link_ctl *ctl);
void link_host_wait_card_attach(void);
void link_command(uint64_t op, uint64_t tok, uint64_t p);
void link_host_done(void); /* this phase is finished */
void link_wait_card(void); /* the card's phase is finished */
void link_host_stats(double *wait_s, long *count); /* since the last call, then cleared */

/* The split form's profile build (rt.h): the card's waits and copies, in generated code. */
#if defined(RT_PROFILE) && defined(APL_RT_H)
#define link_wait_host() RT_TIMED(P_LINK_WAIT, (link_wait_host)())
#define link_pull(o, b) RT_TIMED(P_LINK_COPY, (link_pull)(o, b))
#define link_finish() RT_TIMED(P_LINK_COPY, (link_finish)())
#define link_push(o, b) RT_TIMED(P_LINK_COPY, (link_push)(o, b))
#define link_card_done() RT_TIMED(P_LINK_DONE, (link_card_done)())
#endif

#endif
