/* link_host.c: the host's side of the exchange with the card (src/card/link.h). */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "link.h"

#define WATCHDOG_S 60.0 /* a dead card must not leave the host's worker spinning */

static struct link_ctl *ctl;
static uint64_t host_seq, card_seen, cmd_seq;
static double waited_s;
static long waits;

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

void link_host_stats(double *wait_s, long *count)
{
    *wait_s = waited_s;
    *count = waits;
    waited_s = 0;
    waits = 0;
}

static void wait_word(const volatile uint64_t *word, uint64_t value, const char *what)
{
    if (*word >= value)
        return;
    const double t0 = now_s();
    waits++;
    for (unsigned long i = 1; *word < value; i++) {
        _mm_pause();
        if (!(i & 0xFFFFF) && now_s() - t0 > WATCHDOG_S) {
            fprintf(stderr, "link: no %s from the card after %.0f s\n", what, WATCHDOG_S);
            exit(3);
        }
    }
    waited_s += now_s() - t0;
}

void link_host_open(struct link_ctl *c)
{
    ctl = c;
    host_seq = card_seen = cmd_seq = 0;
    ctl->host_seq = 0;
    ctl->card_seq = 0;
    ctl->cmd_seq = 0;
    ctl->card_ready = 0;
}

void link_host_wait_card_attach(void)
{
    /* the card first loads the whole model: the 9B takes about 5 s */
    wait_word(&ctl->card_ready, 1, "attach");
}

void link_command(uint64_t op, uint64_t tok, uint64_t p)
{
    ctl->op = op;
    ctl->tok = tok;
    ctl->p = p;
    _mm_sfence();
    ctl->cmd_seq = ++cmd_seq;
}

void link_host_done(void)
{
    _mm_sfence();
    ctl->host_seq = ++host_seq;
}

void link_wait_card(void)
{
    wait_word(&ctl->card_seq, ++card_seen, "phase");
}
