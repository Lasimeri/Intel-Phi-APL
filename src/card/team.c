/* team.c: see team.h. A round runs down and up a tree, fan-out 4, with one cache line per
 * thread for each direction: a parent writes each child's `go` line, a child spins only on
 * its own, and on the way back a parent spins on each child's `done` line. Nothing is
 * contended, so a round costs the tree's depth in line transfers, not one transfer per
 * thread on a shared counter. The spin uses DELAY, so a waiting thread gives its issue
 * slots to the core's other threads. */
#define _GNU_SOURCE
#include "team.h"

#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "knc.h"

#define LINE __attribute__((aligned(64)))
#define SPIN_CYCLES 16
#define FANOUT 4
#define FUTEX_WAIT_OP 0
#define FUTEX_WAKE_OP 1

struct worker {
    unsigned go LINE;   /* written by the parent: the round to run */
    unsigned done LINE; /* written by this thread: its subtree has finished the round */
    int awake LINE;     /* team_resize's: 0 = sleep on this word until it is not */
    struct team *t LINE;
    int tid, cpu;
    pthread_t thread;
};

struct team {
    int n, active, pin;
    struct worker *w;
    team_fn fn;
    void *arg;
    unsigned round;
    int count LINE; /* team_barrier's arrivals */
    int sense LINE; /* team_barrier's release */
};

static void pin_to(int cpu)
{
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET((size_t)cpu, &s);
    sched_setaffinity(0, sizeof s, &s);
}

/* Node i's children: 4i+1 .. 4i+4, those below the active count. */
static inline void wake_children(struct team *t, int i, unsigned round)
{
    for (int c = FANOUT * i + 1; c <= FANOUT * i + FANOUT && c < t->active; c++)
        __atomic_store_n(&t->w[c].go, round, __ATOMIC_RELEASE);
}

static inline void wait_children(struct team *t, int i, unsigned round)
{
    for (int c = FANOUT * i + 1; c <= FANOUT * i + FANOUT && c < t->active; c++)
        while (__atomic_load_n(&t->w[c].done, __ATOMIC_ACQUIRE) != round)
            knc_delay(SPIN_CYCLES);
}

static void *worker_main(void *p)
{
    struct worker *w = p;
    struct team *t = w->t;
    unsigned seen = 0;
    if (t->pin)
        pin_to(w->cpu);
    for (;;) {
        unsigned g;
        while ((g = __atomic_load_n(&w->go, __ATOMIC_ACQUIRE)) == seen) {
            if (__atomic_load_n(&w->awake, __ATOMIC_ACQUIRE))
                knc_delay(SPIN_CYCLES);
            else
                syscall(SYS_futex, &w->awake, FUTEX_WAIT_OP, 0, NULL, NULL, 0);
        }
        seen = g;
        wake_children(t, w->tid, g);
        if (!t->fn)
            return NULL; /* team_destroy */
        t->fn(t->arg, w->tid, t->active);
        wait_children(t, w->tid, g);
        __atomic_store_n(&w->done, g, __ATOMIC_RELEASE);
    }
}

struct team *team_create(const int *cpus, int n, int pin)
{
    struct team *t;
    if (n < 1 || posix_memalign((void **)&t, 64, sizeof *t) != 0)
        return NULL;
    *t = (struct team){.n = n, .active = n, .pin = pin};
    if (posix_memalign((void **)&t->w, 64, sizeof *t->w * (size_t)n) != 0) {
        free(t);
        return NULL;
    }
    for (int i = 0; i < n; i++)
        t->w[i] = (struct worker){.awake = 1, .t = t, .tid = i, .cpu = cpus[i]};
    if (pin)
        pin_to(cpus[0]);
    for (int i = 1; i < n; i++)
        if (pthread_create(&t->w[i].thread, NULL, worker_main, &t->w[i]) != 0)
            abort();
    return t;
}

void team_open(struct team *t, team_fn fn, void *arg)
{
    const unsigned g = ++t->round;
    t->fn = fn;
    t->arg = arg;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    wake_children(t, 0, g);
}

void team_close(struct team *t)
{
    t->fn(t->arg, 0, t->active);
    wait_children(t, 0, t->round);
}

void team_run(struct team *t, team_fn fn, void *arg)
{
    team_open(t, fn, arg);
    team_close(t);
}

void team_resize(struct team *t, int n)
{
    for (int i = n; i < t->active; i++)
        __atomic_store_n(&t->w[i].awake, 0, __ATOMIC_RELEASE);
    for (int i = t->active; i < n; i++) {
        __atomic_store_n(&t->w[i].awake, 1, __ATOMIC_RELEASE);
        syscall(SYS_futex, &t->w[i].awake, FUTEX_WAKE_OP, 1, NULL, NULL, 0);
    }
    t->active = n;
}

void team_barrier(struct team *t)
{
    int s = __atomic_load_n(&t->sense, __ATOMIC_RELAXED);
    if (__atomic_add_fetch(&t->count, 1, __ATOMIC_ACQ_REL) == t->active) {
        __atomic_store_n(&t->count, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&t->sense, !s, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&t->sense, __ATOMIC_ACQUIRE) == s)
            knc_delay(SPIN_CYCLES);
    }
}

void team_destroy(struct team *t)
{
    team_resize(t, t->n);
    const unsigned g = ++t->round;
    t->fn = NULL;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    wake_children(t, 0, g);
    for (int i = 1; i < t->n; i++)
        pthread_join(t->w[i].thread, NULL);
    free(t->w);
    free(t);
}
