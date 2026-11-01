/* team.h: a team of pinned threads that run one function together, for the
 * card's kernels (and, through knc.h, for their host model). Workers spin
 * between jobs: a token's many short parallel steps cannot afford to sleep.
 * Only pointers and integers cross this interface. */
#ifndef APL_TEAM_H
#define APL_TEAM_H

struct team;

/* Runs on every active thread: tid in 0..n-1, n the active count. */
typedef void (*team_fn)(void *arg, int tid, int n);

/* A team of n threads, pinned to cpus[0..n-1] when pin is nonzero. The
 * calling thread is tid 0 and is pinned too. All active. NULL on failure. */
struct team *team_create(const int *cpus, int n, int pin);
void team_destroy(struct team *t);

/* Runs fn on all active threads; returns when every one has finished it. */
void team_run(struct team *t, team_fn fn, void *arg);

/* The same in two halves, with the caller free in between: open starts every other thread on fn;
 * close runs fn on the caller and waits for them. Nothing else may use the team in between. */
void team_open(struct team *t, team_fn fn, void *arg);
void team_close(struct team *t);

/* Makes tids 0..n-1 the active ones, between rounds, from the calling thread. The rest sleep in
 * the kernel: a spinning thread costs its core even when it does no work, 2 % of the 9B's decode
 * (2026-11-01). A wake costs a system call per thread, so this is for a change of phase. */
void team_resize(struct team *t, int n);

/* Inside fn: waits until all n threads have arrived. */
void team_barrier(struct team *t);

#endif
