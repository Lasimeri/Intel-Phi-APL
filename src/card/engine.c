/* engine.c: the driver of a compiled model: brings the file in, repacks the weights on
 * the team, binds them to the compiled code's slots, and steps tokens through it,
 * timing the steps. No math of the model here: that is src/apl/qwen35.apl, compiled
 * into out/gen/MODEL/model.c. Built by knc-cc for the card, by the host cc as the test
 * model; only pointers and integers cross to the compiled code.
 *
 *   engine SOURCE TOKENS N_GEN TPC CORES REPEATS [DUMP_DIR]
 *
 * SOURCE: "stream" (from apl_feed) or the model's path (the host model). TOKENS: a
 * file with a line "prompt ID ID ...". The prompt goes through in batches, layer by
 * layer (at most APL_PROMPT_NB tokens each, test only), then N_GEN tokens are generated
 * greedily, one at a time. The first repeat is the warm-up. Prints the prompt and decode
 * rates, and the tokens generated. DUMP_DIR receives, for the first repeat's prompt:
 * x-LL.f32, each layer's output [T D], and logits.f32. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "arena.h"
#include "bench_util.h"
#include "cdma.h"
#include "gemv.h"
#include "link.h"
#include "rt.h"
#include "stream.h"
#include "team.h"

#define MAX_TOKENS 4096
#define MAX_REPEATS 32

#if defined(RT_PROFILE)
/* engine.c includes rt.h after string.h, so the timing macros would catch its own
 * memcpy: it is undefined below. The ticks are the core clock. */
#undef memcpy

uint64_t rt_prof_ticks[P_N], rt_prof_calls[P_N];

static void prof_report(const char *what, long tokens, double wall_s)
{
    static const char *const name[P_N] = {
        "gemv >=100000 rows", "gemv >=512 rows", "gemv <512 rows", "gemm", "elementwise, team", "elementwise, one thread",
        "+/", "max/", "index-of", "+⌿", "inner 2 1", "inner 1 2", "outer 1 1", "scale 0 n", "embedding row", "catenate",
        "index", "transpose", "memcpy (generated)", "waiting for the host", "DMA copies, submit + wait", "phase done, submit"};
    double total = 0;
    for (int i = 0; i < P_N; i++)
        total += (double)rt_prof_ticks[i];
    printf("profile of %s, %ld tokens, %.3f ms per token of wall time:\n", what, tokens, 1e3 * wall_s / (double)tokens);
    for (int i = 0; i < P_N; i++)
        if (rt_prof_calls[i])
            printf("  %-26s %8.1f calls/token %9.3f ms/token %5.1f %%\n", name[i], (double)rt_prof_calls[i] / (double)tokens,
                   (double)rt_prof_ticks[i] / 1238094.0 / (double)tokens,
                   100.0 * (double)rt_prof_ticks[i] / 1238094.0 / 1e3 / wall_s);
    printf("  %-26s %8s             %9.3f ms/token (at 1238.094 MHz)\n", "sum of the above", "", total / 1238094.0 / (double)tokens);
}
#define PROF_REPORT(what, tokens, wall) prof_report(what, tokens, wall)
#define PROF_START(tokens, t0)                                                                                         \
    do {                                                                                                               \
        memset(rt_prof_ticks, 0, sizeof rt_prof_ticks);                                                                \
        memset(rt_prof_calls, 0, sizeof rt_prof_calls);                                                                \
        tokens = 0;                                                                                                    \
        t0 = now_s();                                                                                                  \
    } while (0)
#else
#define PROF_REPORT(what, tokens, wall) ((void)0)
#define PROF_START(tokens, t0) ((void)0)
#endif

/* ---- one step. x and the entries' results live in separate places: an entry's frame
 * is the next entry's scratch. */
static float **slots_of(float **slots, int layer)
{
    return slots + (size_t)layer * (size_t)MODEL.n_slots;
}

/* The phases' shares of the team: a prompt's products are compute-bound and take every thread;
 * a decode step's are memory-bound, and the rest of the team sleeps through it. */
static int step_threads, batch_threads;

/* The next token from the last layer's output x, keeping the logits for a dump */
static long choose(struct rt *R, float *const *top, const float *x, FILE **dump)
{
    const struct batch_bind *const one = &MODEL.batches[MODEL.n_batches - 1];
    const float *y = one->logits(R, top, one->hidden(R, top, x));
    if (dump && dump[MODEL.n_layer]) {
        rewind(dump[MODEL.n_layer]);
        if (fwrite(y, sizeof(float), (size_t)MODEL.vocab, dump[MODEL.n_layer]) != (size_t)MODEL.vocab)
            exit(1);
    }
    return MODEL.next(R, top, y);
}

/* nb tokens from position p through every layer, layer by layer; x [nb d] ends holding them */
static void batch(struct rt *R, float **slots, const struct batch_bind *b, float *x, long p, FILE **dump)
{
    const size_t bytes = sizeof(float) * (size_t)MODEL.d * (size_t)b->nb;
    for (int il = 0; il < MODEL.n_layer; il++) {
        memcpy(x, (MODEL.full[il] ? b->attn : b->delta)(R, slots_of(slots, il), p, x), bytes);
        if (dump && fwrite(x, bytes, 1, dump[il]) != 1)
            exit(1);
    }
}

/* A prompt, in batches of at most limit: only its last token's logits are ever read */
static long prompt_run(struct rt *R, float **slots, float *x, const long *tok, long np, long limit, FILE **dump)
{
    float *const *top = slots_of(slots, MODEL.n_layer);
    long nb = 1;
    for (long p = 0; p < np; p += nb) {
        const struct batch_bind *b = &MODEL.batches[batch_for(np - p, limit)];
        nb = b->nb;
        for (long t = 0; t < nb; t++)
            memcpy(x + t * MODEL.d, MODEL.embed(R, top, tok[p + t]), sizeof(float) * (size_t)MODEL.d);
        team_resize(R->team, batch_threads);
        batch(R, slots, b, x, p, dump);
    }
    team_resize(R->team, step_threads);
    return choose(R, top, x + (nb - 1) * MODEL.d, dump);
}

static long step(struct rt *R, float **slots, float *x, long tok, long p)
{
    float *const *top = slots_of(slots, MODEL.n_layer);
    memcpy(x, MODEL.embed(R, top, tok), sizeof(float) * (size_t)MODEL.d);
    batch(R, slots, &MODEL.batches[MODEL.n_batches - 1], x, p, NULL);
    return choose(R, top, x, NULL);
}

/* ---- the split form: the host's worker drives; each command runs the card's half */
static int follow(struct rt *R, float **slots, char *state, long state_bytes)
{
    long tokens = 0;
    double t0 = now_s();
    float *const *top = slots_of(slots, MODEL.n_layer);
    const struct batch_bind *const one = &MODEL.batches[MODEL.n_batches - 1];
    for (;;) {
        uint64_t op, tok, p;
        link_next(&op, &tok, &p);
        if (op == LINK_END)
            break;
        switch (op) {
        case LINK_RESET:
            if (state_bytes)
                memset(state, 0, (size_t)state_bytes);
            link_card_done();
            PROF_START(tokens, t0);
            break;
        case LINK_EMBED:
            MODEL.embed(R, top, (long)tok);
            break;
        case LINK_BATCH: {
            const struct batch_bind *b = &MODEL.batches[tok];
            team_resize(R->team, b->nb >= 4 ? batch_threads : step_threads); /* below 4 tokens, memory-bound */
            tokens += b->nb;
            for (int il = 0; il < MODEL.n_layer; il++)
                (MODEL.full[il] ? b->attn : b->delta)(R, slots_of(slots, il), (long)p, NULL);
            break;
        }
        case LINK_MTP:
            MODEL.batches[tok].mtp(R, slots_of(slots, MODEL.n_layer + 1), (long)p, NULL);
            break;
        case LINK_LOGITS:
            MODEL.batches[tok].logits(R, top, NULL);
            break;
        case LINK_DRAFT_LOGITS:
            MODEL.draft_logits(R, top, NULL);
            break;
        case LINK_MARK: /* the prompt has ended */
            team_resize(R->team, step_threads);
            link_card_done(); /* acknowledged: the mailbox holds one command */
            PROF_REPORT("the prompt", tokens, now_s() - t0);
            PROF_START(tokens, t0);
            break;
        default: /* LINK_STEP */
            tokens++;
            MODEL.embed(R, top, (long)tok);
            for (int il = 0; il < MODEL.n_layer; il++)
                (MODEL.full[il] ? one->attn : one->delta)(R, slots_of(slots, il), (long)p, NULL);
            one->logits(R, top, NULL);
            break;
        }
    }
    PROF_REPORT("the last decode", tokens, now_s() - t0);
    (void)tokens;
    (void)t0;
    link_close();
    team_destroy(R->team);
    return 0;
}

static long read_prompt(const char *path, long *tok)
{
    static char line[1 << 20];
    FILE *f = fopen(path, "r");
    if (!f) {
        perror(path);
        exit(1);
    }
    long n = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "prompt ", 7) != 0)
            continue;
        for (char *s = line + 7, *e; n < MAX_TOKENS; s = e) {
            const long v = strtol(s, &e, 10);
            if (e == s)
                break;
            tok[n++] = v;
        }
        break;
    }
    fclose(f);
    return n;
}

int main(int argc, char **argv)
{
    if (argc < 7 || argc > 8) {
        fprintf(stderr, "usage: engine SOURCE TOKENS N_GEN TPC CORES REPEATS [DUMP_DIR]\n");
        return 2;
    }
    const char *source = argv[1], *dump_dir = argc == 8 ? argv[7] : NULL;
    const long n_gen = atol(argv[3]);
    const int tpc = atoi(argv[4]), cores = atoi(argv[5]);
    int repeats = atoi(argv[6]);
    if (repeats > MAX_REPEATS)
        repeats = MAX_REPEATS;
    static long prompt[MAX_TOKENS], gen[MAX_REPEATS + 1][MAX_TOKENS];
    const long np = read_prompt(argv[2], prompt);
    if (np <= 0 || np + n_gen > MODEL.ctx || n_gen > MAX_TOKENS || tpc * cores > 256) {
        fprintf(stderr, "engine: prompt %ld + N_GEN %ld against CTX %ld; at most 256 threads\n", np, n_gen, MODEL.ctx);
        return 2;
    }
    /* APL_PROMPT_TPC: threads per core for a prompt; the extra ones come after the rest in tid
     * order, so a decode step's active threads are exactly tpc per core */
    const int prompt_tpc = getenv("APL_PROMPT_TPC") ? atoi(getenv("APL_PROMPT_TPC")) : (tpc > 3 ? tpc : 3);
    const int threads = (prompt_tpc > tpc ? prompt_tpc : tpc) * cores;
    int cpus[256];
    const int *const step_cpus = cpu_list(tpc, cores);
    for (int i = 0; i < cores * tpc; i++)
        cpus[i] = step_cpus[i];
    for (int e = tpc; e < threads / cores; e++)
        for (int i = 0; i < cores; i++)
            cpus[cores * e + i] = 4 * i + 1 + e;
    step_threads = tpc * cores;
    batch_threads = threads;
    struct rt R = {.team = team_create(cpus, threads, 1)};
    if (!R.team || rt_init(&R, threads))
        return 1;
    team_resize(R.team, step_threads);

    /* ---- the file */
    const int streaming = strcmp(source, "stream") == 0;
    struct stream s = {.blk = -1};
    struct stat st;
    uint64_t total;
    if (streaming) {
        if (stream_open(&s))
            return 1;
        total = s.total;
    } else {
        if (stat(source, &st)) {
            perror(source);
            return 1;
        }
        total = (uint64_t)st.st_size;
    }
    const size_t arena_bytes = (size_t)((total + STREAM_PAGE - 1) & ~(uint64_t)(STREAM_PAGE - 1));
    char *file = big_alloc(arena_bytes, "arena");
    struct touch_job tj = {.p = file, .bytes = arena_bytes};
    team_run(R.team, touch_thread, &tj);
    double t0 = now_s();
    if (streaming ? stream_receive(&s, file) : read_all(source, file, (size_t)total))
        return 1;
    double dt = now_s() - t0;
    printf("received %llu B in %.3f s = %.2f GB/s\n", (unsigned long long)total, dt, (double)total / dt / 1e9);
    if (streaming)
        stream_close(&s);
    struct sum_job sj = {.w = (const uint64_t *)file, .words = (long)(total / 8)};
    team_run(R.team, sum_thread, &sj);
    uint64_t sum = 0;
    for (int i = 0; i < step_threads; i++)
        sum += sj.part[i];
    printf("file sum %016llx\n", (unsigned long long)sum);

    /* ---- the weights: Q8_0 repacked in place on the team, each matrix once (a tied
     * output shares the embedding's); F32 copied to 64-byte alignment */
    long f32_bytes = 0, max_cols = 0, q8_bytes = 0;
    for (int i = 0; i < MODEL.n_tensors; i++) {
        const struct tensor_bind *b = &MODEL.tensors[i];
        if (b->kind != KIND_Q8)
            f32_bytes += (b->rows * b->cols * (long)sizeof(float) + 63) & ~63L;
        else if (b->cols > max_cols)
            max_cols = b->cols;
    }
    char *f32 = big_alloc((size_t)f32_bytes, "F32 weights");
    char *scratch = big_alloc((size_t)gemv_i_bytes(16, max_cols) * (size_t)step_threads, "repack scratch");
    float **slots = calloc((size_t)(MODEL.n_layer + 2) * (size_t)MODEL.n_slots, sizeof(float *));
    if (!slots)
        return 1;
    t0 = now_s();
    for (int i = 0; i < MODEL.n_tensors; i++) {
        const struct tensor_bind *b = &MODEL.tensors[i];
        const char *at = file + b->offset;
        if (b->kind == KIND_Q8) {
            int seen = 0;
            for (int k = 0; k < i; k++)
                seen |= MODEL.tensors[k].kind == KIND_Q8 && MODEL.tensors[k].offset == b->offset;
            if (!seen) {
                struct gemv_repack_job rj = {.w = (void *)at, .rows = b->rows, .cols = b->cols, .tmp = scratch};
                team_run(R.team, gemv_i_repack_thread, &rj);
                q8_bytes += gemv_i_bytes(b->rows, b->cols);
            }
            slots_of(slots, b->layer)[b->slot] = (float *)at;
            continue;
        }
        float *z = (float *)f32;
        f32 += (b->rows * b->cols * (long)sizeof(float) + 63) & ~63L;
        if (b->kind == KIND_F32) {
            memcpy(z, at, sizeof(float) * (size_t)(b->rows * b->cols));
        } else {
            for (long c = 0; c < b->cols; c++)
                for (long r = 0; r < b->rows; r++)
                    memcpy(z + c * b->rows + r, at + sizeof(float) * (size_t)(r * b->cols + c), sizeof(float));
        }
        slots_of(slots, b->layer)[b->slot] = z;
    }
    dt = now_s() - t0;
    printf("weights ready in %.3f s: %ld B of Q8_0 repacked in place = %.2f GB/s\n", dt, q8_bytes, (double)q8_bytes / dt / 1e9);

    /* ---- state and frame */
    long state_bytes = 0;
    for (int i = 0; i < MODEL.n_states; i++)
        if (!MODEL.split || MODEL.states[i].side == 0)
            state_bytes += (MODEL.states[i].elems * (long)sizeof(float) + 63) & ~63L;
    /* a split model may keep no state on the card at all */
    char *state = state_bytes ? big_alloc((size_t)state_bytes, "state") : NULL;
    char *sp = state;
    for (int i = 0; i < MODEL.n_states; i++) {
        const struct state_bind *b = &MODEL.states[i];
        if (MODEL.split && b->side != 0) {
            /* in system memory: the card knows it by its region offset (generated code) */
            if (b->region_off >= 0)
                slots_of(slots, b->layer)[b->slot] = (float *)(uintptr_t)b->region_off;
            continue;
        }
        slots_of(slots, b->layer)[b->slot] = (float *)sp;
        sp += (b->elems * (long)sizeof(float) + 63) & ~63L;
    }
    if (MODEL.split) {
        static uint64_t frame_phys[64];
        R.frame = huge_alloc((size_t)MODEL.frame_bytes, frame_phys, 64);
        if (!R.frame || link_open(frame_phys))
            return 1;
        printf("following the host's worker\n");
        fflush(stdout);
        return follow(&R, slots, state, state_bytes);
    }
    R.frame = big_alloc((size_t)MODEL.frame_bytes, "frame");
    struct touch_job fj = {.p = R.frame, .bytes = (size_t)MODEL.frame_bytes};
    team_run(R.team, touch_thread, &fj);
    float *x;
    if (posix_memalign((void **)&x, 64, sizeof(float) * (size_t)MODEL.d * (size_t)MODEL.batches[0].nb))
        return 1;
    const long limit = getenv("APL_PROMPT_NB") ? atol(getenv("APL_PROMPT_NB")) : MODEL.batches[0].nb;

    /* ---- the dumps: one file per layer, then the logits */
    static FILE *dump[256];
    if (dump_dir) {
        char path[512];
        mkdir(dump_dir, 0777);
        for (int il = 0; il < MODEL.n_layer; il++) {
            snprintf(path, sizeof path, "%s/x-%02d.f32", dump_dir, il);
            dump[il] = fopen(path, "wb");
        }
        snprintf(path, sizeof path, "%s/logits.f32", dump_dir);
        dump[MODEL.n_layer] = fopen(path, "wb");
    }

    /* ---- the runs. A step of the decode: one token in, the next one out. */
    double prompt_s[MAX_REPEATS + 1], decode_s[MAX_REPEATS + 1];
    for (int rep = 0; rep <= repeats; rep++) {
        memset(state, 0, (size_t)state_bytes);
        t0 = now_s();
        long tok = prompt_run(&R, slots, x, prompt, np, limit, rep == 0 && dump_dir ? dump : NULL);
        prompt_s[rep] = now_s() - t0;
        gen[rep][0] = tok;
#if defined(RT_PROFILE)
        memset(rt_prof_ticks, 0, sizeof rt_prof_ticks);
        memset(rt_prof_calls, 0, sizeof rt_prof_calls);
#endif
        t0 = now_s();
        for (long i = 1; i < n_gen; i++) {
            tok = step(&R, slots, x, tok, np + i - 1);
            gen[rep][i] = tok;
        }
        decode_s[rep] = now_s() - t0;
        printf("run %d%s: prompt %ld tokens %.3f s = %.3f tok/s; decode %ld tokens %.3f s = %.3f tok/s\n", rep,
               rep ? "" : " (warm-up)", np, prompt_s[rep], (double)np / prompt_s[rep], n_gen - 1, decode_s[rep],
               (double)(n_gen - 1) / decode_s[rep]);
        fflush(stdout);
    }
#if defined(RT_PROFILE)
    prof_report("the last decode", n_gen - 1, decode_s[repeats]);
#endif
    if (dump_dir)
        for (int il = 0; il <= MODEL.n_layer; il++)
            fclose(dump[il]);

    /* ---- the verdict */
    int same = 1;
    for (int rep = 1; rep <= repeats; rep++)
        same &= memcmp(gen[rep], gen[0], sizeof(long) * (size_t)n_gen) == 0;
    printf("gen");
    for (long i = 0; i < n_gen; i++)
        printf(" %ld", gen[0][i]);
    printf("\nall repeats generated the same tokens: %s\n", same ? "yes" : "NO");
    if (repeats > 0) {
        qsort(prompt_s + 1, (size_t)repeats, sizeof(double), cmp_d);
        qsort(decode_s + 1, (size_t)repeats, sizeof(double), cmp_d);
        const double med = decode_s[1 + repeats / 2];
        const double tps = (double)(n_gen - 1) / med;
        printf("DECODE %.3f tok/s (median of %d; min %.3f max %.3f); PROMPT %.3f tok/s; tpc %d cores %d threads %d; "
               "Q8_0 %.2f GB per token = %.1f GB/s (arithmetic, not all of a token's time)\n",
               tps, repeats, (double)(n_gen - 1) / decode_s[repeats], (double)(n_gen - 1) / decode_s[1],
               (double)np / prompt_s[1 + repeats / 2], tpc, cores, threads, (double)q8_bytes / 1e9, (double)q8_bytes * tps / 1e9);
    }
    team_destroy(R.team);
    return 0;
}
