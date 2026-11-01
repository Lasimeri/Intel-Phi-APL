/* engine_host.c: the host's worker: the half of the compiled model the placement gave
 * the host, in lockstep with the card's half over the link. The host drives: it knows
 * the prompt, picks each next token, and commands the card's step. It holds the F32
 * weights (a few MiB, read from the file) and the KV cache, in the pinned region where
 * the card's DMA reaches it; it never holds the Q8_0 matrices, which stream from the
 * NVMe straight to the card (apl_feed).
 *
 *   engine_host MODEL TOKENS N_GEN THREADS REPEATS [DUMP_DIR]
 *
 * THREADS, 1 to 14: CPUs 1-7, then their siblings 9-15; core 0's two threads, CPUs 0 and
 * 8, stay with the scheduler. The first repeat is the warm-up. Prints the same lines as
 * the card-only engine, so runs compare directly. Start it before the card. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "link.h"
#include "rt.h"
#include "team.h"

#define MAX_TOKENS 4096
#define MAX_REPEATS 32
#define MAX_THREADS 14
#define MAX_DRAFT 3 /* a batch of at most 4 has logits (main.apl) */


static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static float **slots_of(float **slots, int layer)
{
    return slots + (size_t)layer * (size_t)MODEL.n_slots;
}

static void *zalloc64(size_t bytes)
{
    void *p;
    if (posix_memalign(&p, 64, (bytes + 63) & ~(size_t)63)) {
        perror("posix_memalign");
        exit(1);
    }
    memset(p, 0, (bytes + 63) & ~(size_t)63);
    return p;
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

/* The F32 weights, from the file; a transposed one is laid out as APL has it. */
static void bind_weights(float **slots, const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        exit(1);
    }
    for (int i = 0; i < MODEL.n_tensors; i++) {
        const struct tensor_bind *b = &MODEL.tensors[i];
        if (b->kind == KIND_Q8)
            continue;
        const size_t bytes = sizeof(float) * (size_t)(b->rows * b->cols);
        float *raw = zalloc64(bytes);
        if (pread(fd, raw, bytes, (off_t)b->offset) != (ssize_t)bytes) {
            perror("pread");
            exit(1);
        }
        if (b->kind == KIND_F32T) {
            float *z = zalloc64(bytes);
            for (long c = 0; c < b->cols; c++)
                for (long r = 0; r < b->rows; r++)
                    z[c * b->rows + r] = raw[r * b->cols + c];
            free(raw);
            raw = z;
        }
        slots_of(slots, b->layer)[b->slot] = raw;
    }
    close(fd);
}

/* ---- the working rows of a prompt or a decode */
struct buffers {
    float *e;       /* [nb d] the tokens' embeddings */
    float *h;       /* [nb d] their hidden states */
    float *m;       /* [nb 2d] the MTP block's input */
    float *pending; /* [d] the hidden state before the next position */
    float *draft;   /* [d] a draft's hidden state */
};

static long mtp_k, drafted, accepted, cycles;

static int batch_index(long nb)
{
    int i = 0;
    while (MODEL.batches[i].nb != nb)
        i++;
    return i;
}

/* A token's embedding into row t of e; the card knows it from the command */
static void embed_row(struct rt *R, float *const *top, long tok, float *e, long t)
{
    link_command(LINK_EMBED, (uint64_t)tok, 0);
    memcpy(e + t * MODEL.d, MODEL.embed(R, top, tok), sizeof(float) * (size_t)MODEL.d);
}

/* The next token from a hidden state, keeping the logits for a dump */
static long choose_hidden(struct rt *R, float *const *top, const float *h, FILE **dump)
{
    const struct batch_bind *const one = &MODEL.batches[MODEL.n_batches - 1];
    link_command(LINK_LOGITS, (uint64_t)(MODEL.n_batches - 1), 0);
    const float *y = one->logits(R, top, h);
    if (dump) {
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

/* The MTP block's own history, positions p .. p+nb-1: each token with the true hidden state of the
 * position before it, over whatever a draft left there. The next draft reads this. */
static void mtp_follow(struct rt *R, float **slots, const struct buffers *B, long nb, long p)
{
    const long d = MODEL.d;
    const size_t row = sizeof(float) * (size_t)d;
    for (long k = 0; k < nb; k++) {
        memcpy(B->m + k * 2 * d, B->e + k * d, row);
        memcpy(B->m + k * 2 * d + d, k ? B->h + (k - 1) * d : B->pending, row);
    }
    const int i = batch_index(nb);
    link_command(LINK_MTP, (uint64_t)i, (uint64_t)p);
    MODEL.batches[i].mtp(R, slots_of(slots, MODEL.n_layer + 1), p, B->m);
    memcpy(B->pending, B->h + (nb - 1) * d, row);
}

/* A prompt, in batches of at most limit: only its last token's logits are ever read */
static long prompt_run(struct rt *R, float **slots, float *x, const struct buffers *B, const long *tok, long np, long limit, FILE **dump)
{
    float *const *top = slots_of(slots, MODEL.n_layer);
    const long d = MODEL.d;
    memset(B->pending, 0, sizeof(float) * (size_t)d);
    long nb = 1;
    for (long p = 0; p < np; p += nb) {
        const int i = batch_for(np - p, limit);
        const struct batch_bind *b = &MODEL.batches[i];
        nb = b->nb;
        for (long t = 0; t < nb; t++)
            embed_row(R, top, tok[p + t], B->e, t);
        memcpy(x, B->e, sizeof(float) * (size_t)d * (size_t)nb);
        link_command(LINK_BATCH, (uint64_t)i, (uint64_t)p);
        batch(R, slots, b, x, p, dump);
        memcpy(B->h, b->hidden(R, top, x), sizeof(float) * (size_t)d * (size_t)nb);
        if (mtp_k > 0)
            mtp_follow(R, slots, B, nb, p);
        else
            memcpy(B->pending, B->h + (nb - 1) * d, sizeof(float) * (size_t)d);
    }
    link_command(LINK_MARK, 0, 0);
    link_wait_card();
    return choose_hidden(R, top, B->pending, dump);
}

static long step(struct rt *R, float **slots, float *x, long tok, long p)
{
    float *const *top = slots_of(slots, MODEL.n_layer);
    const struct batch_bind *const one = &MODEL.batches[MODEL.n_batches - 1];
    link_command(LINK_STEP, (uint64_t)tok, (uint64_t)p);
    memcpy(x, MODEL.embed(R, top, tok), sizeof(float) * (size_t)MODEL.d);
    batch(R, slots, one, x, p, NULL);
    memcpy(x, one->hidden(R, top, x), sizeof(float) * (size_t)MODEL.d);
    return MODEL.next(R, top, one->logits(R, top, x));
}

/* A decode by speculation: the MTP block drafts the tokens after cur, one pass over the model checks
 * them all, and what it agrees with stands. The model's greedy tokens, each one checked; a position
 * passed in error is simply passed again (qwen35.apl: its state is indexed by position). */
static void decode_mtp(struct rt *R, float **slots, float *x, const struct buffers *B, long cur, long q, long *gen, long n_gen)
{
    float *const *top = slots_of(slots, MODEL.n_layer);
    float *const *mtp = slots_of(slots, MODEL.n_layer + 1);
    const struct batch_bind *const one = &MODEL.batches[MODEL.n_batches - 1];
    const int one_i = MODEL.n_batches - 1;
    const long d = MODEL.d;
    const size_t row = sizeof(float) * (size_t)d;
    long i = 1;
    while (i < n_gen) {
        const long left = n_gen - i - 1, kk = mtp_k < left ? mtp_k : left;
        long draft[MAX_DRAFT + 1], agree[MAX_DRAFT + 1];
        const float *h = B->pending;
        embed_row(R, top, cur, B->e, 0);
        for (long j = 0; j < kk; j++) {
            memcpy(B->m, B->e + j * d, row);
            memcpy(B->m + d, h, row);
            link_command(LINK_MTP, (uint64_t)one_i, (uint64_t)(q + j));
            memcpy(B->draft, one->mtp(R, mtp, q + j, B->m), row);
            h = B->draft;
            link_command(LINK_DRAFT_LOGITS, 0, 0);
            draft[j + 1] = MODEL.draft_next(R, top, MODEL.draft_logits(R, top, h));
            embed_row(R, top, draft[j + 1], B->e, j + 1);
        }
        const long t = kk + 1;
        const int bi = batch_index(t);
        const struct batch_bind *b = &MODEL.batches[bi];
        memcpy(x, B->e, row * (size_t)t);
        link_command(LINK_BATCH, (uint64_t)bi, (uint64_t)q);
        batch(R, slots, b, x, q, NULL);
        memcpy(B->h, b->hidden(R, top, x), row * (size_t)t);
        link_command(LINK_LOGITS, (uint64_t)bi, 0);
        const float *y = b->logits(R, top, B->h);
        for (long k = 0; k < t; k++)
            agree[k] = MODEL.next(R, top, y + k * MODEL.vocab);
        const long m = kk ? b->accepted(R, top, draft + 1, agree) : 0; /* qwen35.apl: how many leading drafts stand */
        drafted += kk;
        accepted += m;
        cycles++;
        for (long k = 0; k <= m; k++)
            gen[i++] = agree[k];
        mtp_follow(R, slots, B, m + 1, q);
        q += m + 1;
        cur = agree[m];
    }
}

int main(int argc, char **argv)
{
    if (argc < 6 || argc > 7) {
        fprintf(stderr, "usage: engine_host MODEL TOKENS N_GEN THREADS REPEATS [DUMP_DIR]\n");
        return 2;
    }
    const char *model = argv[1], *dump_dir = argc == 7 ? argv[6] : NULL;
    const long n_gen = atol(argv[3]);
    const int threads = atoi(argv[4]);
    int repeats = atoi(argv[5]);
    if (repeats > MAX_REPEATS)
        repeats = MAX_REPEATS;
    static long prompt[MAX_TOKENS], gen[MAX_REPEATS + 1][MAX_TOKENS];
    const long np = read_prompt(argv[2], prompt);
    if (np <= 0 || np + n_gen > MODEL.ctx || threads < 1 || threads > MAX_THREADS) {
        fprintf(stderr, "engine_host: prompt %ld + N_GEN %ld against CTX %ld; 1 to %d threads\n", np, n_gen, MODEL.ctx, MAX_THREADS);
        return 2;
    }
    if ((uint64_t)MODEL.frame_bytes > LINK_BUF_BYTES) {
        fprintf(stderr, "engine_host: the frame, %ld B, does not fit the link\n", MODEL.frame_bytes);
        return 1;
    }
    if (getenv("APL_HOST_PAR"))
        rt_host_set_par(atol(getenv("APL_HOST_PAR")));

    /* ---- the region: the host's frame and the counters */
    int fd = open("/dev/shm/phi-hostmem", O_RDWR);
    if (fd < 0) {
        perror("/dev/shm/phi-hostmem");
        return 1;
    }
    char *part = mmap(NULL, STREAM_PART_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)STREAM_REGION_OFFSET);
    if (part == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    link_host_open((struct link_ctl *)(part + (LINK_CTL_OFFSET - STREAM_REGION_OFFSET)));

    /* ---- the team: CPUs 1-7, then their siblings; 0 and 8 stay with the scheduler */
    int cpus[MAX_THREADS];
    for (int i = 0; i < MAX_THREADS; i++)
        cpus[i] = i < 7 ? i + 1 : i + 2;
    struct rt R = {.team = threads > 1 ? team_create(cpus, threads, 1) : NULL,
                   .frame = part + (LINK_FRAME_OFFSET - STREAM_REGION_OFFSET)};
    if (threads > 1 && !R.team)
        return 1;
    rt_host_bind(&R, threads);

    /* ---- the weights and the host's state */
    float **slots = calloc((size_t)(MODEL.n_layer + 2) * (size_t)MODEL.n_slots, sizeof(float *));
    if (!slots)
        return 1;
    bind_weights(slots, model);
    size_t kv_bytes = 0;
    for (int i = 0; i < MODEL.n_states; i++) {
        const struct state_bind *b = &MODEL.states[i];
        if (b->side != 1)
            continue;
        if (b->region_off >= 0) {
            if ((uint64_t)b->region_off + sizeof(float) * (size_t)b->elems > LINK_KV_OFFSET + LINK_KV_BYTES) {
                fprintf(stderr, "engine_host: the KV cache does not fit its %llu B of the region\n", (unsigned long long)LINK_KV_BYTES);
                return 1;
            }
            slots_of(slots, b->layer)[b->slot] = (float *)(part + ((uint64_t)b->region_off - STREAM_REGION_OFFSET));
            kv_bytes += sizeof(float) * (size_t)b->elems;
        } else {
            slots_of(slots, b->layer)[b->slot] = zalloc64(sizeof(float) * (size_t)b->elems);
        }
    }
    const size_t row = sizeof(float) * (size_t)MODEL.d, rows = (size_t)MODEL.batches[0].nb;
    float *x = zalloc64(row * rows);
    struct buffers B = {zalloc64(row * rows), zalloc64(row * rows), zalloc64(2 * row * rows), zalloc64(row), zalloc64(row)};
    /* one draft a pass: the best measured on both models (2026-11-01: 9B 20.13 tok/s against
     * 15.41, 2B 64.33 against 51.84). APL_MTP=0 turns speculation off. */
    mtp_k = getenv("APL_MTP") ? atol(getenv("APL_MTP")) : MODEL.batches[0].mtp != NULL;
    if (mtp_k < 0 || mtp_k > MAX_DRAFT) {
        fprintf(stderr, "engine_host: APL_MTP from 0 to %d\n", MAX_DRAFT);
        return 2;
    }
    const long limit = getenv("APL_PROMPT_NB") ? atol(getenv("APL_PROMPT_NB")) : MODEL.batches[0].nb;

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

    printf("host worker: %d threads, waiting for the card\n", threads);
    fflush(stdout);
    link_host_wait_card_attach();

    /* ---- the runs: the same measure as the card-only engine */
    double prompt_s[MAX_REPEATS + 1], decode_s[MAX_REPEATS + 1];
    for (int rep = 0; rep <= repeats; rep++) {
        link_command(LINK_RESET, 0, 0);
        link_wait_card();
        for (int i = 0; i < MODEL.n_states; i++) {
            const struct state_bind *b = &MODEL.states[i];
            if (b->side == 1)
                memset(slots_of(slots, b->layer)[b->slot], 0, sizeof(float) * (size_t)b->elems);
        }
        double wait_s;
        long waits;
        link_host_stats(&wait_s, &waits);
        double t0 = now_s();
        long tok = prompt_run(&R, slots, x, &B, prompt, np, limit, rep == 0 && dump_dir ? dump : NULL);
        prompt_s[rep] = now_s() - t0;
        gen[rep][0] = tok;
        link_host_stats(&wait_s, &waits);
        printf("host in prompt: worked %.3f ms/token, waited for the card %.3f ms/token, %.1f exchanges/token\n", (prompt_s[rep] - wait_s) * 1e3 / (double)np,
               wait_s * 1e3 / (double)np, (double)waits / (double)np);
        t0 = now_s();
        if (mtp_k > 0) {
            drafted = accepted = cycles = 0;
            decode_mtp(&R, slots, x, &B, tok, np, gen[rep], n_gen);
        } else {
            for (long i = 1; i < n_gen; i++) {
                tok = step(&R, slots, x, tok, np + i - 1);
                gen[rep][i] = tok;
            }
        }
        decode_s[rep] = now_s() - t0;
        link_host_stats(&wait_s, &waits);
        const double per = 1e3 / (double)(n_gen - 1);
        printf("host in decode: worked %.3f ms/token, waited for the card %.3f ms/token, %.1f exchanges/token\n", (decode_s[rep] - wait_s) * per,
               wait_s * per, (double)waits / (double)(n_gen - 1));
        if (mtp_k > 0)
            printf("mtp: %ld drafts per pass: drafted %ld, accepted %ld (%.1f %%), %ld passes, %.2f tokens per pass\n", mtp_k, drafted, accepted,
                   drafted ? 100.0 * (double)accepted / (double)drafted : 0.0, cycles, cycles ? (double)(n_gen - 1) / (double)cycles : 0.0);
        printf("run %d%s: prompt %ld tokens %.3f s = %.3f tok/s; decode %ld tokens %.3f s = %.3f tok/s\n", rep,
               rep ? "" : " (warm-up)", np, prompt_s[rep], (double)np / prompt_s[rep], n_gen - 1, decode_s[rep],
               (double)(n_gen - 1) / decode_s[rep]);
        fflush(stdout);
    }
    link_command(LINK_END, 0, 0);
    if (dump_dir)
        for (int il = 0; il <= MODEL.n_layer; il++)
            fclose(dump[il]);

    int same = 1;
    for (int rep = 1; rep <= repeats; rep++)
        same &= memcmp(gen[rep], gen[0], sizeof(long) * (size_t)n_gen) == 0;
    /* placement: what the host holds */
    FILE *st = fopen("/proc/self/status", "r");
    char sline[128];
    while (st && fgets(sline, sizeof sline, st))
        if (strncmp(sline, "VmHWM:", 6) == 0)
            printf("host memory: peak RSS %s", sline + 6);
    if (st)
        fclose(st);
    printf("KV cache in system memory: %zu B reserved in the region\n", kv_bytes);
    printf("gen");
    for (long i = 0; i < n_gen; i++)
        printf(" %ld", gen[0][i]);
    printf("\nall repeats generated the same tokens: %s\n", same ? "yes" : "NO");
    if (repeats > 0) {
        qsort(prompt_s + 1, (size_t)repeats, sizeof(double), cmp_d);
        qsort(decode_s + 1, (size_t)repeats, sizeof(double), cmp_d);
        const double tps = (double)(n_gen - 1) / decode_s[1 + repeats / 2];
        printf("DECODE %.3f tok/s (median of %d; min %.3f max %.3f); PROMPT %.3f tok/s; host threads %d\n", tps, repeats,
               (double)(n_gen - 1) / decode_s[repeats], (double)(n_gen - 1) / decode_s[1], (double)np / prompt_s[1 + repeats / 2], threads);
    }
    if (R.team)
        team_destroy(R.team);
    return 0;
}
