/* ref_llama.c: the llama.cpp reference for the APL engine (host, TEST ONLY,
 * never an inference path). Links the libllama of an existing llama.cpp
 * build read-only and runs on the host CPU (the caller sets
 * CUDA_VISIBLE_DEVICES= so no GPU is touched).
 *
 *   ref_llama MODEL OUTDIR N_GEN THREADS REPEATS PROMPT
 *
 * Tokenizes PROMPT with the model's own vocabulary, then REPEATS + 1 times
 * (the first is a warm-up, not reported): clears the memory, decodes the
 * prompt as one batch, and generates N_GEN tokens greedily (argmax, lowest
 * id on ties), one llama_decode per token. Writes
 *   OUTDIR/tokens.txt   "prompt ID ID ..." and "gen ID ID ..." lines
 *   OUTDIR/logits.f32   the full logits of the first REF_STEPS steps (step s
 *                       = the distribution that picked gen[s]), n_vocab F32 each
 *   OUTDIR/text.txt     the prompt and the generated text
 * and prints, per repeat, prompt and decode timings and tok/s. Exit 1 if a
 * repeat generates different tokens than the first (nondeterminism). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "llama.h"

#define REF_STEPS 8
#define MAX_TOK 4096

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static llama_token argmax(const float *l, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (l[i] > l[best])
            best = i;
    return best;
}

static void write_tokens(const char *dir, const llama_token *p, int np, const llama_token *g, int ng)
{
    char path[4096];
    snprintf(path, sizeof path, "%s/tokens.txt", dir);
    FILE *f = fopen(path, "w");
    if (!f) {
        perror(path);
        exit(1);
    }
    fprintf(f, "prompt");
    for (int i = 0; i < np; i++)
        fprintf(f, " %d", p[i]);
    fprintf(f, "\ngen");
    for (int i = 0; i < ng; i++)
        fprintf(f, " %d", g[i]);
    fprintf(f, "\n");
    fclose(f);
}

static void write_text(const char *dir, const struct llama_vocab *v, const char *prompt, const llama_token *g, int ng)
{
    char path[4096], piece[256];
    snprintf(path, sizeof path, "%s/text.txt", dir);
    FILE *f = fopen(path, "w");
    if (!f) {
        perror(path);
        exit(1);
    }
    fprintf(f, "%s", prompt);
    for (int i = 0; i < ng; i++) {
        int n = llama_token_to_piece(v, g[i], piece, (int)sizeof piece, 0, true);
        if (n > 0)
            fwrite(piece, 1, (size_t)n, f);
    }
    fprintf(f, "\n");
    fclose(f);
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        fprintf(stderr, "usage: %s MODEL OUTDIR N_GEN THREADS REPEATS PROMPT\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1], *dir = argv[2], *prompt = argv[6];
    int n_gen = atoi(argv[3]), threads = atoi(argv[4]), repeats = atoi(argv[5]);
    if (n_gen < 2 || n_gen > MAX_TOK || threads < 1 || repeats < 1) {
        fprintf(stderr, "bad N_GEN (2..%d), THREADS or REPEATS\n", MAX_TOK);
        return 2;
    }

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", model_path);
        return 1;
    }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = MAX_TOK;
    cp.n_batch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "cannot create a context\n");
        return 1;
    }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    static llama_token ptok[MAX_TOK], gen[MAX_TOK], gen0[MAX_TOK];
    int np = llama_tokenize(vocab, prompt, (int)strlen(prompt), ptok, MAX_TOK, true, false);
    if (np <= 0 || np + n_gen > MAX_TOK) {
        fprintf(stderr, "tokenize failed or too long (%d)\n", np);
        return 1;
    }
    float *dump = calloc((size_t)REF_STEPS * (size_t)n_vocab, sizeof(float));
    if (!dump)
        return 1;

    printf("model %s\nvocab %d  prompt %d tokens  gen %d tokens  threads %d  repeats %d (+1 warm-up)\n",
           model_path, n_vocab, np, n_gen, threads, repeats);
    int bad = 0;
    for (int r = 0; r <= repeats; r++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        double t0 = now_s();
        if (llama_decode(ctx, llama_batch_get_one(ptok, np)) != 0) {
            fprintf(stderr, "prompt decode failed\n");
            return 1;
        }
        double t1 = now_s();
        const float *lg = llama_get_logits_ith(ctx, -1);
        for (int i = 0; i < n_gen; i++) {
            if (r == repeats && i < REF_STEPS)
                memcpy(dump + (size_t)i * (size_t)n_vocab, lg, (size_t)n_vocab * sizeof(float));
            gen[i] = argmax(lg, n_vocab);
            if (i == n_gen - 1)
                break;
            if (llama_decode(ctx, llama_batch_get_one(&gen[i], 1)) != 0) {
                fprintf(stderr, "decode failed at %d\n", i);
                return 1;
            }
            lg = llama_get_logits_ith(ctx, -1);
        }
        double t2 = now_s();
        if (r == 0)
            memcpy(gen0, gen, sizeof(llama_token) * (size_t)n_gen);
        else if (memcmp(gen0, gen, sizeof(llama_token) * (size_t)n_gen) != 0)
            bad = 1;
        printf("%s %d  prompt %4d tok %9.2f ms = %8.2f tok/s (one batch)  decode %4d tok %9.2f ms = %7.3f tok/s\n",
               r ? "repeat" : "warmup", r, np, 1e3 * (t1 - t0), np / (t1 - t0), n_gen - 1, 1e3 * (t2 - t1),
               (n_gen - 1) / (t2 - t1));
    }

    write_tokens(dir, ptok, np, gen, n_gen);
    write_text(dir, vocab, prompt, gen, n_gen);
    char path[4096];
    snprintf(path, sizeof path, "%s/logits.f32", dir);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(dump, sizeof(float), (size_t)REF_STEPS * (size_t)n_vocab, f) != (size_t)REF_STEPS * (size_t)n_vocab) {
        perror(path);
        return 1;
    }
    fclose(f);
    printf("%s\n", bad ? "NONDETERMINISTIC: repeats generated different tokens" : "deterministic: every repeat generated the same tokens");

    free(dump);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return bad;
}
