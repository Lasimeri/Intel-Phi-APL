/* logit_cmp.c: hold one set of engine logits against the llama.cpp reference
 * (TEST ONLY). Reads the engine's logits as text (APL high minus understood)
 * or as raw F32 (a .f32 file) and step STEP of ref_llama's logits.f32.
 *
 *   logit_cmp ENGINE REF_LOGITS_F32 STEP N_VOCAB
 *
 * The gate, fixed before the first run (2026-11-01):
 *   same top-1 token,
 *   NRMSE = RMS(engine - ref) / stddev(ref) <= 0.10,
 *   at least 8 of the reference's top 10 tokens in the engine's top 10.
 * llama.cpp's Q8_0 path quantizes activations to 8 bits before each dot
 * product, so a systematic difference from an F32 or F64 engine is expected;
 * the reference's own top1-top2 margin is printed, since a tiny margin makes a
 * top-1 disagreement inconclusive. Exit 0 when the gate passes. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOP 10

static double *load_text(const char *path, long n)
{
    FILE *f = fopen(path, "r");
    double *v = malloc((size_t)n * sizeof *v);
    char tok[128];
    long i = 0;
    if (!f || !v) {
        perror(path);
        exit(2);
    }
    while (i < n && fscanf(f, "%127s", tok) == 1) {
        char *s = tok;
        int neg = strncmp(s, "\xC2\xAF", 2) == 0;
        char buf[260];
        if (neg)
            s += 2;
        /* an exponent's high minus */
        char *e = strstr(s, "E\xC2\xAF");
        if (e) {
            *e = 0;
            snprintf(buf, sizeof buf, "%sE-%s", s, e + 3);
            s = buf;
        }
        v[i] = strtod(s, NULL);
        if (neg)
            v[i] = -v[i];
        i++;
    }
    fclose(f);
    if (i != n) {
        fprintf(stderr, "logit_cmp: %s: %ld numbers, want %ld\n", path, i, n);
        exit(2);
    }
    return v;
}

static double *load_f32(const char *path, long skip, long n)
{
    FILE *f = fopen(path, "rb");
    float *b = malloc((size_t)n * sizeof *b);
    double *v = malloc((size_t)n * sizeof *v);
    if (!f || !b || !v || fseek(f, skip * (long)sizeof(float), SEEK_SET) != 0 || fread(b, sizeof(float), (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "logit_cmp: cannot read %ld F32 values from %s\n", n, path);
        exit(2);
    }
    for (long i = 0; i < n; i++)
        v[i] = (double)b[i];
    free(b);
    fclose(f);
    return v;
}

/* The TOP largest values' indices, largest first, lowest id on ties. */
static void top_k(const double *v, long n, long out[TOP])
{
    for (int k = 0; k < TOP; k++)
        out[k] = -1;
    for (long i = 0; i < n; i++) {
        for (int k = 0; k < TOP; k++) {
            if (out[k] < 0 || v[i] > v[out[k]]) {
                memmove(&out[k + 1], &out[k], (size_t)(TOP - 1 - k) * sizeof out[0]);
                out[k] = i;
                break;
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: logit_cmp ENGINE REF_LOGITS_F32 STEP N_VOCAB\n");
        return 2;
    }
    long step = atol(argv[3]), n = atol(argv[4]);
    size_t len = strlen(argv[1]);
    double *e = len > 4 && strcmp(argv[1] + len - 4, ".f32") == 0 ? load_f32(argv[1], 0, n) : load_text(argv[1], n);
    double *r = load_f32(argv[2], step * n, n);

    double sum = 0, sq = 0, err = 0, max_abs = 0;
    for (long i = 0; i < n; i++) {
        double d = e[i] - r[i];
        sum += r[i];
        sq += r[i] * r[i];
        err += d * d;
        if (fabs(d) > max_abs)
            max_abs = fabs(d);
    }
    double mean = sum / (double)n, std = sqrt(sq / (double)n - mean * mean), rms = sqrt(err / (double)n), nrmse = rms / std;

    long te[TOP], tr[TOP];
    top_k(e, n, te);
    top_k(r, n, tr);
    int overlap = 0;
    for (int a = 0; a < TOP; a++)
        for (int b = 0; b < TOP; b++)
            overlap += te[a] == tr[b];

    printf("reference: top1 %ld (%.4f), top2 %ld (%.4f), margin %.4f, stddev %.4f\n", tr[0], r[tr[0]], tr[1], r[tr[1]], r[tr[0]] - r[tr[1]], std);
    printf("engine:    top1 %ld (%.4f), top2 %ld (%.4f), margin %.4f\n", te[0], e[te[0]], te[1], e[te[1]], e[te[0]] - e[te[1]]);
    printf("error:     RMS %.4f, NRMSE %.4f, max abs %.4f, top-%d overlap %d\n", rms, nrmse, max_abs, TOP, overlap);
    printf("top %d ref:", TOP);
    for (int k = 0; k < TOP; k++)
        printf(" %ld", tr[k]);
    printf("\ntop %d eng:", TOP);
    for (int k = 0; k < TOP; k++)
        printf(" %ld", te[k]);
    printf("\n");
    int pass = te[0] == tr[0] && nrmse <= 0.10 && overlap >= 8;
    printf("%s (top-1 %s, NRMSE %.4f vs 0.10, overlap %d vs 8)\n", pass ? "PASS" : "FAIL", te[0] == tr[0] ? "same" : "DIFFERENT", nrmse, overlap);
    free(e);
    free(r);
    return !pass;
}
