/* upconv_cmp.c: gate 2.1, the judge (host cc, -lm). Recomputes every
 * result from the inputs of upconv_common.h in plain scalar code that shares
 * nothing with knc.h (glibc fmaf, the compiler's _Float16 conversion), then
 * compares bit by bit:
 *
 *   upconv_cmp --ref               print the reference hash lines (the golden file)
 *   upconv_cmp HOST.bin            the host model against the reference
 *   upconv_cmp HOST.bin CARD.bin   both, and the card against both
 *
 * Per array: lanes that differ, NaN-vs-NaN differences (payload or quiet
 * bit) counted apart; for k_f16 the differing half patterns by class
 * (zero, subnormal, normal, inf, NaN); the first differing lanes in full.
 * Also how many lanes or blocks the gate can tell apart (fused vs unfused,
 * tree vs sequential order), so a pass means something. Exit 0 only when
 * every compared array matches the reference in every bit. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "upconv_common.h"

static struct g21_in in;
static struct g21_out ref, host, card;

static float half_ref(uint16_t h)
{
    _Float16 f;
    memcpy(&f, &h, 2);
    return (float)f;
}

static float tree16(const float *x)
{
    float v[16], t[16];
    memcpy(v, x, sizeof v);
    for (int k = 1; k < 16; k <<= 1) {
        for (int i = 0; i < 16; i++)
            t[i] = v[i] + v[i ^ k];
        memcpy(v, t, sizeof v);
    }
    return v[0];
}

static void reference(void)
{
    for (int i = 0; i < G21_N8; i++) {
        ref.s8[i] = fmaf(in.x8[i], (float)in.q[i], in.c8[i]);
        ref.u8fma[i] = fmaf(in.x8[i], (float)in.u[i], in.c8[i]);
        volatile float p = in.x8[i] * (float)in.u[i];
        ref.u8mul[i] = p + in.c8[i];
    }
    for (int i = 0; i < G21_NB; i++)
        ref.b16[i] = ref.bcast[i] = fmaf(in.xb[i], in.s[i / 16], in.cb[i]);
    for (int i = 0; i < G21_NF; i++)
        ref.f16[i] = fmaf(in.xf[i], half_ref(in.h[i]), in.cf[i]);
    for (int b = 0; b < G21_NH / 16; b++)
        ref.hsum[b] = tree16(in.xh + 16 * b);
}

static int load(const char *path, struct g21_out *o)
{
    unsigned char h[64], want[64];
    struct g21_array a[7];
    int k = g21_arrays(o, a), ok;
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return 1;
    }
    g21_header(want);
    ok = fread(h, 1, 64, f) == 64 && memcmp(h, want, 64) == 0;
    if (!ok)
        fprintf(stderr, "%s: not a gate 2.1 result file of this build (header)\n", path);
    for (int j = 0; ok && j < k; j++)
        ok = fread(a[j].p, sizeof(float), (size_t)a[j].n, f) == (size_t)a[j].n;
    ok = ok && fgetc(f) == EOF;
    fclose(f);
    if (!ok)
        fprintf(stderr, "%s: wrong size\n", path);
    return !ok;
}

static int is_nan(uint32_t b) { return (b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu); }

static const char *half_class(uint16_t h)
{
    unsigned e = (h >> 10) & 31, m = h & 1023;
    if (e == 0)
        return m ? "subnormal" : "zero";
    if (e == 31)
        return m ? "nan" : "inf";
    return "normal";
}

/* Input description of lane i of array j (for the mismatch listing). */
static void describe(int j, int i, char *s, size_t n)
{
    switch (j) {
    case 0: snprintf(s, n, "q %4d x %08x c %08x", in.q[i], gate_bits(in.x8[i]), gate_bits(in.c8[i])); break;
    case 1:
    case 2: snprintf(s, n, "u %3u x %08x c %08x", in.u[i], gate_bits(in.x8[i]), gate_bits(in.c8[i])); break;
    case 3:
    case 4: snprintf(s, n, "s %08x x %08x c %08x", gate_bits(in.s[i / 16]), gate_bits(in.xb[i]), gate_bits(in.cb[i])); break;
    case 5: snprintf(s, n, "h %04x (%s) x %08x c %08x", in.h[i], half_class(in.h[i]), gate_bits(in.xf[i]), gate_bits(in.cf[i])); break;
    default: snprintf(s, n, "block %d", i); break;
    }
}

/* Compare one array of `got` against the reference; print one line, and
 * the first differing lanes when show > 0. Returns the differing lanes. */
static int compare(const char *who, int j, const float *got, const float *want, int n, int show)
{
    int diff = 0, nan_diff = 0, cls[5] = {0};
    static const char *cname[5] = {"zero", "subnormal", "normal", "inf", "nan"};
    for (int i = 0; i < n; i++) {
        uint32_t g = gate_bits(got[i]), w = gate_bits(want[i]);
        if (g == w)
            continue;
        if (is_nan(g) && is_nan(w))
            nan_diff++;
        else
            diff++;
        if (j == 5) {
            const char *c = half_class(in.h[i]);
            for (int k = 0; k < 5; k++)
                cls[k] += strcmp(c, cname[k]) == 0;
        }
        if (show-- > 0) {
            char d[96];
            describe(j, i, d, sizeof d);
            printf("        lane %6d  %s  got %08x want %08x\n", i, d, g, w);
        }
    }
    printf("  %-5s n %6d  differ %6d  nan-vs-nan %6d", who, n, diff, nan_diff);
    if (j == 5 && diff + nan_diff)
        printf("  by half class: zero %d subnormal %d normal %d inf %d nan %d", cls[0], cls[1], cls[2], cls[3], cls[4]);
    printf("\n");
    return diff + nan_diff;
}

int main(int argc, char **argv)
{
    struct g21_array r[7], hh[7], cc[7];
    int k, bad = 0, discr_fused = 0, discr_tree = 0;

    g21_fill(&in);
    reference();
    if (argc == 2 && strcmp(argv[1], "--ref") == 0) {
        g21_summary(stdout, &ref);
        return 0;
    }
    if (argc < 2 || argc > 3 || argv[1][0] == '-') {
        fprintf(stderr, "usage: %s --ref | HOST.bin [CARD.bin]\n", argv[0]);
        return 2;
    }
    if (load(argv[1], &host) || (argc == 3 && load(argv[2], &card)))
        return 1;

    for (int i = 0; i < G21_N8; i++)
        discr_fused += gate_bits(ref.u8fma[i]) != gate_bits(ref.u8mul[i]);
    for (int b = 0; b < G21_NH / 16; b++) {
        float seq = 0.0f;
        for (int i = 0; i < 16; i++)
            seq += in.xh[16 * b + i];
        discr_tree += gate_bits(seq) != gate_bits(ref.hsum[b]);
    }
    printf("discriminating: u8fma vs u8mul differ in %d of %d lanes; tree vs sequential hsum differ in %d of %d blocks\n",
           discr_fused, G21_N8, discr_tree, G21_NH / 16);

    k = g21_arrays(&ref, r);
    g21_arrays(&host, hh);
    g21_arrays(&card, cc);
    for (int j = 0; j < k; j++) {
        printf("%s\n", r[j].name);
        bad |= compare("host", j, hh[j].p, r[j].p, r[j].n, 8) != 0;
        if (argc == 3) {
            bad |= compare("card", j, cc[j].p, r[j].p, r[j].n, 8) != 0;
            compare("c-v-h", j, cc[j].p, hh[j].p, r[j].n, 0);
        }
    }
    printf("%s\n", bad ? "GATE 2.1 FAIL" : (argc == 3 ? "GATE 2.1 PASS (host model and card == reference, every bit)" : "host model == reference, every bit (card not compared)"));
    return bad;
}
