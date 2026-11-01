/* upconv_common.h: gate 2.1's inputs, output layout and file format, shared
 * by the program (upconv_main.c: card and host builds) and the comparator
 * (upconv_cmp.c), so all three see the same bits.
 *
 * Inputs, exhaustive where the domain is small:
 *   8-bit sets  (k_s8, k_u8fma, k_u8mul), G21_N8 = 512 lanes: lanes 0..255
 *               take every weight value with x = 1, c = +0 (the conversion
 *               alone), lanes 256..511 every value again with random x, c.
 *   f16 set     (k_f16), G21_NF = 131072 lanes: all 65536 half patterns with
 *               x = 1, c = +0, then all 65536 again with random x, c.
 *   scale set   (k_b16, k_bcast), G21_NB = 4096 lanes, 256 random scales.
 *   hsum set    (k_hsum), G21_NH = 4096 values spread over 2^-20 .. 2^20
 *               so that the summation order changes the rounded sum.
 * Every value is exact in binary32 on any FPU (x87 included): integers
 * below 2^24 scaled by powers of two; no libm.
 *
 * File: a 64-byte header (magic "APLG21v1", then n8, nb, nf, nh as
 * uint32 LE, zero padding), then the result arrays in g21_arrays() order
 * as raw binary32. */
#ifndef APL_UPCONV_COMMON_H
#define APL_UPCONV_COMMON_H
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "gate_common.h"

#define G21_N8 512
#define G21_NB 4096
#define G21_NF 131072
#define G21_NH 4096
#define G21_SEED 0x2545F491u
#define G21_MAGIC "APLG21v1"
#define A64 __attribute__((aligned(64)))

struct g21_in {
    int8_t q[G21_N8 + 256] A64; /* + 256: k_s8 prefetches ahead */
    uint8_t u[G21_N8] A64;
    uint16_t h[G21_NF] A64;
    float x8[G21_N8] A64, c8[G21_N8] A64;
    float xb[G21_NB] A64, cb[G21_NB] A64, s[G21_NB / 16] A64;
    float xf[G21_NF] A64, cf[G21_NF] A64;
    float xh[G21_NH] A64;
};

struct g21_out {
    float s8[G21_N8] A64, u8fma[G21_N8] A64, u8mul[G21_N8] A64;
    float b16[G21_NB] A64, bcast[G21_NB] A64;
    float f16[G21_NF] A64;
    float hsum[G21_NH / 16] A64;
};

/* 2^e as a binary32 built from its bits, |e| <= 126. */
static inline float g21_pow2(int e)
{
    uint32_t b = (uint32_t)(127 + e) << 23;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

static inline void g21_fill(struct g21_in *in)
{
    uint32_t s = G21_SEED;
    memset(in, 0, sizeof *in);
    for (int i = 0; i < G21_N8; i++) {
        int half = i >= 256;
        in->q[i] = (int8_t)(uint8_t)i;   /* 0..127, then -128..-1 */
        in->u[i] = (uint8_t)i;
        in->x8[i] = half ? gate_unit(&s) : 1.0f;
        in->c8[i] = half ? gate_unit(&s) : 0.0f;
    }
    for (int i = 0; i < G21_NF; i++) {
        int half = i >= 65536;
        in->h[i] = (uint16_t)i;
        in->xf[i] = half ? gate_unit(&s) : 1.0f;
        in->cf[i] = half ? gate_unit(&s) : 0.0f;
    }
    for (int i = 0; i < G21_NB; i++) {
        in->xb[i] = gate_unit(&s);
        in->cb[i] = gate_unit(&s);
    }
    for (int i = 0; i < G21_NB / 16; i++)
        in->s[i] = gate_unit(&s);
    for (int i = 0; i < G21_NH; i++) {
        float m = gate_unit(&s);
        in->xh[i] = m * g21_pow2((int)(gate_xs32(&s) % 41u) - 20);
    }
}

/* The result arrays in file order. */
struct g21_array {
    const char *name;
    float *p;
    int n;
};
static inline int g21_arrays(struct g21_out *o, struct g21_array a[7])
{
    struct g21_array t[7] = {
        {"s8", o->s8, G21_N8},       {"u8fma", o->u8fma, G21_N8}, {"u8mul", o->u8mul, G21_N8},
        {"b16", o->b16, G21_NB},     {"bcast", o->bcast, G21_NB}, {"f16", o->f16, G21_NF},
        {"hsum", o->hsum, G21_NH / 16},
    };
    memcpy(a, t, sizeof t);
    return 7;
}

static inline void g21_header(unsigned char h[64])
{
    uint32_t n[4] = {G21_N8, G21_NB, G21_NF, G21_NH};
    memset(h, 0, 64);
    memcpy(h, G21_MAGIC, 8);
    memcpy(h + 8, n, sizeof n);
}

/* One summary line per array: name, lanes, FNV-1a hash. */
static inline void g21_summary(FILE *f, struct g21_out *o)
{
    struct g21_array a[7];
    int k = g21_arrays(o, a);
    for (int j = 0; j < k; j++)
        fprintf(f, "%-6s n %6d hash %016llx\n", a[j].name, a[j].n, (unsigned long long)gate_hash(a[j].p, a[j].n));
}
#endif
