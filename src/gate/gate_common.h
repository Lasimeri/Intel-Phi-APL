/* gate_common.h: the input generator and the result hash shared by the
 * card program (fma_main.c) and the host reference (fma_host.c), so both
 * sides build the same input bits and hash the result the same way.
 * Inputs: xorshift32, the top 24 bits as a signed integer in [-2^23, 2^23)
 * scaled by 2^-23: exact in binary32 on any FPU (x87 or SSE), no rounding. */
#ifndef APL_GATE_COMMON_H
#define APL_GATE_COMMON_H
#include <stdint.h>
#include <string.h>

#define GATE_N 4096
#define GATE_SEED 0x9E3779B9u

static inline uint32_t gate_xs32(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *s = x;
}

static inline float gate_unit(uint32_t *s)
{
    int32_t v = (int32_t)(gate_xs32(s) >> 8) - 8388608;
    return (float)v * (1.0f / 8388608.0f);
}

static inline void gate_fill(float *a, float *b, float *c, int n)
{
    uint32_t s = GATE_SEED;
    for (int i = 0; i < n; i++) {
        a[i] = gate_unit(&s);
        b[i] = gate_unit(&s);
        c[i] = gate_unit(&s);
    }
}

static inline uint32_t gate_bits(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

/* FNV-1a over the 32-bit words of the result. */
static inline uint64_t gate_hash(const float *c, int n)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (int i = 0; i < n; i++) {
        h ^= gate_bits(c[i]);
        h *= 0x100000001b3ull;
    }
    return h;
}
#endif
