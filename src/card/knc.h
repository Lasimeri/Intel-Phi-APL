/* knc.h: every Knights Corner vector operation the APL engine's kernels
 * use, behind one name each, in two bodies:
 *   - under __KNC__ (the Intel-Phi-LLVM clang with -march=knc) the KNC
 *     builtin itself, one instruction per operation, v16f a 512-bit vector;
 *   - anywhere else (the host cc, clangd) a lane-by-lane model that rounds
 *     exactly as the card does: fmaf for every fused multiply-add, one IEEE
 *     add per lane pair in the reduction tree's own pairing; v16f a plain
 *     struct of 16 floats.
 * So a kernel written against this header compiles for the card and, the
 * same source, for the host as a bit-exact model of itself: the host build
 * is each kernel's reference. Kernels touch a v16f only through these
 * functions (no operators, no subscripts): the host struct makes any other
 * use a compile error, so nothing can bypass the model. The struct also
 * keeps the host ABI fixed (a GCC vector of 64 bytes would change ABI
 * without AVX-512, gcc -Wpsabi), and it carries no 64-byte alignment for
 * the same reason (gcc notes a changed ABI for 64-byte-aligned parameters).
 * Assumes round-to-nearest-even and gradual underflow. Gate 2.1 measured
 * the float16 up-convert (half subnormals, inf, NaN bits) and found the card
 * equal to this model; the VPU's handling of F32 subnormal inputs and
 * results (flush-to-zero or not) is NOT measured yet.
 *
 * Memory-operand conversions are the MVEX SSS codes. For a load-op (FMA,
 * mul) clang accepts 0 none, 1 {1to16}, 2 {4to16}, 3 float16, 4 uint8,
 * 6 uint16, 7 sint16 (SemaX86KNC.inc mask 0xdfdf): NO sint8. A plain load
 * (vmovaps) accepts 0, 3 float16, 4 uint8, 5 sint8, 6, 7 (mask 0xf9f9).
 * Pointers to vector memory are 64-byte aligned unless a function says
 * otherwise; an up-converting operand reads 16 elements (16 B for 8-bit,
 * 32 B for 16-bit). */
#ifndef APL_KNC_H
#define APL_KNC_H

#include <stdint.h>

#define KNC_SSS_NONE 0
#define KNC_SSS_BCAST16 1
#define KNC_SSS_FLOAT16 3
#define KNC_SSS_UINT8 4
#define KNC_SSS_SINT8 5
#define KNC_SWZ_CDAB 1 /* lane i takes lane i ^ 1 within each 4-lane group */
#define KNC_SWZ_BADC 2 /* lane i takes lane i ^ 2 */
#define KNC_SWZ_AAAA 4 /* every lane takes lane 0 of its 4-lane group; BBBB 5, CCCC 6, DDDD 7 */
#define KNC_SWZ_BBBB 5
#define KNC_SWZ_CCCC 6
#define KNC_SWZ_DDDD 7
#define KNC_PERM_SWAP1 0xB1 /* 128-bit group g takes group g ^ 1: lane i ^ 4 */
#define KNC_PERM_SWAP2 0x4E /* 128-bit group g takes group g ^ 2: lane i ^ 8 */

#if defined(__KNC__)

#include <immintrin.h>

typedef float v16f __attribute__((vector_size(64), aligned(64)));

#define KNC_ALL ((unsigned short)0xffff)
#define KNC_U ((__v16sf)_mm512_undefined_ps())

static inline v16f knc_zero(void) { return (v16f)_mm512_setzero_ps(); }
static inline v16f knc_load(const float *p) { return (v16f)_mm512_load_ps(p); }
static inline void knc_store(float *p, v16f v) { _mm512_store_ps(p, (__m512)v); }
/* vbroadcastss: p[0] in every lane. */
static inline v16f knc_bcast(const float *p) { return (v16f)__builtin_ia32_knc_vbroadcastss_zm(KNC_U, KNC_ALL, p, KNC_SSS_NONE); }
/* vmovaps {sint8}: 16 signed bytes to 16 floats (exact). */
static inline v16f knc_load_s8(const int8_t *p) { return (v16f)__builtin_ia32_knc_vmovaps_zm(KNC_U, KNC_ALL, p, KNC_SSS_SINT8); }
static inline v16f knc_add(v16f a, v16f b) { return (v16f)__builtin_ia32_knc_vaddps_zzz(KNC_U, KNC_ALL, (__v16sf)a, (__v16sf)b, 0); }
/* Four floats in every 128-bit group: then four FMAs with register swizzles, no memory operand. */
static inline v16f knc_load_4(const float *p) { return (v16f)__builtin_ia32_knc_vbroadcastf32x4_zm(KNC_U, KNC_ALL, p, KNC_SSS_NONE); }
#define KNC_FMA_SWZ(a, b, c, s) ((v16f)__builtin_ia32_knc_vfmadd231ps_zzz((__v16sf)(c), KNC_ALL, (__v16sf)(a), (__v16sf)(b), (s)))
/* a * b + c, one rounding (vfmadd231ps). */
static inline v16f knc_fma(v16f a, v16f b, v16f c)
{
    return (v16f)__builtin_ia32_knc_vfmadd231ps_zzz((__v16sf)c, KNC_ALL, (__v16sf)a, (__v16sf)b, KNC_SSS_NONE);
}
/* a * (float)p[0..15] + c, the uint8 load fused into the FMA. */
static inline v16f knc_fma_u8(v16f a, const uint8_t *p, v16f c)
{
    return (v16f)__builtin_ia32_knc_vfmadd231ps_zzm((__v16sf)c, KNC_ALL, (__v16sf)a, p, KNC_SSS_UINT8);
}
/* a * (float)p[0..15], the uint8 load fused into the multiply. */
static inline v16f knc_mul_u8(v16f a, const uint8_t *p)
{
    return (v16f)__builtin_ia32_knc_vmulps_zzm(KNC_U, KNC_ALL, (__v16sf)a, p, KNC_SSS_UINT8);
}
/* a * half(p[0..15]) + c, the float16 load fused into the FMA. */
static inline v16f knc_fma_f16(v16f a, const uint16_t *p, v16f c)
{
    return (v16f)__builtin_ia32_knc_vfmadd231ps_zzm((__v16sf)c, KNC_ALL, (__v16sf)a, p, KNC_SSS_FLOAT16);
}
/* a * p[0] + c, the scalar broadcast {1to16} fused into the FMA. */
static inline v16f knc_fma_b16(v16f a, const float *p, v16f c)
{
    return (v16f)__builtin_ia32_knc_vfmadd231ps_zzm((__v16sf)c, KNC_ALL, (__v16sf)a, p, KNC_SSS_BCAST16);
}
/* Sum of the 16 lanes: lane pairs i^1, i^2 (register swizzles folded into
 * the add), then i^4, i^8 (vpermf32x4); every lane ends with the sum. */
static inline float knc_hsum(v16f v)
{
    v = (v16f)__builtin_ia32_knc_vaddps_zzz(KNC_U, KNC_ALL, (__v16sf)v, (__v16sf)v, KNC_SWZ_CDAB);
    v = (v16f)__builtin_ia32_knc_vaddps_zzz(KNC_U, KNC_ALL, (__v16sf)v, (__v16sf)v, KNC_SWZ_BADC);
    v = knc_add(v, (v16f)__builtin_ia32_knc_vpermf32x4_zzi(KNC_U, KNC_ALL, (__v16sf)v, KNC_PERM_SWAP1));
    v = knc_add(v, (v16f)__builtin_ia32_knc_vpermf32x4_zzi(KNC_U, KNC_ALL, (__v16sf)v, KNC_PERM_SWAP2));
    return v[0];
}
static inline void knc_prefetch0(const void *p) { __builtin_ia32_knc_vprefetch0_m(p); }
static inline void knc_prefetch1(const void *p) { __builtin_ia32_knc_vprefetch1_m(p); }
/* A spin wait of about n cycles that yields the core's issue slots to its
 * other threads: DELAY r32. KNC has no PAUSE to speak of; this is the
 * instruction made for it. */
static inline void knc_delay(unsigned n) { __asm__ volatile("delay %0" : : "r"(n)); }
static inline v16f knc_sub(v16f a, v16f b) { return (v16f)__builtin_ia32_knc_vsubps_zzz(KNC_U, KNC_ALL, (__v16sf)a, (__v16sf)b, 0); }
static inline v16f knc_mul(v16f a, v16f b) { return (v16f)__builtin_ia32_knc_vmulps_zzz(KNC_U, KNC_ALL, (__v16sf)a, (__v16sf)b, 0); }
/* c - a * b, one rounding. */
static inline v16f knc_fnma(v16f a, v16f b, v16f c)
{
    return (v16f)__builtin_ia32_knc_vfnmadd231ps_zzz((__v16sf)c, KNC_ALL, (__v16sf)a, (__v16sf)b, 0);
}
static inline v16f knc_max(v16f a, v16f b) { return (v16f)__builtin_ia32_knc_vgmaxps_zzz(KNC_U, KNC_ALL, (__v16sf)a, (__v16sf)b, 0); }
static inline v16f knc_min(v16f a, v16f b) { return (v16f)__builtin_ia32_knc_vgminps_zzz(KNC_U, KNC_ALL, (__v16sf)a, (__v16sf)b, 0); }
/* The hardware's F32 functions, measured on this card (2026-10-25): vrcp23ps
 * max 0.90 ULP, vrsqrt23ps 0.70, vlog2ps 1.04 (abs 9.5e-8 in [0.5, 2]), 2^x
 * through Q8.24 + vexp223ps 1.17. The conversion saturates: x below -128 gives
 * +0, at or above 128 gives +inf. Not bit-reproduced by the host model. */
#define KNC_FXP_8_24_RN 0x50
static inline v16f knc_rcp(v16f x) { return (v16f)__builtin_ia32_knc_vrcp23ps_zz(KNC_U, KNC_ALL, (__v16sf)x, 0); }
static inline v16f knc_rsqrt(v16f x) { return (v16f)__builtin_ia32_knc_vrsqrt23ps_zz(KNC_U, KNC_ALL, (__v16sf)x, 0); }
static inline v16f knc_log2(v16f x) { return (v16f)__builtin_ia32_knc_vlog2ps_zz(KNC_U, KNC_ALL, (__v16sf)x, 0); }
static inline v16f knc_exp2(v16f x)
{
    __v16si q = (__v16si)__builtin_ia32_knc_vcvtfxpntps2dq_zzi((__v16si)_mm512_undefined_epi32(), KNC_ALL, (__v16sf)x, KNC_FXP_8_24_RN, 0);
    return (v16f)__builtin_ia32_knc_vexp223ps_zz(KNC_U, KNC_ALL, (__v16sf)q, 0);
}
/* Every lane = the sum of the 16 lanes (knc_hsum's tree, kept as a vector). */
static inline v16f knc_hsum_v(v16f v)
{
    v = (v16f)__builtin_ia32_knc_vaddps_zzz(KNC_U, KNC_ALL, (__v16sf)v, (__v16sf)v, KNC_SWZ_CDAB);
    v = (v16f)__builtin_ia32_knc_vaddps_zzz(KNC_U, KNC_ALL, (__v16sf)v, (__v16sf)v, KNC_SWZ_BADC);
    v = knc_add(v, (v16f)__builtin_ia32_knc_vpermf32x4_zzi(KNC_U, KNC_ALL, (__v16sf)v, KNC_PERM_SWAP1));
    return knc_add(v, (v16f)__builtin_ia32_knc_vpermf32x4_zzi(KNC_U, KNC_ALL, (__v16sf)v, KNC_PERM_SWAP2));
}
/* Every lane = the largest of the 16 lanes. */
static inline v16f knc_hmax_v(v16f v)
{
    v = (v16f)__builtin_ia32_knc_vgmaxps_zzz(KNC_U, KNC_ALL, (__v16sf)v, (__v16sf)v, KNC_SWZ_CDAB);
    v = (v16f)__builtin_ia32_knc_vgmaxps_zzz(KNC_U, KNC_ALL, (__v16sf)v, (__v16sf)v, KNC_SWZ_BADC);
    v = knc_max(v, (v16f)__builtin_ia32_knc_vpermf32x4_zzi(KNC_U, KNC_ALL, (__v16sf)v, KNC_PERM_SWAP1));
    return knc_max(v, (v16f)__builtin_ia32_knc_vpermf32x4_zzi(KNC_U, KNC_ALL, (__v16sf)v, KNC_PERM_SWAP2));
}

#else /* the host model */

#include <math.h>
#include <string.h>

typedef struct {
    float l[16];
} v16f;

/* IEEE binary16 to binary32, exact for every value (NaN payload shifted
 * into place, quiet bit untouched). */
static inline float knc_half(uint16_t h)
{
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, u;
    float f;
    if (e == 31) {
        u = s | 0x7f800000u | m << 13;
    } else if (e != 0) {
        u = s | (e + 112) << 23 | m << 13;
    } else {
        f = ldexpf((float)m, -24);
        return s ? -f : f;
    }
    memcpy(&f, &u, 4);
    return f;
}

static inline v16f knc_zero(void)
{
    v16f r;
    memset(&r, 0, sizeof r);
    return r;
}
static inline v16f knc_load(const float *p)
{
    v16f r;
    memcpy(r.l, p, sizeof r.l);
    return r;
}
static inline void knc_store(float *p, v16f v) { memcpy(p, v.l, sizeof v.l); }
static inline v16f knc_bcast(const float *p)
{
    v16f r;
    for (int i = 0; i < 16; i++)
        r.l[i] = p[0];
    return r;
}
static inline v16f knc_load_s8(const int8_t *p)
{
    v16f r;
    for (int i = 0; i < 16; i++)
        r.l[i] = (float)p[i];
    return r;
}
static inline v16f knc_add(v16f a, v16f b)
{
    v16f r;
    for (int i = 0; i < 16; i++)
        r.l[i] = a.l[i] + b.l[i];
    return r;
}
static inline v16f knc_fma(v16f a, v16f b, v16f c)
{
    v16f r;
    for (int i = 0; i < 16; i++)
        r.l[i] = fmaf(a.l[i], b.l[i], c.l[i]);
    return r;
}
static inline v16f knc_fma_u8(v16f a, const uint8_t *p, v16f c)
{
    v16f r;
    for (int i = 0; i < 16; i++)
        r.l[i] = fmaf(a.l[i], (float)p[i], c.l[i]);
    return r;
}
static inline v16f knc_mul_u8(v16f a, const uint8_t *p)
{
    v16f r;
    for (int i = 0; i < 16; i++)
        r.l[i] = a.l[i] * (float)p[i];
    return r;
}
static inline v16f knc_fma_f16(v16f a, const uint16_t *p, v16f c)
{
    v16f r;
    for (int i = 0; i < 16; i++)
        r.l[i] = fmaf(a.l[i], knc_half(p[i]), c.l[i]);
    return r;
}
static inline v16f knc_fma_b16(v16f a, const float *p, v16f c)
{
    v16f r;
    for (int i = 0; i < 16; i++)
        r.l[i] = fmaf(a.l[i], p[0], c.l[i]);
    return r;
}
static inline float knc_hsum(v16f v)
{
    for (int k = 1; k < 16; k <<= 1) {
        v16f t;
        for (int i = 0; i < 16; i++)
            t.l[i] = v.l[i] + v.l[i ^ k];
        v = t;
    }
    return v.l[0];
}
static inline void knc_prefetch0(const void *p) { (void)p; }
static inline void knc_prefetch1(const void *p) { (void)p; }
/* on the host a spin must give its SMT sibling the core: the sibling may be doing the work */
static inline void knc_delay(unsigned n)
{
    (void)n;
    __builtin_ia32_pause();
}
#define KNC_LANES(expr)                                                                                                \
    do {                                                                                                               \
        for (int i = 0; i < 16; i++)                                                                                   \
            r.l[i] = (expr);                                                                                           \
    } while (0)
static inline v16f knc_sub(v16f a, v16f b) { v16f r; KNC_LANES(a.l[i] - b.l[i]); return r; }
static inline v16f knc_mul(v16f a, v16f b) { v16f r; KNC_LANES(a.l[i] * b.l[i]); return r; }
static inline v16f knc_load_4(const float *p) { v16f r; KNC_LANES(p[i & 3]); return r; }
static inline v16f knc_fma_swz(v16f a, v16f b, v16f c, int s) { v16f r; KNC_LANES(fmaf(a.l[i], b.l[(i & ~3) | (s - 4)], c.l[i])); return r; }
#define KNC_FMA_SWZ(a, b, c, s) knc_fma_swz(a, b, c, s)
static inline v16f knc_fnma(v16f a, v16f b, v16f c) { v16f r; KNC_LANES(fmaf(-a.l[i], b.l[i], c.l[i])); return r; }
static inline v16f knc_max(v16f a, v16f b) { v16f r; KNC_LANES(fmaxf(a.l[i], b.l[i])); return r; }
static inline v16f knc_min(v16f a, v16f b) { v16f r; KNC_LANES(fminf(a.l[i], b.l[i])); return r; }
static inline v16f knc_rcp(v16f x) { v16f r; KNC_LANES(1.0f / x.l[i]); return r; }
static inline v16f knc_rsqrt(v16f x) { v16f r; KNC_LANES(1.0f / sqrtf(x.l[i])); return r; }
static inline v16f knc_log2(v16f x) { v16f r; KNC_LANES(log2f(x.l[i])); return r; }
static inline v16f knc_exp2(v16f x) { v16f r; KNC_LANES(exp2f(x.l[i])); return r; }
static inline v16f knc_hsum_v(v16f v)
{
    for (int k = 1; k < 16; k <<= 1) {
        v16f t;
        for (int i = 0; i < 16; i++)
            t.l[i] = v.l[i] + v.l[i ^ k];
        v = t;
    }
    return v;
}
static inline v16f knc_hmax_v(v16f v)
{
    for (int k = 1; k < 16; k <<= 1) {
        v16f t;
        for (int i = 0; i < 16; i++)
            t.l[i] = fmaxf(v.l[i], v.l[i ^ k]);
        v = t;
    }
    return v;
}

#endif
#endif
