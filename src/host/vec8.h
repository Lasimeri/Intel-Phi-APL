/* vec8.h: the host's AVX2 lane functions, shared by rt_host.c and the fused loops the compiler
 * generates (src/apl/compiler/fz.apl). exp and log hold about 2 ULP by their truncation error
 * (2^f by its series to degree 6 on [-0.5, 0.5]; ln m by atanh's series to s^9 for |s| <= 0.172),
 * not correctly rounded; the card's hardware functions are 1.04 to 1.17 ULP. Held to the F64
 * reference by the engine's gate. */
#ifndef APL_VEC8_H
#define APL_VEC8_H

#include <immintrin.h>

/* ---- e^x */
static inline __m256 exp8(__m256 x)
{
    const __m256 z = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));
    const __m256 c = _mm256_max_ps(_mm256_min_ps(z, _mm256_set1_ps(127.99f)), _mm256_set1_ps(-126.0f));
    const __m256 n = _mm256_round_ps(c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 f = _mm256_sub_ps(c, n);
    __m256 p = _mm256_set1_ps(1.5403530e-4f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.3333558e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(9.6181291e-3f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(5.5504109e-2f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(2.4022651e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(6.9314718e-1f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

/* ---- ln x, x > 0 and normal */
static inline __m256 log8(__m256 x)
{
    const __m256i bits = _mm256_castps_si256(x);
    /* m in [sqrt(1/2), sqrt(2)): shift the exponent's boundary down by that much */
    const __m256i shifted = _mm256_sub_epi32(bits, _mm256_set1_epi32(0x3f3504f3));
    const __m256i ei = _mm256_srai_epi32(shifted, 23);
    const __m256i mi = _mm256_add_epi32(_mm256_and_si256(shifted, _mm256_set1_epi32(0x007fffff)), _mm256_set1_epi32(0x3f3504f3));
    const __m256 m = _mm256_castsi256_ps(mi);
    const __m256 e = _mm256_cvtepi32_ps(ei);
    const __m256 s = _mm256_div_ps(_mm256_sub_ps(m, _mm256_set1_ps(1.0f)), _mm256_add_ps(m, _mm256_set1_ps(1.0f)));
    const __m256 s2 = _mm256_mul_ps(s, s);
    __m256 p = _mm256_set1_ps(2.0f / 9.0f);
    p = _mm256_fmadd_ps(p, s2, _mm256_set1_ps(2.0f / 7.0f));
    p = _mm256_fmadd_ps(p, s2, _mm256_set1_ps(2.0f / 5.0f));
    p = _mm256_fmadd_ps(p, s2, _mm256_set1_ps(2.0f / 3.0f));
    p = _mm256_fmadd_ps(p, s2, _mm256_set1_ps(2.0f));
    return _mm256_fmadd_ps(e, _mm256_set1_ps(0.6931471805599453f), _mm256_mul_ps(p, s));
}

static inline float hsum8(__m256 v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

#endif
