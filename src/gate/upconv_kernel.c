/* upconv_kernel.c: gate 2.1, the vector half. One kernel per memory-operand
 * conversion the Q8 matmul variants are built from, written once against
 * knc.h: on the card each line is the named KNC instruction, on the host
 * the lane-exact model of it. Contracts in gate_kernels.h.
 *
 *   k_s8     vmovaps {sint8} load, then vfmadd231ps (the FMA itself has no
 *            sint8 form: SemaX86KNC.inc mask 0xdfdf), vprefetch0 ahead
 *   k_u8fma  vfmadd231ps with a {uint8} memory operand: dequant fused
 *   k_u8mul  vmulps with a {uint8} memory operand, then vaddps
 *   k_b16    vfmadd231ps with a {1to16} memory operand (per-block scale)
 *   k_bcast  vbroadcastss, then the register vfmadd231ps
 *   k_f16    vfmadd231ps with a {float16} memory operand (Q8_0 block scales)
 *   k_hsum   the 16-lane reduction tree: swizzle adds i^1, i^2, then
 *            vpermf32x4 adds i^4, i^8 */
#include "gate_kernels.h"
#include "knc.h"

void k_s8(const int8_t *q, const float *x, const float *c, float *out, long n)
{
    for (long i = 0; i < n; i += 16) {
        knc_prefetch0(q + i + 256);
        knc_store(out + i, knc_fma(knc_load(x + i), knc_load_s8(q + i), knc_load(c + i)));
    }
}

void k_u8fma(const uint8_t *u, const float *x, const float *c, float *out, long n)
{
    for (long i = 0; i < n; i += 16)
        knc_store(out + i, knc_fma_u8(knc_load(x + i), u + i, knc_load(c + i)));
}

void k_u8mul(const uint8_t *u, const float *x, const float *c, float *out, long n)
{
    for (long i = 0; i < n; i += 16)
        knc_store(out + i, knc_add(knc_mul_u8(knc_load(x + i), u + i), knc_load(c + i)));
}

void k_b16(const float *s, const float *x, const float *c, float *out, long n)
{
    for (long i = 0; i < n; i += 16)
        knc_store(out + i, knc_fma_b16(knc_load(x + i), s + i / 16, knc_load(c + i)));
}

void k_bcast(const float *s, const float *x, const float *c, float *out, long n)
{
    for (long i = 0; i < n; i += 16)
        knc_store(out + i, knc_fma(knc_load(x + i), knc_bcast(s + i / 16), knc_load(c + i)));
}

void k_f16(const uint16_t *h, const float *x, const float *c, float *out, long n)
{
    for (long i = 0; i < n; i += 16)
        knc_store(out + i, knc_fma_f16(knc_load(x + i), h + i, knc_load(c + i)));
}

void k_hsum(const float *x, float *out, long n)
{
    for (long i = 0; i < n; i += 16)
        out[i / 16] = knc_hsum(knc_load(x + i));
}
