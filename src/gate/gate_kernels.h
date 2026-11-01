/* gate_kernels.h: the entry points of the gate kernels, as the programs
 * that call them see them. Only pointers and integers cross this boundary:
 * the kernels are compiled by the KNC clang (card) or the host cc (host
 * model) against knc.h, the callers by knc-cc (x87 ABI) or the host cc, and
 * no v16f may ever be passed by value between the two ABIs.
 *
 * Every n is a multiple of 16. Vector arrays are 64-byte aligned; the
 * up-converting operands (q, u, h) need only their operand alignment (16 B
 * for 8-bit, 32 B for 16-bit: gate 2.1 probes exactly that), and the
 * broadcast operand s needs 4 B. */
#ifndef APL_GATE_KERNELS_H
#define APL_GATE_KERNELS_H
#include <stdint.h>

/* Gate A: c[i] = a[i] * b[i] + c[i], one rounding. */
void apl_gate_fma16(const float *a, const float *b, float *c, long n);

/* Gate 2.1: one kernel per memory-operand conversion; out[i] = ... */
void k_s8(const int8_t *q, const float *x, const float *c, float *out, long n);     /* fma(x, (float)q, c), q readable to q + n + 256 */
void k_u8fma(const uint8_t *u, const float *x, const float *c, float *out, long n); /* fma(x, (float)u, c), uint8 fused */
void k_u8mul(const uint8_t *u, const float *x, const float *c, float *out, long n); /* x * (float)u, then + c: two roundings */
void k_b16(const float *s, const float *x, const float *c, float *out, long n);     /* fma(x, s[i / 16], c), {1to16} fused */
void k_bcast(const float *s, const float *x, const float *c, float *out, long n);   /* fma(x, s[i / 16], c), vbroadcastss load */
void k_f16(const uint16_t *h, const float *x, const float *c, float *out, long n);  /* fma(x, half(h), c), float16 fused */
void k_hsum(const float *x, float *out, long n);                                     /* out[i / 16] = sum of x[i .. i + 15], tree order */

#endif
