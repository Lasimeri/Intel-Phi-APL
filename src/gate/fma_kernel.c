/* fma_kernel.c: gate A, the vector half. c[i] = a[i] * b[i] + c[i], 16
 * lanes a step, one rounding: the +.x step of an inner product. Through
 * knc.h this is vfmadd231ps on the card and fmaf per lane on the host, so
 * the same file is the card kernel and its own bit-exact model. */
#include "gate_kernels.h"
#include "knc.h"

void apl_gate_fma16(const float *a, const float *b, float *c, long n)
{
    for (long i = 0; i < n; i += 16)
        knc_store(c + i, knc_fma(knc_load(a + i), knc_load(b + i), knc_load(c + i)));
}
